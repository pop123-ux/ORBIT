from __future__ import annotations

import math

import torch

from .baselines import Muon
from .optimizer import newton_schulz

PAPER_BASELINE_NAMES = ("adamw", "muon", "normuon", "adamuon")


class PaperAdamW(torch.optim.AdamW):
    """AdamW with the paper campaign's GPT decay routing.

    Two-dimensional parameters decay; vectors, biases, and normalization
    parameters do not. This mirrors the shared routing used in the paper.
    """

    def __init__(
        self,
        model,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.05,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        setter = getattr(model, "set_orbit_stat_collection", None)
        if callable(setter):
            setter(False)
        decay, nodecay = [], []
        for param in model.parameters():
            (decay if param.ndim >= 2 else nodecay).append(param)
        groups = []
        if nodecay:
            groups.append({"params": nodecay, "weight_decay": 0.0})
        if decay:
            groups.append({"params": decay, "weight_decay": weight_decay})
        super().__init__(groups, lr=lr, betas=(0.9, beta2), eps=eps)


class NorMuon(Muon):
    """Paper NorMuon: post-orthogonalization row adaptation with norm restoration."""

    def _spectral_candidate(self, param: torch.Tensor, group: dict) -> torch.Tensor:
        grad = param.grad
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        momentum = state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
        lookahead = grad.add(momentum, alpha=group["momentum"])
        update = newton_schulz(lookahead, group["ns_steps"])

        step = int(state.get("normuon_step", 0)) + 1
        state["normuon_step"] = step
        if "row_second_moment" not in state:
            state["row_second_moment"] = torch.zeros(
                update.size(0), dtype=update.dtype, device=update.device
            )
        beta2 = group["betas"][1]
        row = state["row_second_moment"]
        row.mul_(beta2).add_(update.square().mean(dim=1), alpha=1.0 - beta2)
        denominator = (row / (1.0 - beta2**step)).sqrt().add_(group["eps"])
        scaled = update / denominator.unsqueeze(1)
        update = scaled * (update.norm() / scaled.norm().clamp_min(1e-12))

        update.mul_(max(1.0, update.size(0) / update.size(1)) ** 0.5)
        return update.to(param.dtype)


class PublishedAdaMuon(Muon):
    """Reference AdaMuon rule used by the paper campaign.

    The matrix path applies Nesterov momentum, sign before Newton--Schulz,
    an elementwise post-polar second moment without bias correction, and
    0.2-RMS alignment. The non-matrix path is the same auxiliary AdamW routing
    used by the other Muon-family controls in the paper.
    """

    def _spectral_candidate(self, param: torch.Tensor, group: dict) -> torch.Tensor:
        grad = param.grad
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        momentum = state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
        lookahead = grad.add(momentum, alpha=group["momentum"])
        update = newton_schulz(torch.sign(lookahead), group["ns_steps"])

        if "adamuon_second_moment" not in state:
            state["adamuon_second_moment"] = torch.zeros_like(update)
        variance = state["adamuon_second_moment"]
        beta = group["momentum"]
        variance.mul_(beta).addcmul_(update, update, value=1.0 - beta)
        update = update / variance.sqrt().add_(group["eps"])

        rms_scale = (
            0.2
            * math.sqrt(update.size(0) * update.size(1))
            / update.norm().clamp_min(group["eps"])
        )
        return (update * rms_scale).to(param.dtype)


AdaMuon = PublishedAdaMuon


def build_paper_baseline(
    name: str,
    model,
    *,
    lr: float,
    scalar_lr_mult: float = 0.1,
    weight_decay: float = 0.05,
    beta2: float = 0.95,
):
    """Build one of the non-ASTRO baselines used in the ORBIT paper.

    This exposes the tested update definitions without recreating the paper's
    full campaign orchestration, tuning grid, or evidence pipeline.
    """

    key = name.lower()
    if key == "adamuon_ref":
        key = "adamuon"
    if key not in PAPER_BASELINE_NAMES:
        raise ValueError(
            f"unknown paper baseline {name!r}; choose from {', '.join(PAPER_BASELINE_NAMES)}"
        )

    if key == "adamw":
        return PaperAdamW(
            model,
            lr=lr,
            weight_decay=weight_decay,
            beta2=beta2,
        )

    adamw_lr = lr * scalar_lr_mult
    common = dict(
        lr=lr,
        adamw_lr=adamw_lr,
        weight_decay=weight_decay,
        momentum=0.95,
        ns_steps=5,
        adamw_betas=(0.9, 0.95),
        eps=1e-8,
    )
    if key == "muon":
        return Muon(model, **common)
    if key == "normuon":
        return NorMuon(model, **common)
    return PublishedAdaMuon(model, **common)
