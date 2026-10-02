from __future__ import annotations

import torch

from .optimizer import newton_schulz


class Muon(torch.optim.Optimizer):
    """Muon control with the same parameter-routing policy used by ORBIT.

    Two-dimensional hidden operators use Muon's Newton--Schulz spectral update.
    Embeddings, the tied language-model head, biases, and normalization vectors
    use the auxiliary AdamW path. Unlike ORBIT, this class does not collect Q/K
    covariance statistics and does not apply a function-space preconditioner.
    """

    def __init__(
        self,
        model,
        *,
        lr: float = 0.02,
        adamw_lr: float = 3e-4,
        weight_decay: float = 0.05,
        momentum: float = 0.95,
        ns_steps: int = 5,
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
    ) -> None:
        params = list(model.parameters())
        super().__init__(
            params,
            dict(
                lr=lr,
                adamw_lr=adamw_lr,
                weight_decay=weight_decay,
                momentum=momentum,
                ns_steps=ns_steps,
                betas=adamw_betas,
                eps=eps,
            ),
        )
        self.model = model
        model.set_orbit_stat_collection(False)
        excluded = {id(model.wte.weight), id(model.lm_head.weight)}
        self._meta: dict[int, str] = {}
        for param in params:
            if param.ndim < 2:
                self._meta[id(param)] = "aux_nodecay"
            elif id(param) in excluded:
                self._meta[id(param)] = "aux_decay"
            else:
                self._meta[id(param)] = "spectral"

    def _spectral_candidate(self, param: torch.Tensor, group: dict) -> torch.Tensor:
        grad = param.grad
        state = self.state[param]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(grad)
        momentum = state["momentum_buffer"].mul_(group["momentum"]).add_(grad)
        lookahead = grad.add(momentum, alpha=group["momentum"])
        update = newton_schulz(lookahead, group["ns_steps"])
        update.mul_(max(1.0, update.size(0) / update.size(1)) ** 0.5)
        return update.to(param.dtype)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None if closure is None else closure()
        group = self.param_groups[0]
        candidates: dict[int, torch.Tensor] = {}

        for param in group["params"]:
            if param.grad is None:
                continue
            if self._meta[id(param)] == "spectral":
                candidates[id(param)] = self._spectral_candidate(param, group)

        beta1, beta2 = group["betas"]
        for param in group["params"]:
            if param.grad is None:
                continue
            role = self._meta[id(param)]
            if role == "spectral":
                if group["weight_decay"]:
                    param.mul_(1.0 - group["lr"] * group["weight_decay"])
                param.add_(candidates[id(param)], alpha=-group["lr"])
                continue

            state = self.state[param]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(param)
                state["exp_avg_sq"] = torch.zeros_like(param)
            state["step"] += 1
            step = state["step"]
            grad = param.grad
            exp_avg = state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq = state["exp_avg_sq"].mul_(beta2).addcmul_(
                grad, grad, value=1.0 - beta2
            )
            update = (exp_avg / (1.0 - beta1**step)) / (
                (exp_avg_sq / (1.0 - beta2**step)).sqrt().add_(group["eps"])
            )
            if role == "aux_decay" and group["weight_decay"]:
                param.mul_(1.0 - group["adamw_lr"] * group["weight_decay"])
            param.add_(update, alpha=-group["adamw_lr"])
        return loss
