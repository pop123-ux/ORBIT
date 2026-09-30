#!/usr/bin/env python3
"""ASTRO lab: what the optimizer adds, and whether the additions survive scale.

    !pip -q install transformers datasets
    !python astro_lab.py --mode components          # seconds, no GPU
    !python astro_lab.py --mode scaling --sizes 124M --steps 300 900 \
        --config lr=0.0144 weight_decay=0.02 --seeds 2 --max-minutes 150

Why this exists
---------------
Every ASTRO result so far is 300 steps at 124M -- roughly 0.0005x the
Chinchilla-optimal token budget. That is the regime where optimizer advantages
are widest and least durable: Wen et al. (2509.02046) attribute most published
speedups to exactly this, and we have already watched one component (the
cautious mask) reverse sign between 1.17M and 124M. A 0.04-nat lead at one point
on one axis is a finding, not a result.

So this script sweeps the two axes that decide whether the lead is real:

* **model size** -- 45M to 774M, with the batch shrunk to fit a 16 GB T4 and
  gradient checkpointing above 124M;
* **training length** -- the same cell at 300, 900, 2700 steps.

The artifact is the *trend*, not any single cell. An advantage that decays
toward zero along either axis is an early-training or small-model effect and
should be reported as one.

Modes
-----
``--mode components`` prints the component manifest: every addition, where it
came from, and what the evidence for it currently is. Five of nine have no
measured effect on loss, and saying so is the point -- a paper may only claim
the ones with evidence attached.

``--mode scaling`` runs the grid. Every run is checkpointed to
``astro_lab_state.json`` the moment it finishes, so a dropped Colab session
resumes rather than restarts, and ``--max-minutes`` stops cleanly before a
session limit does and still writes a report.

Cost on a free T4
-----------------
Measured at 1.10 s/step for 124M at batch 8, sequence 512. Time scales with
parameters x batch, so one run is roughly:

    size    batch   300 steps   900    2700
    124M      8         6m      17m     50m
    355M      4         8m      24m     71m
    774M      2         9m      26m     77m

Four optimizers x two seeds multiplies that by eight, which is why the grid is
meant to be run in pieces across sessions rather than in one go.

Fairness
--------
Every optimizer tunes the same number of hyperparameters from ranges that follow
its own update scale -- a Muon-scaled update needs a range about 10x higher than
an Adam-scaled one, and pairing the wrong range with the wrong scale is a silent
handicap this project has shipped twice. With ``--config`` every optimizer
instead runs at one fixed configuration, which is weaker but honest and is
stated in the report.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import random
import statistics
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import torch

#: Resolved once, at import, because --work-dir changes the process's working
#: directory and ``__file__`` is relative when the script is invoked by name.
SOURCE = Path(__file__).resolve()

# ---------------------------------------------------------------------------
# Vendored optimizers, copied verbatim from the tested source so this file runs
# alone. tests/test_astro_lab.py pins the copy against astro.optimizer.
# ---------------------------------------------------------------------------

#: Solved per-step coefficients for a *converging* polar iteration.
#:
#: Muon's quintic is one polynomial applied k times and cannot reach sigma = 1
#: at any budget: its fixed points solve 2.4445 - 4.7750 s^2 + 2.0315 s^4 = 0,
#: at 0.868 and 1.264. A different polynomial per step has no such obstruction.
#: Greedily solved, so a shorter schedule is a prefix of a longer one. The last
#: two entries land near (1.875, -1.25, 0.375), independently reproducing the
#: asymptote the Polar Express publishes.
POLAR_SCHEDULE: tuple[tuple[float, float, float], ...] = (
    (5.741408, -17.016317, 12.623472),
    (4.240444, -6.859093, 2.787935),
    (4.186216, -6.613335, 2.669455),
    (3.958440, -5.645446, 2.206946),
    (2.621392, -2.503740, 0.833594),
    (1.889525, -1.266059, 0.376621),
    (1.777582, -1.055164, 0.277581),
)


def polar_iterate(matrix: torch.Tensor, steps: int = 5, converging: bool = False):
    """Polar factor by Newton-Schulz, optionally with the converging schedule.

    Always float32: a T4 is sm_75 with no bfloat16, and float16 here degrades
    exactly the small singular values the measurement is about.
    """
    x = matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    schedule = (
        POLAR_SCHEDULE[:steps] if converging else ((3.4445, -4.7750, 2.0315),) * steps
    )
    for a, b, c in schedule:
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transposed else x

def newton_schulz(matrix: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Muon's quintic iteration for the polar factor ``msgn(A) = U V^T``.

    Coefficients are Jordan's (3.4445, -4.7750, 2.0315). Always computed in
    float32: on a T4 there is no bfloat16, and float16 here degrades the small
    singular values silently, which is precisely the regime the measurement is
    about.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    x = matrix.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    for _ in range(steps):
        gram = x @ x.T
        correction = b * gram + c * (gram @ gram)
        x = a * x + correction @ x
    return x.T if transposed else x

def cautious_mask(update: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    """Liang et al.'s mask: zero the coordinates that disagree with the gradient."""
    mask = (update * grad > 0).to(update.dtype)
    return update * mask * (mask.numel() / mask.sum().clamp_min(1e-12))


def apply_weight_decay(param, update, rate, *, cautious, rescale=False):
    """Cautious weight decay (Chen et al.): decay only where the step already
    carries the weight toward zero.

    The mask *deletes* decay rather than redistributing it -- measured, the
    condition holds for 0.38 of coordinates, so at one nominal weight_decay a
    masked optimizer gets 38% of the decay an unmasked one does, and because
    decay compounds the weight-norm difference grows without bound in the step
    count (1.06x at 300 steps, 1.62x at 2700). ``rescale`` applies the
    numel/count convention that ``cautious_mask`` already uses, concentrating
    the decay instead of thinning it.
    """
    if rate == 0.0:
        return
    if not cautious:
        param.mul_(1.0 - rate)
        return
    agrees = update * param > 0
    if rescale:
        kept = float(agrees.sum())
        if kept <= 0.0:
            return
        rate = rate * agrees.numel() / kept
    param.add_(torch.where(agrees, param, torch.zeros_like(param)), alpha=-rate)

class Astro(torch.optim.Optimizer):
    """ASTRO's shipped recipe, inlined so the script runs from a bare Colab.

    Faithful to ``astro.optimizer.Astro``'s defaults at every setting that
    matters here: Nesterov momentum at 0.95, the neuron-wise second moment
    applied *after* orthogonalisation and rescaled to preserve the update norm,
    Muon's aspect-ratio update scale applied per block, the cautious mask, the
    QKV split, and cautious weight decay.

    An earlier version of this file inlined the *previous* defaults --
    pre-orthogonalisation variance and Adam-style scaling -- which is the
    configuration the 124M run measured and which we already knew was 0.041
    worse on the CPU benchmark. Keeping this in step with the library is the
    whole point of the class, so it is asserted against it in
    ``tests/test_colab_probe.py``.
    """

    def __init__(self, groups, lr=1e-3, scalar_lr_mult=0.1, betas=(0.95, 0.95),
                 eps=1e-8, weight_decay=0.01, variance_power=1.0,
                 post_normalize=False, cautious_wd=True, converging=False,
                 equilibrate=False, ns_steps=5, cautious=False,
                 update_scale="muon", trust_clip=1e3, split_steps=None,
                 cautious_wd_rescale=False):
        # The scalar path needs its own rate. A Muon-scaled spectral update is
        # 4.4x smaller in Frobenius norm than an Adam-like one at 768 width, and
        # the gap grows with width, so running embeddings and the tied head at
        # the spectral rate overshoots them badly. Muon's reference carries a
        # separate adamw_lr for exactly this reason; an earlier revision of this
        # file gave every group one rate, which handed the baseline an advantage
        # over 31.7% of a GPT-2's parameters.
        for group in groups:
            group.setdefault("lr", lr if group.get("spectral") else lr * scalar_lr_mult)
        super().__init__(groups, dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            variance_power=variance_power, post_normalize=post_normalize,
            cautious_wd=cautious_wd, converging=converging,
            equilibrate=equilibrate, ns_steps=ns_steps, cautious=cautious,
            update_scale=update_scale, trust_clip=trust_clip,
            split_steps=split_steps, cautious_wd_rescale=cautious_wd_rescale,
        ))

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                # HuggingFace GPT-2 stores every projection as a ``Conv1D``,
                # whose weight is (in, out) -- the transpose of the operator. Row
                # statistics, the RMS scale and the QKV split are all defined on
                # the operator, so the whole spectral path runs on a transposed
                # view and the result is transposed back before the write.
                operator = grad.T if group.get("transposed") else grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(operator)
                    if group["spectral"]:
                        state["variance"] = torch.zeros(operator.size(0), device=param.device)
                    else:
                        state["variance"] = torch.zeros_like(operator)
                state["step"] += 1
                step = state["step"]

                momentum = state["momentum"].mul_(beta1).add_(operator, alpha=1 - beta1)

                if group["spectral"]:
                    # Nesterov: the momentum the buffer *would* have after
                    # absorbing this gradient, which is Muon's default.
                    lookahead = operator.lerp(momentum, beta1)
                    if group["equilibrate"]:
                        # MuonEq's R variant: equalise the momentum's row norms
                        # before the polar step, rescaled to preserve its norm.
                        norms = lookahead.norm(dim=1, keepdim=True).clamp_min(group["eps"])
                        balanced = lookahead / norms
                        lookahead = balanced * (
                            lookahead.norm() / balanced.norm().clamp_min(group["eps"])
                        )
                    sizes = group.get("blocks") or (operator.size(0),)
                    # The fused-projection defect is an initialisation-time
                    # phenomenon that halves by the time weights are trained,
                    # and the split costs 1.33x measured, so a schedule relaxes
                    # to a single block once the repair has done its work.
                    limit = group["split_steps"]
                    if limit is not None and step > limit:
                        sizes = (operator.size(0),)
                    pieces = []
                    for chunk in lookahead.split(list(sizes), 0):
                        filtered = polar_iterate(
                            chunk, group["ns_steps"], group["converging"]
                        )
                        if group["post_normalize"]:
                            target = math.sqrt(min(chunk.size(0), chunk.size(1)))
                            filtered = filtered * (target / filtered.norm().clamp_min(1e-12))
                        pieces.append(filtered)
                    direction = torch.cat(pieces, dim=0)

                    # Second moment of the *orthogonalised* update, not of the
                    # gradient: the polar step exists to remove the gradient's
                    # ill-conditioning, so accumulating on the raw gradient
                    # measures the wrong quantity.
                    variance = state["variance"]
                    variance.mul_(beta2).add_(direction.pow(2).mean(dim=1), alpha=1 - beta2)
                    moment = variance / (1 - beta2**step)
                    denominator = moment.pow(0.5 * group["variance_power"]).add_(group["eps"])
                    scaled = direction / denominator.unsqueeze(1)
                    # Norm-preserving: this sets the distribution of step length
                    # across neurons, not the total step size.
                    direction = scaled * (direction.norm() / scaled.norm().clamp_min(1e-12))

                    if group["update_scale"] == "trust":
                        # Angular learning rate: the step is a fixed fraction of
                        # the layer's own norm rather than a function of its
                        # shape. The learning rate *is* the trust ratio, so
                        # there is no second coefficient to tune.
                        reference = param.T if group.get("transposed") else param
                        factor = (reference.norm()
                                  / direction.norm().clamp_min(1e-12))
                        clip = group["trust_clip"]
                        direction = (direction * factor.clamp(1.0 / clip, clip)
                                     ).to(param.dtype)
                    else:
                        # Muon's aspect-ratio scale, per block: under
                        # grouped-query attention the blocks have different row
                        # counts.
                        offset, blocks = 0, []
                        for rows in sizes:
                            piece = direction[offset : offset + rows]
                            blocks.append(piece * max(1.0, rows / direction.size(1)) ** 0.5)
                            offset += rows
                        direction = torch.cat(blocks, dim=0).to(param.dtype)
                else:
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(operator, operator, value=1 - beta2)
                    denominator = (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    direction = (momentum / (1 - beta1**step)) / denominator

                if group.get("transposed"):
                    direction = direction.T
                if group["cautious"]:
                    direction = cautious_mask(direction, grad)
                apply_weight_decay(
                    param, direction, group["lr"] * group["weight_decay"],
                    cautious=group["cautious_wd"],
                    rescale=group["cautious_wd_rescale"],
                )
                param.add_(direction, alpha=-group["lr"])

class Muon(torch.optim.Optimizer):
    """Muon with Nesterov momentum and the ``max(1, m/n)^0.5`` scale, AdamW elsewhere."""

    def __init__(self, groups, lr=0.02, adamw_lr=3e-4, momentum=0.95,
                 betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01):
        super().__init__(groups, dict(lr=lr, adamw_lr=adamw_lr, momentum=momentum,
                                      betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                operator = grad.T if group.get("transposed") else grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(operator)
                    if not group["spectral"]:
                        state["variance"] = torch.zeros_like(operator)
                state["step"] += 1

                if group["spectral"]:
                    buffer = state["momentum"].mul_(group["momentum"]).add_(operator)
                    nesterov = operator.add(buffer, alpha=group["momentum"])
                    direction = newton_schulz(nesterov).to(param.dtype)
                    direction = direction * math.sqrt(
                        max(1.0, operator.size(0) / operator.size(1))
                    )
                    if group.get("transposed"):
                        direction = direction.T
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(direction, alpha=-group["lr"])
                else:
                    beta1, beta2 = group["betas"]
                    step = state["step"]
                    momentum = state["momentum"].mul_(beta1).add_(grad, alpha=1 - beta1)
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    update = (momentum / (1 - beta1**step)) / (
                        (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    )
                    param.mul_(1 - group["adamw_lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["adamw_lr"])

class NorMuon(Muon):
    """Muon plus a neuron-wise second moment applied *after* orthogonalisation.

    Li et al. (arXiv:2510.05491). Muon equalises singular values, which is a
    statement about directions and says nothing about how far any individual
    output neuron moves; NorMuon adds that. The moment is accumulated on the
    orthogonalised update rather than on the gradient, because the polar step
    exists precisely to remove the gradient's conditioning, and the rescale is
    norm-preserving so the rule changes the distribution of step length across
    neurons without changing the total step.
    """

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                operator = grad.T if group.get("transposed") else grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(operator)
                    if group["spectral"]:
                        state["row"] = torch.zeros(operator.size(0), device=param.device)
                    else:
                        state["variance"] = torch.zeros_like(operator)
                state["step"] += 1
                step = state["step"]

                if group["spectral"]:
                    buffer = state["momentum"].mul_(group["momentum"]).add_(operator)
                    nesterov = operator.add(buffer, alpha=group["momentum"])
                    direction = newton_schulz(nesterov)

                    beta2 = group["betas"][1]
                    row = state["row"]
                    row.mul_(beta2).add_(direction.pow(2).mean(dim=1), alpha=1 - beta2)
                    denominator = (row / (1 - beta2**step)).sqrt().add_(group["eps"])
                    scaled = direction / denominator.unsqueeze(1)
                    direction = scaled * (direction.norm() / scaled.norm().clamp_min(1e-12))

                    direction = direction.to(param.dtype) * max(
                        1.0, operator.size(0) / operator.size(1)
                    ) ** 0.5
                    if group.get("transposed"):
                        direction = direction.T
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(direction, alpha=-group["lr"])
                else:
                    beta1, beta2 = group["betas"]
                    momentum = state["momentum"].mul_(beta1).add_(operator, alpha=1 - beta1)
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(operator, operator, value=1 - beta2)
                    update = (momentum / (1 - beta1**step)) / (
                        (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    )
                    if group.get("transposed"):
                        update = update.T
                    param.mul_(1 - group["adamw_lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["adamw_lr"])

class AdaMuon(NorMuon):
    """Muon plus an *elementwise* second moment on the orthogonalised update.

    Si et al. (arXiv:2507.11005). The difference from NorMuon is the axis the
    moment is accumulated on: NorMuon keeps one scalar per output neuron,
    AdaMuon keeps one per weight. Both accumulate on the orthogonalised update
    rather than on the gradient, for the same reason -- the polar step exists to
    remove the gradient's conditioning, so a moment taken before it measures the
    quantity that was about to be discarded.

    The rescale is norm-preserving here too, which is what the paper calls
    RMS-aligned rescaling: without it the elementwise division changes the step
    size as well as its distribution, and the learning rate stops being
    comparable to Muon's.

    Costs one extra tensor the size of every spectral parameter, against
    NorMuon's one float per row. That is the trade the paper is making and it
    is why both are worth having in the table.
    """

    @torch.no_grad()
    def step(self):  # type: ignore[override]
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                operator = grad.T if group.get("transposed") else grad
                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["momentum"] = torch.zeros_like(operator)
                    state["variance"] = torch.zeros_like(operator)
                state["step"] += 1
                step = state["step"]

                if group["spectral"]:
                    buffer = state["momentum"].mul_(group["momentum"]).add_(operator)
                    nesterov = operator.add(buffer, alpha=group["momentum"])
                    direction = newton_schulz(nesterov)

                    beta2 = group["betas"][1]
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(direction, direction, value=1 - beta2)
                    denominator = (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    scaled = direction / denominator
                    direction = scaled * (direction.norm() / scaled.norm().clamp_min(1e-12))

                    direction = direction.to(param.dtype) * max(
                        1.0, operator.size(0) / operator.size(1)
                    ) ** 0.5
                    if group.get("transposed"):
                        direction = direction.T
                    param.mul_(1 - group["lr"] * group["weight_decay"])
                    param.add_(direction, alpha=-group["lr"])
                else:
                    beta1, beta2 = group["betas"]
                    momentum = state["momentum"].mul_(beta1).add_(operator, alpha=1 - beta1)
                    variance = state["variance"]
                    variance.mul_(beta2).addcmul_(operator, operator, value=1 - beta2)
                    update = (momentum / (1 - beta1**step)) / (
                        (variance / (1 - beta2**step)).sqrt().add_(group["eps"])
                    )
                    if group.get("transposed"):
                        update = update.T
                    param.mul_(1 - group["adamw_lr"] * group["weight_decay"])
                    param.add_(update, alpha=-group["adamw_lr"])


def build_groups(model, optimizer_name: str, config) -> list[dict]:
    """Route parameters by module identity, not by name.

    Embeddings, the tied head, norms and biases take the elementwise path; every
    genuine 2-D operator takes the spectral one. Two details matter and both are
    resolved structurally, because the name-based shortcut is wrong on GPT-2:

    ``transposed``
        A HuggingFace ``Conv1D`` stores its weight as (in, out). Every GPT-2
        projection is a ``Conv1D``, so the operator is the transpose and the
        spectral path has to know it.
    ``blocks``
        The fused ``c_attn`` carries Q, K and V in contiguous row blocks of the
        *operator*, which the optimizer splits before orthogonalising.

    Weight tying makes ``wte.weight`` and ``lm_head.weight`` one tensor, so the
    head is excluded by identity rather than by matching the string ``lm_head``,
    which ``named_parameters`` never reports for a tied model.
    """
    try:
        from transformers.pytorch_utils import Conv1D
    except ImportError:  # pragma: no cover - older/newer layouts
        Conv1D = ()  # type: ignore[assignment]

    excluded, transposed_ids, fused_ids = set(), set(), set()
    for module in model.modules():
        if isinstance(module, torch.nn.Embedding):
            excluded.add(id(module.weight))
    output = getattr(model, "lm_head", None)
    if output is not None:
        excluded.add(id(output.weight))

    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if weight is None or weight.ndim != 2:
            continue
        if Conv1D and isinstance(module, Conv1D):
            transposed_ids.add(id(weight))
        if name.endswith("c_attn"):
            fused_ids.add(id(weight))

    width = getattr(config, "n_embd", None)
    buckets: dict[tuple[bool, bool], list[torch.nn.Parameter]] = {}
    scalar: list[torch.nn.Parameter] = []
    fused: list[torch.nn.Parameter] = []
    for param in model.parameters():
        if param.ndim < 2 or id(param) in excluded:
            scalar.append(param)
        elif id(param) in fused_ids and width is not None:
            fused.append(param)
        else:
            buckets.setdefault((True, id(param) in transposed_ids), []).append(param)

    groups: list[dict] = [{"params": scalar, "spectral": False, "transposed": False}]
    for (spectral, transposed), params in buckets.items():
        groups.append({"params": params, "spectral": spectral, "transposed": transposed})
    if fused:
        # ASTRO splits; Muon deliberately does not, since not splitting is the
        # behaviour under test.
        groups.append({
            "params": fused, "spectral": True,
            "transposed": all(id(p) in transposed_ids for p in fused),
            "blocks": (width, width, width) if optimizer_name == "astro" else None,
        })
    return groups



# ---------------------------------------------------------------------------
# What ASTRO adds, and what the evidence for each addition currently is
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Component:
    """One addition to the Muon/NorMuon core, with its provenance and evidence.

    The registry exists so that "what is ASTRO" has a single answer that cannot
    drift from what the code does. ``flag`` is the literal keyword argument, so
    a component that is renamed or removed breaks the manifest rather than
    silently surviving in prose.
    """

    flag: str
    default: object
    origin: str
    what: str
    evidence: str
    status: str  # "kept" | "off" | "unmeasured"


COMPONENTS: tuple[Component, ...] = (
    Component(
        flag="split_fused", default=True, origin="ours (diagnosis); prior art (fix)",
        what="Orthogonalise Q, K and V separately instead of as one fused tensor.",
        evidence="Muon's update puts 65% of its row mass on V at initialisation "
                 "(76% on GPT-2), against 33% for parity, because Q and K reach "
                 "the loss through the softmax Jacobian. Attenuates to 50% on "
                 "trained weights. Measured on 8 checkpoints, 124M-1.4B.",
        status="kept",
    ),
    Component(
        flag="variance_placement", default="post", origin="NorMuon / AdaMuon",
        what="Neuron-wise second moment applied after orthogonalisation, "
             "rescaled to preserve the update norm.",
        evidence="Worth 0.041 over pre-placement on the CPU benchmark. The "
                 "argument is that the polar step exists to remove the "
                 "gradient's conditioning, so accumulating on the raw gradient "
                 "measures the wrong quantity.",
        status="kept",
    ),
    Component(
        flag="update_scale", default="muon", origin="Muon",
        what="Aspect-ratio scale max(1, m/n)^0.5, applied per block.",
        evidence="Per-block is a correctness fix: under grouped-query attention "
                 "the blocks have different shapes, so one scale from the "
                 "largest is wrong for the other two.",
        status="kept",
    ),
    Component(
        flag="cautious_wd", default=True, origin="Chen et al. 2510.12402",
        what="Decay a coordinate only where the step already carries it toward zero.",
        evidence="No new hyperparameter. Not yet isolated at 124M -- it is "
                 "inside the winning recipe but its individual contribution is "
                 "unmeasured.",
        status="unmeasured",
    ),
    Component(
        flag="cautious", default=False, origin="Liang et al. 2411.16085",
        what="Zero update coordinates disagreeing in sign with the gradient, "
             "rescaling survivors by numel/count.",
        evidence="THE SIGN FLIPS WITH SCALE. Worth -0.0291 against Muon on 8/8 "
                 "seeds at 1.17M; costs +0.1341 on 3/3 seeds at 124M. Off by "
                 "default because the target regime is the larger one.",
        status="off",
    ),
    Component(
        flag="converging", default=False, origin="ours (solved); Polar Express (asymptote)",
        what="Per-step polar coefficients that actually converge.",
        evidence="Muon's quintic has fixed points at 0.868 and 1.264 and cannot "
                 "reach sigma=1 at any budget; its update norm drifts 0.688-0.921 "
                 "of target with input conditioning. Solved schedule is exact at "
                 "7 steps. Effect on loss unmeasured.",
        status="unmeasured",
    ),
    Component(
        flag="variance_power", default=1.0, origin="ours",
        what="Interpolates Muon (0) to NorMuon (1) through the induced norm.",
        evidence="Both endpoints published, interior unsearched. Unmeasured.",
        status="unmeasured",
    ),
    Component(
        flag="equilibrate", default=False, origin="MuonEq 2603.28254",
        what="Equalise the momentum's row norms instantaneously before the polar step.",
        evidence="Unmeasured here.", status="unmeasured",
    ),
    Component(
        flag="spectral_blend", default=0.0, origin="ours; Ma et al. 2606.21514",
        what="Blend the unfiltered direction back in, norm-matched.",
        evidence="Motivated by the river-valley result that Muon converges "
                 "slower than gradient descent near the optimum. Unmeasured.",
        status="unmeasured",
    ),
    Component(
        flag="post_normalize", default=False, origin="ours",
        what="Pin each filtered block's norm to the theoretical sqrt(min(m, n)).",
        evidence="Muon's iteration delivers 0.771 of target on a split QKV "
                 "block and about 0.83 fused, and which it delivers depends on "
                 "the conditioning of the momentum -- so the ASTRO/Muon step "
                 "ratio drifts 0.047 over one 40-step run. Pinning makes the "
                 "learning rate name a fixed step size. Turned ON on that "
                 "argument, then turned OFF again when the loss disagreed: at "
                 "124M over 900 steps pinning costs 0.0222. Calibrating the "
                 "step size is not the same as improving the optimizer. NVIDIA "
                 "2606.00371 reached the same conclusion for polar accuracy.",
        status="off",
    ),
    Component(
        flag="update_scale", default="muon", origin="Hyperball; OrScale 2605.07815",
        what="'trust' sets each layer's step from its own norm, not its shape.",
        evidence="Muon's max(1, m/n)^0.5 is a function of shape alone, so two "
                 "identically shaped layers whose weights differ 5x in norm "
                 "undergo relative changes differing 5x. The learning rate then "
                 "IS the trust ratio, so no second knob is added. Registered as "
                 "a swept variant with its own LR range; not default until it "
                 "wins at equal budget.",
        status="unmeasured",
    ),
    Component(
        flag="split_steps", default=None, origin="ours",
        what="Relax the QKV split to a single block after N steps.",
        evidence="The defect the split repairs is an initialisation-time "
                 "phenomenon: V's update share is 0.65 at init against 0.50 "
                 "trained, where uniform is 0.33. Splitting also costs 1.29x by "
                 "operation count and 1.33x measured, so a permanent repair is "
                 "paid for permanently. Schedule unmeasured.",
        status="unmeasured",
    ),
)


#: Variants the scaling study can run. Each is the shipped recipe plus one change.
VARIANTS: dict[str, dict[str, object]] = {
    # ``astro`` now pins the update norm by default (post_normalize=True); the
    # unpinned form is kept as an explicit ablation rather than as the default
    # it used to be. See docs/IMPROVEMENT_LOG.md step 1.
    "astro": {},
    "astro_pinned": {"post_normalize": True},
    # The angular-learning-rate control: each layer's step is set from its own
    # norm rather than its shape. Off by default until it wins at equal budget.
    "astro_trust": {"update_scale": "trust"},
    "astro_cautious": {"cautious": True},
    "astro_converging": {"converging": True},
    "astro_gamma25": {"variance_power": 0.25},
    "astro_gamma50": {"variance_power": 0.5},
    "astro_gamma0": {"variance_power": 0.0},
    "astro_equil": {"equilibrate": True},
    # The decay mask deletes 62% of the decay rather than redistributing it,
    # and decay compounds -- so these two are the direct test of whether the
    # gap that widens with training length is a regularisation mismatch.
    "astro_plain_wd": {"cautious_wd": False},
    "astro_wd_rescaled": {"cautious_wd_rescale": True},
    # Muon runs the elementwise path with Adam beta1 = 0.9; ASTRO inherits its
    # spectral 0.95 for both paths, so a third of GPT-2 124M is trained by a
    # different optimizer than the baseline's. Nobody chose that.
    "astro_muon_betas": {"betas": (0.9, 0.95)},
    # The recipe the session-A ablation points at: every change that earned its
    # place, none that did not. Removing the beta asymmetry was worth 0.0580,
    # dropping post_normalize 0.0222, dropping the cautious decay mask 0.0165;
    # the QKV split is KEPT because removing it cost 0.0460. Whether those add
    # is the question astro_v2 exists to answer -- one seed each, so the
    # combination has to be measured rather than summed.
    "astro_v2": {"betas": (0.9, 0.95), "cautious_wd": False},
    # Same, minus the post-orthogonalisation variance, which measured 0.0032 --
    # inside the noise floor, so it is carried as a free knob rather than a
    # component.
    "astro_v2_gamma0": {"betas": (0.9, 0.95), "cautious_wd": False,
                        "variance_power": 0.0},
    "astro_nosplit": {"split": False},
    # The split repairs an initialisation-time defect that halves by the time
    # weights are trained, so applying it forever is the wrong shape -- and it
    # costs 1.33x measured. These relax to a single block partway through.
    "astro_split100": {"split_steps": 100},
    "astro_split300": {"split_steps": 300},
}

#: Model sizes with a batch that fits a 16 GB T4 in fp16. Time scales with
#: parameters x batch, so the batch shrinks as the model grows -- which also
#: means tokens-seen shrinks, and the report states it rather than hiding it.
SIZES: dict[str, dict[str, int]] = {
    "45M":   {"n_layer": 8,  "n_head": 8,  "n_embd": 512,  "batch": 8},
    "124M":  {"n_layer": 12, "n_head": 12, "n_embd": 768,  "batch": 8},
    "355M":  {"n_layer": 24, "n_head": 16, "n_embd": 1024, "batch": 4},
    "774M":  {"n_layer": 36, "n_head": 20, "n_embd": 1280, "batch": 2},
}

#: Fixed validation slice, so a loss is comparable across step budgets and
#: sessions. 400k tokens is 8 batches x 8 x 512 many times over.
VALIDATION_TOKENS = 400_000

BASELINES = ("adamw", "muon", "normuon", "adamuon")
ADAM_LR = (1e-4, 3e-3)
MUON_LR = (2e-3, 1e-1)


#: ``update_scale="trust"`` makes the step ``lr * ||W||``, so the learning rate
#: names a fractional change in a layer's norm rather than an absolute size.
#: That is a different unit from Muon's, and pairing it with Muon's range is
#: exactly the mistake this project has now shipped twice.
TRUST_LR = (1e-4, 3e-2)

#: Widened from (0.02, 0.5) after two optimizers independently selected
#: 0.4369, which sits at the 96% point of that range in log space. A tuner
#: whose answer is pinned against the top of its range has not found an
#: optimum, it has found a wall -- the same class of error as pairing an
#: update scale with the wrong learning-rate range, which this project has
#: shipped twice.
SCALAR_MULT = (0.02, 1.5)


def space_for(name: str) -> dict[str, tuple[float, float]]:
    """Three tuned hyperparameters for everyone; ranges follow the update scale."""
    if name not in known_optimizers():
        raise SystemExit(
            f"unknown optimizer {name!r}. available: {', '.join(known_optimizers())}"
        )
    if name == "adamw":
        return {"lr": ADAM_LR, "weight_decay": (1e-3, 3e-1), "beta2": (0.9, 0.999)}
    scale = VARIANTS.get(name, {}).get("update_scale", "muon")
    if scale == "trust":
        return {"lr": TRUST_LR, "weight_decay": (1e-3, 3e-1),
                "scalar_lr_mult": SCALAR_MULT}
    if scale != "muon":
        raise ValueError(f"no learning-rate range registered for update scale {scale!r}")
    return {"lr": MUON_LR, "weight_decay": (1e-3, 3e-1), "scalar_lr_mult": SCALAR_MULT}


def known_optimizers() -> tuple[str, ...]:
    return tuple(sorted(set(BASELINES) | set(VARIANTS)))


def build_optimizer(name: str, model, config: dict[str, float]):
    # An unknown name used to fall through every branch and return a default
    # ASTRO. A placeholder in the documentation -- "WINNER" -- was passed
    # literally, tuned, and would have been evaluated for hours under a name
    # that was never an optimizer. Silent fallbacks in a comparison harness are
    # how a table ends up describing something nobody ran.
    if name not in known_optimizers():
        raise SystemExit(
            f"unknown optimizer {name!r}.\n"
            f"  available: {', '.join(known_optimizers())}\n"
            "  (if a document showed you a placeholder like WINNER, substitute "
            "the actual variant name)"
        )
    overrides = dict(VARIANTS.get(name, {}))
    split = overrides.pop("split", True)
    routing = name if name in BASELINES else ("astro" if split else "muon")
    groups = build_groups(model, routing, model.config)
    scalar = config.get("scalar_lr_mult", 1.0)
    if name == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=config["lr"], betas=(0.9, config.get("beta2", 0.95)),
            weight_decay=config["weight_decay"],
        )
    if name == "muon":
        return Muon(groups, lr=config["lr"], adamw_lr=config["lr"] * scalar,
                    weight_decay=config["weight_decay"])
    if name == "normuon":
        return NorMuon(groups, lr=config["lr"], adamw_lr=config["lr"] * scalar,
                       weight_decay=config["weight_decay"])
    if name == "adamuon":
        return AdaMuon(groups, lr=config["lr"], adamw_lr=config["lr"] * scalar,
                       weight_decay=config["weight_decay"])
    return Astro(groups, lr=config["lr"], scalar_lr_mult=scalar,
                 weight_decay=config["weight_decay"], **overrides)


# ---------------------------------------------------------------------------
# Data and training
# ---------------------------------------------------------------------------


def load_tokens(tokenizer, needed: int, cache: Path) -> torch.Tensor:
    """Stream FineWeb-Edu once and cache it. Re-streaming costs minutes a session."""
    if cache.exists():
        tokens = torch.load(cache)
        if tokens.numel() >= needed:
            print(f"cached corpus: {tokens.numel():,} tokens", flush=True)
            return tokens[:needed]
    from datasets import load_dataset

    print(f"streaming FineWeb-Edu for {needed:,} tokens", flush=True)
    stream = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                          split="train", streaming=True)
    collected: list[int] = []
    for record in stream:
        collected.extend(tokenizer(record["text"]).input_ids)
        if len(collected) >= needed:
            break
    tokens = torch.tensor(collected[:needed], dtype=torch.long)
    torch.save(tokens, cache)
    return tokens


def train_once(name: str, config: dict[str, float], seed: int, *, data, size: str,
               steps: int, seq: int, vocab: int, log_every: int) -> tuple[float, float]:
    from transformers import GPT2Config, GPT2LMHeadModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    shape = dict(SIZES[size])
    batch = shape.pop("batch")
    train, validation = data

    torch.manual_seed(seed)
    model = GPT2LMHeadModel(GPT2Config(n_positions=seq, vocab_size=vocab, **shape)).to(device)
    model.train()
    if size in ("355M", "774M"):
        # Activations dominate at these widths on a 16 GB card.
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    optimizer = build_optimizer(name, model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    generator = torch.Generator().manual_seed(seed + 4242)
    warmup = max(1, steps // 10)
    started = time.perf_counter()

    def batches(source, gen, count):
        # HuggingFace shifts labels internally; a pre-shifted target would train
        # two-tokens-ahead prediction.
        for _ in range(count):
            start = torch.randint(0, source.numel() - seq - 1, (batch,), generator=gen)
            yield torch.stack([source[s : s + seq] for s in start]).to(device)

    for step, x in enumerate(batches(train, generator, steps)):
        factor = ((step + 1) / (warmup + 1) if step < warmup else
                  0.1 + 0.45 * (1 + math.cos(math.pi * (step - warmup) /
                                             max(1, steps - warmup))))
        for group in optimizer.param_groups:
            group.setdefault("base_lr", group["lr"])
            group["lr"] = group["base_lr"] * factor
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            loss = model(x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % log_every == 0:
            elapsed = time.perf_counter() - started
            rate = (step + 1) / max(elapsed, 1e-9)
            print(f"      step {step:5d}/{steps} train {float(loss.detach()):.4f} "
                  f"({rate:.2f} it/s, eta {(steps - step) / max(rate, 1e-9) / 60:.0f}m)",
                  flush=True)

    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for x in batches(validation, torch.Generator().manual_seed(777), 20):
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                total += float(model(x, labels=x).loss)
            count += 1
    value = total / max(1, count)

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return value, time.perf_counter() - started


# ---------------------------------------------------------------------------
# Resumable state
# ---------------------------------------------------------------------------

STATE = Path("astro_lab_state.json")


def load_state() -> dict:
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    # ``trials`` was added after several sessions had already been recorded, so
    # every reader must tolerate its absence rather than assume the schema.
    for section in ("runs", "tuned", "trials"):
        state.setdefault(section, {})
    return state


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2))


def key(size: str, steps: int, name: str, seed: int) -> str:
    return f"{size}|{steps}|{name}|{seed}"


def trial_key(size: str, steps: int, name: str, trial: int) -> str:
    return f"{size}|{steps}|{name}|t{trial}"


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def parameter_count(size: str, vocab: int, seq: int) -> int:
    """Non-embedding plus embedding parameters, from the shape rather than a table.

    Derived so the Chinchilla ratio in the report stays correct for any shape,
    including ones added after this was written. Per block: 4d^2 for the fused
    QKV and output projections, 8d^2 for the 4x-expansion MLP.
    """
    shape = SIZES[size]
    d, layers = shape["n_embd"], shape["n_layer"]
    return vocab * d + seq * d + layers * 12 * d * d


def sign_test(deltas: list[float]) -> tuple[int, float]:
    wins = sum(1 for d in deltas if d < 0)
    n = len(deltas)
    if n == 0:
        return 0, float("nan")
    tail = sum(math.comb(n, k) for k in range(min(wins, n - wins) + 1))
    return wins, min(1.0, 2 * tail / 2**n)


def report_scaling(state: dict, grid: list[tuple[str, int]], names: list[str],
                   seeds: int, seq: int, reference: str, vocab: int) -> str:
    lines = ["# ASTRO scaling study", ""]
    trend: list[tuple[str, int, float, float, int, int]] = []

    for size, steps in grid:
        values = {
            n: [state["runs"][key(size, steps, n, s)]["value"]
                for s in range(100, 100 + seeds)
                if key(size, steps, n, s) in state["runs"]]
            for n in names
        }
        seconds = {
            n: [state["runs"][key(size, steps, n, s)]["seconds"]
                for s in range(100, 100 + seeds)
                if key(size, steps, n, s) in state["runs"]]
            for n in names
        }
        done = [n for n in names if values[n]]
        if not done:
            continue

        batch = SIZES[size]["batch"]
        tokens = steps * batch * seq
        params = parameter_count(size, vocab, seq)
        lines += [f"## {size}, {steps} steps -- {params/1e6:.0f}M parameters, "
                  f"{tokens/1e6:.1f}M tokens "
                  f"({tokens / (params * 20):.4f}x Chinchilla-optimal)",
                  "",
                  "| optimizer | val loss | per-seed | s/run |", "|---|---|---|---|"]
        for n in sorted(done, key=lambda m: statistics.fmean(values[m])):
            lines.append(f"| `{n}` | {statistics.fmean(values[n]):.4f} | "
                         f"{', '.join(f'{v:.4f}' for v in values[n])} | "
                         f"{statistics.fmean(seconds[n]):.0f} |")

        if reference in done:
            lines += ["", f"Paired against `{reference}`:", "",
                      "| optimizer | paired Δ | worst seed | seeds won | sign p |",
                      "|---|---|---|---|---|"]
            for n in done:
                if n == reference:
                    continue
                pairs = list(zip(values[n], values[reference], strict=False))
                deltas = [a - b for a, b in pairs]
                wins, p = sign_test(deltas)
                worst = max(deltas) if deltas else float("nan")
                lines.append(f"| `{n}` | {statistics.fmean(deltas):+.4f} | {worst:+.4f} "
                             f"| {wins}/{len(deltas)} | {p:.4f} |")
                if n == "astro":
                    trend.append((size, steps, statistics.fmean(deltas), worst,
                                  wins, len(deltas)))
        lines.append("")

    if len(trend) > 1:
        lines += ["## The question this study exists to answer", "",
                  f"How the `astro` minus `{reference}` gap moves with scale and "
                  "training length. An advantage that shrinks toward zero along "
                  "either axis is an early-training or small-model effect, which "
                  "is what Wen et al. found for most published optimizer gains.",
                  "",
                  "| size | steps | tokens | paired Δ | worst seed | seeds won |",
                  "|---|---|---|---|---|---|"]
        for size, steps, delta, worst, wins, n in trend:
            tokens = steps * SIZES[size]["batch"] * seq
            lines.append(f"| {size} | {steps} | {tokens/1e6:.1f}M | {delta:+.4f} "
                         f"| {worst:+.4f} | {wins}/{n} |")
        lines += ["", f"With {seeds} seeds the sign test cannot go below "
                      f"{2 / 2**seeds:.4f}; read the trend and the worst seed, "
                      "not the p-value.",
                  "",
                  "GPU training is not bit-reproducible: re-running one "
                  "configuration on identical seeds in a later session moved "
                  "it by up to 0.0021. A cell whose gap is under about 0.005 "
                  "is indistinguishable from that floor, whichever sign it "
                  "carries."]
    return "\n".join(lines)


def same_config(a: dict, b: dict) -> bool:
    """Two tuning trials that a paired comparison may legitimately pair.

    Trial *index* is not enough. ``adamw`` tunes ``beta2`` over a different
    learning-rate range and ``astro_trust`` measures its step as a fraction of
    a layer's norm, so their trial 3 is a different point in a different space
    than Muon's trial 3. Pairing on the recorded configuration instead of the
    index means an optimizer with its own space simply contributes no pairs,
    rather than contributing wrong ones.
    """
    if a.keys() != b.keys():
        return False
    return all(math.isclose(a[k], b[k], rel_tol=1e-9) for k in a)


def report_configs(state: dict, names: list[str], reference: str) -> str:
    """Read the tuning sweep back as a paired comparison across configurations.

    Every optimizer with the same search space is offered the *same* drawn
    configurations, so the sweep is a paired design and not merely a search.
    Reporting only each optimizer's best throws that away: a win at one tuned
    point can be the tuner finding a good draw, while a win at every shared
    configuration cannot be. This costs no GPU time -- the runs already
    happened, and were previously discarded.
    """
    cells: dict[tuple[str, int], dict[str, list[tuple[dict, float]]]] = {}
    for slot, entry in state.get("trials", {}).items():
        size, steps, name, index = slot.split("|")
        if not index.startswith("t") or entry.get("value") is None:
            continue
        cells.setdefault((size, int(steps)), {}).setdefault(name, []).append(
            (entry["config"], entry["value"]))

    lines: list[str] = []
    for (size, steps), by_name in sorted(cells.items()):
        if reference not in by_name:
            continue
        rows = []
        for name in names:
            if name == reference or name not in by_name:
                continue
            deltas = [value - ref_value
                      for config, value in by_name[name]
                      for ref_config, ref_value in by_name[reference]
                      if same_config(config, ref_config)]
            if deltas:
                wins, p = sign_test(deltas)
                rows.append(f"| `{name}` | {len(deltas)} | "
                            f"{statistics.fmean(deltas):+.4f} | "
                            f"{max(deltas):+.4f} | {wins}/{len(deltas)} | {p:.4f} |")
        if not rows:
            continue
        lines += [f"## Across shared configurations -- {size}, {steps}-step trials",
                  "",
                  f"Each optimizer was offered the identical drawn configurations "
                  f"as `{reference}`, on seed 0. This pairs on the configuration, "
                  "not on the seed, so it answers a different question than the "
                  "table above: does the advantage hold at every setting, or "
                  "only at the one the tuner happened to select?",
                  "",
                  f"| optimizer | shared configs | paired Δ vs `{reference}` | "
                  "worst config | configs won | sign p |",
                  "|---|---|---|---|---|---|", *rows, ""]
    return "\n".join(lines)


def print_components() -> None:
    print("ASTRO = Muon + NorMuon's placement + the additions below.\n")
    width = max(len(c.flag) for c in COMPONENTS)
    for status in ("kept", "off", "unmeasured"):
        group = [c for c in COMPONENTS if c.status == status]
        if not group:
            continue
        label = {"kept": "KEPT (measured, helps)",
                 "off": "OFF BY DEFAULT (measured, hurts at the target scale)",
                 "unmeasured": "UNMEASURED (implemented, effect on loss unknown)"}[status]
        print(f"--- {label} " + "-" * max(0, 60 - len(label)))
        for c in group:
            print(f"  {c.flag:{width}s} = {c.default!r}")
            print(f"  {'':{width}s}   origin  : {c.origin}")
            print(f"  {'':{width}s}   what    : {c.what}")
            for i, line in enumerate(textwrap.wrap(c.evidence, 66)):
                print(f"  {'':{width}s}   {'evidence:' if i == 0 else '         '} {line}")
            print()
    unmeasured = sum(1 for c in COMPONENTS if c.status == "unmeasured")
    print(f"{unmeasured} of {len(COMPONENTS)} components have no measured effect on "
          "loss.\nThey are implemented and tested, not validated. A paper may only "
          "claim\nthe ones with evidence attached.")


def parse_overrides(items: list[str]) -> dict[str, float]:
    """``["lr=0.01", "weight_decay=0.02"]`` -> ``{"lr": 0.01, ...}``."""
    out: dict[str, float] = {}
    for item in items:
        k, sep, v = item.partition("=")
        if not sep:
            raise SystemExit(f"expected KEY=VALUE, got {item!r}")
        try:
            out[k] = float(v)
        except ValueError:
            raise SystemExit(f"{item!r}: {v!r} is not a number") from None
    return out


def in_colab() -> bool:
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ValueError):
        # find_spec raises rather than returning None when the parent package
        # is itself absent, which is the common case off Colab.
        return False


def enter_work_dir(path: str | None, allow_ephemeral: bool) -> None:
    """Move to the directory that will hold the state file, cache and report.

    Everything this script writes is relative to the working directory. On
    Colab that directory defaults to ``/content``, which the runtime deletes
    when it is reclaimed. A session that ran eight 900-step trials -- two hours
    of a T4 -- and was killed before it finished lost all eight, because the
    state file the resume logic depends on was sitting in ``/content``.

    So a Colab session must name a directory under Drive, and is refused
    otherwise rather than warned: a warning scrolls past in the first second of
    a two-hour cell and is not read again. Drive is mounted here if it is not
    already, so one flag covers the whole thing.
    """
    if path:
        target = Path(path)
        if (in_colab() and target.is_relative_to("/content/drive")
                and not Path("/content/drive/MyDrive").is_dir()):
            from google.colab import drive

            drive.mount("/content/drive")
        target.mkdir(parents=True, exist_ok=True)
        os.chdir(target)

    cwd = Path.cwd().resolve()
    # Off Colab the working directory is the user's own and persists.
    persistent = not in_colab() or cwd.is_relative_to("/content/drive")
    if not persistent and not allow_ephemeral:
        raise SystemExit(
            f"working directory {cwd} does not survive this Colab runtime, and "
            "every completed run would be lost with it.\n"
            "  fix:      --work-dir /content/drive/MyDrive/astro\n"
            "  override: --allow-ephemeral  (only if you do not want the results)"
        )
    print(f"working directory {cwd}"
          f" ({'persistent' if persistent else 'EPHEMERAL -- results will be lost'})",
          flush=True)


def announce(args) -> None:
    """Print the search space actually in effect, and a file fingerprint.

    A previous session's drawn ``scalar_lr_mult`` values all landed inside the
    old (0.02, 0.5) range after that range had been widened to (0.02, 1.5).
    Whether the widened file was the one that ran had to be inferred from the
    numbers, which is guesswork about provenance -- the one thing a benchmark
    harness must never make you guess. The fingerprint is over this file's own
    source, so the report and the code that produced it can be tied together
    after the fact.
    """
    source = SOURCE.read_bytes()
    print(f"astro_lab {hashlib.sha256(source).hexdigest()[:12]} "
          f"({len(source):,} bytes)", flush=True)
    if args.config:
        return
    unknown = set(args.pin) - {k for n in args.optimizers for k in space_for(n)}
    if unknown:
        raise SystemExit(
            f"--pin {sorted(unknown)[0]!r} is not tuned by any of the requested "
            "optimizers, so pinning it would silently do nothing."
        )
    print("search space per optimizer (pinned values marked *):", flush=True)
    for name in args.optimizers:
        terms = []
        for k, (lo, hi) in space_for(name).items():
            terms.append(f"{k}=*{args.pin[k]:g}" if k in args.pin
                         else f"{k}=[{lo:g},{hi:g}]")
        print(f"  {name:16s} " + "  ".join(terms), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="scaling", choices=["scaling", "components"])
    parser.add_argument("--sizes", nargs="+", default=["124M"], choices=sorted(SIZES))
    parser.add_argument("--steps", type=int, nargs="+", default=[300, 900])
    parser.add_argument("--optimizers", nargs="+",
                        default=["adamw", "muon", "normuon", "astro"])
    parser.add_argument("--seeds", type=int, default=2)
    parser.add_argument("--trials", type=int, default=0,
                        help="tuning trials per optimizer per cell. 0 reuses the "
                             "configuration tuned at the smallest cell, which is "
                             "what makes a scaling study affordable; the report "
                             "says which was used.")
    parser.add_argument("--seq", type=int, default=512)
    parser.add_argument("--tune-steps", type=int, default=None,
                        help="step budget for the tuning trials, independent of "
                             "the evaluation grid. Tuning at the evaluation "
                             "budget is what makes a sweep unaffordable: five "
                             "trials for six optimizers at 900 steps is seven "
                             "hours, the same at 300 steps is two. The report "
                             "states which budget selected the configuration.")
    parser.add_argument("--reference", default="muon")
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--stop-after", type=int, default=None,
                        help="finish at most this many runs, then exit "
                             "cleanly. --max-minutes bounds a cell by the "
                             "clock, which does not help when what ends the "
                             "cell is Colab reclaiming the runtime rather "
                             "than the budget expiring. A run count is "
                             "predictable: at ~15.5 min per 900-step run, "
                             "--stop-after 5 is a 78-minute cell, every time.")
    parser.add_argument("--work-dir", default=None,
                        help="directory to run in; created if missing. "
                             "Everything written -- state file, corpus cache, "
                             "report -- lands here. On Colab this must be "
                             "under /content/drive, and Drive is mounted for "
                             "you if it is not already.")
    parser.add_argument("--allow-ephemeral", action="store_true",
                        help="permit a Colab working directory that the "
                             "runtime deletes. Only for a throwaway check.")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--config", nargs="+", metavar="K=V", default=None,
                        help="fixed hyperparameters for every optimizer, e.g. "
                             "--config lr=0.0144 weight_decay=0.02")
    parser.add_argument("--pin", nargs="+", metavar="K=V", default=[],
                        help="hold these hyperparameters fixed for every "
                             "optimizer while still tuning the rest. "
                             "scalar_lr_mult is the intended use: it scales "
                             "the elementwise path, which is the *same* AdamW "
                             "code in Muon, NorMuon, AdaMuon and ASTRO and "
                             "covers 31.7%% of GPT-2 124M. Tuning it "
                             "separately per optimizer makes the comparison "
                             "partly a lottery over a shared subsystem, and "
                             "that lottery measured 0.1144 nats -- larger "
                             "than any gap between the optimizers themselves.")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild the table from astro_lab_state.json and "
                             "exit. Touches no GPU and downloads nothing, so a "
                             "run whose output scrolled away or whose session "
                             "died mid-cell can still be read out.")
    args = parser.parse_args()

    if args.mode == "components":
        print_components()
        return 0

    # Validate names before anything expensive. Catching this inside
    # train_once means the failure arrives after a tokenizer download, a
    # corpus stream and possibly a tuning trial; catching it here costs
    # nothing and arrives in the first second.
    unknown = [n for n in args.optimizers if n not in known_optimizers()]
    if unknown:
        raise SystemExit(
            f"unknown optimizer {unknown[0]!r}.\n"
            f"  available: {', '.join(known_optimizers())}\n"
            "  (if a document showed you a placeholder like WINNER, substitute "
            "the actual variant name)"
        )

    args.pin = parse_overrides(args.pin)
    # Fingerprint first, so the very first line of a session still identifies
    # the code that produced it; then the directory that will hold the results.
    announce(args)
    enter_work_dir(args.work_dir, args.allow_ephemeral)

    if args.report_only:
        if not STATE.exists():
            print(f"no {STATE} here -- nothing was completed, or you are in a "
                  "different working directory than the run was")
            return 1
        state = load_state()
        grid = [(size, steps) for size in args.sizes for steps in sorted(args.steps)]
        # GPT-2's vocabulary, so the parameter counts in the table can be
        # computed without downloading a tokenizer.
        finish(state, grid, args, 50257)
        return 0

    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        print(f"GPU {torch.cuda.get_device_name(0)} sm_{major}{minor} "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB",
              flush=True)
    else:
        print("no CUDA device; this will be unusably slow", flush=True)

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    biggest_batch = max(SIZES[s]["batch"] for s in args.sizes)
    needed = biggest_batch * args.seq * (max(args.steps) + 60)
    # The validation set must not depend on the training budget. It used to be
    # "the last 5% of whatever we streamed", and what we streamed was sized from
    # max(--steps) -- so a 900-step run and a 2700-step run validated on
    # different text and their losses were never comparable. Muon moved 0.0555
    # between two such runs, 26x the noise floor, for that reason alone.
    #
    # Validation is now the *first* VALIDATION_TOKENS of the stream, identical
    # in every run at every budget; training uses everything after it.
    tokens = load_tokens(tokenizer, needed + VALIDATION_TOKENS,
                         Path("fineweb_cache.pt"))
    data = (tokens[VALIDATION_TOKENS:], tokens[:VALIDATION_TOKENS])
    print(f"corpus: {tokens.numel():,} tokens, "
          f"validation pinned to the first {VALIDATION_TOKENS:,}", flush=True)

    fixed: dict[str, float] | None = None
    if args.config:
        fixed = parse_overrides(args.config)
        fixed.setdefault("weight_decay", 0.01)
        fixed.setdefault("scalar_lr_mult", 0.1)
        fixed.setdefault("beta2", 0.95)
        print(f"fixed configuration for every optimizer: {fixed}", flush=True)

    state = load_state()
    started = time.perf_counter()

    finished = [0]          # runs completed in *this* session, not in total

    def elapsed() -> float:
        return (time.perf_counter() - started) / 60

    def stop_reason() -> str | None:
        """Why this session should stop now, or None to keep going."""
        if args.stop_after is not None and finished[0] >= args.stop_after:
            return f"the {args.stop_after}-run limit for this session"
        if args.max_minutes is not None and elapsed() >= args.max_minutes:
            return f"the {args.max_minutes:.0f}-minute budget"
        return None

    def seconds_per_step(size: str, name: str) -> float:
        """Measured cost from this state file, falling back to a T4 estimate.

        Without this the budget is only honoured *between* runs, so a 46-minute
        2700-step cell started at minute 239 of a 240-minute budget finishes at
        minute 285. That is exactly how a --max-minutes 240 run took four hours
        and returned nothing, and it is the failure the caller least expects.
        """
        # Both sections, because a --seeds 0 sweep produces no evaluation runs
        # at all: reading only "runs" left the estimator on its generic T4
        # guess for the entire session, which is the session where the guard
        # matters most.
        seen = [entry["seconds"] / int(k.split("|")[1])
                for section in ("runs", "trials")
                for k, entry in state[section].items()
                if k.startswith(f"{size}|") and k.split("|")[2] == name
                and isinstance(entry, dict) and entry.get("seconds")]
        if seen:
            return max(seen)
        # T4 at batch 8, sequence 512, scaled by parameter count against 124M.
        shape = SIZES[size]
        relative = (shape["n_layer"] * shape["n_embd"] ** 2) / (12 * 768 ** 2)
        return 1.15 * relative * (SIZES[size]["batch"] / 8)

    def would_overrun(size: str, name: str, steps: int) -> float | None:
        """Minutes this run needs, or None if it fits the remaining budget."""
        if args.max_minutes is None:
            return None
        need = seconds_per_step(size, name) * steps / 60 * 1.10   # 10% headroom
        return need if elapsed() + need > args.max_minutes else None

    grid = [(size, steps) for size in args.sizes for steps in sorted(args.steps)]
    planned = [(size, steps, n, s) for size, steps in grid for n in args.optimizers
               for s in range(100, 100 + args.seeds)
               if key(size, steps, n, s) not in state["runs"]]
    # Tuning trials are runs too, and under --seeds 0 they are the *only* runs.
    # Counting evaluations alone printed "0 runs to go" at the start of a
    # twelve-run sweep.
    pending_trials = 0 if fixed else sum(
        1 for size, steps in grid for n in args.optimizers
        for t in range(args.trials)
        if n not in state["tuned"]
        and trial_key(size, args.tune_steps or steps, n, t) not in state["trials"])
    print(f"\n{len(planned)} evaluations and {pending_trials} tuning trials to go "
          f"across {len(grid)} cells", flush=True)

    for size, steps in grid:
        if fixed is None and args.trials <= 0:
            missing = [n for n in args.optimizers if n not in state["tuned"]]
            if missing:
                raise SystemExit(
                    f"no configuration for {missing[0]!r}: pass --config, or "
                    "--trials to tune one."
                )
        elif fixed is None:
            # Run the grid whenever configurations are requested, not only when
            # nothing has been selected yet. Gating on "is this optimizer
            # already in state['tuned']" made the documented ladder --
            # --trials 2, then 3, then 4 -- a no-op after the first cell: every
            # optimizer had a selection, so the extra configurations were never
            # drawn. Recorded trials are skipped inside the sweep, so this
            # costs nothing when there is nothing new to run.
            # Tune at --tune-steps when given, so the search budget is
            # decoupled from the evaluation budget.
            complete = sweep_shared_grid(args, state, data, size,
                                         args.tune_steps or steps, tokenizer,
                                         stop_reason, would_overrun, finished)
            if not complete:
                finish(state, grid, args, len(tokenizer))
                return 0
        for name in args.optimizers:
            config = fixed if fixed is not None else state["tuned"][name]
            for seed in range(100, 100 + args.seeds):
                if key(size, steps, name, seed) in state["runs"]:
                    continue
                reason = stop_reason()
                if reason is not None:
                    print(f"\nstopping at {reason}; re-run the same command to "
                          "continue", flush=True)
                    finish(state, grid, args, len(tokenizer))
                    return 0
                need = would_overrun(size, steps=steps, name=name)
                if need is not None:
                    print(f"\nskipping [{size} {steps}st] {name} seed {seed}: needs "
                          f"~{need:.0f} min, only {args.max_minutes - elapsed():.0f} "
                          "left of the budget. Re-run the same command to continue.",
                          flush=True)
                    finish(state, grid, args, len(tokenizer))
                    return 0
                print(f"[{size} {steps}st] {name} seed {seed}  "
                      f"(~{seconds_per_step(size, name) * steps / 60:.0f} min, "
                      f"{elapsed():.0f}/{args.max_minutes or 0:.0f} used, "
                      f"run {finished[0] + 1}"
                      f"{f'/{args.stop_after}' if args.stop_after else ''})",
                      flush=True)
                value, seconds = train_once(
                    name, config, seed, data=data, size=size, steps=steps,
                    seq=args.seq, vocab=len(tokenizer), log_every=args.log_every)
                finished[0] += 1
                state["runs"][key(size, steps, name, seed)] = {
                    "value": value, "seconds": seconds, "config": config}
                save_state(state)
                # Refresh the report on disk too, so a reclaimed runtime never
                # leaves numbers without a readable table beside them.
                finish(state, grid, args, len(tokenizer), quiet=True)
                print(f"      -> {value:.4f} ({seconds:.0f}s)"
                      f"   [{len(state['runs'])} runs saved -> "
                      f"{Path('astro_lab_report.md').resolve()}]", flush=True)

    finish(state, grid, args, len(tokenizer))
    return 0


def draw_configs(name: str, trials: int, pin: dict[str, float]) -> list[dict[str, float]]:
    """The tuning candidates for one optimizer, as a list rather than a stream.

    Seeded identically for every optimizer, so all of them are offered the
    *same* configurations -- common random numbers, which turns the tuning
    sweep into a paired comparison across configurations instead of a set of
    unrelated searches. Materialising the list also lets a resumed session
    reconstruct trial ``k`` without re-running trials ``0..k-1``.
    """
    rng = random.Random(12345)
    space = {k: v for k, v in space_for(name).items() if k not in pin}
    out = []
    for _ in range(trials):
        config = {k: (lo if lo == hi else math.exp(rng.uniform(math.log(lo), math.log(hi))))
                  for k, (lo, hi) in space.items()}
        config.update(pin)
        out.append(config)
    return out


def sweep_shared_grid(args, state, data, size, steps, tokenizer, stop_reason,
                      would_overrun, finished) -> bool:
    """Run the shared grid **configuration-major**, and select from it.

    The order is the whole point. Walking optimizers in the outer loop means
    any early stop leaves the first optimizers with every configuration and the
    last with none -- and the last is where ASTRO sits, because the reference
    is conventionally listed first. A real session stopped after six runs and
    produced a report comparing Muon, NorMuon and AdaMuon with both ASTRO
    columns simply absent; the numbers were fine and the table answered a
    question nobody asked.

    Configuration-major, every stopping point is balanced to within the one
    configuration in flight, and the paired table is readable at any moment.

    Every completed trial is written to the state file. It used to keep only
    the winner, so a session that hit its budget at trial 7 of 8 threw away six
    finished runs -- an hour and a half of a T4. The trials are also evidence
    in their own right: because every optimizer sharing a space is offered the
    same configurations, the recorded grid is a paired comparison across
    configurations, which ``report_configs`` reads back.

    Returns True if every optimizer now has a selected configuration.
    """
    pending = list(args.optimizers)
    configs = {n: draw_configs(n, args.trials, args.pin) for n in pending}

    for trial in range(args.trials):
        for name in pending:
            slot = trial_key(size, steps, name, trial)
            if slot in state["trials"]:
                value = state["trials"][slot]["value"]
                print(f"[grid {size} {steps}st] config {trial + 1}/{args.trials} "
                      f"{name} -> "
                      f"{'diverged' if value is None else f'{value:.4f}'} "
                      "(already done)", flush=True)
                continue

            reason = stop_reason()
            if reason is None and would_overrun(size, name, steps) is not None:
                reason = (f"~{would_overrun(size, name, steps):.0f} min needed "
                          "and less than that left of the budget")
            if reason is not None:
                print(f"\nstopping before [grid {size} {steps}st] config "
                      f"{trial + 1} {name}: {reason}. Re-run the same command "
                      "to continue -- every finished run is saved.", flush=True)
                return False

            config = configs[name][trial]
            print(f"[grid {size} {steps}st] config {trial + 1}/{args.trials} "
                  f"{name} (run {finished[0] + 1}"
                  f"{f'/{args.stop_after}' if args.stop_after else ''}) "
                  + " ".join(f"{k}={v:.4g}" for k, v in sorted(config.items())),
                  flush=True)
            began = time.perf_counter()
            try:
                # The *reported* duration, not the wall time around the call --
                # the same quantity the evaluation path stores, so the cost
                # estimator divides a like for a like when it reads both.
                value, seconds = train_once(name, config, 0, data=data, size=size,
                                            steps=steps, seq=args.seq,
                                            vocab=len(tokenizer),
                                            log_every=args.log_every)
            except (RuntimeError, ValueError) as error:
                print(f"      diverged: {type(error).__name__}", flush=True)
                value, seconds = math.inf, time.perf_counter() - began
            if math.isnan(value):
                value = math.inf
            finished[0] += 1
            print(f"      -> {value:.4f}", flush=True)
            state["trials"][slot] = {
                "config": config,
                "value": None if math.isinf(value) else value,
                "seconds": seconds,
            }
            save_state(state)
            # The report is a paired table across configurations, so it is
            # meaningful from the very first completed configuration onward.
            finish(state, [(size, steps)], args, len(tokenizer), quiet=True)

    for name in pending:
        select_tuned(name, state, size, steps, configs[name])
    return True


def select_tuned(name: str, state: dict, size: str, steps: int,
                 configs: list[dict]) -> dict:
    """Pick the best recorded configuration. Separate from running them.

    Selection reads the state file rather than a variable held across the
    sweep, so it gives the same answer whether the grid ran in one session or
    five.
    """
    scored = []
    for trial, config in enumerate(configs):
        entry = state["trials"].get(trial_key(size, steps, name, trial))
        if entry is not None and entry["value"] is not None:
            scored.append((entry["value"], trial, config))
    if not scored:
        raise SystemExit(f"every configuration for {name!r} diverged")
    best_value, _, best_config = min(scored, key=lambda row: row[0])
    state["tuned"][name] = best_config
    state["tuned_at"] = dict(state.get("tuned_at", {}),
                             **{name: {"size": size, "steps": steps,
                                       "value": best_value,
                                       "of_configurations": len(scored)}})
    save_state(state)
    print(f"selected {name}: {best_value:.4f} of {len(scored)} configurations "
          f"{best_config}", flush=True)
    return best_config


def finish(state, grid, args, vocab: int, *, quiet: bool = False) -> None:
    """Write the report. Called after every run, not only at the end.

    A Colab session that is reclaimed mid-grid used to leave a state file and
    no report, so the numbers survived but nothing readable did. Formatting a
    dict of a dozen entries costs nothing, so the report on disk is now always
    current: whatever moment the runtime dies, astro_lab_report.md already
    describes every run that finished.
    """
    table = report_scaling(state, grid, args.optimizers, args.seeds, args.seq,
                           args.reference, vocab)
    across = report_configs(state, args.optimizers, args.reference)
    if across:
        table = table + "\n" + across
    Path("astro_lab_report.md").write_text(table + "\n")
    if not quiet:
        print("\n" + table)
        print(f"\nwrote astro_lab_report.md and {STATE}")


if __name__ == "__main__":
    raise SystemExit(main())
