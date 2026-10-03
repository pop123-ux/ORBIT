from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn


@dataclass(frozen=True)
class QKPairSpec:
    """Describe one RoPE query/key projection pair in an external model.

    The first adapter release intentionally targets the same attention geometry
    evaluated in the paper: separate Q/K linear projections with equal output
    width, ordinary multi-head attention, and an even per-head dimension.
    Architectures such as GQA/MQA need an explicit head-mapping rule rather
    than a shape-only guess and should use a native integration for now.
    """

    q_proj: nn.Linear
    k_proj: nn.Linear
    n_head: int
    rope_base: float = 10_000.0
    stats_beta: float = 0.95
    name: str = ""

    def validate(self) -> None:
        if not isinstance(self.q_proj, nn.Linear) or not isinstance(self.k_proj, nn.Linear):
            raise TypeError("QKPairSpec currently requires nn.Linear q_proj and k_proj modules")
        if self.n_head <= 0:
            raise ValueError("n_head must be positive")
        if self.q_proj.out_features != self.k_proj.out_features:
            raise ValueError(
                "Q/K projection widths differ. The generic adapter currently targets MHA; "
                "GQA/MQA require an explicit head-mapping integration."
            )
        if self.q_proj.out_features % self.n_head:
            raise ValueError("Q/K output width must be divisible by n_head")
        head_dim = self.q_proj.out_features // self.n_head
        if head_dim % 2:
            raise ValueError("RoPE head_dim must be even")


class _HookedQKStatistics(nn.Module):
    """Persistent ORBIT statistics collected from external Q/K projections."""

    def __init__(self, spec: QKPairSpec) -> None:
        super().__init__()
        spec.validate()
        self.n_head = int(spec.n_head)
        self.head_dim = spec.q_proj.out_features // self.n_head
        self.n_freq = self.head_dim // 2
        self.stats_beta = float(spec.stats_beta)
        self.name = spec.name
        self.collect_orbit_stats = False
        self._active_forward_id = -1
        self._updated_forward_id = -1
        self._pending_q: torch.Tensor | None = None
        self._pending_k: torch.Tensor | None = None

        inv_freq = 1.0 / (
            float(spec.rope_base)
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        eye = torch.eye(2, dtype=torch.float32).view(1, 1, 2, 2)
        self.register_buffer(
            "orbit_q_second_moment",
            eye.repeat(self.n_head, self.n_freq, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "orbit_k_second_moment",
            eye.repeat(self.n_head, self.n_freq, 1, 1),
            persistent=True,
        )
        self.register_buffer("orbit_stats_seen", torch.zeros((), dtype=torch.long), persistent=True)

        self._q_handle = spec.q_proj.register_forward_hook(self._capture_q)
        self._k_handle = spec.k_proj.register_forward_hook(self._capture_k)

    def begin_forward(self, forward_id: int) -> None:
        self._active_forward_id = int(forward_id)
        self._pending_q = None
        self._pending_k = None

    def _capture_q(self, _module, _inputs, output) -> None:
        self._capture("q", output)

    def _capture_k(self, _module, _inputs, output) -> None:
        self._capture("k", output)

    def _capture(self, which: str, output) -> None:
        if not self.collect_orbit_stats or not self.training:
            return
        if self._updated_forward_id == self._active_forward_id:
            return
        if not torch.is_tensor(output):
            raise TypeError("ORBIT adapter expects Q/K projection outputs to be tensors")
        if output.ndim != 3:
            raise ValueError(
                "ORBIT adapter expects Q/K projection outputs shaped [batch, time, width]; "
                "use a native integration for a different projection layout"
            )
        if output.size(-1) != self.n_head * self.head_dim:
            raise ValueError("Q/K projection output width changed after adapter construction")
        if which == "q":
            self._pending_q = output.detach()
        else:
            self._pending_k = output.detach()
        if self._pending_q is not None and self._pending_k is not None:
            self._update_second_moments(self._pending_q, self._pending_k)
            self._updated_forward_id = self._active_forward_id
            self._pending_q = None
            self._pending_k = None

    @torch.no_grad()
    def _update_second_moments(self, q: torch.Tensor, k: torch.Tensor) -> None:
        if q.shape[:2] != k.shape[:2]:
            raise ValueError("Q/K adapter projections must share batch and sequence dimensions")
        q2 = q.float().view(q.size(0), q.size(1), self.n_head, self.n_freq, 2)
        k2 = k.float().view(k.size(0), k.size(1), self.n_head, self.n_freq, 2)
        q_sum = torch.einsum("bthfi,bthfj->hfij", q2, q2)
        k_sum = torch.einsum("bthfi,bthfj->hfij", k2, k2)
        count = q_sum.new_tensor(float(max(1, q.size(0) * q.size(1))))

        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            packed = torch.stack((q_sum, k_sum), dim=0)
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
            q_sum, k_sum = packed[0], packed[1]

        q_second_moment = q_sum / count.clamp_min(1.0)
        k_second_moment = k_sum / count.clamp_min(1.0)
        beta = (self.orbit_stats_seen > 0).to(q_second_moment.dtype) * self.stats_beta
        self.orbit_q_second_moment.mul_(beta).add_(q_second_moment * (1.0 - beta))
        self.orbit_k_second_moment.mul_(beta).add_(k_second_moment * (1.0 - beta))
        self.orbit_stats_seen.add_(1)

    def _optimizer_transport_rotation(self, delta: int) -> torch.Tensor:
        if delta < 0:
            raise ValueError("ORBIT deltas are causal distances i-j and must be non-negative")
        angle = self.inv_freq.float() * float(delta)
        c, s = angle.cos(), angle.sin()
        return torch.stack(
            (
                torch.stack((c, -s), dim=-1),
                torch.stack((s, c), dim=-1),
            ),
            dim=-2,
        ).unsqueeze(0)

    @torch.no_grad()
    def orbit_metrics(
        self,
        deltas: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        *,
        rotate: bool = True,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q_second_moment = self.orbit_q_second_moment.float()
        k_second_moment = self.orbit_k_second_moment.float()
        if not rotate:
            mq, mk = k_second_moment.clone(), q_second_moment.clone()
        else:
            mq = torch.zeros_like(k_second_moment)
            mk = torch.zeros_like(q_second_moment)
            ds = tuple(int(d) for d in deltas) or (0,)
            for delta in ds:
                r = self._optimizer_transport_rotation(delta)
                rt = r.transpose(-1, -2)
                mq.add_(r @ k_second_moment @ rt)
                mk.add_(rt @ q_second_moment @ r)
            mq.div_(len(ds))
            mk.div_(len(ds))
        eye = torch.eye(2, device=mq.device, dtype=mq.dtype).view(1, 1, 2, 2)
        return mq + eps * eye, mk + eps * eye

    @contextmanager
    def suspend_orbit_stat_collection(self):
        previous = self.collect_orbit_stats
        self.collect_orbit_stats = False
        try:
            yield
        finally:
            self.collect_orbit_stats = previous

    def close(self) -> None:
        self._q_handle.remove()
        self._k_handle.remove()


class OrbitAdapter(nn.Module):
    """Wrap an external RoPE model with the explicit contract required by ORBIT.

    Function-aware optimizers need architectural semantics that cannot be inferred
    safely from tensor shapes alone: which matrices are Q/K partners, how heads map
    to RoPE frequency pairs, which parameters belong on the auxiliary AdamW path,
    and where unrotated Q/K activations can be observed. The adapter makes those
    choices explicit instead of guessing them.

    Use the returned wrapper for both forward passes and optimizer construction.
    The underlying model remains available as ``adapter.model``.
    """

    def __init__(
        self,
        model: nn.Module,
        qk_pairs: Sequence[QKPairSpec],
        *,
        aux_decay_parameters: Iterable[nn.Parameter | nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if not qk_pairs:
            raise ValueError("OrbitAdapter requires at least one Q/K pair")
        self.model = model
        self._specs = tuple(qk_pairs)
        self.collectors = nn.ModuleList([_HookedQKStatistics(spec) for spec in self._specs])
        self._forward_id = 0
        self._aux_decay = self._resolve_aux_decay(aux_decay_parameters)

    def _resolve_aux_decay(
        self, items: Iterable[nn.Parameter | nn.Module] | None
    ) -> tuple[nn.Parameter, ...]:
        params: list[nn.Parameter] = []
        if items is None:
            for getter_name in ("get_input_embeddings", "get_output_embeddings"):
                getter = getattr(self.model, getter_name, None)
                if callable(getter):
                    module = getter()
                    weight = getattr(module, "weight", None) if module is not None else None
                    if isinstance(weight, nn.Parameter):
                        params.append(weight)
        else:
            for item in items:
                if isinstance(item, nn.Parameter):
                    params.append(item)
                elif isinstance(item, nn.Module):
                    weight = getattr(item, "weight", None)
                    if not isinstance(weight, nn.Parameter):
                        raise TypeError("auxiliary-decay modules must expose a Parameter named weight")
                    params.append(weight)
                else:
                    raise TypeError("aux_decay_parameters accepts Parameters or Modules")
        unique: dict[int, nn.Parameter] = {id(param): param for param in params}
        return tuple(unique.values())

    def orbit_aux_decay_parameters(self) -> tuple[nn.Parameter, ...]:
        return self._aux_decay

    def orbit_qk_pairs(self):
        return [
            {
                "layer": index,
                "module": collector,
                "q": spec.q_proj.weight,
                "k": spec.k_proj.weight,
            }
            for index, (spec, collector) in enumerate(zip(self._specs, self.collectors))
        ]

    def set_orbit_stat_collection(self, enabled: bool) -> None:
        for collector in self.collectors:
            collector.collect_orbit_stats = bool(enabled)

    @contextmanager
    def suspend_orbit_stat_collection(self):
        previous = [collector.collect_orbit_stats for collector in self.collectors]
        for collector in self.collectors:
            collector.collect_orbit_stats = False
        try:
            yield
        finally:
            for collector, enabled in zip(self.collectors, previous):
                collector.collect_orbit_stats = enabled

    def forward(self, *args, **kwargs):
        self._forward_id += 1
        for collector in self.collectors:
            collector.begin_forward(self._forward_id)
        return self.model(*args, **kwargs)

    def close(self) -> None:
        for collector in self.collectors:
            collector.close()
