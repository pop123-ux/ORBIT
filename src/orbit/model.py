from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class OrbitGPTConfig:
    vocab_size: int = 50_257
    block_size: int = 512
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    rope_base: float = 10_000.0
    stats_beta: float = 0.95
    gradient_checkpointing: bool = False

    @property
    def head_dim(self) -> int:
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        dim = self.n_embd // self.n_head
        if dim % 2:
            raise ValueError("RoPE head_dim must be even")
        return dim


class RotaryAttention(nn.Module):
    """Causal MHA plus optimizer-only RoPE circuit statistics.

    The buffers are deliberately tiny: for every head and 2D RoPE frequency
    pair we store one 2x2 covariance for Q and K. They are not parameters and
    are irrelevant at inference. ORBIT uses them to approximate the local
    metric induced by movement of pre-softmax attention logits.
    """

    def __init__(self, config: OrbitGPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.n_freq = self.head_dim // 2
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        inv_freq = 1.0 / (
            config.rope_base
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=True)
        eye = torch.eye(2, dtype=torch.float32).view(1, 1, 2, 2)
        self.register_buffer(
            "orbit_q_cov", eye.repeat(self.n_head, self.n_freq, 1, 1), persistent=False
        )
        self.register_buffer(
            "orbit_k_cov", eye.repeat(self.n_head, self.n_freq, 1, 1), persistent=False
        )
        self.register_buffer("orbit_stats_seen", torch.zeros((), dtype=torch.long), persistent=False)
        # Baselines must not pay ORBIT's statistics overhead. Orbit.__init__
        # explicitly enables collection only for ORBIT-family runs.
        self.collect_orbit_stats = False

    def _apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, heads, time, head_dim]
        time = x.size(2)
        angles = torch.arange(time, device=x.device, dtype=torch.float32)[:, None]
        angles = angles * self.inv_freq.to(x.device)[None, :]
        cos = angles.cos().to(x.dtype).view(1, 1, time, self.n_freq, 1)
        sin = angles.sin().to(x.dtype).view(1, 1, time, self.n_freq, 1)
        pair = x.view(*x.shape[:-1], self.n_freq, 2)
        a, b = pair[..., :1], pair[..., 1:]
        rotated = torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1)
        return rotated.flatten(-2)

    @torch.no_grad()
    def _update_covariances(self, q: torch.Tensor, k: torch.Tensor) -> None:
        if not self.collect_orbit_stats or not self.training:
            return
        q2 = q.float().view(q.size(0), self.n_head, q.size(2), self.n_freq, 2)
        k2 = k.float().view(k.size(0), self.n_head, k.size(2), self.n_freq, 2)
        denom = float(max(1, q.size(0) * q.size(2)))
        q_cov = torch.einsum("bhtfi,bhtfj->hfij", q2, q2) / denom
        k_cov = torch.einsum("bhtfi,bhtfj->hfij", k2, k2) / denom
        beta = self.config.stats_beta if int(self.orbit_stats_seen) else 0.0
        self.orbit_q_cov.mul_(beta).add_(q_cov, alpha=1.0 - beta)
        self.orbit_k_cov.mul_(beta).add_(k_cov, alpha=1.0 - beta)
        self.orbit_stats_seen.add_(1)

    @torch.no_grad()
    def orbit_metrics(
        self,
        deltas: Iterable[int] = (1, 2, 4, 8, 16, 32, 64, 128),
        *,
        rotate: bool = True,
        eps: float = 1e-5,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return 2x2 local Q/K logit-sensitivity metrics.

        For one RoPE pair the score contribution is q^T R_delta k. Therefore a
        Q perturbation sees covariance R C_k R^T, while a K perturbation sees
        R^T C_q R. Averaging over relative offsets gives a cheap approximation
        to functional movement in attention-logit space without materializing a
        sequence-by-sequence Jacobian or a dense d_model x d_model circuit.
        """
        q_cov = self.orbit_q_cov.float()
        k_cov = self.orbit_k_cov.float()
        if not rotate:
            mq, mk = k_cov.clone(), q_cov.clone()
        else:
            mq = torch.zeros_like(k_cov)
            mk = torch.zeros_like(q_cov)
            ds = tuple(int(d) for d in deltas)
            if not ds:
                ds = (0,)
            for delta in ds:
                angle = self.inv_freq.float() * float(delta)
                c, s = angle.cos(), angle.sin()
                r = torch.stack(
                    (
                        torch.stack((c, -s), dim=-1),
                        torch.stack((s, c), dim=-1),
                    ),
                    dim=-2,
                ).unsqueeze(0)
                rt = r.transpose(-1, -2)
                mq.add_(r @ k_cov @ rt)
                mk.add_(rt @ q_cov @ r)
            mq.div_(len(ds))
            mk.div_(len(ds))
        eye = torch.eye(2, device=mq.device, dtype=mq.dtype).view(1, 1, 2, 2)
        return mq + eps * eye, mk + eps * eye

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, width = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        self._update_covariances(q.detach(), k.detach())
        q = self._apply_rope(q)
        k = self._apply_rope(k)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.config.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, width)
        return self.o_proj(y)


class MLP(nn.Module):
    def __init__(self, config: OrbitGPTConfig) -> None:
        super().__init__()
        hidden = 4 * config.n_embd
        self.fc1 = nn.Linear(config.n_embd, hidden, bias=config.bias)
        self.fc2 = nn.Linear(hidden, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(F.gelu(self.fc1(x), approximate="tanh")))


class Block(nn.Module):
    def __init__(self, config: OrbitGPTConfig, layer_idx: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = RotaryAttention(config, layer_idx)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class OrbitGPT(nn.Module):
    """GPT-2-sized decoder with RoPE and otherwise conventional pre-norm blocks."""

    def __init__(self, config: OrbitGPTConfig) -> None:
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head.weight = self.wte.weight
        self.apply(self._init_weights)
        for name, param in self.named_parameters():
            if name.endswith(("attn.o_proj.weight", "mlp.fc2.weight")):
                nn.init.normal_(param, mean=0.0, std=0.02 / (2 * config.n_layer) ** 0.5)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def orbit_qk_pairs(self):
        return [
            {
                "layer": i,
                "module": block.attn,
                "q": block.attn.q_proj.weight,
                "k": block.attn.k_proj.weight,
            }
            for i, block in enumerate(self.blocks)
        ]

    def set_orbit_stat_collection(self, enabled: bool) -> None:
        for block in self.blocks:
            block.attn.collect_orbit_stats = enabled

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        if input_ids.size(1) > self.config.block_size:
            raise ValueError(
                f"sequence length {input_ids.size(1)} exceeds block_size {self.config.block_size}"
            )
        x = self.drop(self.wte(input_ids))
        for block in self.blocks:
            if self.config.gradient_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1))
        return SimpleNamespace(logits=logits, loss=loss)