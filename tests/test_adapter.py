from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from orbit import Orbit, OrbitAdapter, QKPairSpec


class ExternalToyRoPE(nn.Module):
    """Tiny external-style model with separate Q/K projections and HF-like getters."""

    def __init__(self, vocab_size: int = 41, width: int = 8, n_head: int = 2) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.width = width
        self.n_head = n_head
        self.head_dim = width // n_head
        self.embed = nn.Embedding(vocab_size, width)
        self.q_proj = nn.Linear(width, width, bias=False)
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.o_proj = nn.Linear(width, width, bias=False)
        self.lm_head = nn.Linear(width, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        inv_freq = 1.0 / (
            10_000.0 ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def _rope(self, x: torch.Tensor) -> torch.Tensor:
        bsz, heads, time, dim = x.shape
        angles = torch.arange(time, device=x.device, dtype=torch.float32)[:, None] * self.inv_freq
        cos = angles.cos().to(x.dtype).view(1, 1, time, dim // 2, 1)
        sin = angles.sin().to(x.dtype).view(1, 1, time, dim // 2, 1)
        pair = x.view(bsz, heads, time, dim // 2, 2)
        a, b = pair[..., :1], pair[..., 1:]
        return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        x = self.embed(input_ids)
        bsz, time, _ = x.shape
        q = self.q_proj(x).view(bsz, time, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, time, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, time, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(self._rope(q), self._rope(k), v, is_causal=True)
        y = y.transpose(1, 2).reshape(bsz, time, self.width)
        logits = self.lm_head(self.o_proj(y))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, self.vocab_size), labels[:, 1:].reshape(-1)
            )
        return SimpleNamespace(logits=logits, loss=loss)


def test_external_adapter_collects_statistics_and_trains():
    torch.manual_seed(5)
    base = ExternalToyRoPE()
    model = OrbitAdapter(
        base,
        [QKPairSpec(base.q_proj, base.k_proj, n_head=base.n_head, name="toy.attn")],
    )
    optimizer = Orbit(model, lr=0.01, adamw_lr=1e-3, weight_decay=0.01)

    assert optimizer._meta[id(base.embed.weight)] == "aux_decay"
    before = base.q_proj.weight.detach().clone()
    x = torch.randint(0, base.vocab_size, (3, 12))
    loss = model(x, labels=x).loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    collector = model.collectors[0]
    assert int(collector.orbit_stats_seen) == 1
    assert not torch.equal(base.q_proj.weight, before)
    assert optimizer.diagnostics()["joint_restore"] > 0.0


def test_adapter_statistics_survive_state_dict_roundtrip():
    base = ExternalToyRoPE()
    model = OrbitAdapter(base, [QKPairSpec(base.q_proj, base.k_proj, n_head=base.n_head)])
    Orbit(model, lr=0.01, adamw_lr=1e-3)
    x = torch.randint(0, base.vocab_size, (2, 9))
    model(x, labels=x)
    state = model.state_dict()
    assert "collectors.0.orbit_q_second_moment" in state
    assert "collectors.0.orbit_k_second_moment" in state
    assert "collectors.0.orbit_stats_seen" in state


def test_adapter_rejects_ambiguous_gqa_shape_mapping():
    q_proj = nn.Linear(8, 8, bias=False)
    k_proj = nn.Linear(8, 4, bias=False)
    spec = QKPairSpec(q_proj, k_proj, n_head=2)
    with pytest.raises(ValueError, match="GQA/MQA"):
        spec.validate()
