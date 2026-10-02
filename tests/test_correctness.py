from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from orbit import Muon, Orbit, OrbitGPT, OrbitGPTConfig
from orbit.model import RotaryAttention


def tiny_config(**overrides):
    values = dict(
        vocab_size=67,
        block_size=16,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
    )
    values.update(overrides)
    return OrbitGPTConfig(**values)


def test_rope_score_uses_causal_negative_relative_rotation():
    attn = RotaryAttention(tiny_config(n_embd=2), layer_idx=0)
    time = 9
    q = torch.zeros(1, 1, time, 2)
    k = torch.zeros(1, 1, time, 2)
    q_vec = torch.tensor([0.8, -1.1])
    k_vec = torch.tensor([1.7, 0.4])
    i, j = 8, 0
    q[0, 0, i] = q_vec
    k[0, 0, j] = k_vec

    q_rot = attn._apply_rope(q)
    k_rot = attn._apply_rope(k)
    score = q_rot[0, 0, i] @ k_rot[0, 0, j]

    r_negative = attn._causal_relative_rotation(i - j)[0, 0]
    expected = q_vec @ (r_negative @ k_vec)

    angle = attn.inv_freq[0] * float(i - j)
    c, s = angle.cos(), angle.sin()
    r_positive = torch.stack((torch.stack((c, -s)), torch.stack((s, c))))
    wrong_sign = q_vec @ (r_positive @ k_vec)

    assert torch.allclose(score, expected, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(score, wrong_sign, atol=1e-3, rtol=1e-3)


def test_orbit_metric_uses_same_causal_rotation_as_attention_score():
    attn = RotaryAttention(tiny_config(n_embd=2), layer_idx=0)
    with torch.no_grad():
        k_cov = torch.tensor([[4.0, 1.3], [1.3, 0.7]])
        q_cov = torch.tensor([[1.1, -0.6], [-0.6, 3.2]])
        attn.orbit_k_second_moment.copy_(k_cov.view(1, 1, 2, 2))
        attn.orbit_q_second_moment.copy_(q_cov.view(1, 1, 2, 2))

    delta = 8
    mq, mk = attn.orbit_metrics((delta,), rotate=True, eps=0.0)
    r = attn._causal_relative_rotation(delta)[0, 0]
    expected_mq = r @ k_cov @ r.T
    expected_mk = r.T @ q_cov @ r

    assert torch.allclose(mq[0, 0], expected_mq, atol=1e-6, rtol=1e-6)
    assert torch.allclose(mk[0, 0], expected_mk, atol=1e-6, rtol=1e-6)


def test_negative_delta_is_rejected_to_keep_convention_unambiguous():
    attn = RotaryAttention(tiny_config(n_embd=2), layer_idx=0)
    with pytest.raises(ValueError, match="causal distances"):
        attn.orbit_metrics((-1,), rotate=True)


def test_gradient_checkpointing_updates_statistics_once_per_forward_backward():
    model = OrbitGPT(tiny_config(gradient_checkpointing=True))
    Orbit(model, lr=0.01, adamw_lr=1e-3)
    x = torch.randint(0, model.config.vocab_size, (2, 12))
    loss = model(x, labels=x).loss
    loss.backward()

    assert [int(block.attn.orbit_stats_seen) for block in model.blocks] == [1]


def test_orbit_statistics_roundtrip_in_model_state_dict():
    config = tiny_config()
    model = OrbitGPT(config)
    Orbit(model, lr=0.01, adamw_lr=1e-3)
    x = torch.randint(0, config.vocab_size, (2, 12))
    model(x, labels=x)

    state = model.state_dict()
    assert "blocks.0.attn.orbit_q_second_moment" in state
    assert "blocks.0.attn.orbit_k_second_moment" in state
    assert "blocks.0.attn.orbit_stats_seen" in state

    restored = OrbitGPT(config)
    restored.load_state_dict(state)

    assert torch.equal(
        restored.blocks[0].attn.orbit_q_second_moment,
        model.blocks[0].attn.orbit_q_second_moment,
    )
    assert torch.equal(
        restored.blocks[0].attn.orbit_k_second_moment,
        model.blocks[0].attn.orbit_k_second_moment,
    )
    assert torch.equal(
        restored.blocks[0].attn.orbit_stats_seen,
        model.blocks[0].attn.orbit_stats_seen,
    )


@pytest.mark.parametrize("optimizer_cls", [Orbit, Muon])
def test_standard_scheduler_scales_matrix_and_auxiliary_learning_rates(optimizer_cls):
    model = OrbitGPT(tiny_config())
    optimizer = optimizer_cls(model, lr=0.02, adamw_lr=0.004)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    matrix_before = optimizer.param_groups[0]["lr"]
    aux_before = optimizer.auxiliary_lr()

    optimizer.step()
    scheduler.step()

    assert optimizer.param_groups[0]["lr"] == pytest.approx(matrix_before * 0.1)
    assert optimizer.auxiliary_lr() == pytest.approx(aux_before * 0.1)




@pytest.mark.parametrize("optimizer_cls", [Orbit, Muon])
def test_legacy_optimizer_checkpoint_migrates_auxiliary_lr_metadata(optimizer_cls):
    model = OrbitGPT(tiny_config())
    optimizer = optimizer_cls(model, lr=0.02, adamw_lr=0.004)
    state = optimizer.state_dict()
    group = state["param_groups"][0]
    ratio = group.pop("adamw_lr_ratio")
    group["adamw_lr"] = group["lr"] * ratio

    restored_model = OrbitGPT(tiny_config())
    restored = optimizer_cls(restored_model, lr=0.02, adamw_lr=0.004)
    restored.load_state_dict(state)

    assert restored.auxiliary_lr() == pytest.approx(0.004)


def _ddp_worker(rank: int, world_size: int, init_file: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        torch.manual_seed(1234)
        config = tiny_config()
        model = OrbitGPT(config)
        optimizer = Orbit(model, lr=0.01, adamw_lr=1e-3)
        ddp = DDP(model, broadcast_buffers=False)

        for step in range(3):
            generator = torch.Generator().manual_seed(10_000 + 100 * step + rank)
            x = torch.randint(
                0,
                config.vocab_size,
                (2, 12),
                generator=generator,
            )
            loss = ddp(x, labels=x).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        q_weight = model.blocks[0].attn.q_proj.weight.detach()
        q_cov = model.blocks[0].attn.orbit_q_second_moment.detach()

        gathered_weights = [torch.empty_like(q_weight) for _ in range(world_size)]
        gathered_covs = [torch.empty_like(q_cov) for _ in range(world_size)]
        dist.all_gather(gathered_weights, q_weight)
        dist.all_gather(gathered_covs, q_cov)

        for other in gathered_weights[1:]:
            assert torch.allclose(gathered_weights[0], other, atol=1e-7, rtol=1e-7)
        for other in gathered_covs[1:]:
            assert torch.allclose(gathered_covs[0], other, atol=1e-7, rtol=1e-7)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed unavailable")
def test_ddp_keeps_orbit_statistics_and_weights_synchronized(tmp_path):
    init_file = os.fspath(tmp_path / "ddp_init")
    mp.spawn(_ddp_worker, args=(2, init_file), nprocs=2, join=True)
