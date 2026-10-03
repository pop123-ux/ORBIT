from __future__ import annotations

import math

import pytest
import torch

from orbit import (
    AdaMuon,
    Muon,
    NorMuon,
    OrbitGPT,
    OrbitGPTConfig,
    PaperAdamW,
    PublishedAdaMuon,
    build_paper_baseline,
)


def tiny_model() -> OrbitGPT:
    return OrbitGPT(
        OrbitGPTConfig(
            vocab_size=64,
            block_size=8,
            n_layer=1,
            n_head=2,
            n_embd=16,
            dropout=0.0,
        )
    )


def backward_once(model: OrbitGPT) -> torch.Tensor:
    batch = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(123))
    loss = model(batch, labels=batch).loss
    loss.backward()
    return loss


@pytest.mark.parametrize("name", ["adamw", "muon", "normuon", "adamuon", "adamuon_ref"])
def test_paper_baseline_builder_smoke(name: str):
    torch.manual_seed(5)
    model = tiny_model()
    lr = 1e-3 if name == "adamw" else 1e-2
    optimizer = build_paper_baseline(
        name,
        model,
        lr=lr,
        scalar_lr_mult=0.1,
        weight_decay=0.01,
    )
    loss = backward_once(model)
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(param).all() for param in model.parameters())


def test_public_types_and_aliases():
    assert AdaMuon is PublishedAdaMuon
    model = tiny_model()
    assert isinstance(build_paper_baseline("muon", model, lr=0.02), Muon)
    model = tiny_model()
    assert isinstance(build_paper_baseline("normuon", model, lr=0.02), NorMuon)
    model = tiny_model()
    assert isinstance(build_paper_baseline("adamuon", model, lr=0.005), PublishedAdaMuon)


def test_paper_adamw_decay_routing():
    model = tiny_model()
    optimizer = PaperAdamW(model, lr=1e-3, weight_decay=0.07)
    seen_decay = False
    seen_nodecay = False
    for group in optimizer.param_groups:
        wd = float(group["weight_decay"])
        for param in group["params"]:
            if param.ndim >= 2:
                assert wd == pytest.approx(0.07)
                seen_decay = True
            else:
                assert wd == 0.0
                seen_nodecay = True
    assert seen_decay and seen_nodecay


def test_normuon_tracks_row_moment_on_spectral_parameters():
    model = tiny_model()
    optimizer = NorMuon(model, lr=0.02, adamw_lr=2e-3, weight_decay=0.01)
    backward_once(model)
    optimizer.step()

    spectral = [
        param
        for param in optimizer.param_groups[0]["params"]
        if optimizer._meta[id(param)] == "spectral"
    ]
    assert spectral
    state = optimizer.state[spectral[0]]
    assert state["row_second_moment"].shape == (spectral[0].shape[0],)
    assert int(state["normuon_step"]) == 1


def test_adamuon_matrix_candidate_is_point_two_rms():
    model = tiny_model()
    optimizer = PublishedAdaMuon(model, lr=0.005, adamw_lr=5e-4, weight_decay=0.01)
    backward_once(model)
    param = next(
        param
        for param in optimizer.param_groups[0]["params"]
        if optimizer._meta[id(param)] == "spectral" and param.grad is not None
    )
    update = optimizer._spectral_candidate(param, optimizer.param_groups[0]).float()
    rms = update.norm() / math.sqrt(update.numel())
    assert float(rms) == pytest.approx(0.2, rel=1e-5, abs=1e-6)
    assert "adamuon_second_moment" in optimizer.state[param]
