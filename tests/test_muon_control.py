import torch

from orbit import Muon, OrbitGPT, OrbitGPTConfig


def test_muon_control_does_not_collect_orbit_statistics():
    torch.manual_seed(3)
    model = OrbitGPT(
        OrbitGPTConfig(
            vocab_size=127,
            block_size=32,
            n_layer=2,
            n_head=4,
            n_embd=64,
            dropout=0.0,
        )
    )
    optimizer = Muon(model, lr=0.01, adamw_lr=3e-4)
    x = torch.randint(0, 127, (2, 16))
    loss = model(x, labels=x).loss
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert all(int(block.attn.orbit_stats_seen) == 0 for block in model.blocks)
