"""One tiny synthetic optimizer step for the paper-tested baselines.

This is an API/update smoke test, not a reproduction of the paper campaign.
"""

from __future__ import annotations

import torch

from orbit import OrbitGPT, OrbitGPTConfig, build_paper_baseline


LRS = {
    "adamw": 1e-3,
    "muon": 2e-2,
    "normuon": 2e-2,
    "adamuon": 5e-3,
}


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


def main() -> None:
    torch.manual_seed(7)
    reference = tiny_model()
    initial_state = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    batch = torch.randint(0, 64, (2, 8), generator=torch.Generator().manual_seed(11))

    for name, lr in LRS.items():
        model = tiny_model()
        model.load_state_dict(initial_state)
        optimizer = build_paper_baseline(
            name,
            model,
            lr=lr,
            scalar_lr_mult=0.1,
            weight_decay=0.01,
        )

        before = float(model(batch, labels=batch).loss.detach())
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, labels=batch).loss
        loss.backward()
        optimizer.step()
        after = float(model(batch, labels=batch).loss.detach())
        print(f"{name:8s} before={before:.4f} after={after:.4f}")


if __name__ == "__main__":
    main()
