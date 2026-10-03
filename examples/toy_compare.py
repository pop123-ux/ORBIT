"""Tiny synthetic comparison of ORBIT and the paper-tested non-ASTRO baselines.

The script exists to make the optimizer APIs easy to exercise. It is not a
paper benchmark, tuning procedure, or reproduction of reported results.
"""

from __future__ import annotations

import argparse

import torch

from orbit import Orbit, OrbitGPT, OrbitGPTConfig, build_paper_baseline


LRS = {
    "adamw": 1e-3,
    "muon": 2e-2,
    "normuon": 2e-2,
    "adamuon": 5e-3,
    "orbit": 2e-2,
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


def make_optimizer(name: str, model: OrbitGPT):
    if name == "orbit":
        return Orbit(model, lr=LRS[name], adamw_lr=2e-3, weight_decay=0.01)
    return build_paper_baseline(
        name,
        model,
        lr=LRS[name],
        scalar_lr_mult=0.1,
        weight_decay=0.01,
    )


def run(name: str, state: dict[str, torch.Tensor], batches: list[torch.Tensor]) -> tuple[float, float]:
    model = tiny_model()
    model.load_state_dict(state)
    optimizer = make_optimizer(name, model)

    initial = float(model(batches[0], labels=batches[0]).loss.detach())
    for batch in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = model(batch, labels=batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    final = float(model(batches[0], labels=batches[0]).loss.detach())
    return initial, final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()
    if args.steps < 1:
        raise SystemExit("--steps must be at least 1")

    torch.manual_seed(7)
    reference = tiny_model()
    state = {key: value.detach().clone() for key, value in reference.state_dict().items()}
    generator = torch.Generator().manual_seed(23)
    batches = [torch.randint(0, 64, (2, 8), generator=generator) for _ in range(args.steps)]

    print("toy synthetic run (not paper reproduction)")
    for name in LRS:
        initial, final = run(name, state, batches)
        print(f"{name:8s} initial={initial:.4f} final={final:.4f}")


if __name__ == "__main__":
    main()
