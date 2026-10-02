"""Minimal ORBIT training example on a deterministic next-token pattern."""

import torch

from orbit import Muon, Orbit, OrbitGPT, OrbitGPTConfig


def patterned_batch(batch: int, length: int, vocab_size: int, device: str) -> torch.Tensor:
    starts = torch.arange(batch, device=device)[:, None] * 3
    positions = torch.arange(length, device=device)[None, :]
    return (starts + positions) % vocab_size


def run(name: str = "orbit", steps: int = 20) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    model = OrbitGPT(
        OrbitGPTConfig(
            vocab_size=64,
            block_size=32,
            n_layer=1,
            n_head=4,
            n_embd=64,
            dropout=0.0,
        )
    ).to(device)

    if name == "orbit":
        optimizer = Orbit(model, lr=0.02, adamw_lr=3e-4, weight_decay=0.01)
    elif name == "muon":
        optimizer = Muon(model, lr=0.02, adamw_lr=3e-4, weight_decay=0.01)
    else:
        raise ValueError("name must be 'orbit' or 'muon'")

    x = patterned_batch(batch=8, length=32, vocab_size=64, device=device)
    first_loss = None

    for step in range(steps):
        loss = model(x, labels=x).loss
        if first_loss is None:
            first_loss = float(loss.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step in {0, steps - 1} or (step + 1) % 5 == 0:
            print(f"{name} step={step + 1:02d} loss={loss.item():.4f}")

    print(f"{name} loss change: {first_loss:.4f} -> {loss.item():.4f}")
    if name == "orbit":
        print("ORBIT diagnostics:", optimizer.diagnostics())


if __name__ == "__main__":
    run("orbit")
