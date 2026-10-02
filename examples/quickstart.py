"""Minimal ORBIT smoke-training example on synthetic tokens."""

import torch

from orbit import Muon, Orbit, OrbitGPT, OrbitGPTConfig


def run(name: str = "orbit", steps: int = 5) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(7)

    model = OrbitGPT(
        OrbitGPTConfig(
            vocab_size=1024,
            block_size=64,
            n_layer=2,
            n_head=4,
            n_embd=128,
            dropout=0.0,
        )
    ).to(device)

    if name == "orbit":
        optimizer = Orbit(model, lr=0.02, adamw_lr=3e-4, weight_decay=0.01)
    elif name == "muon":
        optimizer = Muon(model, lr=0.02, adamw_lr=3e-4, weight_decay=0.01)
    else:
        raise ValueError("name must be 'orbit' or 'muon'")

    for step in range(steps):
        x = torch.randint(0, 1024, (2, 64), device=device)
        loss = model(x, labels=x).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"{name} step={step+1} loss={loss.item():.4f}")

    if name == "orbit":
        print("ORBIT diagnostics:", optimizer.diagnostics())


if __name__ == "__main__":
    run("orbit")
