#!/usr/bin/env python3
"""Minimal ORBIT experiment runner.

The primary frozen Muon/ORBIT paper cells are driven by ``paper_run.py``.
This lower-level runner also exposes the non-ASTRO paper baselines so users can
exercise the tested optimizer definitions on the same model/data loop without
recreating the full broad campaign, tuning grid, or manuscript pipeline.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import torch

from orbit import (
    Muon,
    Orbit,
    OrbitGPT,
    OrbitGPTConfig,
    build_paper_baseline,
)


FINEWEB_DATASET = "HuggingFaceFW/fineweb-edu"
FINEWEB_CONFIG = "sample-10BT"
FINEWEB_REVISION = "v1.0.0"
VALIDATION_TOKENS = 400_000
TRAIN_TOKENS = 12_000_000
TOTAL_TOKENS = VALIDATION_TOKENS + TRAIN_TOKENS

SIZES = {
    "124M": dict(n_layer=12, n_head=12, n_embd=768, batch=8),
    "355M": dict(n_layer=24, n_head=16, n_embd=1024, batch=4),
}

OPTIMIZERS = (
    "adamw",
    "muon",
    "normuon",
    "adamuon",
    "orbit",
    "orbit_identity",
    "orbit_norope",
    "orbit_diag",
)


def source_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "src" / "orbit" / "model.py",
        root / "src" / "orbit" / "optimizer.py",
        root / "src" / "orbit" / "baselines.py",
        root / "src" / "orbit" / "paper_baselines.py",
        root / "experiments" / "matched.py",
    ]
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def environment() -> dict[str, str]:
    try:
        import datasets
        datasets_version = datasets.__version__
    except ImportError:
        datasets_version = "not-installed"
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = "not-installed"
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "datasets": datasets_version,
        "transformers": transformers_version,
    }


def prepare_token_cache(path: Path) -> torch.Tensor:
    if path.exists():
        tokens = torch.load(path, map_location="cpu")
        if tokens.numel() != TOTAL_TOKENS:
            raise RuntimeError(
                f"{path} contains {tokens.numel():,} tokens; expected {TOTAL_TOKENS:,}"
            )
        return tokens

    try:
        from datasets import load_dataset
        from transformers import GPT2TokenizerFast
    except ImportError as exc:
        raise SystemExit('install experiment dependencies with pip install -e ".[experiments]"') from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.model_max_length = 10**30
    stream = load_dataset(
        FINEWEB_DATASET,
        name=FINEWEB_CONFIG,
        split="train",
        streaming=True,
        revision=FINEWEB_REVISION,
    )

    collected: list[int] = []
    for record in stream:
        collected.extend(tokenizer(record["text"], add_special_tokens=False).input_ids)
        if len(collected) >= TOTAL_TOKENS:
            break

    tokens = torch.tensor(collected[:TOTAL_TOKENS], dtype=torch.long)
    if tokens.numel() != TOTAL_TOKENS:
        raise RuntimeError("FineWeb-Edu stream ended before the fixed token pool was built")
    torch.save(tokens, path)
    return tokens


def make_batch(
    source: torch.Tensor,
    *,
    batch: int,
    seq: int,
    generator: torch.Generator,
    device: str,
) -> torch.Tensor:
    starts = torch.randint(0, source.numel() - seq - 1, (batch,), generator=generator)
    x = torch.stack([source[int(s) : int(s) + seq] for s in starts])
    return x.to(device, non_blocking=True)


def lr_factor(step: int, steps: int) -> float:
    warmup = max(1, steps // 10)
    if step < warmup:
        return (step + 1) / (warmup + 1)
    return 0.1 + 0.45 * (
        1.0 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))
    )


def build_optimizer(args, model):
    if args.optimizer in {"adamw", "normuon", "adamuon"}:
        return build_paper_baseline(
            args.optimizer,
            model,
            lr=args.lr,
            scalar_lr_mult=args.scalar_lr_mult,
            weight_decay=args.weight_decay,
            beta2=args.beta2,
        )

    aux_lr = args.lr * args.scalar_lr_mult
    common = dict(
        lr=args.lr,
        adamw_lr=aux_lr,
        weight_decay=args.weight_decay,
        momentum=0.95,
        ns_steps=5,
    )
    if args.optimizer == "muon":
        return Muon(model, **common)
    return Orbit(
        model,
        variant=args.optimizer,
        functional_power=1.0,
        metric_eps=1e-5,
        metric_condition_cap=100.0,
        deltas=(1, 2, 4, 8, 16, 32, 64, 128),
        **common,
    )


def schedule(optimizer, factor: float) -> None:
    for group in optimizer.param_groups:
        group.setdefault("paper_base_lr", group["lr"])
        group["lr"] = group["paper_base_lr"] * factor


def run(args) -> dict:
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    tokens = prepare_token_cache(args.token_cache)
    validation = tokens[:VALIDATION_TOKENS]
    training = tokens[VALIDATION_TOKENS:]

    shape = dict(SIZES[args.size])
    batch = shape.pop("batch")
    torch.manual_seed(args.seed)
    if device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    model = OrbitGPT(
        OrbitGPTConfig(
            vocab_size=50_257,
            block_size=args.seq,
            gradient_checkpointing=args.size == "355M",
            **shape,
        )
    ).to(device)
    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    train_gen = torch.Generator().manual_seed(args.seed + 4242)
    curve = []
    started = time.perf_counter()

    model.train()
    for step in range(args.steps):
        schedule(optimizer, lr_factor(step, args.steps))
        x = make_batch(
            training,
            batch=batch,
            seq=args.seq,
            generator=train_gen,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if device.startswith("cuda")
            else contextlib.nullcontext()
        )
        with ctx:
            loss = model(x, labels=x).loss
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step + 1}")
        if step % args.log_every == 0 or step + 1 == args.steps:
            curve.append(
                {
                    "step": step + 1,
                    "train_loss": float(loss.detach().cpu()),
                    "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                    "elapsed_sec": time.perf_counter() - started,
                }
            )

    model.eval()
    model.set_orbit_stat_collection(False)
    eval_gen = torch.Generator().manual_seed(777)
    values = []
    with torch.no_grad():
        for _ in range(20):
            x = make_batch(
                validation,
                batch=batch,
                seq=args.seq,
                generator=eval_gen,
                device=device,
            )
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if device.startswith("cuda")
                else contextlib.nullcontext()
            )
            with ctx:
                values.append(float(model(x, labels=x).loss.detach().cpu()))

    elapsed = time.perf_counter() - started
    mean_loss = sum(values) / len(values)
    diagnostics = optimizer.diagnostics() if isinstance(optimizer, Orbit) else {}

    config = {
        "lr": args.lr,
        "scalar_lr_mult": args.scalar_lr_mult,
        "weight_decay": args.weight_decay,
        "beta2": args.beta2,
    }
    if args.optimizer != "adamw":
        config.update(
            {
                "momentum": 0.95,
                "newton_schulz_steps": 5,
            }
        )
    if args.optimizer.startswith("orbit"):
        config.update(
            {
                "functional_power": 1.0,
                "metric_eps": 1e-5,
                "metric_condition_cap": 100.0,
                "deltas": [1, 2, 4, 8, 16, 32, 64, 128],
            }
        )

    return {
        "schema_version": 1,
        "status": "ok",
        "optimizer": args.optimizer,
        "size": args.size,
        "steps": args.steps,
        "seed": args.seed,
        "sequence_length": args.seq,
        "batch_size": batch,
        "config": config,
        "val_loss": mean_loss,
        "val_losses": values,
        "seconds": elapsed,
        "peak_cuda_bytes": (
            int(torch.cuda.max_memory_allocated()) if device.startswith("cuda") else 0
        ),
        "curve": curve,
        "optimizer_diagnostics": diagnostics,
        "source_digest": source_digest(),
        "environment": environment(),
        "device": device,
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else None,
        "data": {
            "dataset": FINEWEB_DATASET,
            "config": FINEWEB_CONFIG,
            "revision": FINEWEB_REVISION,
            "validation_tokens": VALIDATION_TOKENS,
            "training_tokens": TRAIN_TOKENS,
        },
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", choices=OPTIMIZERS)
    p.add_argument("--size", choices=tuple(SIZES), default="124M")
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--seed", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--scalar-lr-mult", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument(
        "--beta2",
        type=float,
        default=0.95,
        help="AdamW beta2; the Muon-family auxiliary path remains fixed at the paper value 0.95",
    )
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--device", default="auto")
    p.add_argument("--token-cache", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--prepare-data", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.prepare_data:
        prepare_token_cache(args.token_cache)
        print(args.token_cache)
        return

    required = {
        "--optimizer": args.optimizer,
        "--seed": args.seed,
        "--lr": args.lr,
        "--scalar-lr-mult": args.scalar_lr_mult,
        "--weight-decay": args.weight_decay,
        "--output": args.output,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SystemExit("missing required run arguments: " + ", ".join(missing))

    record = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"{record['optimizer']} seed={record['seed']} val={record['val_loss']:.6f}")
    print(args.output)


if __name__ == "__main__":
    main()
