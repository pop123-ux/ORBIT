#!/usr/bin/env python3
"""Four-shard, paper-grade ORBIT language-model campaign.

This runner deliberately reuses the inherited ASTRO paper campaign's pinned
FineWeb-Edu token cache, baseline optimizer implementations, shared LR schedule,
and decay routing, while using a GPT-2-sized RoPE model required by ORBIT.

The result stream is append-only JSONL per shard. Four Colab instances may write
concurrently because each owns a different file. Merge/freeze configurations
between phases with ``scripts/orbit_merge.py``.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
from pathlib import Path

import torch

import astro_lab as lab
import paper_campaign as paper
from orbit import Orbit, OrbitGPT, OrbitGPTConfig


DEFAULT_OPTIMIZERS = ("adamw", "muon", "normuon", "adamuon_ref", "astro_v2", "orbit")
ORBIT_VARIANTS = ("orbit", "orbit_norope", "orbit_diag", "orbit_identity")
SIZES = {
    "45M": dict(n_layer=8, n_head=8, n_embd=512, batch=16),
    "124M": dict(n_layer=12, n_head=12, n_embd=768, batch=8),
    "355M": dict(n_layer=24, n_head=16, n_embd=1024, batch=4),
}
DEFAULT_SEQ = 512
PHASE_DEFAULTS = {
    "discovery": dict(size="124M", steps=300, seeds=(0,)),
    "confirm": dict(size="124M", steps=900, seeds=(100, 101, 102, 103, 104)),
    "ablation": dict(size="124M", steps=900, seeds=(200, 201, 202)),
    "horizon": dict(size="124M", steps=2700, seeds=(400, 401)),
    "scale": dict(size="355M", steps=900, seeds=(300, 301)),
}


def code_digest() -> str:
    root = Path(__file__).resolve().parents[1]
    paths = [
        Path(__file__).resolve(),
        root / "src/orbit/model.py",
        root / "src/orbit/optimizer.py",
        root / "scripts/paper_campaign.py",
        root / "scripts/astro_lab.py",
    ]
    h = hashlib.sha256()
    for path in paths:
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()


def environment() -> dict[str, str]:
    import datasets
    import transformers

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": str(torch.version.cuda),
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
    }


def stable_rng(*parts: object) -> random.Random:
    payload = "|".join(map(str, parts)).encode()
    seed = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return random.Random(seed)


def sample_value(rng: random.Random, value):
    if isinstance(value, (tuple, list)) and len(value) == 2 and all(
        isinstance(x, (int, float)) for x in value
    ):
        low, high = float(value[0]), float(value[1])
        if low > 0 and high > 0:
            return math.exp(rng.uniform(math.log(low), math.log(high)))
        return rng.uniform(low, high)
    if isinstance(value, (tuple, list)):
        return rng.choice(list(value))
    return value


def search_space(name: str):
    base = "muon" if name in ORBIT_VARIANTS else name
    return paper.paper_space_for(base)


def sampled_config(name: str, trial: int) -> dict[str, float]:
    rng = stable_rng("orbit-discovery", name, trial)
    config = {key: sample_value(rng, value) for key, value in search_space(name).items()}
    config.setdefault("weight_decay", 0.05)
    config.setdefault("scalar_lr_mult", 0.1)
    return config


def load_best_configs(work_dir: Path) -> dict[str, dict]:
    path = work_dir / "merged" / "best_configs.json"
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Finish discovery on all shards, then run "
            "python scripts/orbit_merge.py --work-dir <same-dir> --phase discovery"
        )
    return json.loads(path.read_text())


def task_id(task: dict) -> str:
    payload = json.dumps(task, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def make_tasks(
    phase: str,
    *,
    work_dir: Path,
    trials: int,
    optimizers: tuple[str, ...],
    steps_override: int | None,
) -> list[dict]:
    defaults = PHASE_DEFAULTS[phase]
    size = defaults["size"]
    steps = int(steps_override or defaults["steps"])
    tasks: list[dict] = []

    if phase == "discovery":
        for name in optimizers:
            for trial in range(trials):
                task = {
                    "phase": phase,
                    "optimizer": name,
                    "trial": trial,
                    "seed": 0,
                    "size": size,
                    "steps": steps,
                    "config": sampled_config(name, trial),
                }
                task["task_id"] = task_id(task)
                tasks.append(task)
        return tasks

    best = load_best_configs(work_dir)
    if phase == "ablation":
        orbit_config = dict(best["orbit"]["config"])
        names = ORBIT_VARIANTS
    elif phase in {"horizon", "scale"}:
        names = tuple(name for name in ("muon", "normuon", "orbit") if name in best)
    else:
        names = tuple(name for name in optimizers if name in best)

    for name in names:
        if phase == "ablation":
            config = orbit_config
        else:
            config = dict(best[name]["config"])
        for seed in defaults["seeds"]:
            task = {
                "phase": phase,
                "optimizer": name,
                "trial": None,
                "seed": int(seed),
                "size": size,
                "steps": steps,
                "config": config,
            }
            task["task_id"] = task_id(task)
            tasks.append(task)
    return tasks


def build_optimizer(name: str, model: OrbitGPT, config: dict[str, float]):
    if name in ORBIT_VARIANTS:
        return Orbit(
            model,
            lr=float(config["lr"]),
            adamw_lr=float(config["lr"]) * float(config.get("scalar_lr_mult", 0.1)),
            weight_decay=float(config.get("weight_decay", 0.05)),
            functional_power=1.0,
            metric_condition_cap=100.0,
            variant=name,
        )
    return paper.paper_build_optimizer(name, model, config)


def get_tokens(*, synthetic: bool, seq: int, vocab: int):
    if synthetic:
        gen = torch.Generator().manual_seed(12345)
        total = max(200_000, seq * 500)
        tokens = torch.randint(0, vocab, (total,), generator=gen)
        cut = min(50_000, total // 5)
        return tokens[cut:], tokens[:cut]

    from transformers import GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokens = paper.paper_load_tokens(tokenizer, paper.TOTAL_TOKENS, paper.SHARED_CACHE)
    validation = tokens[: lab.VALIDATION_TOKENS]
    training = tokens[lab.VALIDATION_TOKENS :]
    return training, validation


def make_batch(source: torch.Tensor, *, batch: int, seq: int, gen: torch.Generator, device):
    starts = torch.randint(0, source.numel() - seq - 1, (batch,), generator=gen)
    x = torch.stack([source[int(s) : int(s) + seq] for s in starts])
    return x.to(device, non_blocking=True)


def lr_factor(step: int, steps: int) -> float:
    warmup = max(1, steps // 10)
    if step < warmup:
        return (step + 1) / (warmup + 1)
    return 0.1 + 0.45 * (
        1.0 + math.cos(math.pi * (step - warmup) / max(1, steps - warmup))
    )


def train_once(
    task: dict,
    *,
    data,
    seq: int,
    log_every: int,
    device: str,
) -> dict:
    shape = dict(SIZES[task["size"]])
    batch = shape.pop("batch")
    vocab = 50_257
    use_cuda = device.startswith("cuda")
    torch.manual_seed(task["seed"])
    if use_cuda:
        torch.cuda.manual_seed_all(task["seed"])
        torch.cuda.reset_peak_memory_stats()

    config = OrbitGPTConfig(
        vocab_size=vocab,
        block_size=seq,
        gradient_checkpointing=task["size"] == "355M",
        **shape,
    )
    model = OrbitGPT(config).to(device)
    optimizer = build_optimizer(task["optimizer"], model, task["config"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)
    training, validation = data
    gen = torch.Generator().manual_seed(task["seed"] + 4242)
    curve = []
    started = time.perf_counter()

    model.train()
    for step in range(task["steps"]):
        paper.schedule_optimizer_lrs(optimizer, lr_factor(step, task["steps"]))
        x = make_batch(training, batch=batch, seq=seq, gen=gen, device=device)
        optimizer.zero_grad(set_to_none=True)
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if use_cuda
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
            raise FloatingPointError(f"non-finite loss at step {step}: {float(loss)}")
        if step % log_every == 0 or step + 1 == task["steps"]:
            elapsed = time.perf_counter() - started
            point = {
                "step": step + 1,
                "train_loss": float(loss.detach().cpu()),
                "grad_norm": float(torch.as_tensor(grad_norm).detach().cpu()),
                "elapsed_sec": elapsed,
            }
            curve.append(point)
            print(
                f"[{task['task_id']}] {task['optimizer']:>14} "
                f"{step + 1:4d}/{task['steps']} loss={point['train_loss']:.4f} "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    model.eval()
    model.set_orbit_stat_collection(False)
    eval_gen = torch.Generator().manual_seed(777)
    values = []
    with torch.no_grad():
        for _ in range(20):
            x = make_batch(validation, batch=batch, seq=seq, gen=eval_gen, device=device)
            ctx = (
                torch.autocast(device_type="cuda", dtype=torch.float16)
                if use_cuda
                else contextlib.nullcontext()
            )
            with ctx:
                values.append(float(model(x, labels=x).loss.detach().cpu()))

    elapsed = time.perf_counter() - started
    diagnostics = optimizer.diagnostics() if isinstance(optimizer, Orbit) else {}
    record = dict(task)
    record.update(
        {
            "status": "ok",
            "val_loss": statistics.fmean(values),
            "val_loss_sd": statistics.stdev(values) if len(values) > 1 else 0.0,
            "seconds": elapsed,
            "curve": curve,
            "optimizer_diagnostics": diagnostics,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "code_digest": code_digest(),
            "environment": environment(),
            "device": device,
            "gpu": torch.cuda.get_device_name() if use_cuda else None,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated() if use_cuda else 0,
        }
    )
    del model, optimizer, scaler
    gc.collect()
    if use_cuda:
        torch.cuda.empty_cache()
    return record


def completed_ids(path: Path, *, digest: str, env: dict[str, str]) -> set[str]:
    """Return tasks completed by the exact current implementation/environment.

    After a numerical or algorithmic patch, old rows stay in the append-only log
    for provenance but are intentionally not considered resumable. Re-running the
    same command appends fresh rows for the same task IDs; merge keeps the latest
    successful row. This prevents a post-fix campaign from silently mixing code
    versions while preserving the audit trail.
    """
    if not path.exists():
        return set()
    found = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("status") == "ok"
            and row.get("code_digest") == digest
            and row.get("environment") == env
        ):
            found.add(row["task_id"])
    return found


def append_record(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=tuple(PHASE_DEFAULTS), default="discovery")
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--optimizers", nargs="*", default=list(DEFAULT_OPTIMIZERS))
    p.add_argument("--seq", type=int, default=DEFAULT_SEQ)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--steps-override", type=int)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--prepare-data", action="store_true")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must satisfy 0 <= shard-id < num-shards")
    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    if args.prepare_data:
        get_tokens(synthetic=args.synthetic, seq=args.seq, vocab=50_257)
        print(f"data ready: {paper.SHARED_CACHE}")
        return

    tasks = make_tasks(
        args.phase,
        work_dir=args.work_dir,
        trials=args.trials,
        optimizers=tuple(args.optimizers),
        steps_override=args.steps_override,
    )
    shard_tasks = [task for index, task in enumerate(tasks) if index % args.num_shards == args.shard_id]
    if args.list_tasks:
        print(json.dumps(shard_tasks, indent=2, sort_keys=True))
        return

    output = args.work_dir / "shards" / f"shard-{args.shard_id}" / f"{args.phase}.jsonl"
    digest = code_digest()
    env = environment()
    done = completed_ids(output, digest=digest, env=env)
    pending = [task for task in shard_tasks if task["task_id"] not in done]
    print(
        f"ORBIT phase={args.phase} shard={args.shard_id}/{args.num_shards} "
        f"tasks={len(shard_tasks)} pending={len(pending)} digest={digest[:12]}",
        flush=True,
    )
    if not pending:
        return
    data = get_tokens(synthetic=args.synthetic, seq=args.seq, vocab=50_257)
    campaign_started = time.perf_counter()
    finished_now = 0
    for task in pending:
        if args.max_tasks and finished_now >= args.max_tasks:
            break
        if args.max_minutes and finished_now and (
            time.perf_counter() - campaign_started
        ) / 60 >= args.max_minutes:
            print("max-minutes reached cleanly between tasks")
            break
        try:
            record = train_once(task, data=data, seq=args.seq, log_every=args.log_every, device=device)
        except Exception as exc:
            record = dict(task)
            record.update(
                {
                    "status": "error",
                    "error": repr(exc),
                    "code_digest": digest,
                    "environment": env,
                    "device": device,
                }
            )
            append_record(output, record)
            raise
        append_record(output, record)
        finished_now += 1
        print(
            f"finished {task['task_id']} {task['optimizer']} val={record['val_loss']:.5f} "
            f"time={record['seconds'] / 60:.1f}m",
            flush=True,
        )


if __name__ == "__main__":
    main()
