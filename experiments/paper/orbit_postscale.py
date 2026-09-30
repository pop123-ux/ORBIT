#!/usr/bin/env python3
"""Post-scale ORBIT campaign.

This layer is deliberately additive: it imports the frozen core campaign rather
than modifying it, so all pre-existing discovery/confirm/ablation/horizon/scale
rows remain valid under the original core code digest.

New phases address the four remaining paper-grade questions:
1. xconfig: 2x2 optimizer/config cross-over to separate mechanism from recipe;
2. matched_tune + matched_confirm: Muon and ORBIT receive the same 10 candidate
   hyperparameter configurations at the 900-step target, then are compared on
   10 held-out seeds;
3. ablation_ext: 10-seed mechanism ablation using the matched ORBIT config;
4. astro_horizon + astro_scale: add the strongest ASTRO-v2 baseline to the
   already-completed long-horizon and 355M transfer cells.

Each phase uses four logical shards and can be scheduled across any three Colab
instances. Shard identity is encoded in the output path, not tied to a machine.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import orbit_campaign as base


POST_TUNE_TRIALS = 10
POST_PHASES = {
    "xconfig": dict(size="124M", steps=900, seeds=(100, 101, 102, 103, 104)),
    "matched_tune": dict(size="124M", steps=900, seeds=(0,)),
    "matched_confirm": dict(size="124M", steps=900, seeds=tuple(range(500, 510))),
    "ablation_ext": dict(size="124M", steps=900, seeds=tuple(range(600, 610))),
    "astro_horizon": dict(size="124M", steps=2700, seeds=(400, 401)),
    "astro_scale": dict(size="355M", steps=900, seeds=(300, 301)),
}
POST_PHASE_ORDER = tuple(POST_PHASES)
MATCHED_OPTIMIZERS = ("muon", "orbit")
EXTENDED_ABLATIONS = ("orbit", "orbit_norope", "orbit_diag", "orbit_identity")


def postscale_digest() -> str:
    """Fingerprint only the post-scale orchestration layer.

    The underlying training/model/optimizer implementation retains the core
    digest produced by ``orbit_campaign.code_digest()``.
    """
    path = Path(__file__).resolve()
    h = hashlib.sha256()
    h.update(path.read_bytes())
    h.update(b"\0")
    return h.hexdigest()


def matched_candidate(trial: int) -> dict[str, float]:
    """Return one shared Muon-space candidate used by both Muon and ORBIT."""
    rng = base.stable_rng("orbit-matched-shared-900", int(trial))
    config = {
        key: base.sample_value(rng, value)
        for key, value in base.search_space("muon").items()
    }
    config.setdefault("weight_decay", 0.05)
    config.setdefault("scalar_lr_mult", 0.1)
    return config


def load_matched_best(work_dir: Path) -> dict[str, dict]:
    path = work_dir / "merged" / "matched_best_configs.json"
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist. Finish + merge matched_tune before this phase."
        )
    return json.loads(path.read_text())


def _task(
    *,
    phase: str,
    optimizer: str,
    analysis_label: str,
    config: dict,
    seed: int,
    size: str,
    steps: int,
    trial: int | None = None,
    config_source: str,
    config_id: str | None = None,
) -> dict:
    task = {
        "phase": phase,
        "optimizer": optimizer,
        "analysis_label": analysis_label,
        "trial": trial,
        "seed": int(seed),
        "size": size,
        "steps": int(steps),
        "config": dict(config),
        "config_source": config_source,
    }
    if config_id is not None:
        task["config_id"] = config_id
    task["task_id"] = base.task_id(task)
    return task


def make_tasks(
    phase: str,
    *,
    work_dir: Path,
    steps_override: int | None = None,
) -> list[dict]:
    if phase not in POST_PHASES:
        raise ValueError(f"unknown post-scale phase {phase!r}")

    defaults = POST_PHASES[phase]
    size = defaults["size"]
    steps = int(steps_override or defaults["steps"])
    seeds = defaults["seeds"]
    tasks: list[dict] = []

    if phase == "xconfig":
        best = base.load_best_configs(work_dir)
        cells = (
            ("muon", "muon_at_muon_config", "muon"),
            ("muon", "muon_at_orbit_config", "orbit"),
            ("orbit", "orbit_at_muon_config", "muon"),
            ("orbit", "orbit_at_orbit_config", "orbit"),
        )
        for optimizer, label, source in cells:
            config = dict(best[source]["config"])
            for seed in seeds:
                tasks.append(
                    _task(
                        phase=phase,
                        optimizer=optimizer,
                        analysis_label=label,
                        config=config,
                        seed=seed,
                        size=size,
                        steps=steps,
                        config_source=f"discovery:{source}",
                    )
                )
        return tasks

    if phase == "matched_tune":
        for trial in range(POST_TUNE_TRIALS):
            config = matched_candidate(trial)
            config_id = f"shared-{trial:02d}"
            for optimizer in MATCHED_OPTIMIZERS:
                tasks.append(
                    _task(
                        phase=phase,
                        optimizer=optimizer,
                        analysis_label=optimizer,
                        config=config,
                        seed=0,
                        size=size,
                        steps=steps,
                        trial=trial,
                        config_source="shared-muon-space",
                        config_id=config_id,
                    )
                )
        return tasks

    if phase == "matched_confirm":
        matched = load_matched_best(work_dir)
        for optimizer in MATCHED_OPTIMIZERS:
            config = dict(matched[optimizer]["config"])
            for seed in seeds:
                tasks.append(
                    _task(
                        phase=phase,
                        optimizer=optimizer,
                        analysis_label=optimizer,
                        config=config,
                        seed=seed,
                        size=size,
                        steps=steps,
                        config_source=f"matched_tune:{optimizer}",
                    )
                )
        return tasks

    if phase == "ablation_ext":
        matched = load_matched_best(work_dir)
        orbit_config = dict(matched["orbit"]["config"])
        for optimizer in EXTENDED_ABLATIONS:
            for seed in seeds:
                tasks.append(
                    _task(
                        phase=phase,
                        optimizer=optimizer,
                        analysis_label=optimizer,
                        config=orbit_config,
                        seed=seed,
                        size=size,
                        steps=steps,
                        config_source="matched_tune:orbit",
                    )
                )
        return tasks

    best = base.load_best_configs(work_dir)
    astro_config = dict(best["astro_v2"]["config"])
    for seed in seeds:
        tasks.append(
            _task(
                phase=phase,
                optimizer="astro_v2",
                analysis_label="astro_v2",
                config=astro_config,
                seed=seed,
                size=size,
                steps=steps,
                config_source="discovery:astro_v2",
            )
        )
    return tasks


def completed_ids(
    path: Path,
    *,
    core_digest: str,
    orchestration_digest: str,
    env: dict[str, str],
) -> set[str]:
    if not path.exists():
        return set()
    found: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            row.get("status") == "ok"
            and row.get("code_digest") == core_digest
            and row.get("orchestration_digest") == orchestration_digest
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
    p.add_argument("--phase", choices=POST_PHASE_ORDER, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=4)
    p.add_argument("--seq", type=int, default=base.DEFAULT_SEQ)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--max-minutes", type=float, default=0.0)
    p.add_argument("--max-tasks", type=int, default=0)
    p.add_argument("--steps-override", type=int)
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--list-tasks", action="store_true")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_id < args.num_shards:
        raise SystemExit("shard-id must satisfy 0 <= shard-id < num-shards")

    device = (
        "cuda"
        if args.device == "auto" and base.torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    if device.startswith("cuda") and not base.torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    tasks = make_tasks(
        args.phase,
        work_dir=args.work_dir,
        steps_override=args.steps_override,
    )
    shard_tasks = [
        task
        for index, task in enumerate(tasks)
        if index % args.num_shards == args.shard_id
    ]
    if args.list_tasks:
        print(json.dumps(shard_tasks, indent=2, sort_keys=True))
        return

    output = (
        args.work_dir
        / "shards"
        / f"shard-{args.shard_id}"
        / f"{args.phase}.jsonl"
    )
    core_digest = base.code_digest()
    orchestration_digest = postscale_digest()
    env = base.environment()
    done = completed_ids(
        output,
        core_digest=core_digest,
        orchestration_digest=orchestration_digest,
        env=env,
    )
    pending = [task for task in shard_tasks if task["task_id"] not in done]

    print(
        f"ORBIT post-scale phase={args.phase} "
        f"shard={args.shard_id}/{args.num_shards} "
        f"tasks={len(shard_tasks)} pending={len(pending)} "
        f"core={core_digest[:12]} post={orchestration_digest[:12]}",
        flush=True,
    )
    if not pending:
        return

    data = base.get_tokens(
        synthetic=args.synthetic,
        seq=args.seq,
        vocab=50_257,
    )
    campaign_started = time.perf_counter()
    finished_now = 0

    for task in pending:
        if args.max_tasks and finished_now >= args.max_tasks:
            break
        if (
            args.max_minutes
            and finished_now
            and (time.perf_counter() - campaign_started) / 60 >= args.max_minutes
        ):
            print("max-minutes reached cleanly between tasks")
            break

        try:
            record = base.train_once(
                task,
                data=data,
                seq=args.seq,
                log_every=args.log_every,
                device=device,
            )
            record["orchestration_digest"] = orchestration_digest
        except Exception as exc:
            record = dict(task)
            record.update(
                {
                    "status": "error",
                    "error": repr(exc),
                    "code_digest": core_digest,
                    "orchestration_digest": orchestration_digest,
                    "environment": env,
                    "device": device,
                }
            )
            append_record(output, record)
            raise

        append_record(output, record)
        finished_now += 1
        print(
            f"finished {task['task_id']} {task['analysis_label']} "
            f"val={record['val_loss']:.5f} "
            f"time={record['seconds'] / 60:.1f}m",
            flush=True,
        )
        gc.collect()


if __name__ == "__main__":
    main()
