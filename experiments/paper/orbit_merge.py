#!/usr/bin/env python3
"""Merge append-only ORBIT shard outputs and freeze campaign results safely.

Shard logs are append-only and intentionally retain historical runs. A merge
therefore applies two independent provenance filters:

1. the row must have been produced by the current ORBIT experiment digest; and
2. for post-discovery phases, its task id must be one of the exact tasks implied
   by the currently frozen ``best_configs.json``.

The second rule prevents an early confirmation run (started before discovery was
finally frozen) from contaminating the final confirmation summary even when both
runs used the same code digest.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from orbit_campaign import DEFAULT_OPTIMIZERS, code_digest, make_tasks


DISCOVERY_TRIALS = 5
DISCOVERY_OPTIMIZERS = tuple(DEFAULT_OPTIMIZERS)


def expected_tasks(work_dir: Path, phase: str) -> list[dict]:
    """Reconstruct the exact frozen campaign tasks for ``phase``."""
    return make_tasks(
        phase,
        work_dir=work_dir,
        trials=DISCOVERY_TRIALS,
        optimizers=DISCOVERY_OPTIMIZERS,
        steps_override=None,
    )


def load_rows(
    work_dir: Path,
    phase: str,
    *,
    digest: str,
    allowed_task_ids: set[str] | None = None,
) -> tuple[list[dict], int, int]:
    """Load latest successful row per task after provenance filtering."""
    rows: dict[str, dict] = {}
    stale_digest_rows = 0
    foreign_current_rows = 0
    for path in sorted((work_dir / "shards").glob(f"shard-*/{phase}.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "ok":
                continue
            if row.get("code_digest") != digest:
                stale_digest_rows += 1
                continue
            task_id = row.get("task_id")
            if allowed_task_ids is not None and task_id not in allowed_task_ids:
                foreign_current_rows += 1
                continue
            # A task can appear multiple times after a resume/retry. Keeping the
            # latest successful matching row preserves append-only provenance.
            rows[task_id] = row
    ordered = sorted(
        rows.values(),
        key=lambda r: (r["optimizer"], r["seed"], str(r.get("trial"))),
    )
    return ordered, stale_digest_rows, foreign_current_rows


def summarize(rows: list[dict]) -> dict:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["optimizer"]].append(row)
    summary = {}
    for name, items in sorted(grouped.items()):
        losses = [float(x["val_loss"]) for x in items]
        times = [float(x["seconds"]) for x in items]
        summary[name] = {
            "n": len(items),
            "mean_val_loss": statistics.fmean(losses),
            "sd_val_loss": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "median_val_loss": statistics.median(losses),
            "mean_seconds": statistics.fmean(times),
            "mean_peak_cuda_gb": statistics.fmean(
                [float(x.get("peak_cuda_bytes", 0)) / (1024**3) for x in items]
            ),
        }
    return summary


def freeze_discovery(rows: list[dict]) -> dict:
    best = {}
    for row in rows:
        name = row["optimizer"]
        if name not in best or row["val_loss"] < best[name]["val_loss"]:
            best[name] = {
                "val_loss": row["val_loss"],
                "trial": row.get("trial"),
                "config": row["config"],
                "task_id": row["task_id"],
                "code_digest": row["code_digest"],
                "environment": row["environment"],
            }
    return best


def require_complete(rows: list[dict], tasks: list[dict], *, phase: str) -> None:
    """Refuse to summarize/freeze a partial or unexpected campaign phase."""
    expected = {task["task_id"]: task for task in tasks}
    actual = {row["task_id"]: row for row in rows}
    missing_ids = sorted(set(expected) - set(actual))
    extra_ids = sorted(set(actual) - set(expected))
    if not missing_ids and not extra_ids:
        return

    missing_counts = Counter(expected[task_id]["optimizer"] for task_id in missing_ids)
    extra_counts = Counter(actual[task_id]["optimizer"] for task_id in extra_ids)
    pieces = [f"total={len(actual)}/{len(expected)}"]
    if missing_counts:
        pieces.append(
            "missing=" + ",".join(f"{name}:{count}" for name, count in sorted(missing_counts.items()))
        )
    if extra_counts:
        pieces.append(
            "unexpected=" + ",".join(f"{name}:{count}" for name, count in sorted(extra_counts.items()))
        )
    raise SystemExit(
        f"{phase}: incomplete exact frozen campaign; do not use this summary yet: "
        + "; ".join(pieces)
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument(
        "--phase",
        choices=("discovery", "confirm", "ablation", "horizon", "scale", "all"),
        default="all",
    )
    args = p.parse_args()
    phases = (
        ("discovery", "confirm", "ablation", "horizon", "scale")
        if args.phase == "all"
        else (args.phase,)
    )
    current_digest = code_digest()
    print(f"merge current digest={current_digest[:12]}")

    merged_dir = args.work_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)
    full_summary = {}
    existing_summary = merged_dir / "summary.json"
    if existing_summary.exists():
        full_summary.update(json.loads(existing_summary.read_text()))

    for phase in phases:
        tasks = expected_tasks(args.work_dir, phase)
        allowed_ids = {task["task_id"] for task in tasks}
        rows, stale_digest_rows, foreign_current_rows = load_rows(
            args.work_dir,
            phase,
            digest=current_digest,
            allowed_task_ids=allowed_ids,
        )
        if stale_digest_rows:
            print(
                f"{phase}: ignored {stale_digest_rows} successful historical rows "
                "from stale code digests"
            )
        if foreign_current_rows:
            print(
                f"{phase}: ignored {foreign_current_rows} successful current-digest rows "
                "that are not part of the exact frozen campaign"
            )
        if not rows:
            raise SystemExit(
                f"{phase}: no matching completed rows for current digest "
                f"{current_digest[:12]}"
            )

        digests = {row["code_digest"] for row in rows}
        if digests != {current_digest}:
            raise SystemExit(f"{phase}: internal digest filter failure: {sorted(digests)}")
        envs = {json.dumps(row["environment"], sort_keys=True) for row in rows}
        if len(envs) != 1:
            raise SystemExit(
                f"{phase}: mixed Python/package environments detected. "
                "Recreate matching Colabs before merging."
            )

        require_complete(rows, tasks, phase=phase)

        write_jsonl(merged_dir / f"{phase}.jsonl", rows)
        phase_summary = summarize(rows)
        full_summary[phase] = phase_summary
        (merged_dir / f"{phase}_summary.json").write_text(
            json.dumps(phase_summary, indent=2, sort_keys=True) + "\n"
        )
        if phase == "discovery":
            best = freeze_discovery(rows)
            (merged_dir / "best_configs.json").write_text(
                json.dumps(best, indent=2, sort_keys=True) + "\n"
            )
            print("discovery configurations frozen -> merged/best_configs.json")
        print(
            f"{phase}: merged {len(rows)} exact current-digest frozen-campaign tasks"
        )

    (merged_dir / "summary.json").write_text(
        json.dumps(full_summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
