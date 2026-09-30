#!/usr/bin/env python3
"""Merge and validate post-scale ORBIT campaign phases."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import orbit_campaign as base
import orbit_postscale as post


def expected_tasks(work_dir: Path, phase: str) -> list[dict]:
    return post.make_tasks(phase, work_dir=work_dir, steps_override=None)


def load_rows(
    work_dir: Path,
    phase: str,
    *,
    allowed_task_ids: set[str],
) -> tuple[list[dict], int, int, int]:
    core_digest = base.code_digest()
    orchestration_digest = post.postscale_digest()
    rows: dict[str, dict] = {}
    stale_core = 0
    stale_orchestration = 0
    foreign = 0

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
            if row.get("code_digest") != core_digest:
                stale_core += 1
                continue
            if row.get("orchestration_digest") != orchestration_digest:
                stale_orchestration += 1
                continue
            task_id = row.get("task_id")
            if task_id not in allowed_task_ids:
                foreign += 1
                continue
            rows[task_id] = row

    ordered = sorted(
        rows.values(),
        key=lambda r: (
            r.get("analysis_label", r["optimizer"]),
            r["seed"],
            str(r.get("trial")),
        ),
    )
    return ordered, stale_core, stale_orchestration, foreign


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row.get("analysis_label", row["optimizer"])].append(row)

    summary = {}
    for label, items in sorted(grouped.items()):
        losses = [float(x["val_loss"]) for x in items]
        times = [float(x["seconds"]) for x in items]
        summary[label] = {
            "n": len(items),
            "mean_val_loss": statistics.fmean(losses),
            "median_val_loss": statistics.median(losses),
            "sd_val_loss": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "mean_seconds": statistics.fmean(times),
            "mean_peak_cuda_gb": statistics.fmean(
                float(x.get("peak_cuda_bytes", 0)) / (1024**3) for x in items
            ),
        }
    return summary


def require_complete(rows: list[dict], tasks: list[dict], *, phase: str) -> None:
    expected = {task["task_id"]: task for task in tasks}
    actual = {row["task_id"]: row for row in rows}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    if not missing and not extra:
        return

    missing_counts = Counter(
        expected[task_id].get("analysis_label", expected[task_id]["optimizer"])
        for task_id in missing
    )
    extra_counts = Counter(
        actual[task_id].get("analysis_label", actual[task_id]["optimizer"])
        for task_id in extra
    )
    pieces = [f"total={len(actual)}/{len(expected)}"]
    if missing_counts:
        pieces.append(
            "missing="
            + ",".join(f"{name}:{count}" for name, count in sorted(missing_counts.items()))
        )
    if extra_counts:
        pieces.append(
            "unexpected="
            + ",".join(f"{name}:{count}" for name, count in sorted(extra_counts.items()))
        )
    raise SystemExit(
        f"{phase}: incomplete exact post-scale campaign; do not use this summary: "
        + "; ".join(pieces)
    )


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def freeze_matched(rows: list[dict]) -> dict:
    best: dict[str, dict] = {}
    for row in rows:
        name = row["optimizer"]
        if name not in best or float(row["val_loss"]) < float(best[name]["val_loss"]):
            best[name] = {
                "val_loss": row["val_loss"],
                "trial": row["trial"],
                "config_id": row.get("config_id"),
                "config": row["config"],
                "task_id": row["task_id"],
                "code_digest": row["code_digest"],
                "orchestration_digest": row["orchestration_digest"],
                "environment": row["environment"],
            }
    if set(best) != set(post.MATCHED_OPTIMIZERS):
        raise SystemExit(
            f"matched_tune did not produce winners for {post.MATCHED_OPTIMIZERS}: "
            f"got {sorted(best)}"
        )
    return best


def load_legacy_rows(work_dir: Path, phase: str) -> list[dict]:
    path = work_dir / "merged" / f"{phase}.jsonl"
    if not path.exists():
        raise SystemExit(
            f"{path} is missing. Merge the legacy {phase} phase before adding ASTRO-v2."
        )
    rows = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    expected = {"muon", "normuon", "orbit"}
    names = {row["optimizer"] for row in rows}
    if names != expected:
        raise SystemExit(
            f"legacy {phase} rows changed; expected {sorted(expected)}, got {sorted(names)}"
        )
    digests = {row.get("code_digest") for row in rows}
    if digests != {base.code_digest()}:
        raise SystemExit(
            f"legacy {phase} core digest mismatch: {sorted(str(x) for x in digests)}"
        )
    return rows


def combine_strong_baseline(
    work_dir: Path,
    *,
    legacy_phase: str,
    astro_rows: list[dict],
) -> tuple[list[dict], dict]:
    legacy = load_legacy_rows(work_dir, legacy_phase)
    combined = legacy + astro_rows

    expected_seeds = set(post.POST_PHASES[f"astro_{legacy_phase}"]["seeds"])
    grouped: dict[str, set[int]] = defaultdict(set)
    for row in combined:
        grouped[row.get("analysis_label", row["optimizer"])].add(int(row["seed"]))

    expected_names = {"muon", "normuon", "orbit", "astro_v2"}
    if set(grouped) != expected_names:
        raise SystemExit(
            f"{legacy_phase}+ASTRO incomplete optimizer set: {sorted(grouped)}"
        )
    for name in expected_names:
        if grouped[name] != expected_seeds:
            raise SystemExit(
                f"{legacy_phase}+ASTRO seed mismatch for {name}: "
                f"{sorted(grouped[name])} vs {sorted(expected_seeds)}"
            )
    return combined, summarize(combined)


def paired_delta(rows: list[dict], a: str, b: str) -> dict:
    by_label: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        label = row.get("analysis_label", row["optimizer"])
        by_label[label][int(row["seed"])] = float(row["val_loss"])

    shared = sorted(set(by_label[a]) & set(by_label[b]))
    if not shared:
        raise ValueError(f"no paired seeds for {a} vs {b}")
    diffs = [by_label[a][seed] - by_label[b][seed] for seed in shared]
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0

    tcrit = {
        1: 12.706,
        2: 4.303,
        3: 3.182,
        4: 2.776,
        5: 2.571,
        6: 2.447,
        7: 2.365,
        8: 2.306,
        9: 2.262,
    }
    df = len(diffs) - 1
    if df >= 1:
        half = tcrit.get(df, 1.96) * sd / math.sqrt(len(diffs))
        ci95 = [mean - half, mean + half]
    else:
        ci95 = None

    return {
        "a": a,
        "b": b,
        "definition": "a_minus_b",
        "n": len(diffs),
        "mean_delta": mean,
        "sd_delta": sd,
        "ci95": ci95,
        "a_wins": sum(delta < 0 for delta in diffs),
        "b_wins": sum(delta > 0 for delta in diffs),
        "ties": sum(delta == 0 for delta in diffs),
        "min_delta": min(diffs),
        "max_delta": max(diffs),
        "per_seed": {str(seed): diff for seed, diff in zip(shared, diffs)},
    }


def interaction_delta(rows: list[dict]) -> dict:
    """2x2 optimizer-by-config interaction on the shared xconfig seeds."""
    by_label: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        label = row.get("analysis_label", row["optimizer"])
        by_label[label][int(row["seed"])] = float(row["val_loss"])
    labels = (
        "orbit_at_orbit_config",
        "muon_at_orbit_config",
        "orbit_at_muon_config",
        "muon_at_muon_config",
    )
    shared = sorted(set.intersection(*(set(by_label[label]) for label in labels)))
    values = []
    for seed in shared:
        mechanism_orbit_cfg = (
            by_label["orbit_at_orbit_config"][seed]
            - by_label["muon_at_orbit_config"][seed]
        )
        mechanism_muon_cfg = (
            by_label["orbit_at_muon_config"][seed]
            - by_label["muon_at_muon_config"][seed]
        )
        values.append(mechanism_orbit_cfg - mechanism_muon_cfg)
    return {
        "definition": "(ORBIT-Muon at ORBIT config) - (ORBIT-Muon at Muon config)",
        "n": len(values),
        "mean_interaction": statistics.fmean(values),
        "sd_interaction": statistics.stdev(values) if len(values) > 1 else 0.0,
        "per_seed": {str(seed): value for seed, value in zip(shared, values)},
    }


def write_phase_analysis(work_dir: Path, phase: str, rows: list[dict]) -> None:
    merged = work_dir / "merged"
    analysis: dict[str, object] = {}

    if phase == "xconfig":
        analysis["mechanism_at_orbit_config"] = paired_delta(
            rows, "orbit_at_orbit_config", "muon_at_orbit_config"
        )
        analysis["mechanism_at_muon_config"] = paired_delta(
            rows, "orbit_at_muon_config", "muon_at_muon_config"
        )
        analysis["recipe_effect_on_muon"] = paired_delta(
            rows, "muon_at_orbit_config", "muon_at_muon_config"
        )
        analysis["recipe_effect_on_orbit"] = paired_delta(
            rows, "orbit_at_orbit_config", "orbit_at_muon_config"
        )
        analysis["optimizer_by_config_interaction"] = interaction_delta(rows)
    elif phase == "matched_confirm":
        analysis["orbit_vs_muon"] = paired_delta(rows, "orbit", "muon")
    elif phase == "ablation_ext":
        for baseline in ("orbit_identity", "orbit_norope", "orbit_diag"):
            analysis[f"orbit_vs_{baseline}"] = paired_delta(rows, "orbit", baseline)

    if analysis:
        (merged / f"{phase}_analysis.json").write_text(
            json.dumps(analysis, indent=2, sort_keys=True) + "\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=post.POST_PHASE_ORDER + ("all",),
        default="all",
    )
    args = parser.parse_args()

    phases = post.POST_PHASE_ORDER if args.phase == "all" else (args.phase,)
    merged = args.work_dir / "merged"
    merged.mkdir(parents=True, exist_ok=True)

    for phase in phases:
        tasks = expected_tasks(args.work_dir, phase)
        allowed = {task["task_id"] for task in tasks}
        rows, stale_core, stale_orch, foreign = load_rows(
            args.work_dir,
            phase,
            allowed_task_ids=allowed,
        )
        if stale_core:
            print(f"{phase}: ignored {stale_core} stale-core rows")
        if stale_orch:
            print(f"{phase}: ignored {stale_orch} stale-orchestration rows")
        if foreign:
            print(f"{phase}: ignored {foreign} foreign current rows")
        if not rows:
            raise SystemExit(f"{phase}: no matching successful rows")

        envs = {json.dumps(row["environment"], sort_keys=True) for row in rows}
        if len(envs) != 1:
            raise SystemExit(f"{phase}: mixed Python/package environments")

        require_complete(rows, tasks, phase=phase)
        write_jsonl(merged / f"{phase}.jsonl", rows)
        summary = summarize(rows)
        (merged / f"{phase}_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

        if phase == "matched_tune":
            best = freeze_matched(rows)
            (merged / "matched_best_configs.json").write_text(
                json.dumps(best, indent=2, sort_keys=True) + "\n"
            )
            print("matched_tune winners frozen -> merged/matched_best_configs.json")

        write_phase_analysis(args.work_dir, phase, rows)

        if phase == "astro_horizon":
            combined, combined_summary = combine_strong_baseline(
                args.work_dir,
                legacy_phase="horizon",
                astro_rows=rows,
            )
            write_jsonl(merged / "horizon_with_astro.jsonl", combined)
            (merged / "horizon_with_astro_summary.json").write_text(
                json.dumps(combined_summary, indent=2, sort_keys=True) + "\n"
            )
            strong_analysis = {
                "orbit_vs_astro_v2": paired_delta(combined, "orbit", "astro_v2"),
                "orbit_vs_muon": paired_delta(combined, "orbit", "muon"),
                "orbit_vs_normuon": paired_delta(combined, "orbit", "normuon"),
            }
            (merged / "horizon_with_astro_analysis.json").write_text(
                json.dumps(strong_analysis, indent=2, sort_keys=True) + "\n"
            )

        if phase == "astro_scale":
            combined, combined_summary = combine_strong_baseline(
                args.work_dir,
                legacy_phase="scale",
                astro_rows=rows,
            )
            write_jsonl(merged / "scale_with_astro.jsonl", combined)
            (merged / "scale_with_astro_summary.json").write_text(
                json.dumps(combined_summary, indent=2, sort_keys=True) + "\n"
            )
            strong_analysis = {
                "orbit_vs_astro_v2": paired_delta(combined, "orbit", "astro_v2"),
                "orbit_vs_muon": paired_delta(combined, "orbit", "muon"),
                "orbit_vs_normuon": paired_delta(combined, "orbit", "normuon"),
            }
            (merged / "scale_with_astro_analysis.json").write_text(
                json.dumps(strong_analysis, indent=2, sort_keys=True) + "\n"
            )

        print(f"{phase}: merged {len(rows)} exact post-scale tasks")


if __name__ == "__main__":
    main()
