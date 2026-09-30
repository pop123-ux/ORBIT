#!/usr/bin/env python3
"""Freeze and validate the empirical evidence used by the ORBIT paper.

This script is the single source of truth between experiment JSONLs and paper
figures/tables. It intentionally validates the *final* experimental hierarchy:

1. matched Muon-vs-ORBIT confirmation (primary causal comparison);
2. cross-configuration isolation (hyperparameter confound);
3. expanded mechanism ablation;
4. long-horizon and 355M transfer including ASTRO;
5. broad independently tuned confirmation (secondary context).

The script writes a compact paper_results.json plus a provenance manifest. Paper
generation should consume these outputs rather than the legacy merged/summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

CORE_DIGEST = "de8b994a734276871770c6c67648117d3613d0954f3ae90d7e7b69246308c200"

EXPECTED = {
    "confirm": {
        "path": "confirm.jsonl",
        "group_key": "optimizer",
        "groups": {
            "adamw": (100, 101, 102, 103, 104),
            "adamuon_ref": (100, 101, 102, 103, 104),
            "muon": (100, 101, 102, 103, 104),
            "normuon": (100, 101, 102, 103, 104),
            "astro_v2": (100, 101, 102, 103, 104),
            "orbit": (100, 101, 102, 103, 104),
        },
    },
    "xconfig": {
        "path": "xconfig.jsonl",
        "group_key": "analysis_label",
        "groups": {
            "muon_at_muon_config": (100, 101, 102, 103, 104),
            "muon_at_orbit_config": (100, 101, 102, 103, 104),
            "orbit_at_muon_config": (100, 101, 102, 103, 104),
            "orbit_at_orbit_config": (100, 101, 102, 103, 104),
        },
    },
    "matched_tune": {
        "path": "matched_tune.jsonl",
        "group_key": "optimizer",
        "groups": {"muon": (0,) * 10, "orbit": (0,) * 10},
        "trials": tuple(range(10)),
    },
    "matched_confirm": {
        "path": "matched_confirm.jsonl",
        "group_key": "optimizer",
        "groups": {
            "muon": tuple(range(500, 510)),
            "orbit": tuple(range(500, 510)),
        },
    },
    "ablation_ext": {
        "path": "ablation_ext.jsonl",
        "group_key": "optimizer",
        "groups": {
            "orbit": tuple(range(600, 610)),
            "orbit_identity": tuple(range(600, 610)),
            "orbit_norope": tuple(range(600, 610)),
            "orbit_diag": tuple(range(600, 610)),
        },
    },
    "horizon_with_astro": {
        "path": "horizon_with_astro.jsonl",
        "group_key": "optimizer",
        "groups": {
            "muon": (400, 401),
            "normuon": (400, 401),
            "astro_v2": (400, 401),
            "orbit": (400, 401),
        },
    },
    "scale_with_astro": {
        "path": "scale_with_astro.jsonl",
        "group_key": "optimizer",
        "groups": {
            "muon": (300, 301),
            "normuon": (300, 301),
            "astro_v2": (300, 301),
            "orbit": (300, 301),
        },
    },
}

T95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{lineno}: invalid JSON: {exc}") from exc
    return rows


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finite_positive(value) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x) and x > 0


def grouped(rows: Iterable[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(row[key])].append(row)
    return dict(out)


def summary(rows: list[dict], key: str = "optimizer") -> dict:
    out = {}
    for name, items in sorted(grouped(rows, key).items()):
        losses = [float(x["val_loss"]) for x in items]
        times = [float(x["seconds"]) for x in items]
        memory = [float(x.get("peak_cuda_bytes", 0)) / (1024 ** 3) for x in items]
        out[name] = {
            "n": len(items),
            "mean_val_loss": statistics.fmean(losses),
            "median_val_loss": statistics.median(losses),
            "sd_val_loss": statistics.stdev(losses) if len(losses) > 1 else 0.0,
            "mean_seconds": statistics.fmean(times),
            "mean_peak_cuda_gb": statistics.fmean(memory),
        }
    return out


def paired(rows: list[dict], a: str, b: str, key: str = "optimizer") -> dict:
    by_group = grouped(rows, key)
    aa = {int(x["seed"]): float(x["val_loss"]) for x in by_group[a]}
    bb = {int(x["seed"]): float(x["val_loss"]) for x in by_group[b]}
    seeds = sorted(set(aa) & set(bb))
    if not seeds:
        raise ValueError(f"no paired seeds for {a} vs {b}")
    diffs = [aa[s] - bb[s] for s in seeds]
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if len(diffs) > 1 else 0.0
    df = len(diffs) - 1
    half = T95.get(df, 1.96) * sd / math.sqrt(len(diffs)) if df >= 1 else None
    return {
        "a": a,
        "b": b,
        "definition": "a_minus_b",
        "n": len(diffs),
        "mean_delta": mean,
        "sd_delta": sd,
        "ci95": [mean - half, mean + half] if half is not None else None,
        "a_wins": sum(x < 0 for x in diffs),
        "b_wins": sum(x > 0 for x in diffs),
        "ties": sum(x == 0 for x in diffs),
        "per_seed": {str(s): d for s, d in zip(seeds, diffs)},
    }


def validate_phase(name: str, rows: list[dict]) -> list[str]:
    spec = EXPECTED[name]
    warnings: list[str] = []
    expected_groups = spec["groups"]
    key = spec["group_key"]
    actual_groups = grouped(rows, key)

    if set(actual_groups) != set(expected_groups):
        raise ValueError(
            f"{name}: groups mismatch; expected {sorted(expected_groups)}, "
            f"got {sorted(actual_groups)}"
        )

    seen_tasks = [x.get("task_id") for x in rows]
    if None in seen_tasks or len(set(seen_tasks)) != len(seen_tasks):
        raise ValueError(f"{name}: missing or duplicate task_id")

    for group_name, expected_seeds in expected_groups.items():
        items = actual_groups[group_name]
        actual_seeds = sorted(int(x["seed"]) for x in items)
        if name == "matched_tune":
            if len(items) != 10 or set(actual_seeds) != {0}:
                raise ValueError(f"{name}:{group_name}: expected ten seed-0 trials")
            trials = sorted(int(x["trial"]) for x in items)
            if trials != list(spec["trials"]):
                raise ValueError(
                    f"{name}:{group_name}: trials {trials} != {list(spec['trials'])}"
                )
        elif actual_seeds != sorted(expected_seeds):
            raise ValueError(
                f"{name}:{group_name}: seeds {actual_seeds} != {sorted(expected_seeds)}"
            )

    for row in rows:
        if row.get("status") != "ok":
            raise ValueError(f"{name}: non-success row {row.get('task_id')}")
        if row.get("code_digest") != CORE_DIGEST:
            raise ValueError(
                f"{name}: core digest changed for {row.get('task_id')}: "
                f"{row.get('code_digest')}"
            )
        if not finite_positive(row.get("val_loss")):
            raise ValueError(f"{name}: invalid val_loss in {row.get('task_id')}")
        if not finite_positive(row.get("seconds")):
            raise ValueError(f"{name}: invalid runtime in {row.get('task_id')}")

    envs = {json.dumps(x.get("environment", {}), sort_keys=True) for x in rows}
    if len(envs) != 1:
        raise ValueError(f"{name}: mixed software environments")

    gpus = {x.get("gpu") or x.get("gpu_name") or x.get("device_name") for x in rows}
    gpus.discard(None)
    if len(gpus) > 1:
        warnings.append(f"{name}: multiple GPU labels observed: {sorted(gpus)}")

    return warnings


def validate_matched_configs(merged: Path, phases: dict[str, list[dict]]) -> dict:
    best = load_json(merged / "matched_best_configs.json")
    muon_cfg = best["muon"]["config"]
    orbit_cfg = best["orbit"]["config"]
    if muon_cfg != orbit_cfg:
        raise ValueError(
            "matched_tune winners differ; matched_confirm is not a strict identical-config test"
        )
    if best["muon"].get("config_id") != best["orbit"].get("config_id"):
        raise ValueError("matched_tune winners do not share the same config_id")

    for row in phases["matched_confirm"]:
        if row["config"] != muon_cfg:
            raise ValueError(
                f"matched_confirm:{row['task_id']}: config differs from frozen shared winner"
            )

    for row in phases["ablation_ext"]:
        if row["config"] != orbit_cfg:
            raise ValueError(
                f"ablation_ext:{row['task_id']}: config differs from frozen ORBIT winner"
            )
    return best


def interaction(phases: dict[str, list[dict]]) -> dict:
    rows = phases["xconfig"]
    by = grouped(rows, "analysis_label")
    labels = (
        "orbit_at_orbit_config",
        "muon_at_orbit_config",
        "orbit_at_muon_config",
        "muon_at_muon_config",
    )
    maps = {
        label: {int(x["seed"]): float(x["val_loss"]) for x in by[label]}
        for label in labels
    }
    seeds = sorted(set.intersection(*(set(maps[label]) for label in labels)))
    vals = []
    for seed in seeds:
        mech_orbit = maps["orbit_at_orbit_config"][seed] - maps["muon_at_orbit_config"][seed]
        mech_muon = maps["orbit_at_muon_config"][seed] - maps["muon_at_muon_config"][seed]
        vals.append(mech_orbit - mech_muon)
    return {
        "definition": "(ORBIT-Muon at ORBIT config) - (ORBIT-Muon at Muon config)",
        "n": len(vals),
        "mean_interaction": statistics.fmean(vals),
        "sd_interaction": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "per_seed": {str(s): v for s, v in zip(seeds, vals)},
    }



POSTSCALE_REPAIR = {
    "xconfig": "xconfig",
    "matched_tune": "matched_tune",
    "matched_confirm": "matched_confirm",
    "ablation_ext": "ablation_ext",
    "horizon_with_astro": "astro_horizon",
    "scale_with_astro": "astro_scale",
}


def repair_missing_postscale_merges(work_dir: Path) -> None:
    """Reconstruct missing merged artifacts from completed shard records.

    Fresh notebook sessions occasionally expose the persistent shard files before a
    previously generated merged JSONL appears through the Drive mount. Re-merging is
    deterministic and performs no training, so the evidence freeze can safely repair
    these derived files before validation.
    """
    shard_root = work_dir / "shards"
    merged = work_dir / "merged"
    merge_script = Path(__file__).with_name("orbit_postscale_merge.py")

    for target_phase, source_phase in POSTSCALE_REPAIR.items():
        target = merged / EXPECTED[target_phase]["path"]
        if target.exists():
            continue
        sources = list(shard_root.glob(f"shard-*/{source_phase}.jsonl"))
        if not sources:
            continue
        print(
            f"repairing missing merged artifact {target.name} "
            f"from {len(sources)} persisted shard file(s)"
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(merge_script),
                    "--work-dir",
                    str(work_dir),
                    "--phase",
                    source_phase,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Could not reconstruct {target.name} from persisted shard records."
            ) from exc


def build(work_dir: Path, *, allow_incomplete: bool = False) -> tuple[dict, dict]:
    repair_missing_postscale_merges(work_dir)
    merged = work_dir / "merged"
    artifact_dir = work_dir / "paper_artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    phases: dict[str, list[dict]] = {}
    warnings: list[str] = []
    missing: list[str] = []

    for phase, spec in EXPECTED.items():
        path = merged / spec["path"]
        if not path.exists():
            missing.append(str(path))
            continue
        rows = load_jsonl(path)
        try:
            warnings.extend(validate_phase(phase, rows))
        except Exception as exc:
            if allow_incomplete:
                warnings.append(f"{phase}: {exc}")
            else:
                raise
        phases[phase] = rows

    if missing and not allow_incomplete:
        raise SystemExit(
            "Paper artifact freeze incomplete. Missing required files:\n  - "
            + "\n  - ".join(missing)
        )

    best = {}
    if {"matched_confirm", "ablation_ext"}.issubset(phases) and (
        merged / "matched_best_configs.json"
    ).exists():
        best = validate_matched_configs(merged, phases)

    results: dict[str, object] = {
        "schema_version": 2,
        "core_digest": CORE_DIGEST,
        "evidence_order": [
            "matched_confirmation",
            "cross_configuration_isolation",
            "mechanism_ablation",
            "long_horizon_transfer",
            "scale_transfer",
            "broad_independently_tuned_context",
        ],
        "warnings": warnings,
    }

    if "matched_confirm" in phases:
        rows = phases["matched_confirm"]
        sm = summary(rows)
        effect = paired(rows, "orbit", "muon")
        runtime_overhead = sm["orbit"]["mean_seconds"] / sm["muon"]["mean_seconds"] - 1.0
        memory_overhead = (
            sm["orbit"]["mean_peak_cuda_gb"] / sm["muon"]["mean_peak_cuda_gb"] - 1.0
        )
        results["matched_confirmation"] = {
            "summary": sm,
            "orbit_vs_muon": effect,
            "runtime_overhead_fraction": runtime_overhead,
            "memory_overhead_fraction": memory_overhead,
            "shared_config": best.get("orbit", {}).get("config"),
            "shared_config_id": best.get("orbit", {}).get("config_id"),
            "interpretation": "primary",
        }

    if "xconfig" in phases:
        rows = phases["xconfig"]
        results["cross_configuration_isolation"] = {
            "summary": summary(rows, key="analysis_label"),
            "mechanism_at_muon_config": paired(
                rows, "orbit_at_muon_config", "muon_at_muon_config", key="analysis_label"
            ),
            "mechanism_at_orbit_config": paired(
                rows, "orbit_at_orbit_config", "muon_at_orbit_config", key="analysis_label"
            ),
            "recipe_effect_on_muon": paired(
                rows, "muon_at_orbit_config", "muon_at_muon_config", key="analysis_label"
            ),
            "recipe_effect_on_orbit": paired(
                rows, "orbit_at_orbit_config", "orbit_at_muon_config", key="analysis_label"
            ),
            "optimizer_by_config_interaction": interaction(phases),
            "interpretation": "confound_isolation",
        }

    if "ablation_ext" in phases:
        rows = phases["ablation_ext"]
        results["mechanism_ablation"] = {
            "summary": summary(rows),
            "orbit_vs_identity": paired(rows, "orbit", "orbit_identity"),
            "orbit_vs_norope": paired(rows, "orbit", "orbit_norope"),
            "orbit_vs_diag": paired(rows, "orbit", "orbit_diag"),
            "interpretation": "mechanism",
        }

    if "horizon_with_astro" in phases:
        rows = phases["horizon_with_astro"]
        results["long_horizon_transfer"] = {
            "summary": summary(rows),
            "orbit_vs_astro_v2": paired(rows, "orbit", "astro_v2"),
            "orbit_vs_muon": paired(rows, "orbit", "muon"),
            "orbit_vs_normuon": paired(rows, "orbit", "normuon"),
            "interpretation": "secondary_n2",
        }

    if "scale_with_astro" in phases:
        rows = phases["scale_with_astro"]
        results["scale_transfer"] = {
            "summary": summary(rows),
            "orbit_vs_astro_v2": paired(rows, "orbit", "astro_v2"),
            "orbit_vs_muon": paired(rows, "orbit", "muon"),
            "orbit_vs_normuon": paired(rows, "orbit", "normuon"),
            "interpretation": "secondary_n2",
        }

    if "confirm" in phases:
        results["broad_independently_tuned_context"] = {
            "summary": summary(phases["confirm"]),
            "interpretation": "secondary_independently_tuned",
        }

    sources = {}
    for phase, spec in EXPECTED.items():
        path = merged / spec["path"]
        if path.exists():
            rows = phases.get(phase, [])
            sources[phase] = {
                "path": str(path.relative_to(work_dir)),
                "sha256": file_sha256(path),
                "rows": len(rows),
                "task_ids": len({x.get("task_id") for x in rows}),
                "code_digests": sorted({str(x.get("code_digest")) for x in rows}),
                "environments": sorted(
                    {json.dumps(x.get("environment", {}), sort_keys=True) for x in rows}
                ),
            }

    manifest = {
        "schema_version": 2,
        "status": "paper_ready" if not missing and not warnings else "incomplete_or_warn",
        "core_digest": CORE_DIGEST,
        "sources": sources,
        "missing": missing,
        "warnings": warnings,
        "matched_config_id": best.get("orbit", {}).get("config_id"),
        "matched_config": best.get("orbit", {}).get("config"),
    }

    (artifact_dir / "paper_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n"
    )
    (artifact_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return results, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    results, manifest = build(args.work_dir, allow_incomplete=args.allow_incomplete)
    print(json.dumps({
        "status": manifest["status"],
        "artifact_dir": str(args.work_dir / "paper_artifacts"),
        "evidence_order": results["evidence_order"],
        "warnings": manifest["warnings"],
        "missing": manifest["missing"],
    }, indent=2))


if __name__ == "__main__":
    main()
