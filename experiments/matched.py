#!/usr/bin/env python3
"""Deterministic matched-config selection for the audited ORBIT paper protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


TRIALS = 10
SELECTION_RULE = "muon_winner_from_shared_grid"
SEARCH_SPACE = {
    "lr": (2e-3, 1e-1),
    "weight_decay": (1e-3, 3e-1),
    "scalar_lr_mult": (0.02, 1.5),
}


def stable_rng(*parts: object) -> random.Random:
    payload = "|".join(map(str, parts)).encode()
    seed = int(hashlib.sha256(payload).hexdigest()[:16], 16)
    return random.Random(seed)


def sample_value(rng: random.Random, bounds: tuple[float, float]) -> float:
    low, high = map(float, bounds)
    return math.exp(rng.uniform(math.log(low), math.log(high)))


def candidate_config(trial: int) -> dict[str, float]:
    if not 0 <= trial < TRIALS:
        raise ValueError(f"trial must be in [0, {TRIALS - 1}]")
    rng = stable_rng("orbit-matched-shared-900", trial)
    return {key: sample_value(rng, bounds) for key, bounds in SEARCH_SPACE.items()}


def candidate_table() -> list[dict]:
    return [
        {
            "config_id": f"shared-{trial:02d}",
            "trial": trial,
            "config": candidate_config(trial),
        }
        for trial in range(TRIALS)
    ]


def configs_match(a: dict, b: dict, tol: float = 1e-15) -> bool:
    if set(a) != set(b):
        return False
    return all(
        math.isclose(float(a[key]), float(b[key]), rel_tol=tol, abs_tol=0.0)
        for key in a
    )


def select_muon_winner(paths: list[Path]) -> dict:
    if len(paths) != TRIALS:
        raise ValueError(f"expected exactly {TRIALS} Muon tuning records, got {len(paths)}")

    expected = candidate_table()
    by_trial: dict[int, dict] = {}
    digests: set[str] = set()

    for path in paths:
        row = json.loads(path.read_text())
        if row.get("status") != "ok":
            raise ValueError(f"{path}: run status is not ok")
        if row.get("optimizer") != "muon":
            raise ValueError(f"{path}: matched tuning must contain Muon only")
        if row.get("size") != "124M" or int(row.get("steps", -1)) != 900:
            raise ValueError(f"{path}: expected the 124M/900-step tuning protocol")
        if int(row.get("seed", -1)) != 0:
            raise ValueError(f"{path}: matched tuning seed must be 0")

        matches = [
            item for item in expected
            if configs_match(row.get("config", {}), item["config"])
        ]
        if len(matches) != 1:
            raise ValueError(f"{path}: config is not one of the deterministic matched candidates")
        trial = matches[0]["trial"]
        if trial in by_trial:
            raise ValueError(f"duplicate matched-tuning record for trial {trial}")
        by_trial[trial] = row
        digests.add(str(row.get("source_digest")))

    if set(by_trial) != set(range(TRIALS)):
        missing = sorted(set(range(TRIALS)) - set(by_trial))
        raise ValueError(f"missing matched-tuning trial(s): {missing}")
    if len(digests) != 1:
        raise ValueError("matched-tuning records were produced by different source digests")

    winner_trial = min(by_trial, key=lambda t: float(by_trial[t]["val_loss"]))
    winner = by_trial[winner_trial]
    return {
        "selection_rule": SELECTION_RULE,
        "selected_by": "muon",
        "config_id": f"shared-{winner_trial:02d}",
        "trial": winner_trial,
        "tuning_val_loss": float(winner["val_loss"]),
        "config": candidate_config(winner_trial),
        "source_digest": next(iter(digests)),
    }


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    show = sub.add_parser("candidates", help="write or print the ten deterministic Muon candidates")
    show.add_argument("--output", type=Path)

    select = sub.add_parser("select", help="freeze the lowest-loss Muon candidate")
    select.add_argument("records", nargs="+", type=Path)
    select.add_argument("--output", type=Path, required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "candidates":
        payload = {
            "selection_rule": SELECTION_RULE,
            "optimizer": "muon",
            "size": "124M",
            "steps": 900,
            "seed": 0,
            "candidates": candidate_table(),
        }
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
            print(args.output)
        else:
            print(text, end="")
        return

    selected = select_muon_winner(args.records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    print(f"{selected['config_id']} val={selected['tuning_val_loss']:.6f}")
    print(args.output)


if __name__ == "__main__":
    main()
