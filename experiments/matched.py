#!/usr/bin/env python3
"""Deterministic matched-config selection for the ORBIT paper protocol.

The paper evaluates the same ten deterministic 124M/900-step candidate recipes for
Muon and ORBIT on tuning seed 0. Each optimizer selects its own lowest-loss candidate.
The historical paper campaign selected the same candidate for both methods; this script
verifies that coincidence before emitting one shared matched recipe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path


TRIALS = 10
SELECTION_RULE = "independent_winners_from_shared_grid"
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


def _select_winner(paths: list[Path], optimizer: str) -> dict:
    if len(paths) != TRIALS:
        raise ValueError(f"expected exactly {TRIALS} {optimizer} tuning records, got {len(paths)}")

    expected = candidate_table()
    by_trial: dict[int, dict] = {}
    digests: set[str] = set()

    for path in paths:
        row = json.loads(path.read_text())
        if row.get("status") != "ok":
            raise ValueError(f"{path}: run status is not ok")
        if row.get("optimizer") != optimizer:
            raise ValueError(f"{path}: expected optimizer={optimizer}")
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
            raise ValueError(f"duplicate {optimizer} matched-tuning record for trial {trial}")
        by_trial[trial] = row
        digests.add(str(row.get("source_digest", row.get("code_digest"))))

    if set(by_trial) != set(range(TRIALS)):
        missing = sorted(set(range(TRIALS)) - set(by_trial))
        raise ValueError(f"missing {optimizer} matched-tuning trial(s): {missing}")
    if len(digests) != 1:
        raise ValueError(f"{optimizer} matched-tuning records used different source digests")

    winner_trial = min(by_trial, key=lambda t: float(by_trial[t]["val_loss"]))
    winner = by_trial[winner_trial]
    return {
        "optimizer": optimizer,
        "config_id": f"shared-{winner_trial:02d}",
        "trial": winner_trial,
        "tuning_val_loss": float(winner["val_loss"]),
        "config": candidate_config(winner_trial),
        "source_digest": next(iter(digests)),
    }


def select_shared_winner(muon_paths: list[Path], orbit_paths: list[Path]) -> dict:
    muon = _select_winner(muon_paths, "muon")
    orbit = _select_winner(orbit_paths, "orbit")
    if muon["config_id"] != orbit["config_id"] or not configs_match(muon["config"], orbit["config"]):
        raise ValueError(
            "Muon and ORBIT selected different matched-grid winners; there is no single shared recipe"
        )
    return {
        "selection_rule": SELECTION_RULE,
        "config_id": muon["config_id"],
        "config": muon["config"],
        "muon": muon,
        "orbit": orbit,
    }


def parse_args():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)

    show = sub.add_parser("candidates", help="write or print the ten deterministic shared candidates")
    show.add_argument("--output", type=Path)

    select = sub.add_parser("select", help="verify the independent Muon/ORBIT winners coincide")
    select.add_argument("--muon", nargs=TRIALS, type=Path, required=True)
    select.add_argument("--orbit", nargs=TRIALS, type=Path, required=True)
    select.add_argument("--output", type=Path, required=True)

    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "candidates":
        payload = {
            "selection_rule": SELECTION_RULE,
            "optimizers": ["muon", "orbit"],
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

    selected = select_shared_winner(args.muon, args.orbit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")
    print(
        f"{selected['config_id']} "
        f"muon={selected['muon']['tuning_val_loss']:.6f} "
        f"orbit={selected['orbit']['tuning_val_loss']:.6f}"
    )
    print(args.output)


if __name__ == "__main__":
    main()
