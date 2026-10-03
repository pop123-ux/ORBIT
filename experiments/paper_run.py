#!/usr/bin/env python3
"""Run one frozen ORBIT paper cell from the release config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_124m_matched.json"
PRIMARY_OPTIMIZERS = {"muon", "orbit"}
ABLATION_OPTIMIZERS = {"orbit_identity", "orbit_norope", "orbit_diag"}
OPTIMIZERS = tuple(sorted(PRIMARY_OPTIMIZERS | ABLATION_OPTIMIZERS))


def expected_seeds(config: dict, optimizer: str) -> list[int]:
    """Return the frozen paper seed set appropriate for one optimizer cell."""
    if optimizer in PRIMARY_OPTIMIZERS:
        key = "primary_confirmation_seeds"
    elif optimizer in ABLATION_OPTIMIZERS:
        key = "ablation_seeds"
    else:  # defensive guard for programmatic callers
        raise ValueError(f"unsupported paper optimizer {optimizer!r}")
    return [int(x) for x in config[key]]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", choices=OPTIMIZERS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--token-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--allow-nonprimary-seed",
        action="store_true",
        help="allow a seed outside the frozen paper seed set for the selected optimizer",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    seeds = expected_seeds(config, args.optimizer)
    if args.seed not in seeds and not args.allow_nonprimary_seed:
        raise SystemExit(
            f"seed {args.seed} is not in the frozen paper seed set for {args.optimizer}; "
            f"expected one of {seeds}. Pass --allow-nonprimary-seed for an exploratory run."
        )

    model = config["model"]
    opt = config["optimizer"]
    command = [
        sys.executable,
        str(ROOT / "experiments" / "run.py"),
        "--optimizer",
        args.optimizer,
        "--size",
        str(model["size"]),
        "--steps",
        str(model["steps"]),
        "--seq",
        str(model["sequence_length"]),
        "--seed",
        str(args.seed),
        "--lr",
        repr(float(opt["lr"])),
        "--scalar-lr-mult",
        repr(float(opt["scalar_lr_mult"])),
        "--weight-decay",
        repr(float(opt["weight_decay"])),
        "--token-cache",
        str(args.token_cache),
        "--output",
        str(args.output),
    ]
    print("Running frozen paper recipe:")
    print(" ".join(command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
