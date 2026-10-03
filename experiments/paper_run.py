#!/usr/bin/env python3
"""Run one paper-matched ORBIT/Muon cell from the frozen release config."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_124m_matched.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--optimizer", choices=("muon", "orbit", "orbit_identity", "orbit_norope", "orbit_diag"), required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--token-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--allow-nonprimary-seed",
        action="store_true",
        help="allow a seed outside the paper's 500-509 primary confirmation set",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    seeds = [int(x) for x in config["primary_confirmation_seeds"]]
    if args.seed not in seeds and not args.allow_nonprimary_seed:
        raise SystemExit(
            f"seed {args.seed} is not a primary paper seed; expected one of {seeds}. "
            "Pass --allow-nonprimary-seed for an exploratory run."
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
