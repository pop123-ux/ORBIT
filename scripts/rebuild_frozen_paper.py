#!/usr/bin/env python3
"""Rebuild the ORBIT manuscript from the committed frozen evidence snapshot."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    work = root / ".orbit-frozen-work"
    merged = work / "merged"
    raw = root / "results" / "paper-v1" / "raw"

    if work.exists():
        shutil.rmtree(work)
    merged.mkdir(parents=True)

    for path in raw.glob("*.jsonl"):
        shutil.copy2(path, merged / path.name)
    shutil.copy2(
        root / "configs" / "matched_best_configs.json",
        merged / "matched_best_configs.json",
    )

    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "orbit_build_paper.py"),
            "--work-dir",
            str(work),
            "--paper-dir",
            str(root / "docs" / "paper"),
        ],
        check=True,
        cwd=root,
    )
    print(root / "docs" / "paper" / "main.pdf")


if __name__ == "__main__":
    main()
