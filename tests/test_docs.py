from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = [
    ROOT / "README.md",
    ROOT / "docs" / "METHOD.md",
    ROOT / "docs" / "ARCHITECTURE.md",
    ROOT / "docs" / "EXPERIMENTS.md",
    ROOT / "docs" / "TRAINING.md",
    ROOT / "experiments" / "README.md",
]


def test_github_math_uses_supported_delimiters():
    for path in DOCS:
        text = path.read_text()
        display_delimiter = "$" * 2
        assert display_delimiter not in text, f"use fenced math blocks in {path}"
        assert r"\(" not in text, f"use $...$ for inline math in {path}"
        assert r"\)" not in text, f"use $...$ for inline math in {path}"
        assert not any(line.strip() == "$" for line in text.splitlines()), (
            f"use a fenced math block instead of multiline single-dollar math in {path}"
        )


def test_markdown_fences_are_balanced():
    for path in DOCS:
        text = path.read_text()
        assert text.count("```") % 2 == 0, f"unbalanced backtick fence in {path}"
        assert text.count("~~~") % 2 == 0, f"unbalanced tilde fence in {path}"


def test_docs_have_no_control_characters():
    allowed = {"\n", "\r", "\t"}
    for path in DOCS:
        text = path.read_text()
        bad = [ch for ch in text if ord(ch) < 32 and ch not in allowed]
        assert not bad, f"control characters found in {path}"


def test_relative_markdown_links_resolve():
    link_pattern = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
    for path in DOCS:
        text = path.read_text()
        for target in link_pattern.findall(text):
            target = target.strip()
            if (
                not target
                or target.startswith(("http://", "https://", "mailto:", "#"))
            ):
                continue
            target = target.split("#", 1)[0]
            resolved = (path.parent / target).resolve()
            assert resolved.exists(), f"broken relative link in {path}: {target}"
