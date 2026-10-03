from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import orbit


ROOT = Path(__file__).resolve().parents[1]


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _matched_module():
    return _load_script_module(
        "orbit_matched_release_check", ROOT / "experiments" / "matched.py"
    )


def _paper_run_module():
    return _load_script_module(
        "orbit_paper_run_release_check", ROOT / "experiments" / "paper_run.py"
    )


def test_release_versions_are_consistent():
    pyproject = (ROOT / "pyproject.toml").read_text()
    citation = (ROOT / "CITATION.cff").read_text()
    config = json.loads((ROOT / "configs" / "paper_124m_matched.json").read_text())

    version_match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert version_match is not None
    version = version_match.group(1)
    assert version == "0.1.0"
    assert orbit.__version__ == version
    assert f"version: {version}" in citation
    assert config["package_version"] == version
    assert (ROOT / "CHANGELOG.md").exists()
    assert (ROOT / "RELEASE.md").exists()


def test_frozen_paper_config_is_exact_shared_04_candidate():
    matched = _matched_module()
    config = json.loads((ROOT / "configs" / "paper_124m_matched.json").read_text())
    frozen = config["optimizer"]
    expected = matched.candidate_config(4)

    assert config["matched_tuning"]["selected_config_id"] == "shared-04"
    assert config["matched_tuning"]["seed"] == 0
    assert config["matched_tuning"]["candidate_count"] == 10
    assert frozen["lr"] == expected["lr"]
    assert frozen["weight_decay"] == expected["weight_decay"]
    assert frozen["scalar_lr_mult"] == expected["scalar_lr_mult"]
    assert math.isclose(
        frozen["adamw_lr"],
        frozen["lr"] * frozen["scalar_lr_mult"],
        rel_tol=0.0,
        abs_tol=1e-18,
    )


def test_paper_seed_sets_match_campaign_protocol():
    config = json.loads((ROOT / "configs" / "paper_124m_matched.json").read_text())
    assert config["primary_confirmation_seeds"] == list(range(500, 510))
    assert config["ablation_seeds"] == list(range(600, 610))
    assert config["crossed_recipe_seeds"] == list(range(100, 105))
    assert config["long_horizon_seeds"] == [400, 401]


def test_paper_runner_routes_primary_and_ablation_seed_sets():
    paper_run = _paper_run_module()
    config = json.loads((ROOT / "configs" / "paper_124m_matched.json").read_text())

    for optimizer in ("muon", "orbit"):
        assert paper_run.expected_seeds(config, optimizer) == list(range(500, 510))
    for optimizer in ("orbit_identity", "orbit_norope", "orbit_diag"):
        assert paper_run.expected_seeds(config, optimizer) == list(range(600, 610))
