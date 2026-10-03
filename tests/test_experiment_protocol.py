from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "orbit_matched_protocol",
    ROOT / "experiments" / "matched.py",
)
matched = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(matched)


def test_matched_candidate_grid_is_deterministic_and_unique():
    first = matched.candidate_table()
    second = matched.candidate_table()
    assert first == second
    assert len(first) == 10
    assert len({json.dumps(x["config"], sort_keys=True) for x in first}) == 10
    assert [x["config_id"] for x in first] == [f"shared-{i:02d}" for i in range(10)]
    assert first[0]["config"] == {
        "lr": 0.002510810677061948,
        "weight_decay": 0.09862407335797245,
        "scalar_lr_mult": 0.028701450292747916,
    }

    for item in first:
        cfg = item["config"]
        assert 2e-3 <= cfg["lr"] <= 1e-1
        assert 1e-3 <= cfg["weight_decay"] <= 3e-1
        assert 0.02 <= cfg["scalar_lr_mult"] <= 1.5


def _write_tuning_records(tmp_path: Path, optimizer: str, winner_trial: int) -> list[Path]:
    digest = "paper-source"
    paths: list[Path] = []
    for item in matched.candidate_table():
        trial = item["trial"]
        row = {
            "status": "ok",
            "optimizer": optimizer,
            "size": "124M",
            "steps": 900,
            "seed": 0,
            "config": item["config"],
            "val_loss": 5.0 - (0.1 if trial == winner_trial else 0.0) + 0.001 * trial,
            "source_digest": digest,
        }
        path = tmp_path / f"{optimizer}-trial-{trial:02d}.json"
        path.write_text(json.dumps(row))
        paths.append(path)
    return paths


def test_matched_selection_freezes_coincident_independent_winner(tmp_path):
    muon_paths = _write_tuning_records(tmp_path, "muon", winner_trial=6)
    orbit_paths = _write_tuning_records(tmp_path, "orbit", winner_trial=6)

    selected = matched.select_shared_winner(muon_paths, orbit_paths)

    assert selected["selection_rule"] == "independent_winners_from_shared_grid"
    assert selected["config_id"] == "shared-06"
    assert selected["config"] == matched.candidate_config(6)
    assert selected["muon"]["trial"] == 6
    assert selected["orbit"]["trial"] == 6
    assert selected["muon"]["source_digest"] == "paper-source"
    assert selected["orbit"]["source_digest"] == "paper-source"


def test_matched_selection_rejects_different_independent_winners(tmp_path):
    muon_paths = _write_tuning_records(tmp_path, "muon", winner_trial=6)
    orbit_paths = _write_tuning_records(tmp_path, "orbit", winner_trial=4)

    with pytest.raises(ValueError, match="different matched-grid winners"):
        matched.select_shared_winner(muon_paths, orbit_paths)


def test_optimizer_specific_tuning_records_are_not_interchangeable(tmp_path):
    muon_paths = _write_tuning_records(tmp_path, "muon", winner_trial=6)
    orbit_paths = _write_tuning_records(tmp_path, "orbit", winner_trial=6)

    # The selector validates optimizer identity inside each ten-run set.
    with pytest.raises(ValueError, match="expected optimizer=muon"):
        matched.select_shared_winner(orbit_paths, orbit_paths)
    with pytest.raises(ValueError, match="expected optimizer=orbit"):
        matched.select_shared_winner(muon_paths, muon_paths)
