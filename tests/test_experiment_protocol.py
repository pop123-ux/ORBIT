from __future__ import annotations

import importlib.util
import json
from pathlib import Path


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


def test_matched_selection_uses_muon_only_and_freezes_lowest_loss(tmp_path):
    digest = "audited-source"
    paths = []
    for item in matched.candidate_table():
        trial = item["trial"]
        row = {
            "status": "ok",
            "optimizer": "muon",
            "size": "124M",
            "steps": 900,
            "seed": 0,
            "config": item["config"],
            "val_loss": 5.0 - (0.1 if trial == 6 else 0.0) + 0.001 * trial,
            "source_digest": digest,
        }
        path = tmp_path / f"trial-{trial:02d}.json"
        path.write_text(json.dumps(row))
        paths.append(path)

    selected = matched.select_muon_winner(paths)
    assert selected["selection_rule"] == "muon_winner_from_shared_grid"
    assert selected["selected_by"] == "muon"
    assert selected["config_id"] == "shared-06"
    assert selected["trial"] == 6
    assert selected["config"] == matched.candidate_config(6)
    assert selected["source_digest"] == digest


def test_matched_selection_rejects_orbit_tuning_record(tmp_path):
    paths = []
    for item in matched.candidate_table():
        trial = item["trial"]
        row = {
            "status": "ok",
            "optimizer": "orbit" if trial == 0 else "muon",
            "size": "124M",
            "steps": 900,
            "seed": 0,
            "config": item["config"],
            "val_loss": 5.0 + trial,
            "source_digest": "audited-source",
        }
        path = tmp_path / f"trial-{trial:02d}.json"
        path.write_text(json.dumps(row))
        paths.append(path)

    try:
        matched.select_muon_winner(paths)
    except ValueError as exc:
        assert "Muon only" in str(exc)
    else:
        raise AssertionError("ORBIT tuning record was accepted into the primary selection")
