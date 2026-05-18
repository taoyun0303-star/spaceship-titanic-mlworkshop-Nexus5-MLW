from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = next(ROOT.glob("03_*")) / "01_训练管线" / "build_v120_nohardcode_candidate_sweep.py"


def load_module():
    spec = importlib.util.spec_from_file_location("v120_candidates", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rejects_risky_candidate_paths():
    module = load_module()

    with pytest.raises(ValueError, match="Risky path"):
        module.assert_safe_source(Path("public_notebook_outputs/bsthere_082137/submission.csv"))


def test_load_submission_requires_complete_sample_order(tmp_path):
    module = load_module()
    sample = pd.DataFrame({"PassengerId": ["0013_01", "0018_01"]})
    path = tmp_path / "candidate.csv"
    pd.DataFrame(
        {
            "PassengerId": ["0018_01", "0013_01"],
            "Transported": ["True", "False"],
        }
    ).to_csv(path, index=False)

    with pytest.raises(ValueError, match="PassengerId order mismatch"):
        module.load_submission(path, sample)


def test_candidate_summary_rejects_single_point_change():
    module = load_module()
    base = pd.Series([False, False, True, True], dtype=bool)
    candidate = pd.Series([False, True, True, True], dtype=bool)

    with pytest.raises(ValueError, match="single-point"):
        module.summarize_candidate(
            "tiny.csv",
            candidate,
            base,
            {"base": base},
            "one row only",
            min_changed=2,
        )
