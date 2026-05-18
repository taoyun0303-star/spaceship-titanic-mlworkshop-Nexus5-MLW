from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "03_代码"
    / "01_训练管线"
    / "pipeline_v110_clean_public_model_nohardcode.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("pipeline_v110", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_rejects_hardcoded_source_terms():
    module = load_module()
    with pytest.raises(ValueError, match="hardcode risk"):
        module.assert_no_hardcode_risk(Path("public_notebook_outputs/bsthere_082137/submission.csv"))


def test_validate_submission_requires_boolean_transport_column(tmp_path):
    module = load_module()
    test_ids = np.array(["0013_01", "0018_01"])
    good_path = tmp_path / "good.csv"
    pd.DataFrame({"PassengerId": test_ids, "Transported": [True, False]}).to_csv(good_path, index=False)

    result = module.validate_submission(good_path, test_ids)

    assert result["valid"] is True
    assert result["rows"] == 2
    assert result["true_rate"] == 0.5


def test_weighted_vote_uses_complete_prediction_arrays():
    module = load_module()
    preds = [
        np.array([True, False, True, False]),
        np.array([True, True, False, False]),
        np.array([False, True, True, False]),
    ]

    voted = module.weighted_vote(preds, weights=[0.5, 0.3, 0.2], threshold=0.5)

    assert voted.tolist() == [True, False, True, False]
