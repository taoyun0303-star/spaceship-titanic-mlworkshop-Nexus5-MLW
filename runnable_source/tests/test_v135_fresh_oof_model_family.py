from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "03_代码" / "01_训练管线" / "build_v135_fresh_oof_model_family_nohardcode.py"
OUT_DIR = ROOT / "04_实验输出" / "v135_fresh_oof_model_family_nohardcode"
DATA_DIR = ROOT / "02_数据与特征"


def load_module():
    spec = importlib.util.spec_from_file_location("v135", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_risk_pattern_blocks_forbidden_tokens():
    module = load_module()
    risky_paths = [
        "generate_v79_single_flip_probes.py",
        "pipeline_v64_titanic_anchor.py",
        "fixed_actual_pids.csv",
        "best_public_label_override.csv",
    ]
    for value in risky_paths:
        assert module.RISK_PATTERN.search(value), value


def test_smoke_run_outputs_valid_complete_candidates():
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--smoke",
        "--max-candidates",
        "3",
    ]
    result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    summary = pd.read_csv(OUT_DIR / "candidate_summary.csv")
    validation = pd.read_csv(OUT_DIR / "format_validation.csv")
    family = pd.read_csv(OUT_DIR / "model_family_summary.csv")

    assert 1 <= len(summary) <= 3
    assert validation["valid"].all()
    assert {"file", "family", "oof_accuracy", "threshold", "positive_rate", "compliance"}.issubset(summary.columns)
    assert {"model", "seed", "fold", "accuracy", "threshold"}.issubset(family.columns)
    assert summary["compliance"].str.contains("complete model", case=False).all()

    for filename in summary["file"]:
        assert filename.endswith(".csv")
        sub = pd.read_csv(OUT_DIR / filename)
        assert list(sub.columns) == ["PassengerId", "Transported"]
        assert sub["PassengerId"].astype(str).tolist() == sample["PassengerId"].astype(str).tolist()
        assert set(sub["Transported"].astype(str).str.lower()).issubset({"true", "false"})

    for array_name in ["v135_oof_stack.npy", "v135_test_stack.npy"]:
        arr = np.load(OUT_DIR / array_name)
        assert np.isfinite(arr).all()
        assert ((0.0 <= arr) & (arr <= 1.0)).all()


def test_run_summary_records_nohardcode_policy():
    module = load_module()
    assert module.CFG.run_dir_name == "v135_fresh_oof_model_family_nohardcode"
    assert "no PassengerId" in module.COMPLIANCE_NOTE
    assert "leaderboard-derived row fixes" in module.COMPLIANCE_NOTE
