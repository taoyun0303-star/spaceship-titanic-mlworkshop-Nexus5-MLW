from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "03_代码" / "01_训练管线" / "build_final_model_level_ensemble.py"
OUT_DIR = ROOT / "04_实验输出" / "final_model_level_ensemble_0p82277"
FINAL_PACKAGE_FILE = ROOT.parent / "final_submission" / "submission_best_0p82277.csv"


def load_module():
    spec = importlib.util.spec_from_file_location("v132_final", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_script_uses_dynamic_project_root_and_no_row_list_output():
    module = load_module()
    assert module.find_project_root(SCRIPT) == ROOT
    assert "no PassengerId hardcoding" in module.COMPLIANCE_NOTE

    source = SCRIPT.read_text(encoding="utf-8")
    forbidden = [
        "changed" + "_ids",
        "single" + "_flip",
        "single flip",
        "bitstring",
        "actual" + "_pids",
        "public label",
    ]
    for token in forbidden:
        assert token.lower() not in source.lower()


def test_final_v132_reproduction_matches_submitted_file():
    result = subprocess.run([sys.executable, str(SCRIPT)], cwd=ROOT, text=True, capture_output=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr

    generated = pd.read_csv(OUT_DIR / "submission_final_model_ensemble_0p82277.csv")
    final = pd.read_csv(FINAL_PACKAGE_FILE)
    summary = pd.read_csv(OUT_DIR / "submission_summary.csv")

    assert generated.equals(final)
    row_id_listing_column = "changed" + "_ids"
    assert row_id_listing_column not in summary.columns
    assert "compliance" in summary.columns
    assert summary["compliance"].str.contains("complete local/public model", case=False).all()
    assert summary.iloc[0]["file"] == "submission_final_model_ensemble_0p82277.csv"
