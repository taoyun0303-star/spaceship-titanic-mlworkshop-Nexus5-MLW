from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUNNABLE_ROOT = ROOT / "runnable_source"


def test_readme_contains_real_github_link():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "https://github.com/taoyun0303-star/spaceship-titanic-mlworkshop-Nexus5-MLW" in readme
    assert "github.com/<team>/<repository>" not in readme


def test_runner_uses_dynamic_script_discovery():
    runner = (RUNNABLE_ROOT / "run_final_pipeline.ps1").read_text(encoding="utf-8")
    assert "01_" not in runner
    assert "Get-ChildItem" in runner
    assert "exit $LASTEXITCODE" in runner
    assert "build_final_model_level_ensemble.py" in runner


def test_no_local_absolute_paths_in_package_docs():
    checked_suffixes = {".md", ".csv", ".json", ".ps1", ".py", ".txt"}
    drive_marker = chr(68) + ":\\"
    users_marker = chr(67) + ":\\Users\\"
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in checked_suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if drive_marker in text or users_marker in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_method_wording_is_balanced_and_attributed():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    method = (ROOT / "docs" / "final_method_summary.md").read_text(encoding="utf-8")
    combined = readme + "\n" + method

    assert "Selected external prediction sources are used only as additional ensemble-level support signals" in combined
    assert "rather than labels or direct replacements" in combined
    assert "local OOF probability model" in combined
    assert "References" in method
    assert "JimLiu" in method
    assert "Ravi" in method
    assert "PassengerId corrections" in method
    assert "ensemble robustness" in combined
    assert "uncertainty analysis" in combined
    assert "cross-model consensus" in combined
    assert "0.81318" in combined
    assert "0.82277" in combined


def test_readme_uses_final_v132_output_path():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "runnable_source\\04_实验输出\\final_model_level_ensemble_0p82277" in readme
    assert "0p82207" not in readme


def test_final_submission_matches_reproducible_v132_output():
    final_path = ROOT / "final_submission" / "submission_best_0p82277.csv"
    generated_path = (
        RUNNABLE_ROOT
        / "04_实验输出"
        / "final_model_level_ensemble_0p82277"
        / "submission_final_model_ensemble_0p82277.csv"
    )
    sample_path = RUNNABLE_ROOT / "02_数据与特征" / "sample_submission.csv"

    final = pd.read_csv(final_path)
    generated = pd.read_csv(generated_path)
    sample = pd.read_csv(sample_path)

    assert final.equals(generated)
    assert list(final.columns) == ["PassengerId", "Transported"]
    assert final["PassengerId"].astype(str).equals(sample["PassengerId"].astype(str))
    assert set(final["Transported"].astype(str).str.lower()).issubset({"true", "false"})


def test_v132_summary_contains_no_passengerid_listing():
    summary_path = (
        RUNNABLE_ROOT
        / "04_实验输出"
        / "final_model_level_ensemble_0p82277"
        / "submission_summary.csv"
    )
    if not summary_path.exists():
        return
    summary = pd.read_csv(summary_path)
    row_id_listing_column = "changed" + "_ids"
    assert row_id_listing_column not in summary.columns
    assert "compliance" in summary.columns
    assert summary["compliance"].str.contains("no PassengerId", case=False).all()
