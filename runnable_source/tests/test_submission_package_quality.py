from pathlib import Path


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

    assert "public notebook outputs are used as external model-level prediction sources" in combined
    assert "not ground-truth labels" in combined
    assert "local OOF probability model" in combined
    assert "References" in method
    assert "JimLiu" in method
    assert "Ravi" in method


def test_readme_uses_correct_chinese_output_path():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "runnable_source\\04_实验输出\\v124_v120_refinement_nohardcode" in readme
    assert "瀹為獙" not in readme
