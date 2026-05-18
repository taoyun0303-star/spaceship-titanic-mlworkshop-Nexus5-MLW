# Spaceship Titanic Code Submission Package

This repository contains the readable implementation code, runnable source files, final submission file, and concise documentation for the AI3023 Machine Learning Workshop Spaceship Titanic project.

The main technical workflow is a reproducible tabular machine-learning pipeline: feature engineering, local OOF validation, probability calibration, and a final global ensemble rule. Selected public notebook outputs are used as external model-level prediction sources, not ground-truth labels, and are combined only through global confidence and agreement rules.

## Contents

- `runnable_source/`: runnable project source with the directory layout expected by the scripts.
- `final_submission/submission_best_0p82207.csv`: final Kaggle submission file.
- `docs/final_method_summary.md`: concise method summary for report writing.
- `requirements.txt`: Python dependencies.

## Environment

Recommended Python version: Python 3.10 or newer.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional libraries `catboost`, `lightgbm`, and `xgboost` are used by the training pipeline when available. The scripts skip unavailable optional model families where supported.

## Reproduce the Final Pipeline

From this package root:

```powershell
cd runnable_source
.\run_final_pipeline.ps1
```

The final candidate family is written under:

```text
runnable_source\04_实验输出\v124_v120_refinement_nohardcode
```

The submitted best file is also provided at:

```text
final_submission\submission_best_0p82207.csv
```

## Run Checks

From `runnable_source`:

```powershell
.\run_checks.ps1
```

The tests check submission formatting, complete prediction arrays, no PassengerId-level hardcoding, no local absolute-path leakage, and rejection of risky probe or single-flip source patterns.

## Compliance Note

The final method uses complete model outputs, local OOF probability estimates, model-level probability blending, global thresholds, and complete-source agreement rules. It does not use PassengerId-level hardcoding, public-label overwrite, bitstrings, single-row flips, or leaderboard probe logic.

## Public Notebook Attribution

We reviewed selected public Kaggle notebook solutions and used their complete prediction outputs as external model-level prediction sources. These outputs are treated as predictions rather than labels and are cited in the report references.

## GitHub Link

https://github.com/taoyun0303-star/spaceship-titanic-mlworkshop-Nexus5-MLW
