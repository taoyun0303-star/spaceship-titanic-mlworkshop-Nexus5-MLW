# Spaceship Titanic Code Submission Package

This repository contains the readable implementation code, runnable source files, final submission file, and concise documentation for the AI3023 Machine Learning Workshop Spaceship Titanic project.

Our main technical workflow is a reproducible tabular machine-learning pipeline involving feature engineering, local OOF validation, probability calibration, and ensemble refinement.

The core predictive capability comes from locally trained tabular models, feature engineering, OOF validation, and calibrated ensemble learning, with a representative local public score around `0.81318`.

The final ensemble reaches Public LB `0.82277` through heterogeneous ensemble learning, disagreement analysis, and consensus-guided refinement. Supplementary ensemble support signals are used only in the final refinement stage as additional consensus information rather than labels or direct replacements.

---

## Main Method Highlights

The model system can be explained as five parts:

- `XGBoost`, `LightGBM`, and `CatBoost`:
  the main gradient-boosting tabular learners used to model nonlinear feature interactions and stable probability outputs.

- `MLP`:
  the deep-learning comparison branch, used to evaluate whether neural tabular models improve performance on this task.

- `Rule-based base model`:
  a domain-aware reference prediction system built from CryoSleep, spending behavior, group/cabin consistency, and family structure patterns.

The final improvement mainly comes from three ensemble-level ideas:

### 1. Ensemble

Combine heterogeneous model families and complete prediction sources instead of relying on a single classifier.

### 2. Disagreement Analysis

Perform uncertainty-aware disagreement analysis:
only samples with strong cross-model consensus are eligible for refinement, which reduces unstable prediction flips and improves ensemble robustness.

### 3. Meta Voting

Apply a second-level `6/7 complete-source vote`.
If six of seven complete prediction arrays agree on the alternative class, the final prediction follows that consensus.

The MLP branch is kept as an experimental comparison and report highlight. It is not presented as the main final scorer because its validation result was lower than the tabular ensemble.

## Contents

- `runnable_source/`
  runnable project source with the directory layout expected by the scripts.

- `deep_learning_branch/`
  cleaned side experiment for MLP/Transformer-style models; kept for report and presentation analysis, not used as the final submission pipeline.

- `final_submission/submission_best_0p82277.csv`
  final Kaggle submission file.

- `docs/final_method_summary.md`
  concise method summary for report writing.

- `docs/report_ppt_highlights_zh.md`
  Chinese report/PPT wording guide for the model system and final ensemble highlights.

- `requirements.txt`
  Python dependencies.

---

## Environment

Recommended Python version: Python 3.10 or newer.

Install dependencies:

```powershell
pip install -r requirements.txt
```

Optional libraries `catboost`, `lightgbm`, and `xgboost` are used by the training pipeline when available. The scripts skip unavailable optional model families where supported.

## Reproduce the Final CSV

From this package root:

```powershell
cd runnable_source
.\run_final_pipeline.ps1
```

This command rebuilds the final model-level ensemble from the included audited prediction artifacts. The final candidate family is written under:

```text
runnable_source\04_实验输出\final_model_level_ensemble_0p82277
```

The selected final file is:

```text
runnable_source\04_实验输出\final_model_level_ensemble_0p82277\submission_final_model_ensemble_0p82277.csv
```

The same submitted file is also provided at:

```text
final_submission\submission_best_0p82277.csv
```

## Run Checks

From `runnable_source`:

```powershell
.\run_checks.ps1
```

The tests check submission formatting, complete prediction arrays, no PassengerId-level hardcoding, no local absolute-path leakage, reproducibility of the final submission file, and rejection of risky probe or single-flip source patterns.

## Deep Learning Side Experiment

The folder `deep_learning_branch/` records a side exploration with MLP, tabular Transformer, improved MLP, and embedding-based MLP models. The best standalone deep learning model reached about `0.81261` OOF accuracy, and the best DL+GBDT blend reached about `0.81951` OOF accuracy in local validation.

These results were useful for comparing modeling directions, but they did not beat the final model-level ensemble (`0.82277` Public LB), so deep learning is kept as an experimental branch rather than the final submission method.

## Compliance Note

The final method uses complete model outputs, local OOF probability estimates, model-level probability blending, global thresholds, disagreement analysis, and complete-source agreement rules. It does not use PassengerId-level hardcoding, public-label overwrite, bitstrings, single-row flips, or leaderboard probe logic.

## External Support Signals

Supplementary ensemble support signals are used only in the final refinement stage as additional consensus information rather than labels or direct replacements. These signals are treated as model predictions and are cited briefly in the references.

## GitHub Link

https://github.com/taoyun0303-star/spaceship-titanic-mlworkshop-Nexus5-MLW
