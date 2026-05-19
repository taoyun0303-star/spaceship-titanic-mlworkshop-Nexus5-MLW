# Deep Learning Branch Experiment

This folder contains a cleaned side experiment for the Spaceship Titanic project. It is included for report and presentation evidence only. The final submitted Kaggle CSV is still produced by the main tabular model-level ensemble in `runnable_source/`.

## Purpose

We tested whether neural networks could improve the tabular-model pipeline. The explored models include:

- Baseline MLP: dense layers with BatchNorm, SiLU activation, and Dropout.
- Tabular Transformer: feature tokenizer plus self-attention blocks.
- Improved MLP: StandardScaler, OneCycleLR, accuracy-based early stopping, gradient clipping, Mixup, label smoothing, and delayed SWA.
- Native Embedding MLP: categorical embeddings plus numeric features built from the raw data.

The conclusion is negative but useful: deep learning models were competitive as auxiliary experiments, but they did not beat the final tree/probability/ensemble strategy.

## Key Results

| Experiment | Main Idea | OOF Accuracy |
|---|---|---:|
| Baseline MLP | 3-layer MLP on engineered features | 0.80824 |
| Transformer | Feature-token Transformer | 0.80950 |
| Improved MLP | MLP with Mixup, label smoothing, OneCycleLR, SWA | 0.81261 |
| Native Embedding MLP | Raw categorical embeddings + numeric features | 0.80651 |
| Best DL + GBDT blend | v34 DL blended with GBDT baseline | 0.81951 |
| Final submitted main ensemble | Main package model-level ensemble | 0.82277 |

## Why It Was Not Used As Final Method

The dataset is small, heterogeneous, and strongly suited to tree-based tabular models. More complex neural architectures tended to overfit or underuse the structured categorical/spending relations. The best deep learning model improved after training fixes, but its standalone OOF accuracy remained below the local tabular family and below the final ensemble.

Deep learning therefore appears in the report as an ablation and exploration branch:

1. It demonstrates that we compared more than one modeling family.
2. It explains why tree-based and model-level ensemble methods were preferred.
3. It provides negative evidence that more complex neural networks were not automatically better for this tabular dataset.

## Contents

- `scripts/baseline/dl_mlp_train.py`: baseline MLP.
- `scripts/baseline/dl_transformer_train.py`: tabular Transformer.
- `scripts/improved/dl_improved_train.py`: best standalone DL experiment.
- `scripts/improved/dl_native_train.py`: raw-feature embedding MLP.
- `scripts/improved/dl_v34_train.py`, `dl_v35_train.py`, `dl_v37_train.py`, `dl_v40_train.py`: architecture and feature variants.
- `features/dl_features_v3/`: saved DL feature matrices for the v40-style experiments.
- `outputs/best/`: compact result artifacts used for report tables.
- `docs/`: detailed Chinese experiment notes and summary.

## Running A Smoke Test

Full GPU training can take a long time. For a quick CPU/GPU smoke test, run from this folder:

```powershell
python .\scripts\baseline\dl_mlp_train.py --train features\dl_features_v3\train_features_dl_v3.csv --test features\dl_features_v3\test_features_dl_v3.csv --epochs 1 --patience 1 --seeds 42 --folds 2 --out-dir outputs\smoke\dl_mlp
```

The smoke test checks that the training script, feature files, and output writing work. It is not intended to reproduce the full reported score.

## Compliance Note

This branch does not use PassengerId-level hardcoding, public-label overwrite, bitstrings, single-row flips, or leaderboard probing. It is a modeling experiment branch and is not the final submission pipeline.
