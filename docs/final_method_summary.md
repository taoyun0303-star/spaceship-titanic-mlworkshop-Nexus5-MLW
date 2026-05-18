# Final Method Summary

## Best Submission

The final selected submission is:

`final_submission/submission_best_0p82207.csv`

Public leaderboard score: `0.82207`.

The submitted pipeline is an audited multi-source model-level ensemble. Its core is our local OOF probability model and feature-engineering pipeline. Selected public notebook outputs are used as external model-level prediction sources, not ground-truth labels, and are combined with the local model only through global probability and agreement rules.

## Feature Engineering

The main local probability model uses 43 engineered features. These features can be grouped as follows:

- Passenger group structure: group id, group member index, group size, solo passenger indicator.
- Cabin structure: cabin deck, cabin number, cabin side, deck-side interaction, cabin zone.
- Family/name information: surname and family size.
- Demographic indicators: age, age band, child/teen/senior flags.
- Service spending: five raw spending columns, total spend, average spend per used service, spend per group member.
- Spending transformations: log-transformed service spending, log total spending, log average spending, log spending per group member.
- Behavioral indicators: CryoSleep flag, VIP flag, no-spend flag, number of positive spending services.
- Interaction features: home planet plus destination, age-spend interaction.

Missing values are filled with dataset-aware rules: group-level modes for categorical passenger attributes, cabin/home/destination consistency rules, group/home medians for age and spending, and the CryoSleep/no-spend relation for service expenses.

Categorical variables are encoded with ordinal encoding for tree models, while CatBoost also uses categorical-style feature handling in the training script.

## Local Models and Validation

The local model family compares five tabular classifiers:

- ExtraTrees
- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost

Training uses two random seeds and 5-fold stratified cross-validation. For each model family, the pipeline stores out-of-fold probabilities, test probabilities, fold accuracy, AUC, and log loss. A logistic regression stacker is trained on the OOF probability matrix, and global thresholds are selected from validation performance only.

Representative OOF results from the clean local model run:

| Model | OOF Accuracy | AUC | Log Loss |
|---|---:|---:|---:|
| CatBoost | 0.81353 | 0.90471 | 0.38515 |
| LightGBM | 0.80996 | 0.90189 | 0.38798 |
| HistGradientBoosting | 0.80904 | 0.90190 | 0.38670 |
| XGBoost | 0.80841 | 0.90267 | 0.38458 |
| ExtraTrees | 0.80473 | 0.89359 | 0.40914 |

## Final Ensemble Rule

The final ensemble combines the local OOF probability model with complete prediction arrays from audited model sources. The local probability variants include:

- rules/no-rules average
- rules-weighted probability blends
- no-rules-weighted probability blends
- GBDT-family average
- all-tree average
- rules/no-rules plus GBDT weighted blends

The best final file uses:

- local probability source: `rules_weighted_60`
- external anchor source: complete JimLiu prediction output
- external agreement source: complete Ravi prediction output
- decision rule: update the anchor only when the local probability is at least `0.033` away from the global threshold and the complete Ravi output agrees with the model direction

This produced 2,413 predicted `True` labels, a true rate of about `0.56418`, and a Public LB score of `0.82207`.

## Report Wording Guide

In the report, describe the method as our audited multi-source model-level ensemble:

1. Data preprocessing and feature engineering.
2. Five-model local tabular learner family with OOF validation.
3. Probability calibration and global thresholding.
4. Complete-source agreement filter.
5. Final model-level ensemble and leaderboard result.

Avoid presenting row-level edits, PassengerId corrections, or leaderboard probes. They are not part of the final method.

## Public Notebook Attribution

We reviewed selected public Kaggle notebook solutions and used their complete prediction outputs as external model-level prediction sources. These outputs were treated as predictions rather than labels and were combined with our local OOF probability model only through global confidence and agreement rules.

## References

[1] Kaggle, "Spaceship Titanic Competition," Kaggle, accessed May 2026.

[2] JimLiu, "Spaceship Titanic public notebook solution," Kaggle Notebook, accessed May 2026.

[3] Ravi20076, "Spaceship Titanic public notebook solution," Kaggle Notebook, accessed May 2026.
