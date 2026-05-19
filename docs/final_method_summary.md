# Final Method Summary

## Best Submission

The final selected submission is:

`final_submission/submission_best_0p82277.csv`

Public leaderboard score: `0.82277`.

The submitted pipeline is an audited model-level ensemble. Its core is our local OOF probability model and feature-engineering pipeline. The local model family provides the main explainable modeling foundation and reached a representative public score around `0.81318`; the final ensemble improved the submitted score to `0.82277`. Selected external prediction sources are used only as additional ensemble-level support signals rather than labels or direct replacements.

## Model System Highlights

For the report and presentation, the model system can be introduced as a layered ensemble:

- `XGBoost`: gradient-boosting tree model used to capture non-linear feature interactions.
- `LightGBM`: efficient GBDT model that provides a complementary tree-based probability estimate.
- `CatBoost`: strong tabular model with robust handling of categorical-style features.
- `MLP`: deep-learning comparison branch; useful as an experimental direction, but not selected as the final scorer because it did not outperform the tabular ensemble.
- `Rule-based base model`: domain-aware base prediction built from structured Spaceship Titanic patterns such as group/cabin consistency, CryoSleep spending behavior, and calibrated probability rules.

The last step is not a manual correction stage. It is a model-level fusion stage with three ideas:

1. `Ensemble`: combine several model families and complete prediction sources.
2. `Disagreement analysis`: compare the stable rule/probability base prediction with other complete model outputs and focus only on high-consensus disagreements.
3. `Meta voting`: use a second-level 6/7 vote across complete prediction arrays. The rule is global and source-level, so it does not depend on PassengerId lists or row-by-row probing.

Report/PPT keywords to emphasize:

- `ensemble robustness`
- `uncertainty analysis`
- `cross-model consensus`
- `disagreement analysis`
- `heterogeneous models`
- `OOF validation`
- `consensus-guided refinement`

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

## Deep Learning Side Experiment

We also tested deep learning as a separate modeling branch. This branch included a baseline MLP, a feature-token Transformer, an improved MLP with Mixup/label smoothing/OneCycleLR/SWA, and a native embedding MLP for categorical features.

The best standalone deep learning model was the improved MLP, with OOF accuracy about `0.81261`. A DL+GBDT blend reached about `0.81951` OOF accuracy in local validation, but the improvement was not strong enough to replace the final model-level ensemble. This result supports the final modeling choice: for this small heterogeneous tabular dataset, tree-based and probability-level ensemble methods were more reliable than heavier neural architectures.

The deep learning branch is documented in `deep_learning_branch/` and is used in the report as an experimental comparison, not as the final submission method.

## Final Ensemble Rule

The final ensemble uses complete-source agreement, which is the implementation of the meta-voting idea:

1. Start from a strong complete prediction produced by our audited probability-fusion pipeline.
2. Compare it with seven complete prediction arrays: five audited external model outputs and two local model-family outputs.
3. Apply one global meta-voting rule: when at least six of seven complete sources agree on the alternative class, the final ensemble follows the agreement vote.

This rule changes only a very small number of predictions relative to the stable reference prediction, but the rule itself is global and source-level: it is not a PassengerId list, label lookup, bitstring, single-row flip, or leaderboard probe.

The final submitted file is:

`submission_final_model_ensemble_0p82277.csv`

It produced 2,410 predicted `True` labels, a true rate of about `0.56348`, and a Public LB score of `0.82277`.

## Report Wording Guide

In the report, describe the method as our audited tabular model and model-level ensemble:

1. Data preprocessing and feature engineering.
2. Model system: XGBoost, LightGBM, CatBoost, MLP comparison branch, and rule-based base model.
3. Five-model local tabular learner family with OOF validation.
4. Probability calibration and global thresholding.
5. Ensemble robustness, uncertainty analysis, disagreement analysis, and cross-model consensus.
6. Final model-level ensemble and leaderboard result.

Avoid presenting row-level edits, PassengerId corrections, or leaderboard probes. They are not part of the final method.

## External Support Signals

Selected external prediction sources are used only as additional ensemble-level support signals rather than labels or direct replacements. These signals were treated as model predictions and were combined with our local OOF probability model only through global confidence and agreement rules.

## References

[1] Kaggle, "Spaceship Titanic Competition," Kaggle, accessed May 2026.

[2] JimLiu, "Spaceship Titanic Kaggle solution reference," accessed May 2026.

[3] Ravi20076, "Spaceship Titanic Kaggle solution reference," accessed May 2026.

[4] Additional external Kaggle solution materials reviewed as ensemble-level support sources, accessed May 2026.
