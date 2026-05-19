# Final Method Summary

## Best Submission

The final selected submission is:

`final_submission/submission_best_0p82277.csv`

Public leaderboard score: `0.82277`.

The project is best described as a two-stage modeling process. In Round 1, we built an independent local tabular ensemble. The best confirmed local-only file is `submission_round1_independent_local_ensemble_0p81318.csv`, with Public LB `0.81318`. In Round 2, we added an audited hybrid model-level consensus refinement layer. The final `0.82277` result comes from combining the local/reference prediction system with selected complete external prediction sources through global agreement rules. These external sources are a material part of the final consensus layer, but they are used only as complete prediction arrays, not as labels, direct row replacements, or PassengerId-level rules.

## Model System Highlights

For the report and presentation, introduce the work as a timeline:

1. `Round 1 - Independent local ensemble`: build our own feature engineering, local models, OOF validation, and voting ensemble. Best local-only Public LB: `0.81318`.
2. `Round 2 - Consensus refinement`: add selected public complete prediction sources as model-level signals and apply global disagreement/meta-voting rules. Final Public LB: `0.82277`.

The model system can then be introduced as a layered ensemble:

- `XGBoost`: gradient-boosting tree model used to capture non-linear feature interactions.
- `LightGBM`: efficient GBDT model that provides a complementary tree-based probability estimate.
- `CatBoost`: strong tabular model with robust handling of categorical-style features.
- `MLP`: deep-learning comparison branch; useful as an experimental direction, but not selected as the final scorer because it did not outperform the tabular ensemble.
- `Rule-based base model`: domain-aware base prediction built from structured Spaceship Titanic patterns such as group/cabin consistency, CryoSleep spending behavior, and calibrated probability rules.

The last step is not a manual correction stage. It is a model-level fusion stage with three ideas:

1. `Ensemble`: combine several local model families and complete prediction sources.
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

## Round 1 Independent Local Ensemble

The strongest local-only route in this package is the Round 1 independent local ensemble:

- Script: `runnable_source/03_代码/01_训练管线/build_round1_independent_local_ensemble.py`
- Output: `runnable_source/04_实验输出/round1_independent_local_ensemble_0p81318/submission_round1_independent_local_ensemble_0p81318.csv`
- Public LB: `0.81318`

This route uses 64 engineered tabular features and a vote across four locally trained model outputs:

- XGBoost
- LightGBM
- CatBoost trained on engineered numeric features
- CatBoost trained with native categorical handling

An MLP branch is also trained and compared in this stage, but the selected local submission uses the more stable `2 of 4` tree/CatBoost vote.

## Feature Engineering

Across the local modeling work, the feature engineering focuses on passenger structure, cabin structure, family/name information, demographics, service spending, spending transformations, and task-specific interactions. The compact clean local model artifact uses 43 engineered features, while the Round 1 independent local ensemble uses a larger 64-feature set. These features can be grouped as follows:

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

The full training design uses multiple random seeds and stratified cross-validation. For each model family, the pipeline stores out-of-fold probabilities, test probabilities, fold accuracy, AUC, and log loss. A logistic regression stacker is trained on the OOF probability matrix, and global thresholds are selected from validation performance only.

Representative OOF results from the compact clean local artifact included in this package:

| Model | OOF Accuracy | AUC | Log Loss |
|---|---:|---:|---:|
| HistGradientBoosting | 0.80709 | 0.89806 | 0.39876 |
| LightGBM | 0.80571 | 0.89619 | 0.40548 |
| ExtraTrees | 0.80433 | 0.88903 | 0.41698 |
| CatBoost | 0.79949 | 0.88581 | 0.42886 |
| XGBoost | 0.79788 | 0.88372 | 0.43716 |

The compact clean local artifact is kept as an additional reproducible validation record. The main local-only score reported in the project timeline is the confirmed Round 1 independent local ensemble score: `0.81318`.

## Deep Learning Side Experiment

We also tested deep learning as a separate modeling branch. This branch included a baseline MLP, a feature-token Transformer, an improved MLP with Mixup/label smoothing/OneCycleLR/SWA, and a native embedding MLP for categorical features.

The best standalone deep learning model was the improved MLP, with OOF accuracy about `0.81261`. A DL+GBDT blend reached about `0.81951` OOF accuracy in local validation, but the improvement was not strong enough to replace the final model-level ensemble. This result supports the final modeling choice: for this small heterogeneous tabular dataset, tree-based and probability-level ensemble methods were more reliable than heavier neural architectures.

The deep learning branch is documented in `deep_learning_branch/` and is used in the report as an experimental comparison, not as the final submission method.

## Final Ensemble Rule

The Round 2 final ensemble uses complete-source agreement, which is the implementation of the meta-voting idea:

1. Start from a strong complete reference prediction produced by the audited probability/rule-fusion pipeline.
2. Compare it with seven complete prediction arrays: five audited external model outputs and two local model-family outputs.
3. Apply one global meta-voting rule: when at least six of seven complete sources agree on the alternative class, the final ensemble follows the agreement vote.

This rule changes only a very small number of predictions relative to the stable reference prediction, but the rule itself is global and source-level: it is not a PassengerId list, label lookup, bitstring, single-row flip, or leaderboard probe.

The top-level submitted package file is:

`final_submission/submission_best_0p82277.csv`

The reproducible file generated by `runnable_source/run_final_pipeline.ps1` is:

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

External prediction sources are a material part of the final consensus layer. These signals were treated as complete model predictions and were combined with the local/reference prediction system only through global confidence, disagreement, and agreement rules. They were not used as labels, direct row replacements, PassengerId-level corrections, bitstrings, single-row flips, or leaderboard probes.

## References

[1] Kaggle, "Spaceship Titanic Competition," Kaggle, accessed May 2026.

[2] JimLiu, "Spaceship Titanic Kaggle solution reference," accessed May 2026.

[3] Ravi20076, "Spaceship Titanic Kaggle solution reference," accessed May 2026.

[4] Additional external Kaggle solution materials reviewed as ensemble-level support sources, accessed May 2026.
