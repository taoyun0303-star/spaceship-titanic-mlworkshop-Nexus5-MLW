# v110 clean public-model reproduction

This run reproduces the public LR-style ensemble as a pure model prediction and does not apply the embedded best-public override from the public notebook.
Candidate ensembles use only complete prediction arrays from clean model submissions plus the v110 raw model output.

OOF CV accuracy: 0.817324
OOF threshold: 0.5025
Stack weight: 0.800

Top local-ranked candidates by closeness to the 0.81669 JimLiu anchor:
- submission_v110_maj_jimliu_v108_raw.csv: diff_jimliu=21, true_rate=0.544073, three complete-model majority: JimLiu, v108, v110 raw
- submission_v110_maj7_clean_public_raw_need4.csv: diff_jimliu=46, true_rate=0.549918, seven complete-model sources, require four true votes
- submission_v110_jimliu_overlay_raw_ravi_agree_margin_0p12.csv: diff_jimliu=83, true_rate=0.558569, JimLiu anchor updated only where high-confidence v110 agrees with Ravi, margin 0.12
- submission_v110_jimliu_overlay_raw_ravi_agree_margin_0p10.csv: diff_jimliu=89, true_rate=0.559972, JimLiu anchor updated only where high-confidence v110 agrees with Ravi, margin 0.10
- submission_v110_jimliu_overlay_raw_ravi_agree_margin_0p08.csv: diff_jimliu=95, true_rate=0.561375, JimLiu anchor updated only where high-confidence v110 agrees with Ravi, margin 0.08
- submission_v110_jimliu_overlay_raw_ravi_agree_margin_0p06.csv: diff_jimliu=115, true_rate=0.566051, JimLiu anchor updated only where high-confidence v110 agrees with Ravi, margin 0.06
- submission_v110_maj_jimliu_ravi_raw.csv: diff_jimliu=136, true_rate=0.570961, three complete-model majority: JimLiu, Ravi, v110 raw
- submission_v110_jimliu_overlay_raw_margin_0p12.csv: diff_jimliu=324, true_rate=0.515314, JimLiu anchor overwritten only by high-confidence v110 model probabilities, margin 0.12
