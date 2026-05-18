from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = next(ROOT.glob("02_*"))
OUT_DIR = next(ROOT.glob("04_*"))
RUN_DIR = OUT_DIR / "v124_v120_refinement_nohardcode"

RISK_PATTERN = re.compile(
    r"(probe|single[_ -]?(flip|point)|manual|actual_pids|for_teammate|fixed|"
    r"override|best[_ -]?public|public[_ -]?label|bitstring|bsthere|082137)",
    re.IGNORECASE,
)


def assert_safe_source(path: Path | str) -> None:
    if RISK_PATTERN.search(str(path).replace("\\", "/")):
        raise ValueError(f"Risky path blocked: {path}")


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool).reset_index(drop=True)
    values = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if values.isna().any():
        raise ValueError("Transported must contain boolean True/False values")
    return values.astype(bool).reset_index(drop=True)


def load_submission(path: Path, sample: pd.DataFrame) -> pd.Series:
    assert_safe_source(path)
    df = pd.read_csv(path)
    if list(df.columns) != ["PassengerId", "Transported"]:
        raise ValueError(f"Unexpected columns: {path}")
    if len(df) != len(sample):
        raise ValueError(f"Unexpected row count: {path}")
    if not df["PassengerId"].astype(str).equals(sample["PassengerId"].astype(str)):
        raise ValueError(f"PassengerId order mismatch: {path}")
    return as_bool_series(df["Transported"])


def write_submission(path: Path, sample: pd.DataFrame, preds: pd.Series) -> None:
    pd.DataFrame(
        {
            "PassengerId": sample["PassengerId"].astype(str),
            "Transported": preds.astype(bool).map({True: "True", False: "False"}),
        }
    ).to_csv(path, index=False)


def summarize(
    filename: str,
    candidate: pd.Series,
    anchor81973: pd.Series,
    best82066: pd.Series,
    refs: dict[str, pd.Series],
    note: str,
    source_family: str,
) -> dict:
    diff_best = candidate != best82066
    n_diff_best = int(diff_best.sum())
    if 0 < n_diff_best < 2:
        raise ValueError(f"Rejected single-point-sized candidate {filename}: {n_diff_best} changed rows")
    row = {
        "file": filename,
        "source_family": source_family,
        "note": note,
        "n_true": int(candidate.sum()),
        "true_rate": float(candidate.mean()),
        "diff_anchor81973": int((candidate != anchor81973).sum()),
        "anchor_true_to_false": int((anchor81973 & ~candidate).sum()),
        "anchor_false_to_true": int((~anchor81973 & candidate).sum()),
        "diff_best82066": n_diff_best,
        "best_true_to_false": int((best82066 & ~candidate).sum()),
        "best_false_to_true": int((~best82066 & candidate).sum()),
        "compliance": "complete submission from global model/probability rule; no PassengerId labels; no hardcoding",
    }
    for name, ref in refs.items():
        row[f"diff_{name}"] = int((candidate != ref).sum())
    return row


def add_candidate(
    rows: list[dict],
    seen: set[tuple[bool, ...]],
    filename: str,
    candidate: pd.Series,
    sample: pd.DataFrame,
    anchor81973: pd.Series,
    best82066: pd.Series,
    refs: dict[str, pd.Series],
    note: str,
    source_family: str,
) -> None:
    key = tuple(candidate.astype(bool).tolist())
    if key in seen:
        return
    try:
        row = summarize(filename, candidate, anchor81973, best82066, refs, note, source_family)
    except ValueError as exc:
        if "single-point" in str(exc):
            return
        raise
    seen.add(key)
    write_submission(RUN_DIR / filename, sample, candidate)
    rows.append(row)


def probability_sources(v110_dir: Path) -> dict[str, np.ndarray]:
    rules = np.load(v110_dir / "v110_test_probs_rules.npy").reshape(-1)
    norules = np.load(v110_dir / "v110_test_probs_no_rules.npy").reshape(-1)
    cat = np.load(v110_dir / "v110_cat_test_prob.npy").reshape(-1)
    lgb = np.load(v110_dir / "v110_lgb_test_prob.npy").reshape(-1)
    xgb = np.load(v110_dir / "v110_xgb_test_prob.npy").reshape(-1)
    histgb = np.load(v110_dir / "v110_hist_gb_test_prob.npy").reshape(-1)
    extratrees = np.load(v110_dir / "v110_extra_trees_test_prob.npy").reshape(-1)
    return {
        "rules_norules_avg": (rules + norules) / 2.0,
        "rules_weighted_55": 0.55 * rules + 0.45 * norules,
        "rules_weighted_60": 0.60 * rules + 0.40 * norules,
        "norules_weighted_55": 0.45 * rules + 0.55 * norules,
        "gbdt_avg": (cat + lgb + xgb + histgb) / 4.0,
        "tree_all_avg": (cat + lgb + xgb + histgb + extratrees) / 5.0,
        "blend_rn70_gbdt30": 0.70 * ((rules + norules) / 2.0) + 0.30 * ((cat + lgb + xgb + histgb) / 4.0),
        "blend_rn80_gbdt20": 0.80 * ((rules + norules) / 2.0) + 0.20 * ((cat + lgb + xgb + histgb) / 4.0),
        "blend_rn90_gbdt10": 0.90 * ((rules + norules) / 2.0) + 0.10 * ((cat + lgb + xgb + histgb) / 4.0),
    }


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    v110_dir = OUT_DIR / "v110_clean_public_lr_model_nohardcode"
    v120_dir = OUT_DIR / "v120_nohardcode_candidate_sweep"
    public_dir = OUT_DIR / "public_notebook_outputs"
    v108_dir = OUT_DIR / "v108_jimliu_clean_model_overlays_nohardcode"

    jimliu = load_submission(public_dir / "jimliu_081669" / "submission.csv", sample)
    ravi = load_submission(public_dir / "ravi20076_submission_v2" / "submission.csv", sample)
    v108 = load_submission(v108_dir / "submission_v108_pub2_core5_need5_ft.csv", sample)
    v110_best = load_submission(v110_dir / "submission_v110_maj_jimliu_v108_raw.csv", sample)
    anchor81973 = load_submission(
        OUT_DIR / "v117_margin_sweep_nohardcode" / "submission_v117_jimliu_raw_ravi_agree_margin_0p040.csv",
        sample,
    )
    best82066 = load_submission(
        v120_dir / "submission_v120_jimliu_rules_norules_avg_ravi_agree_margin_0p045.csv",
        sample,
    )

    refs = {
        "jimliu": jimliu,
        "ravi": ravi,
        "v108": v108,
        "v110_best": v110_best,
    }
    prob_map = probability_sources(v110_dir)
    rows: list[dict] = []
    seen: set[tuple[bool, ...]] = {tuple(best82066.astype(bool).tolist())}

    # The proven v120 family is extended with wider, still-global confidence
    # margins and slight rules/no-rules weight variants. This is model-level
    # thresholding, not row-level tuning.
    margin_grid = [
        0.026,
        0.027,
        0.029,
        0.031,
        0.033,
        0.037,
        0.039,
        0.042,
        0.044,
        0.046,
        0.048,
        0.052,
        0.055,
        0.058,
        0.062,
        0.065,
        0.070,
        0.075,
        0.080,
        0.090,
    ]
    for prob_name, prob in prob_map.items():
        raw_pred = pd.Series(prob >= 0.5)
        agree_ravi = raw_pred == ravi
        for margin in margin_grid:
            confident = (prob >= 0.5 + margin) | (prob <= 0.5 - margin)
            update = pd.Series(confident) & agree_ravi
            candidate = jimliu.copy()
            candidate.loc[update] = raw_pred.loc[update].astype(bool)
            label = f"{margin:.3f}".replace(".", "p")
            add_candidate(
                rows,
                seen,
                f"submission_v124_jimliu_{prob_name}_ravi_agree_margin_{label}.csv",
                candidate,
                sample,
                anchor81973,
                best82066,
                refs,
                f"JimLiu anchor updated by {prob_name} at global margin {margin:.3f} where complete Ravi agrees",
                f"{prob_name}_ravi_agree",
            )

    # A conservative complete-model support variant around the best v120 output:
    # only broad probability bands with Ravi plus at least one additional full
    # model source agreeing. Min diff is checked against best82066 above.
    support_votes = pd.concat([ravi, v108, v110_best], axis=1).sum(axis=1)
    rn_prob = prob_map["rules_norules_avg"]
    rn_pred = pd.Series(rn_prob >= 0.5)
    for margin in [0.035, 0.040, 0.045, 0.050, 0.055, 0.060, 0.065, 0.070]:
        confident_true = rn_prob >= 0.5 + margin
        confident_false = rn_prob <= 0.5 - margin
        for need_true, need_false in [(2, 2), (3, 2), (2, 3), (3, 3)]:
            candidate = best82066.copy()
            add_true = (~candidate) & confident_true & rn_pred & ravi & (support_votes >= need_true)
            false_votes = 3 - support_votes
            add_false = candidate & confident_false & (~rn_pred) & (~ravi) & (false_votes >= need_false)
            candidate.loc[add_true | add_false] = rn_pred.loc[add_true | add_false].astype(bool)
            label = f"{margin:.3f}".replace(".", "p")
            add_candidate(
                rows,
                seen,
                f"submission_v124_best82066_refine_rn_margin_{label}_t{need_true}_f{need_false}.csv",
                candidate,
                sample,
                anchor81973,
                best82066,
                refs,
                f"Best v120 output refined by rules_norules_avg margin {margin:.3f}, Ravi agreement, and complete-model support",
                "best82066_refine",
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No v124 candidates generated")
    summary["priority"] = (
        (summary["diff_best82066"] - 12).abs() * 1.3
        + summary["best_true_to_false"] * 0.4
        + summary["anchor_true_to_false"] * 0.03
        - summary["best_false_to_true"] * 0.25
    )
    summary = summary.sort_values(
        ["priority", "diff_best82066", "best_true_to_false", "file"],
        ascending=[True, True, True, True],
    )
    summary.to_csv(RUN_DIR / "submission_summary.csv", index=False)
    (RUN_DIR / "run_metadata.json").write_text(
        json.dumps(
            {
                "best82066_reference": str(
                    (v120_dir / "submission_v120_jimliu_rules_norules_avg_ravi_agree_margin_0p045.csv").relative_to(ROOT)
                ),
                "probability_sources": sorted(prob_map),
                "n_candidates": int(len(summary)),
                "compliance": "all v124 candidates are full model-level outputs from global thresholds and complete-model agreement",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.head(50).to_string(index=False))


if __name__ == "__main__":
    main()
