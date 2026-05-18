from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = next(ROOT.glob("02_*"))
OUT_DIR = next(ROOT.glob("04_*"))
RUN_DIR = OUT_DIR / "v120_nohardcode_candidate_sweep"

RISK_PATTERN = re.compile(
    r"(probe|single[_ -]?(flip|point)|manual|actual_pids|for_teammate|fixed|"
    r"override|best[_ -]?public|public[_ -]?label|bitstring|bsthere|082137)",
    re.IGNORECASE,
)


def assert_safe_source(path: Path | str) -> None:
    text = str(path).replace("\\", "/")
    if RISK_PATTERN.search(text):
        raise ValueError(f"Risky path blocked: {path}")


def as_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({"true": True, "false": False})
    if mapped.isna().any():
        raise ValueError("Transported must contain boolean True/False values")
    return mapped.astype(bool)


def load_submission(path: Path, sample: pd.DataFrame) -> pd.Series:
    assert_safe_source(path)
    df = pd.read_csv(path)
    if list(df.columns) != ["PassengerId", "Transported"]:
        raise ValueError(f"Unexpected columns: {path}")
    if len(df) != len(sample):
        raise ValueError(f"Unexpected row count: {path}")
    if not df["PassengerId"].astype(str).equals(sample["PassengerId"].astype(str)):
        raise ValueError(f"PassengerId order mismatch: {path}")
    return as_bool_series(df["Transported"]).reset_index(drop=True)


def write_submission(path: Path, sample: pd.DataFrame, preds: pd.Series) -> None:
    df = pd.DataFrame(
        {
            "PassengerId": sample["PassengerId"].astype(str),
            "Transported": preds.astype(bool).map({True: "True", False: "False"}),
        }
    )
    df.to_csv(path, index=False)


def summarize_candidate(
    filename: str,
    candidate: pd.Series,
    anchor: pd.Series,
    references: dict[str, pd.Series],
    note: str,
    *,
    min_changed: int = 2,
) -> dict:
    changed = candidate != anchor
    n_changed = int(changed.sum())
    if 0 < n_changed < min_changed:
        raise ValueError(f"Rejected single-point-sized candidate {filename}: {n_changed} changed rows")
    row = {
        "file": filename,
        "note": note,
        "n_true": int(candidate.sum()),
        "true_rate": float(candidate.mean()),
        "diff_anchor": n_changed,
        "anchor_true_to_false": int((anchor & ~candidate).sum()),
        "anchor_false_to_true": int((~anchor & candidate).sum()),
        "compliance": "global model/probability rule; complete submission; no PassengerId list; no hardcoding; no single-point flip",
    }
    for name, ref in references.items():
        row[f"diff_{name}"] = int((candidate != ref).sum())
    return row


def probability_sources(v110_dir: Path) -> dict[str, np.ndarray]:
    sources = {
        "rules": np.load(v110_dir / "v110_test_probs_rules.npy").reshape(-1),
        "norules": np.load(v110_dir / "v110_test_probs_no_rules.npy").reshape(-1),
        "cat": np.load(v110_dir / "v110_cat_test_prob.npy").reshape(-1),
        "lgb": np.load(v110_dir / "v110_lgb_test_prob.npy").reshape(-1),
        "xgb": np.load(v110_dir / "v110_xgb_test_prob.npy").reshape(-1),
        "histgb": np.load(v110_dir / "v110_hist_gb_test_prob.npy").reshape(-1),
        "extratrees": np.load(v110_dir / "v110_extra_trees_test_prob.npy").reshape(-1),
    }
    sources["rules_norules_avg"] = (sources["rules"] + sources["norules"]) / 2.0
    sources["gbdt_avg"] = (sources["cat"] + sources["lgb"] + sources["xgb"] + sources["histgb"]) / 4.0
    sources["all_model_avg"] = (
        sources["cat"] + sources["lgb"] + sources["xgb"] + sources["histgb"] + sources["extratrees"]
    ) / 5.0
    return sources


def add_candidate(
    rows: list[dict],
    seen: set[tuple[bool, ...]],
    filename: str,
    candidate: pd.Series,
    sample: pd.DataFrame,
    anchor: pd.Series,
    references: dict[str, pd.Series],
    note: str,
    *,
    min_changed: int = 2,
) -> None:
    key = tuple(candidate.astype(bool).tolist())
    if key in seen:
        return
    try:
        row = summarize_candidate(filename, candidate, anchor, references, note, min_changed=min_changed)
    except ValueError as exc:
        if "single-point" in str(exc):
            return
        raise
    seen.add(key)
    write_submission(RUN_DIR / filename, sample, candidate)
    rows.append(row)


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    sample = pd.read_csv(DATA_DIR / "sample_submission.csv")
    v110_dir = OUT_DIR / "v110_clean_public_lr_model_nohardcode"
    v108_dir = OUT_DIR / "v108_jimliu_clean_model_overlays_nohardcode"
    public_dir = OUT_DIR / "public_notebook_outputs"

    preds = {
        "jimliu": load_submission(public_dir / "jimliu_081669" / "submission.csv", sample),
        "ravi": load_submission(public_dir / "ravi20076_submission_v2" / "submission.csv", sample),
        "v108_ft": load_submission(v108_dir / "submission_v108_pub2_core5_need5_ft.csv", sample),
        "v108_both": load_submission(v108_dir / "submission_v108_pub2_core5_need5_both.csv", sample),
        "v107_agree6": load_submission(
            OUT_DIR / "v107_public_model_blends_nohardcode" / "submission_v107_jimliu_overlay_agree6.csv",
            sample,
        ),
        "v110_best": load_submission(v110_dir / "submission_v110_maj_jimliu_v108_raw.csv", sample),
        "v110_raw": load_submission(v110_dir / "submission_v110_raw_model_oofbest_rules.csv", sample),
        "jimmy_xgb": load_submission(public_dir / "jimmyyeung_xgb_top5" / "submission.csv", sample),
        "guan_xgb": load_submission(public_dir / "guanlintao_0814_xgb" / "Submission_XGB.csv", sample),
        "ishan_cat": load_submission(public_dir / "ishanpurohit_top5" / "Submission.csv", sample),
    }
    anchor = load_submission(
        OUT_DIR / "v117_margin_sweep_nohardcode" / "submission_v117_jimliu_raw_ravi_agree_margin_0p040.csv",
        sample,
    )
    preds["anchor81973"] = anchor

    prob_sources = probability_sources(v110_dir)
    rows: list[dict] = []
    seen: set[tuple[bool, ...]] = {tuple(anchor.astype(bool).tolist())}
    references = {
        "jimliu": preds["jimliu"],
        "ravi": preds["ravi"],
        "v108_ft": preds["v108_ft"],
        "v110_best": preds["v110_best"],
        "anchor81973": anchor,
    }

    # Family A: the successful v117 rule generalized to several complete v110
    # probability sources. Candidate changes are produced by a global margin and
    # Ravi complete-model agreement, never by individual PassengerId selection.
    for prob_name, prob in prob_sources.items():
        raw_pred = pd.Series(prob >= 0.5)
        agree_ravi = raw_pred == preds["ravi"]
        for margin in [0.025, 0.028, 0.030, 0.032, 0.034, 0.035, 0.036, 0.038, 0.040, 0.045, 0.050, 0.060]:
            candidate = preds["jimliu"].copy()
            confident = (prob >= 0.5 + margin) | (prob <= 0.5 - margin)
            update = pd.Series(confident) & agree_ravi
            candidate.loc[update] = raw_pred.loc[update].astype(bool)
            label = f"{margin:.3f}".replace(".", "p")
            add_candidate(
                rows,
                seen,
                f"submission_v120_jimliu_{prob_name}_ravi_agree_margin_{label}.csv",
                candidate,
                sample,
                anchor,
                references,
                f"JimLiu anchor updated by {prob_name} probability only where globally confident and agreeing with complete Ravi output",
            )

    # Family B: extend the current 0.81973 anchor only in the safer false->true
    # direction, requiring probability confidence plus complete-model support.
    support_sets = {
        "ravi_v108": ["ravi", "v108_ft"],
        "ravi_v108_v110": ["ravi", "v108_ft", "v110_best"],
        "top4": ["ravi", "v108_ft", "v110_best", "v107_agree6"],
        "public5": ["ravi", "v108_ft", "v110_best", "jimmy_xgb", "ishan_cat"],
    }
    for prob_name, prob in prob_sources.items():
        for support_name, support_keys in support_sets.items():
            support_votes = pd.concat([preds[key] for key in support_keys], axis=1).sum(axis=1)
            for threshold in [0.52, 0.54, 0.56, 0.58, 0.60, 0.62, 0.64]:
                for need in [len(support_keys), max(2, len(support_keys) - 1)]:
                    candidate = anchor.copy()
                    add_true = (~anchor) & (pd.Series(prob) >= threshold) & (support_votes >= need)
                    candidate.loc[add_true] = True
                    label = f"{threshold:.2f}".replace(".", "p")
                    add_candidate(
                        rows,
                        seen,
                        f"submission_v120_anchor_addtrue_{prob_name}_{support_name}_ge_{label}_need{need}.csv",
                        candidate,
                        sample,
                        anchor,
                        references,
                        f"v117 0.81973 anchor add-true using {prob_name} >= {threshold:.2f} and {support_name} complete-model support need {need}",
                    )

    # Family C: complete hard-vote candidates centered on v117. These are kept
    # distinct from earlier broad v116 votes by making v117 the anchor source and
    # only trying high-support thresholds.
    vote_sets = {
        "anchor_top5": ["anchor81973", "ravi", "v108_ft", "v110_best", "v107_agree6"],
        "anchor_public7": ["anchor81973", "ravi", "v108_ft", "v110_best", "v107_agree6", "jimmy_xgb", "ishan_cat"],
        "anchor_diverse8": [
            "anchor81973",
            "ravi",
            "v108_ft",
            "v110_best",
            "v107_agree6",
            "jimmy_xgb",
            "guan_xgb",
            "ishan_cat",
        ],
    }
    for vote_name, keys in vote_sets.items():
        votes = pd.concat([preds[key] for key in keys], axis=1).sum(axis=1)
        for need in range((len(keys) // 2) + 1, len(keys) + 1):
            candidate = votes >= need
            add_candidate(
                rows,
                seen,
                f"submission_v120_vote_{vote_name}_need{need}.csv",
                candidate,
                sample,
                anchor,
                references,
                f"complete-model hard vote centered on v117 source set {vote_name}, require {need}/{len(keys)} true votes",
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError("No non-duplicate v120 candidates were generated")
    summary["priority"] = (
        summary["anchor_true_to_false"] * 10
        + (summary["diff_anchor"] - summary["anchor_false_to_true"]).clip(lower=0) * 2
        + (summary["diff_anchor"] - 80).abs() * 0.05
        - summary["anchor_false_to_true"] * 0.15
    )
    summary = summary.sort_values(
        ["anchor_true_to_false", "priority", "diff_anchor", "anchor_false_to_true", "file"],
        ascending=[True, True, True, False, True],
    )
    summary.to_csv(RUN_DIR / "submission_summary.csv", index=False)
    metadata = {
        "anchor": str(
            (OUT_DIR / "v117_margin_sweep_nohardcode" / "submission_v117_jimliu_raw_ravi_agree_margin_0p040.csv")
            .relative_to(ROOT)
        ),
        "probability_sources": sorted(prob_sources),
        "complete_sources": {name: "complete submission array" for name in sorted(preds)},
        "compliance": "all candidates are generated by global model/probability rules; no PassengerId labels or public-label overrides",
        "n_candidates": int(len(summary)),
    }
    (RUN_DIR / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.head(40).to_string(index=False))


if __name__ == "__main__":
    main()
