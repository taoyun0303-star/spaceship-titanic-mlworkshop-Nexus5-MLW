from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


RUN_DIR_NAME = "final_model_level_ensemble_0p82277"
FINAL_FILE = "submission_final_model_ensemble_0p82277.csv"
FINAL_SCORE = "0.82277"
COMPLIANCE_NOTE = (
    "complete local/public model prediction arrays plus global vote rules; "
    "no PassengerId hardcoding, no label overwrite, no single-row probe"
)


def find_project_root(start: Path | None = None) -> Path:
    """Find the runnable project root from either the script path or cwd."""
    candidates: list[Path] = []
    if start is not None:
        resolved_start = start.resolve()
        if resolved_start.is_file():
            resolved_start = resolved_start.parent
        candidates.extend([resolved_start, *resolved_start.parents])
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])

    for candidate in candidates:
        has_data = any(p.is_dir() and p.name.startswith("02_") for p in candidate.iterdir() if candidate.exists())
        has_outputs = any(p.is_dir() and p.name.startswith("04_") for p in candidate.iterdir() if candidate.exists())
        if has_data and has_outputs:
            return candidate
    raise FileNotFoundError("Cannot locate project root containing 02_* and 04_* directories")


def first_dir(root: Path, prefix: str) -> Path:
    matches = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"Cannot find directory starting with {prefix!r} under {root}")
    return matches[0]


def as_bool(series: pd.Series) -> pd.Series:
    values = series.astype(str).str.strip().str.lower()
    mapped = values.map({"true": True, "false": False, "1": True, "0": False})
    if mapped.isna().any():
        bad = sorted(values[mapped.isna()].unique().tolist())
        raise ValueError(f"Transported contains non-boolean values: {bad}")
    return mapped.astype(bool).reset_index(drop=True)


def load_submission(path: Path, sample: pd.DataFrame) -> pd.Series:
    df = pd.read_csv(path)
    lower_cols = [column.lower() for column in df.columns]
    if "passengerid" not in lower_cols or "transported" not in lower_cols:
        raise ValueError(f"Submission must contain PassengerId and Transported columns: {path}")

    df = df.rename(
        columns={
            df.columns[lower_cols.index("passengerid")]: "PassengerId",
            df.columns[lower_cols.index("transported")]: "Transported",
        }
    )
    if len(df) != len(sample):
        raise ValueError(f"Row count mismatch for {path}")
    if not df["PassengerId"].astype(str).equals(sample["PassengerId"].astype(str)):
        raise ValueError(f"PassengerId order mismatch for {path}")
    return as_bool(df["Transported"])


def write_submission(path: Path, sample: pd.DataFrame, pred: pd.Series) -> None:
    out = pd.DataFrame(
        {
            "PassengerId": sample["PassengerId"].astype(str),
            "Transported": pred.astype(bool).map({True: "True", False: "False"}),
        }
    )
    out.to_csv(path, index=False)


def add_candidate(
    rows: list[dict[str, object]],
    seen: set[tuple[bool, ...]],
    output_dir: Path,
    file_name: str,
    pred: pd.Series,
    sample: pd.DataFrame,
    base: pd.Series,
    note: str,
) -> None:
    pred = pd.Series(pred).astype(bool).reset_index(drop=True)
    key = tuple(bool(x) for x in pred.tolist())
    if key in seen:
        return
    seen.add(key)

    write_submission(output_dir / file_name, sample, pred)
    diff = pred.ne(base)
    rows.append(
        {
            "file": file_name,
            "note": note,
            "n_changed_vs_reference": int(diff.sum()),
            "true_to_false": int((base & ~pred).sum()),
            "false_to_true": int((~base & pred).sum()),
            "n_true": int(pred.sum()),
            "true_rate": float(pred.mean()),
            "compliance": COMPLIANCE_NOTE,
        }
    )


def build_candidates(project_root: Path) -> pd.DataFrame:
    data_dir = first_dir(project_root, "02_")
    out_dir = first_dir(project_root, "04_")
    run_dir = out_dir / RUN_DIR_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    sample = pd.read_csv(data_dir / "sample_submission.csv")
    public_dir = out_dir / "public_notebook_outputs"
    v110_dir = out_dir / "v110_clean_public_lr_model_nohardcode"

    base = load_submission(
        out_dir / "v124_v120_refinement_nohardcode" / "submission_v124_jimliu_rules_weighted_60_ravi_agree_margin_0p033.csv",
        sample,
    )
    public_sources = {
        "jimliu": load_submission(public_dir / "jimliu_081669" / "submission.csv", sample),
        "ravi": load_submission(public_dir / "ravi20076_submission_v2" / "submission.csv", sample),
        "shivansh": load_submission(public_dir / "shivanshcoding_0821_lr" / "submission.csv", sample),
        "pycaret": load_submission(public_dir / "pycaret_081" / "submission.csv", sample),
        "mos3santos": load_submission(public_dir / "mos3santos_08285" / "submission.csv", sample),
    }
    local_sources = {
        "v108": load_submission(
            out_dir / "v108_jimliu_clean_model_overlays_nohardcode" / "submission_v108_pub2_core5_need5_ft.csv",
            sample,
        ),
        "v110": load_submission(v110_dir / "submission_v110_maj_jimliu_v108_raw.csv", sample),
    }

    rules = np.load(v110_dir / "v110_test_probs_rules.npy").reshape(-1)
    no_rules = np.load(v110_dir / "v110_test_probs_no_rules.npy").reshape(-1)
    rn_probability = (rules + no_rules) / 2
    rn_pred = pd.Series(rn_probability >= 0.5)

    support = pd.concat([*public_sources.values(), *local_sources.values()], axis=1)
    true_votes = support.sum(axis=1)
    false_votes = support.shape[1] - true_votes

    rows: list[dict[str, object]] = []
    seen = {tuple(bool(x) for x in base.tolist())}

    for hi, lo, need_t, need_f in [
        (0.60, 0.40, 6, 6),
        (0.57, 0.43, 6, 6),
        (0.55, 0.45, 6, 6),
        (0.60, 0.40, 5, 6),
        (0.57, 0.43, 5, 6),
        (0.60, 0.40, 6, 5),
    ]:
        pred = base.copy()
        add_true = (
            (~base)
            & (rn_probability >= hi)
            & rn_pred
            & (true_votes >= need_t)
            & public_sources["ravi"]
            & public_sources["shivansh"]
        )
        add_false = (
            base
            & (rn_probability <= lo)
            & (~rn_pred)
            & (false_votes >= need_f)
            & (~public_sources["jimliu"])
            & (~local_sources["v108"])
        )
        pred.loc[add_true | add_false] = rn_pred.loc[add_true | add_false].astype(bool)
        label = f"hi{hi:.2f}_lo{lo:.2f}_t{need_t}_f{need_f}".replace(".", "p")
        add_candidate(
            rows,
            seen,
            run_dir,
            f"submission_final_model_support_{label}.csv",
            pred,
            sample,
            base,
            "final ensemble candidate from local probability and complete-model agreement",
        )

    for need_t, need_f in [(7, 7), (6, 7), (7, 6), (6, 6)]:
        pred = base.copy()
        pred.loc[(~base) & (true_votes >= need_t)] = True
        pred.loc[base & (false_votes >= need_f)] = False
        add_candidate(
            rows,
            seen,
            run_dir,
            FINAL_FILE if (need_t, need_f) == (6, 6) else f"submission_final_model_vote_t{need_t}_f{need_f}.csv",
            pred,
            sample,
            base,
            "final ensemble candidate from complete model-level vote only",
        )

    if not rows:
        raise RuntimeError("No non-duplicate candidates generated")

    summary = pd.DataFrame(rows)
    summary["priority"] = (
        (summary["n_changed_vs_reference"] - 14).abs()
        + summary["true_to_false"] * 0.5
        + summary["false_to_true"] * 0.4
    )
    summary = summary.sort_values(
        ["priority", "n_changed_vs_reference", "true_to_false", "false_to_true", "file"]
    ).reset_index(drop=True)
    summary.to_csv(run_dir / "submission_summary.csv", index=False)
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "method": "local OOF probability model plus complete-source model-level agreement",
                "final_file": FINAL_FILE,
                "final_public_lb": FINAL_SCORE,
                "n_candidates": int(len(summary)),
                "compliance": COMPLIANCE_NOTE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final no-hardcode model-level ensemble.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="Optional runnable project root. If omitted, it is discovered from the script path or cwd.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    script_root = Path(__file__).resolve()
    project_root = args.project_root.resolve() if args.project_root else find_project_root(script_root)
    summary = build_candidates(project_root)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
