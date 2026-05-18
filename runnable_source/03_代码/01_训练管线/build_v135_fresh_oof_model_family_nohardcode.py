from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


def optional_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


HAS_XGB = optional_module("xgboost")
HAS_LGB = optional_module("lightgbm")
HAS_CAT = optional_module("catboost")

if HAS_XGB:
    import xgboost as xgb
if HAS_LGB:
    import lightgbm as lgb
if HAS_CAT:
    from catboost import CatBoostClassifier


RISK_PATTERN = re.compile(
    r"(probe|single[_ -]?(flip|point)|manual|actual_pids|for_teammate|fixed_actual|"
    r"override|best[_ -]?public|public[_ -]?label|bitstring|bsthere|082137|"
    r"titanic_anchor|compare_to_titanic|changed_points|candidate_top)",
    re.IGNORECASE,
)

COMPLIANCE_NOTE = (
    "complete model outputs and global OOF probability rules only; "
    "no PassengerId hardcoding; no leaderboard-derived row fixes"
)


@dataclass(frozen=True)
class CFG:
    run_dir_name: str = "v135_fresh_oof_model_family_nohardcode"
    target: str = "Transported"
    competition: str = "spaceship-titanic"
    seeds: tuple[int, ...] = (42, 2024)
    smoke_seeds: tuple[int, ...] = (42,)
    n_splits: int = 5
    smoke_splits: int = 3
    spend_cols: tuple[str, ...] = ("RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck")
    candidate_limit: int = 6


def find_prefixed_dir(root: Path, prefix: str) -> Path:
    matches = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    if not matches:
        raise FileNotFoundError(f"Missing directory with prefix {prefix} under {root}")
    return matches[0]


def resolve_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    data_dir = find_prefixed_dir(root, "02_")
    out_root = find_prefixed_dir(root, "04_")
    run_dir = out_root / CFG.run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "root": root,
        "data_dir": data_dir,
        "out_root": out_root,
        "run_dir": run_dir,
        "train": data_dir / "train.csv",
        "test": data_dir / "test.csv",
        "sample": data_dir / "sample_submission.csv",
    }


def parse_bool_target(series: pd.Series) -> np.ndarray:
    values = series.astype(str).str.strip().str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
    if values.isna().any():
        raise ValueError("Transported contains non-boolean values")
    return values.astype(int).to_numpy()


def split_passenger_id(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["PassengerId"].astype(str).str.split("_", expand=True)
    df["GroupId"] = parts[0].astype(int)
    df["GroupMember"] = parts[1].astype(int)
    return df


def split_cabin(df: pd.DataFrame) -> pd.DataFrame:
    cabin = df["Cabin"].astype("string").fillna("U/-1/U").str.split("/", expand=True)
    df["CabinDeck"] = cabin[0].fillna("U").astype(str)
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce").fillna(-1).astype(int)
    df["CabinSide"] = cabin[2].fillna("U").astype(str)
    df["CabinNumBin"] = pd.cut(
        df["CabinNum"],
        bins=[-2, -1, 100, 300, 600, 1000, 2000],
        labels=["missing", "0_100", "101_300", "301_600", "601_1000", "1000_plus"],
    ).astype(str)
    return df


def add_name_features(df: pd.DataFrame) -> pd.DataFrame:
    name = df["Name"].astype("string").fillna("Unknown Unknown")
    pieces = name.str.rsplit(" ", n=1, expand=True)
    df["Surname"] = pieces[1].fillna("Unknown").astype(str)
    df["NameMissing"] = df["Name"].isna().astype(int)
    return df


def build_features(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    train_x = train.drop(columns=[CFG.target]).copy()
    test_x = test.copy()
    combined = pd.concat([train_x, test_x], axis=0, ignore_index=True)
    combined = split_passenger_id(combined)
    combined = split_cabin(combined)
    combined = add_name_features(combined)

    group_sizes = combined.groupby("GroupId")["PassengerId"].transform("size")
    combined["GroupSize"] = group_sizes.astype(int)
    combined["IsAlone"] = (combined["GroupSize"] == 1).astype(int)
    surname_sizes = combined.groupby("Surname")["PassengerId"].transform("size")
    combined["SurnameSize"] = surname_sizes.astype(int)

    for col in CFG.spend_cols:
        combined[f"{col}Missing"] = combined[col].isna().astype(int)
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)
        combined[f"Log{col}"] = np.log1p(combined[col])

    combined["TotalSpend"] = combined[list(CFG.spend_cols)].sum(axis=1)
    combined["LogTotalSpend"] = np.log1p(combined["TotalSpend"])
    combined["AnySpend"] = (combined["TotalSpend"] > 0).astype(int)
    combined["NoSpend"] = (combined["TotalSpend"] == 0).astype(int)
    combined["SpendMissingCount"] = combined[[f"{c}Missing" for c in CFG.spend_cols]].sum(axis=1)

    combined["AgeMissing"] = combined["Age"].isna().astype(int)
    combined["Age"] = pd.to_numeric(combined["Age"], errors="coerce")
    combined["AgeFill"] = combined["Age"].fillna(combined["Age"].median())
    combined["AgeBand"] = pd.cut(
        combined["AgeFill"],
        bins=[-1, 12, 18, 25, 35, 50, 80],
        labels=["child", "teen", "young", "adult", "middle", "senior"],
    ).astype(str)

    for col in ["HomePlanet", "CryoSleep", "Destination", "VIP"]:
        combined[col] = combined[col].astype("string").fillna("Missing").astype(str)

    combined["CryoNoSpend"] = ((combined["CryoSleep"] == "True") & (combined["TotalSpend"] == 0)).astype(int)
    combined["CryoSpendConflict"] = ((combined["CryoSleep"] == "True") & (combined["TotalSpend"] > 0)).astype(int)
    combined["HomeDest"] = combined["HomePlanet"] + "_" + combined["Destination"]
    combined["DeckSide"] = combined["CabinDeck"] + "_" + combined["CabinSide"]
    combined["HomeDeck"] = combined["HomePlanet"] + "_" + combined["CabinDeck"]
    combined["DestDeck"] = combined["Destination"] + "_" + combined["CabinDeck"]

    categorical_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "CabinNumBin",
        "Surname",
        "AgeBand",
        "HomeDest",
        "DeckSide",
        "HomeDeck",
        "DestDeck",
    ]
    numeric_cols = [
        "GroupId",
        "GroupMember",
        "GroupSize",
        "IsAlone",
        "SurnameSize",
        "CabinNum",
        "AgeFill",
        "AgeMissing",
        "TotalSpend",
        "LogTotalSpend",
        "AnySpend",
        "NoSpend",
        "SpendMissingCount",
        "CryoNoSpend",
        "CryoSpendConflict",
    ]
    numeric_cols += list(CFG.spend_cols)
    numeric_cols += [f"Log{c}" for c in CFG.spend_cols]
    numeric_cols += [f"{c}Missing" for c in CFG.spend_cols]

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1)
    combined[categorical_cols] = encoder.fit_transform(combined[categorical_cols]).astype(float)
    combined[numeric_cols] = combined[numeric_cols].replace([np.inf, -np.inf], np.nan).fillna(-1.0)

    feature_cols = categorical_cols + numeric_cols
    train_features = combined.iloc[: len(train)].reset_index(drop=True)
    test_features = combined.iloc[len(train) :].reset_index(drop=True)
    return train_features[feature_cols], test_features[feature_cols], feature_cols, categorical_cols


def bool_submission(sample: pd.DataFrame, preds: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "PassengerId": sample["PassengerId"].astype(str),
            "Transported": pd.Series(preds.astype(bool)).map({True: "True", False: "False"}),
        }
    )


def validate_submission(path: Path, sample: pd.DataFrame) -> dict[str, object]:
    sub = pd.read_csv(path)
    valid = True
    reason = "ok"
    if list(sub.columns) != ["PassengerId", "Transported"]:
        valid = False
        reason = "bad_columns"
    elif len(sub) != len(sample):
        valid = False
        reason = "bad_length"
    elif sub["PassengerId"].astype(str).tolist() != sample["PassengerId"].astype(str).tolist():
        valid = False
        reason = "bad_passenger_order"
    elif not set(sub["Transported"].astype(str).str.lower()).issubset({"true", "false"}):
        valid = False
        reason = "bad_target_values"
    return {"file": path.name, "valid": bool(valid), "reason": reason}


def make_models(seed: int, smoke: bool) -> list[tuple[str, object]]:
    models: list[tuple[str, object]] = []
    if HAS_CAT:
        models.append(
            (
                "cat",
                CatBoostClassifier(
                    iterations=80 if smoke else 450,
                    depth=5,
                    learning_rate=0.045,
                    loss_function="Logloss",
                    eval_metric="Accuracy",
                    random_seed=seed,
                    verbose=False,
                    allow_writing_files=False,
                ),
            )
        )
    if HAS_LGB:
        models.append(
            (
                "lgb",
                lgb.LGBMClassifier(
                    n_estimators=80 if smoke else 500,
                    learning_rate=0.035,
                    num_leaves=31,
                    subsample=0.82,
                    colsample_bytree=0.82,
                    reg_lambda=2.0,
                    random_state=seed,
                    objective="binary",
                    verbosity=-1,
                ),
            )
        )
    if HAS_XGB:
        models.append(
            (
                "xgb",
                xgb.XGBClassifier(
                    n_estimators=70 if smoke else 420,
                    max_depth=4,
                    learning_rate=0.035,
                    subsample=0.84,
                    colsample_bytree=0.84,
                    reg_lambda=2.5,
                    eval_metric="logloss",
                    random_state=seed,
                    n_jobs=2,
                    tree_method="hist",
                ),
            )
        )
    models.extend(
        [
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=80 if smoke else 650,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
            (
                "hist_gb",
                HistGradientBoostingClassifier(
                    max_iter=70 if smoke else 320,
                    learning_rate=0.045,
                    l2_regularization=0.05,
                    random_state=seed,
                ),
            ),
        ]
    )
    if not smoke:
        models.append(
            (
                "rf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    random_state=seed,
                    n_jobs=-1,
                ),
            )
        )
    return models


def best_threshold(y_true: np.ndarray, prob: np.ndarray) -> tuple[float, float]:
    best_thr = 0.5
    best_acc = -1.0
    for thr in np.linspace(0.35, 0.65, 121):
        acc = accuracy_score(y_true, prob >= thr)
        if acc > best_acc:
            best_acc = float(acc)
            best_thr = float(thr)
    return best_thr, best_acc


def predict_proba_positive(model: object, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)
    return np.asarray(proba)[:, 1]


def train_model_family(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    y: np.ndarray,
    smoke: bool,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[dict[str, object]]]:
    seeds = CFG.smoke_seeds if smoke else CFG.seeds
    splits = CFG.smoke_splits if smoke else CFG.n_splits
    skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=2026)
    model_probs = []
    model_test_probs = []
    rows: list[dict[str, object]] = []

    for seed in seeds:
        for model_name, model in make_models(seed, smoke):
            oof = np.zeros(len(train_x), dtype=float)
            test_fold_probs = []
            for fold, (tr_idx, va_idx) in enumerate(skf.split(train_x, y), start=1):
                x_tr = train_x.iloc[tr_idx]
                x_va = train_x.iloc[va_idx]
                y_tr = y[tr_idx]
                y_va = y[va_idx]
                model.fit(x_tr, y_tr)
                va_prob = predict_proba_positive(model, x_va)
                test_prob = predict_proba_positive(model, test_x)
                oof[va_idx] = va_prob
                test_fold_probs.append(test_prob)
                fold_thr, fold_acc = best_threshold(y_va, va_prob)
                rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "fold": fold,
                        "accuracy": fold_acc,
                        "threshold": fold_thr,
                        "auc": roc_auc_score(y_va, va_prob),
                        "logloss": log_loss(y_va, np.clip(va_prob, 1e-6, 1 - 1e-6)),
                    }
                )
            test_avg = np.mean(test_fold_probs, axis=0)
            model_thr, model_acc = best_threshold(y, oof)
            rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    "fold": "oof",
                    "accuracy": model_acc,
                    "threshold": model_thr,
                    "auc": roc_auc_score(y, oof),
                    "logloss": log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)),
                }
            )
            model_probs.append(oof)
            model_test_probs.append(test_avg)

    oof_stack = np.vstack(model_probs).T
    test_stack = np.vstack(model_test_probs).T
    summary = pd.DataFrame(rows)
    return summary, oof_stack, test_stack, rows


def save_probability_assets(run_dir: Path, oof_stack: np.ndarray, test_stack: np.ndarray) -> None:
    np.save(run_dir / "v135_oof_stack.npy", oof_stack)
    np.save(run_dir / "v135_test_stack.npy", test_stack)


def load_optional_v110(paths: dict[str, Path], y_len: int, test_len: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    v110_dir = paths["out_root"] / "v110_clean_public_lr_model_nohardcode"
    oof_path = v110_dir / "v110_oof_probs.npy"
    test_path = v110_dir / "v110_test_probs_rules.npy"
    if not oof_path.exists() or not test_path.exists():
        return None, None
    if RISK_PATTERN.search(str(oof_path)) or RISK_PATTERN.search(str(test_path)):
        return None, None
    oof = np.load(oof_path)
    test = np.load(test_path)
    if len(oof) != y_len or len(test) != test_len:
        return None, None
    return np.asarray(oof, dtype=float), np.asarray(test, dtype=float)


def load_v124_anchor(paths: dict[str, Path], sample: pd.DataFrame) -> np.ndarray | None:
    anchor_path = (
        paths["out_root"]
        / "v124_v120_refinement_nohardcode"
        / "submission_v124_jimliu_rules_weighted_60_ravi_agree_margin_0p033.csv"
    )
    if not anchor_path.exists():
        return None
    sub = pd.read_csv(anchor_path)
    if sub["PassengerId"].astype(str).tolist() != sample["PassengerId"].astype(str).tolist():
        return None
    return sub["Transported"].astype(str).str.lower().map({"true": 1.0, "false": 0.0}).to_numpy()


def rank_average(stack: np.ndarray) -> np.ndarray:
    ranks = np.zeros_like(stack, dtype=float)
    for i in range(stack.shape[1]):
        order = np.argsort(stack[:, i])
        ranks[order, i] = np.linspace(0.0, 1.0, len(stack))
    return ranks.mean(axis=1)


def fit_logistic_stack(y: np.ndarray, oof_stack: np.ndarray, test_stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    meta = LogisticRegression(C=0.5, max_iter=1000, solver="lbfgs")
    meta.fit(oof_stack, y)
    return meta.predict_proba(oof_stack)[:, 1], meta.predict_proba(test_stack)[:, 1]


def build_candidate_probs(
    y: np.ndarray,
    oof_stack: np.ndarray,
    test_stack: np.ndarray,
    paths: dict[str, Path],
    sample: pd.DataFrame,
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    mean_oof = oof_stack.mean(axis=1)
    mean_test = test_stack.mean(axis=1)
    rank_oof = rank_average(oof_stack)
    rank_test = rank_average(test_stack)
    stack_oof, stack_test = fit_logistic_stack(y, oof_stack, test_stack)

    base_items = [
        ("v135_soft_mean", mean_oof, mean_test),
        ("v135_rank_average", rank_oof, rank_test),
        ("v135_logistic_stack", stack_oof, stack_test),
    ]
    v110_oof, v110_test = load_optional_v110(paths, len(y), len(sample))
    if v110_oof is not None and v110_test is not None:
        base_items.append(("v135_v110_blend65", 0.65 * mean_oof + 0.35 * v110_oof, 0.65 * mean_test + 0.35 * v110_test))
        base_items.append(
            ("v135_stack_v110_blend70", 0.70 * stack_oof + 0.30 * v110_oof, 0.70 * stack_test + 0.30 * v110_test)
        )
    for name, oof_prob, test_prob in base_items:
        thr, acc = best_threshold(y, np.asarray(oof_prob))
        candidates.append(
            {
                "family": name,
                "oof_prob": np.asarray(oof_prob),
                "test_prob": np.asarray(test_prob),
                "threshold": thr,
                "oof_accuracy": acc,
            }
        )
    anchor = load_v124_anchor(paths, sample)
    if anchor is not None:
        thr, acc = best_threshold(y, stack_oof)
        candidates.append(
            {
                "family": "v135_anchor_prob_blend55",
                "oof_prob": stack_oof,
                "test_prob": 0.55 * stack_test + 0.45 * anchor,
                "threshold": thr,
                "oof_accuracy": acc,
            }
        )
    candidates.sort(key=lambda row: float(row["oof_accuracy"]), reverse=True)
    return candidates


def write_candidates(
    candidates: list[dict[str, object]],
    run_dir: Path,
    sample: pd.DataFrame,
    max_candidates: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    for old_submission in run_dir.glob("submission_v135_*.csv"):
        old_submission.unlink()
    summary_rows = []
    validation_rows = []
    for rank, item in enumerate(candidates[:max_candidates], start=1):
        family = str(item["family"])
        threshold = float(item["threshold"])
        test_prob = np.asarray(item["test_prob"])
        preds = test_prob >= threshold
        threshold_text = f"{threshold:.3f}".replace(".", "p")
        filename = f"submission_v135_{rank:02d}_{family}_thr_{threshold_text}.csv"
        path = run_dir / filename
        bool_submission(sample, preds).to_csv(path, index=False)
        validation = validate_submission(path, sample)
        validation_rows.append(validation)
        summary_rows.append(
            {
                "file": filename,
                "rank": rank,
                "family": family,
                "oof_accuracy": float(item["oof_accuracy"]),
                "threshold": threshold,
                "positive_rate": float(preds.mean()),
                "true_count": int(preds.sum()),
                "compliance": COMPLIANCE_NOTE,
            }
        )
    return pd.DataFrame(summary_rows), pd.DataFrame(validation_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="run reduced folds/models for tests")
    parser.add_argument("--max-candidates", type=int, default=CFG.candidate_limit)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.time()
    paths = resolve_paths()
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    sample = pd.read_csv(paths["sample"])
    y = parse_bool_target(train[CFG.target])
    train_x, test_x, feature_cols, categorical_cols = build_features(train, test)

    family_summary, oof_stack, test_stack, _ = train_model_family(train_x, test_x, y, smoke=args.smoke)
    save_probability_assets(paths["run_dir"], oof_stack, test_stack)
    family_summary.to_csv(paths["run_dir"] / "model_family_summary.csv", index=False)

    candidates = build_candidate_probs(y, oof_stack, test_stack, paths, sample)
    candidate_summary, validation = write_candidates(candidates, paths["run_dir"], sample, args.max_candidates)
    candidate_summary.to_csv(paths["run_dir"] / "candidate_summary.csv", index=False)
    validation.to_csv(paths["run_dir"] / "format_validation.csv", index=False)

    run_summary = {
        "run_dir": paths["run_dir"].relative_to(paths["root"]).as_posix(),
        "smoke": bool(args.smoke),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_features": int(len(feature_cols)),
        "categorical_features": categorical_cols,
        "n_models": int(oof_stack.shape[1]),
        "n_candidates": int(len(candidate_summary)),
        "all_valid": bool(validation["valid"].all()),
        "compliance": COMPLIANCE_NOTE,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    (paths["run_dir"] / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
