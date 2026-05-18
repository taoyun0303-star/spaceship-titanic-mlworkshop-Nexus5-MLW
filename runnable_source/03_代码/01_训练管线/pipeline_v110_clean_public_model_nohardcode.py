from __future__ import annotations

import argparse
import importlib.util
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


RISK_PATTERN = re.compile(
    r"(override|bitstring|best[_-]?public|public[_ -]?label|probe|single[_ -]?(flip|point)|"
    r"manual|actual_pids|for_teammate|fixed_actual|lb_probes|bsthere|082137)",
    re.IGNORECASE,
)


def optional_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


XGB_AVAILABLE = optional_module("xgboost")
LGB_AVAILABLE = optional_module("lightgbm")
CAT_AVAILABLE = optional_module("catboost")

if XGB_AVAILABLE:
    import xgboost as xgb

if LGB_AVAILABLE:
    import lightgbm as lgb

if CAT_AVAILABLE:
    from catboost import CatBoostClassifier


class CFG:
    target = "Transported"
    competition = "spaceship-titanic"
    run_dir_name = "v110_clean_public_lr_model_nohardcode"
    seeds = (42, 2024)
    n_splits = 5
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    categorical_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "HomeDest",
        "DeckSide",
        "CabinZone",
        "AgeBand",
        "Surname",
    ]
    feature_cols = [
        "HomePlanet",
        "CryoSleep",
        "Destination",
        "VIP",
        "CabinDeck",
        "CabinSide",
        "HomeDest",
        "DeckSide",
        "CabinZone",
        "AgeBand",
        "Surname",
        "GroupId",
        "GroupMember",
        "GroupSize",
        "Solo",
        "FamilySize",
        "Age",
        "CabinNum",
        "CryoFlag",
        "VipFlag",
        "IsChild",
        "IsTeen",
        "IsSenior",
        "SpendPositiveCount",
        "NoSpend",
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck",
        "TotalSpend",
        "AvgSpendPerService",
        "SpendPerGroupMember",
        "Log_RoomService",
        "Log_FoodCourt",
        "Log_ShoppingMall",
        "Log_Spa",
        "Log_VRDeck",
        "Log_TotalSpend",
        "Log_AvgSpendPerService",
        "Log_SpendPerGroupMember",
        "AgeSpendInteraction",
    ]


def find_prefixed_dir(root: Path, prefix: str) -> Path:
    matches = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    if not matches:
        raise FileNotFoundError(f"Cannot find directory starting with {prefix!r} under {root}")
    return matches[0]


def project_paths() -> dict[str, Path]:
    root = Path(__file__).resolve().parents[2]
    data_dir = find_prefixed_dir(root, "02_")
    output_root = find_prefixed_dir(root, "04_")
    run_dir = output_root / CFG.run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return {"root": root, "data": data_dir, "output_root": output_root, "run": run_dir}


def display_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def assert_no_hardcode_risk(path: Path | str) -> None:
    text = str(path).replace("\\", "/")
    if RISK_PATTERN.search(text):
        raise ValueError(f"hardcode risk in source path: {path}")


def parse_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(bool)
    text = series.astype(str).str.strip().str.lower()
    mapped = text.map({"true": True, "false": False})
    if mapped.notna().all():
        return mapped.astype(bool)
    raise ValueError("Transported must contain boolean True/False values")


def validate_submission(path: Path, test_ids: np.ndarray) -> dict[str, Any]:
    row: dict[str, Any] = {"file": path.name, "path": str(path), "valid": False}
    try:
        df = pd.read_csv(path)
        if list(df.columns) != ["PassengerId", "Transported"]:
            row["reason"] = "bad_columns"
            return row
        if len(df) != len(test_ids):
            row["reason"] = "bad_row_count"
            return row
        ids = df["PassengerId"].astype(str).to_numpy()
        if not np.array_equal(ids, test_ids.astype(str)):
            row["reason"] = "passenger_order_mismatch"
            return row
        if not df["PassengerId"].is_unique:
            row["reason"] = "duplicate_passenger_id"
            return row
        pred = parse_bool_series(df["Transported"]).to_numpy(dtype=bool)
    except Exception as exc:
        row["reason"] = str(exc)
        return row
    row.update(
        {
            "valid": True,
            "rows": int(len(df)),
            "true_count": int(pred.sum()),
            "true_rate": float(pred.mean()),
            "reason": "",
        }
    )
    return row


def mode_or_nan(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    modes = non_null.mode(dropna=True)
    if modes.empty:
        return non_null.iloc[0]
    return modes.iloc[0]


def fill_from_group_mode(frame: pd.DataFrame, key_col: str, value_col: str) -> None:
    mapping = frame.groupby(key_col)[value_col].agg(mode_or_nan)
    frame[value_col] = frame[value_col].fillna(frame[key_col].map(mapping))


def parse_cabin(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    cabin = series.fillna("U/9999/U").astype(str).str.split("/", expand=True)
    deck = cabin[0].replace("nan", "U")
    num = pd.to_numeric(cabin[1], errors="coerce")
    side = cabin[2].replace("nan", "U")
    return deck, num, side


def engineer_features(train_frame: pd.DataFrame, test_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train_frame.copy()
    test = test_frame.copy()
    train["_is_train"] = 1
    test["_is_train"] = 0
    test[CFG.target] = np.nan
    full = pd.concat([train, test], ignore_index=True)

    group_parts = full["PassengerId"].str.split("_", expand=True)
    full["GroupId"] = pd.to_numeric(group_parts[0], errors="coerce")
    full["GroupMember"] = pd.to_numeric(group_parts[1], errors="coerce")
    full["GroupSize"] = full.groupby("GroupId")["PassengerId"].transform("size").astype(int)
    full["Solo"] = (full["GroupSize"] == 1).astype(int)
    full["CabinDeck"], full["CabinNum"], full["CabinSide"] = parse_cabin(full["Cabin"])

    name_parts = full["Name"].fillna("Unknown Unknown").astype(str).str.split(" ", n=1, expand=True)
    full["FirstName"] = name_parts[0].fillna("Unknown")
    full["Surname"] = name_parts[1].fillna("Unknown")
    full["FamilySize"] = full.groupby("Surname")["PassengerId"].transform("size").astype(int)

    spend_total_initial = full[CFG.spend_cols].fillna(0).sum(axis=1)
    full.loc[full["CryoSleep"].isna() & (spend_total_initial > 0), "CryoSleep"] = False
    full.loc[full["CryoSleep"].isna() & (spend_total_initial == 0), "CryoSleep"] = True

    for col in ["HomePlanet", "Destination", "CabinDeck", "CabinSide", "Surname"]:
        fill_from_group_mode(full, "GroupId", col)

    deck_home = full.groupby("CabinDeck")["HomePlanet"].agg(mode_or_nan)
    full["HomePlanet"] = full["HomePlanet"].fillna(full["CabinDeck"].map(deck_home))
    full["HomePlanet"] = full["HomePlanet"].fillna(mode_or_nan(full["HomePlanet"]))

    home_dest = full.groupby("HomePlanet")["Destination"].agg(mode_or_nan)
    full["Destination"] = full["Destination"].fillna(full["HomePlanet"].map(home_dest))
    full["Destination"] = full["Destination"].fillna(mode_or_nan(full["Destination"]))

    home_deck = full.groupby("HomePlanet")["CabinDeck"].agg(mode_or_nan)
    full["CabinDeck"] = full["CabinDeck"].fillna(full["HomePlanet"].map(home_deck))
    full["CabinDeck"] = full["CabinDeck"].fillna("U")
    full["CabinSide"] = full["CabinSide"].fillna(mode_or_nan(full["CabinSide"]))

    full["CabinNum"] = full["CabinNum"].fillna(full.groupby("GroupId")["CabinNum"].transform("median"))
    full["CabinNum"] = full["CabinNum"].fillna(full["CabinNum"].median())

    full["Age"] = full["Age"].fillna(full.groupby("GroupId")["Age"].transform("median"))
    full["Age"] = full["Age"].fillna(full.groupby("HomePlanet")["Age"].transform("median"))
    full["Age"] = full["Age"].fillna(full["Age"].median())
    full["VIP"] = full["VIP"].fillna(False)

    for col in CFG.spend_cols:
        full.loc[full["CryoSleep"] == True, col] = full.loc[full["CryoSleep"] == True, col].fillna(0.0)
        home_median = full.groupby("HomePlanet")[col].transform("median")
        full[col] = full[col].fillna(home_median)
        full[col] = full[col].fillna(full[col].median())
        full.loc[full["CryoSleep"] == True, col] = 0.0

    full["TotalSpend"] = full[CFG.spend_cols].sum(axis=1)
    full["SpendPositiveCount"] = (full[CFG.spend_cols] > 0).sum(axis=1).astype(int)
    full["NoSpend"] = (full["TotalSpend"] == 0).astype(int)
    full["AvgSpendPerService"] = full["TotalSpend"] / np.maximum(full["SpendPositiveCount"], 1)
    full["SpendPerGroupMember"] = full["TotalSpend"] / np.maximum(full["GroupSize"], 1)

    for col in CFG.spend_cols + ["TotalSpend", "AvgSpendPerService", "SpendPerGroupMember"]:
        full[f"Log_{col}"] = np.log1p(full[col])

    full["CryoFlag"] = full["CryoSleep"].astype(int)
    full["VipFlag"] = full["VIP"].astype(int)
    full["IsChild"] = (full["Age"] < 13).astype(int)
    full["IsTeen"] = ((full["Age"] >= 13) & (full["Age"] < 18)).astype(int)
    full["IsSenior"] = (full["Age"] >= 60).astype(int)
    full["AgeSpendInteraction"] = full["Age"] * full["Log_TotalSpend"]

    full["AgeBand"] = pd.cut(
        full["Age"],
        bins=[-1, 12, 18, 25, 40, 60, 120],
        labels=["child", "teen", "young_adult", "adult", "midlife", "senior"],
    ).astype(str)
    full["CabinZone"] = pd.qcut(full["CabinNum"], q=6, duplicates="drop").astype(str)
    full["HomeDest"] = full["HomePlanet"].astype(str) + "_" + full["Destination"].astype(str)
    full["DeckSide"] = full["CabinDeck"].astype(str) + "_" + full["CabinSide"].astype(str)
    full["CryoSleep"] = full["CryoSleep"].map({True: "True", False: "False"}).fillna("False")
    full["VIP"] = full["VIP"].map({True: "True", False: "False"}).fillna("False")

    train_out = full[full["_is_train"] == 1].drop(columns=["_is_train"]).reset_index(drop=True)
    test_out = full[full["_is_train"] == 0].drop(columns=["_is_train"]).reset_index(drop=True)
    return train_out, test_out.drop(columns=[CFG.target])


def encode_ordinal(
    train_x: pd.DataFrame, test_x: pd.DataFrame, categorical_cols: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    enc = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
    )
    train_out = train_x.copy()
    test_out = test_x.copy()
    train_cat = train_x[categorical_cols].fillna("__MISSING__").astype(str)
    test_cat = test_x[categorical_cols].fillna("__MISSING__").astype(str)
    enc.fit(pd.concat([train_cat, test_cat], ignore_index=True))
    train_out[categorical_cols] = enc.transform(train_cat)
    test_out[categorical_cols] = enc.transform(test_cat)
    return train_out.astype(float), test_out.astype(float)


def optimize_threshold(y_true: pd.Series | np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0
    for threshold in np.linspace(0.35, 0.65, 121):
        score = accuracy_score(y_true, probs >= threshold)
        if score > best_score:
            best_score = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_score


def apply_test_group_rules(test_frame: pd.DataFrame, probs: np.ndarray, threshold: float) -> np.ndarray:
    adjusted = probs.copy()
    cryo_mask = (test_frame["CryoFlag"].to_numpy() == 1) & (test_frame["NoSpend"].to_numpy() == 1)
    cryo_uncertain = cryo_mask & (adjusted > threshold - 0.10) & (adjusted < threshold + 0.08)
    adjusted[cryo_uncertain] = np.maximum(adjusted[cryo_uncertain], threshold + 0.06)

    group_ids = test_frame["GroupId"].to_numpy()
    for group_id in np.unique(group_ids):
        member_idx = np.where(group_ids == group_id)[0]
        if len(member_idx) <= 1:
            continue
        group_probs = adjusted[member_idx]
        confident = (group_probs <= threshold - 0.18) | (group_probs >= threshold + 0.18)
        if not confident.any():
            continue
        majority = int((group_probs[confident] >= threshold).mean() >= 0.5)
        uncertain_idx = member_idx[~confident]
        if len(uncertain_idx) == 0:
            continue
        adjusted[uncertain_idx] = (threshold + 0.12) if majority else (threshold - 0.12)
    return np.clip(adjusted, 0.0, 1.0)


def train_public_lr_style_model(
    train_frame: pd.DataFrame, test_frame: pd.DataFrame, y_true: pd.Series, fast: bool = False
) -> dict[str, Any]:
    x_train_raw = train_frame[CFG.feature_cols].copy()
    x_test_raw = test_frame[CFG.feature_cols].copy()
    x_train_num, x_test_num = encode_ordinal(x_train_raw, x_test_raw, CFG.categorical_cols)
    cat_indices = [x_train_raw.columns.get_loc(col) for col in CFG.categorical_cols]

    model_names = ["extra_trees", "hist_gb"]
    if XGB_AVAILABLE:
        model_names.append("xgb")
    if LGB_AVAILABLE:
        model_names.append("lgb")
    if CAT_AVAILABLE:
        model_names.append("cat")

    seeds = (42,) if fast else CFG.seeds
    n_splits = 3 if fast else CFG.n_splits
    oof_store = {name: np.zeros(len(y_true), dtype=float) for name in model_names}
    count_store = {name: np.zeros(len(y_true), dtype=float) for name in model_names}
    test_store: dict[str, list[np.ndarray]] = {name: [] for name in model_names}
    fold_rows = []

    for seed in seeds:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_train_num, y_true), start=1):
            x_tr_num = x_train_num.iloc[train_idx]
            x_val_num = x_train_num.iloc[valid_idx]
            y_tr = y_true.iloc[train_idx]
            y_val = y_true.iloc[valid_idx]

            x_tr_cat = x_train_raw.iloc[train_idx].copy()
            x_val_cat = x_train_raw.iloc[valid_idx].copy()
            x_test_cat = x_test_raw.copy()
            for col in CFG.categorical_cols:
                x_tr_cat[col] = x_tr_cat[col].astype(str)
                x_val_cat[col] = x_val_cat[col].astype(str)
                x_test_cat[col] = x_test_cat[col].astype(str)

            et_model = ExtraTreesClassifier(
                n_estimators=250 if fast else 500,
                min_samples_leaf=2,
                random_state=seed * 10 + fold_idx,
                n_jobs=4,
            )
            et_model.fit(x_tr_num, y_tr)
            add_model_predictions("extra_trees", et_model, x_val_num, x_test_num, valid_idx, y_val, oof_store, count_store, test_store, fold_rows, seed, fold_idx)

            hgb_model = HistGradientBoostingClassifier(
                max_depth=6,
                learning_rate=0.04,
                max_iter=180 if fast else 350,
                random_state=seed * 10 + fold_idx,
            )
            hgb_model.fit(x_tr_num, y_tr)
            add_model_predictions("hist_gb", hgb_model, x_val_num, x_test_num, valid_idx, y_val, oof_store, count_store, test_store, fold_rows, seed, fold_idx)

            if XGB_AVAILABLE:
                xgb_model = xgb.XGBClassifier(
                    n_estimators=180 if fast else 350,
                    max_depth=5,
                    learning_rate=0.03,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    min_child_weight=3,
                    reg_alpha=0.05,
                    reg_lambda=1.0,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    random_state=seed * 10 + fold_idx,
                    n_jobs=4,
                )
                xgb_model.fit(x_tr_num, y_tr)
                add_model_predictions("xgb", xgb_model, x_val_num, x_test_num, valid_idx, y_val, oof_store, count_store, test_store, fold_rows, seed, fold_idx)

            if LGB_AVAILABLE:
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=220 if fast else 450,
                    learning_rate=0.03,
                    num_leaves=31,
                    subsample=0.85,
                    colsample_bytree=0.80,
                    min_child_samples=18,
                    random_state=seed * 10 + fold_idx,
                    verbosity=-1,
                )
                lgb_model.fit(x_tr_num, y_tr)
                add_model_predictions("lgb", lgb_model, x_val_num, x_test_num, valid_idx, y_val, oof_store, count_store, test_store, fold_rows, seed, fold_idx)

            if CAT_AVAILABLE:
                cat_model = CatBoostClassifier(
                    iterations=220 if fast else 400,
                    depth=6,
                    learning_rate=0.03,
                    l2_leaf_reg=4.0,
                    loss_function="Logloss",
                    random_seed=seed * 10 + fold_idx,
                    verbose=False,
                    allow_writing_files=False,
                )
                cat_model.fit(x_tr_cat, y_tr, cat_features=cat_indices, verbose=False)
                add_model_predictions("cat", cat_model, x_val_cat, x_test_cat, valid_idx, y_val, oof_store, count_store, test_store, fold_rows, seed, fold_idx)

    for name in model_names:
        oof_store[name] = oof_store[name] / np.maximum(count_store[name], 1.0)

    oof_matrix = np.column_stack([oof_store[name] for name in model_names])
    test_matrix = np.column_stack([np.mean(test_store[name], axis=0) for name in model_names])

    meta = LogisticRegression(C=1.0, max_iter=2000)
    meta.fit(oof_matrix, y_true)
    stack_oof = meta.predict_proba(oof_matrix)[:, 1]
    stack_test = meta.predict_proba(test_matrix)[:, 1]

    simple_oof = oof_matrix.mean(axis=1)
    simple_test = test_matrix.mean(axis=1)

    best_weight = 0.5
    best_threshold = 0.5
    best_cv = -1.0
    best_oof = simple_oof
    best_test = simple_test
    for weight in np.linspace(0.2, 0.8, 13):
        candidate_oof = weight * stack_oof + (1.0 - weight) * simple_oof
        threshold, score = optimize_threshold(y_true, candidate_oof)
        if score > best_cv:
            best_cv = score
            best_weight = float(weight)
            best_threshold = float(threshold)
            best_oof = candidate_oof
            best_test = weight * stack_test + (1.0 - weight) * simple_test

    adjusted_test = apply_test_group_rules(test_frame, best_test, best_threshold)
    return {
        "model_names": model_names,
        "oof_probs": best_oof,
        "test_probs_no_rules": best_test,
        "test_probs_rules": adjusted_test,
        "threshold": best_threshold,
        "cv_accuracy": best_cv,
        "stack_weight": best_weight,
        "fold_scores": pd.DataFrame(fold_rows),
        "model_oof": oof_store,
        "model_test": {name: np.mean(test_store[name], axis=0) for name in model_names},
    }


def add_model_predictions(
    name: str,
    model: Any,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    valid_idx: np.ndarray,
    y_val: pd.Series,
    oof_store: dict[str, np.ndarray],
    count_store: dict[str, np.ndarray],
    test_store: dict[str, list[np.ndarray]],
    fold_rows: list[dict[str, Any]],
    seed: int,
    fold_idx: int,
) -> None:
    val_prob = model.predict_proba(x_val)[:, 1]
    test_prob = model.predict_proba(x_test)[:, 1]
    oof_store[name][valid_idx] += val_prob
    count_store[name][valid_idx] += 1.0
    test_store[name].append(test_prob)
    fold_rows.append(
        {
            "seed": seed,
            "fold": fold_idx,
            "model": name,
            "acc": float(accuracy_score(y_val, val_prob >= 0.5)),
            "auc": float(roc_auc_score(y_val, val_prob)),
            "logloss": float(log_loss(y_val, np.clip(val_prob, 1e-5, 1 - 1e-5))),
        }
    )


def weighted_vote(preds: list[np.ndarray], weights: list[float], threshold: float = 0.5) -> np.ndarray:
    matrix = np.column_stack([np.asarray(pred, dtype=bool) for pred in preds]).astype(float)
    weight_arr = np.asarray(weights, dtype=float)
    weight_arr = weight_arr / weight_arr.sum()
    return (matrix @ weight_arr) > threshold


def majority_vote(preds: list[np.ndarray], need: int | None = None) -> np.ndarray:
    matrix = np.column_stack([np.asarray(pred, dtype=bool) for pred in preds]).astype(int)
    if need is None:
        need = (matrix.shape[1] // 2) + 1
    return matrix.sum(axis=1) >= need


def confidence_overlay(
    base: np.ndarray,
    model_prob: np.ndarray,
    model_threshold: float,
    min_margin: float,
    require_agree: np.ndarray | None = None,
) -> np.ndarray:
    model_pred = model_prob >= model_threshold
    confident = np.abs(model_prob - model_threshold) >= min_margin
    if require_agree is not None:
        confident &= model_pred == require_agree
    out = base.copy()
    out[confident] = model_pred[confident]
    return out


def save_submission(run_dir: Path, test_ids: np.ndarray, name: str, pred: np.ndarray) -> Path:
    path = run_dir / f"submission_v110_{name}.csv"
    pd.DataFrame({"PassengerId": test_ids, "Transported": np.asarray(pred, dtype=bool)}).to_csv(path, index=False)
    return path


def load_submission(path: Path, test_ids: np.ndarray) -> np.ndarray:
    assert_no_hardcode_risk(path)
    validation = validate_submission(path, test_ids)
    if not validation["valid"]:
        raise ValueError(f"Invalid source submission {path}: {validation.get('reason')}")
    df = pd.read_csv(path)
    return parse_bool_series(df["Transported"]).to_numpy(dtype=bool)


def clean_public_sources(paths: dict[str, Path], test_ids: np.ndarray) -> dict[str, dict[str, Any]]:
    public = paths["output_root"] / "public_notebook_outputs"
    sources = {
        "jimliu_081669": {"path": public / "jimliu_081669" / "submission.csv", "score": 0.81669},
        "ravi20076_v2": {"path": public / "ravi20076_submission_v2" / "submission.csv", "score": 0.81669},
        "jimmyyeung_xgb_top5": {"path": public / "jimmyyeung_xgb_top5" / "submission.csv", "score": 0.81505},
        "guanlintao_xgb": {"path": public / "guanlintao_0814_xgb" / "Submission_XGB.csv", "score": 0.81400},
        "v108_core5_need5_ft": {
            "path": paths["output_root"]
            / "v108_jimliu_clean_model_overlays_nohardcode"
            / "submission_v108_pub2_core5_need5_ft.csv",
            "score": 0.81646,
        },
        "v109_cryo_nospend_me": {
            "path": paths["output_root"]
            / "v109_clean_feature_rule_overlays_nohardcode"
            / "submission_v109_jimliu_cryo_nospend_me_nontrap_settrue.csv",
            "score": 0.81599,
        },
    }
    loaded: dict[str, dict[str, Any]] = {}
    for label, meta in sources.items():
        try:
            pred = load_submission(meta["path"], test_ids)
            loaded[label] = {**meta, "pred": pred, "valid": True, "reason": ""}
        except Exception as exc:
            loaded[label] = {**meta, "pred": None, "valid": False, "reason": str(exc)}
    return loaded


def build_candidates(paths: dict[str, Path], test_ids: np.ndarray, results: dict[str, Any]) -> pd.DataFrame:
    run_dir = paths["run"]
    clean_sources = clean_public_sources(paths, test_ids)
    source_rows = []
    for label, meta in clean_sources.items():
        pred = meta["pred"]
        source_rows.append(
            {
                "label": label,
                "path": display_path(meta["path"], paths["root"]),
                "valid": meta["valid"],
                "score": meta["score"],
                "true_rate": float(pred.mean()) if pred is not None else "",
                "reason": meta["reason"],
            }
        )
    pd.DataFrame(source_rows).to_csv(run_dir / "source_audit.csv", index=False, encoding="utf-8-sig")

    threshold = float(results["threshold"])
    raw_no_rules = results["test_probs_no_rules"] >= threshold
    raw_rules = results["test_probs_rules"] >= threshold
    jimliu = clean_sources["jimliu_081669"]["pred"]
    ravi = clean_sources["ravi20076_v2"]["pred"]
    v108 = clean_sources["v108_core5_need5_ft"]["pred"]
    v109 = clean_sources["v109_cryo_nospend_me"]["pred"]
    jimmy = clean_sources["jimmyyeung_xgb_top5"]["pred"]
    guanlin = clean_sources["guanlintao_xgb"]["pred"]

    candidate_specs: list[tuple[str, np.ndarray, str, list[str]]] = []
    candidate_specs.append(("raw_model_oofbest_rules", raw_rules, "clean reproduction of public LR-style model, no override", ["v110_raw_model_rules"]))
    candidate_specs.append(("raw_model_oofbest_no_rules", raw_no_rules, "same model without global test group rules", ["v110_raw_model_no_rules"]))
    candidate_specs.append(("raw_model_t050_rules", results["test_probs_rules"] >= 0.50, "same model with fixed 0.50 threshold", ["v110_raw_model_rules"]))

    for delta in [-0.010, -0.005, 0.005, 0.010]:
        candidate_specs.append(
            (
                f"raw_model_thr_{threshold + delta:.3f}".replace(".", "p").replace("-", "m"),
                results["test_probs_rules"] >= (threshold + delta),
                f"same model with OOF threshold shifted by {delta:+.3f}",
                ["v110_raw_model_rules"],
            )
        )

    candidate_specs.extend(
        [
            (
                "maj_jimliu_ravi_raw",
                majority_vote([jimliu, ravi, raw_rules]),
                "three complete-model majority: JimLiu, Ravi, v110 raw",
                ["jimliu_081669", "ravi20076_v2", "v110_raw_model_rules"],
            ),
            (
                "maj_jimliu_v108_raw",
                majority_vote([jimliu, v108, raw_rules]),
                "three complete-model majority: JimLiu, v108, v110 raw",
                ["jimliu_081669", "v108_core5_need5_ft", "v110_raw_model_rules"],
            ),
            (
                "w_jimliu_ravi_v108_v109_raw",
                weighted_vote([jimliu, ravi, v108, v109, raw_rules], [0.81669, 0.81669, 0.81646, 0.81599, results["cv_accuracy"]]),
                "weighted complete-model vote around two 0.81669 anchors plus v110 raw",
                ["jimliu_081669", "ravi20076_v2", "v108_core5_need5_ft", "v109_cryo_nospend_me", "v110_raw_model_rules"],
            ),
            (
                "maj6_clean_public_raw_need4",
                majority_vote([jimliu, ravi, v108, v109, jimmy, raw_rules], need=4),
                "six complete-model sources, require four true votes",
                ["jimliu_081669", "ravi20076_v2", "v108_core5_need5_ft", "v109_cryo_nospend_me", "jimmyyeung_xgb_top5", "v110_raw_model_rules"],
            ),
            (
                "maj7_clean_public_raw_need4",
                majority_vote([jimliu, ravi, v108, v109, jimmy, guanlin, raw_rules], need=4),
                "seven complete-model sources, require four true votes",
                ["jimliu_081669", "ravi20076_v2", "v108_core5_need5_ft", "v109_cryo_nospend_me", "jimmyyeung_xgb_top5", "guanlintao_xgb", "v110_raw_model_rules"],
            ),
        ]
    )

    for margin in [0.06, 0.08, 0.10, 0.12]:
        candidate_specs.append(
            (
                f"jimliu_overlay_raw_margin_{margin:.2f}".replace(".", "p"),
                confidence_overlay(jimliu, results["test_probs_rules"], threshold, margin),
                f"JimLiu anchor overwritten only by high-confidence v110 model probabilities, margin {margin:.2f}",
                ["jimliu_081669", "v110_raw_model_rules"],
            )
        )
        candidate_specs.append(
            (
                f"jimliu_overlay_raw_ravi_agree_margin_{margin:.2f}".replace(".", "p"),
                confidence_overlay(jimliu, results["test_probs_rules"], threshold, margin, require_agree=ravi),
                f"JimLiu anchor updated only where high-confidence v110 agrees with Ravi, margin {margin:.2f}",
                ["jimliu_081669", "ravi20076_v2", "v110_raw_model_rules"],
            )
        )

    rows = []
    seen: set[bytes] = set()
    for name, pred, note, source_labels in candidate_specs:
        pred = np.asarray(pred, dtype=bool)
        key = pred.tobytes()
        if key in seen:
            continue
        seen.add(key)
        path = save_submission(run_dir, test_ids, name, pred)
        diff_jimliu = pred != jimliu
        diff_ravi = pred != ravi
        rows.append(
            {
                "file": path.name,
                "note": note,
                "sources": json.dumps(source_labels, ensure_ascii=False),
                "true_rate": float(pred.mean()),
                "true_count": int(pred.sum()),
                "diff_jimliu": int(diff_jimliu.sum()),
                "jimliu_true_to_false": int((jimliu & ~pred).sum()),
                "jimliu_false_to_true": int((~jimliu & pred).sum()),
                "diff_ravi": int(diff_ravi.sum()),
                "ravi_true_to_false": int((ravi & ~pred).sum()),
                "ravi_false_to_true": int((~ravi & pred).sum()),
            }
        )
    summary = pd.DataFrame(rows).sort_values(["diff_jimliu", "diff_ravi", "file"])
    summary.to_csv(run_dir / "submission_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def write_model_reports(paths: dict[str, Path], results: dict[str, Any], elapsed: float) -> None:
    run_dir = paths["run"]
    np.save(run_dir / "v110_oof_probs.npy", results["oof_probs"])
    np.save(run_dir / "v110_test_probs_no_rules.npy", results["test_probs_no_rules"])
    np.save(run_dir / "v110_test_probs_rules.npy", results["test_probs_rules"])
    for name, arr in results["model_oof"].items():
        np.save(run_dir / f"v110_{name}_oof_prob.npy", arr)
    for name, arr in results["model_test"].items():
        np.save(run_dir / f"v110_{name}_test_prob.npy", arr)

    results["fold_scores"].to_csv(run_dir / "fold_scores.csv", index=False, encoding="utf-8-sig")
    model_scores = (
        results["fold_scores"]
        .groupby("model", as_index=False)
        .agg(acc=("acc", "mean"), auc=("auc", "mean"), logloss=("logloss", "mean"))
        .sort_values("acc", ascending=False)
    )
    model_scores.to_csv(run_dir / "model_score_summary.csv", index=False, encoding="utf-8-sig")
    run_summary = {
        "elapsed_seconds": elapsed,
        "models": results["model_names"],
        "cv_accuracy": float(results["cv_accuracy"]),
        "threshold": float(results["threshold"]),
        "stack_weight": float(results["stack_weight"]),
        "xgboost_available": XGB_AVAILABLE,
        "lightgbm_available": LGB_AVAILABLE,
        "catboost_available": CAT_AVAILABLE,
        "compliance": "No embedded public-label override, bitstring, PassengerId rule, single-point probe, or manual flip is used.",
    }
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False), encoding="utf-8")


def write_format_validation(paths: dict[str, Path], test_ids: np.ndarray) -> pd.DataFrame:
    rows = [validate_submission(path, test_ids) for path in sorted(paths["run"].glob("submission_v110_*.csv"))]
    validation = pd.DataFrame(rows)
    validation.to_csv(paths["run"] / "format_validation.csv", index=False, encoding="utf-8-sig")
    return validation


def submit_candidates(paths: dict[str, Path], summary: pd.DataFrame, max_submissions: int) -> None:
    if max_submissions <= 0:
        return
    log_rows = []
    for _, row in summary.head(max_submissions).iterrows():
        file_path = paths["run"] / row["file"]
        message = row["file"].replace("submission_", "").replace(".csv", "")
        cmd = ["kaggle", "competitions", "submit", "-c", CFG.competition, "-f", str(file_path), "-m", message]
        started = time.strftime("%Y-%m-%d %H:%M:%S")
        try:
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
            log_rows.append(
                {
                    "time": started,
                    "file": row["file"],
                    "message": message,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout.strip(),
                    "stderr": proc.stderr.strip(),
                }
            )
        except Exception as exc:
            log_rows.append({"time": started, "file": row["file"], "message": message, "returncode": -1, "stdout": "", "stderr": str(exc)})
    pd.DataFrame(log_rows).to_csv(paths["run"] / "kaggle_submit_attempts.csv", index=False, encoding="utf-8-sig")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="Use fewer folds/trees for a quick smoke run.")
    parser.add_argument("--submit", action="store_true", help="Submit ranked v110 candidates using the currently authenticated Kaggle CLI.")
    parser.add_argument("--max-submissions", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = project_paths()
    start = time.time()
    train_df = pd.read_csv(paths["data"] / "train.csv")
    test_df = pd.read_csv(paths["data"] / "test.csv")
    test_ids = test_df["PassengerId"].astype(str).to_numpy()
    y = train_df[CFG.target].astype(int)

    train_feat, test_feat = engineer_features(train_df, test_df)
    results = train_public_lr_style_model(train_feat, test_feat, y, fast=args.fast)
    write_model_reports(paths, results, time.time() - start)
    summary = build_candidates(paths, test_ids, results)
    validation = write_format_validation(paths, test_ids)

    readme = [
        "# v110 clean public-model reproduction",
        "",
        "This run reproduces the public LR-style ensemble as a pure model prediction and does not apply the embedded best-public override from the public notebook.",
        "Candidate ensembles use only complete prediction arrays from clean model submissions plus the v110 raw model output.",
        "",
        f"OOF CV accuracy: {results['cv_accuracy']:.6f}",
        f"OOF threshold: {results['threshold']:.4f}",
        f"Stack weight: {results['stack_weight']:.3f}",
        "",
        "Top local-ranked candidates by closeness to the 0.81669 JimLiu anchor:",
    ]
    for _, row in summary.head(8).iterrows():
        readme.append(f"- {row['file']}: diff_jimliu={row['diff_jimliu']}, true_rate={row['true_rate']:.6f}, {row['note']}")
    (paths["run"] / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    print("v110 generation complete")
    print("run_dir:", paths["run"])
    print("valid submissions:", int(validation["valid"].sum()), "/", len(validation))
    print(summary.head(12).to_string(index=False))

    if args.submit:
        submit_candidates(paths, summary, args.max_submissions)


if __name__ == "__main__":
    main()
