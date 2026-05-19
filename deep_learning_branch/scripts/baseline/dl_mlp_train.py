import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, n_features, hidden=(256, 128, 64), dropout=0.25):
        super().__init__()
        layers = []
        in_dim = n_features
        for out_dim in hidden:
            layers.extend(
                [
                    nn.Linear(in_dim, out_dim),
                    nn.BatchNorm1d(out_dim),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


def parse_bool_target(s):
    if s.dtype == bool:
        return s.astype(np.float32).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.float32).to_numpy()
    mapped = (
        s.astype(str)
        .str.strip()
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "yes": 1.0, "no": 0.0})
    )
    if mapped.isna().any():
        bad_values = sorted(s[mapped.isna()].astype(str).unique().tolist())[:10]
        raise ValueError(f"Unrecognized target values in Transported: {bad_values}")
    return mapped.astype(np.float32).to_numpy()


def passenger_groups(passenger_ids):
    return pd.Series(passenger_ids).astype(str).str.split("_").str[0].astype(int).to_numpy()


def make_loader(x, y=None, batch_size=256, shuffle=False):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(x_tensor)
    else:
        ds = TensorDataset(x_tensor, torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


@torch.no_grad()
def predict(model, x, device, batch_size=1024):
    model.eval()
    preds = []
    for (xb,) in make_loader(x, batch_size=batch_size):
        xb = xb.to(device)
        preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(x, y, train_idx, valid_idx, x_test, args, seed, fold, device):
    seed_everything(seed + fold * 1000)
    model = MLP(x.shape[1], hidden=tuple(args.hidden), dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = make_loader(x[train_idx], y[train_idx], args.batch_size, shuffle=True)
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]

    best_loss = float("inf")
    best_state = None
    stale = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        valid_prob = predict(model, x_valid, device)
        valid_loss = log_loss(y_valid, np.clip(valid_prob, 1e-6, 1 - 1e-6))
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best_state)
    valid_prob = predict(model, x_valid, device)
    test_prob = predict(model, x_test, device)
    valid_acc = accuracy_score(y_valid, valid_prob >= 0.5)
    return valid_prob, test_prob, {"fold": fold, "best_logloss": best_loss, "acc": valid_acc}


def save_submission(passenger_id, probs, out_path, threshold=0.5):
    sub = pd.DataFrame({"PassengerId": passenger_id, "Transported": probs >= threshold})
    sub.to_csv(out_path, index=False)


def maybe_blend(out_dir, y, test_ids, dl_oof, dl_test):
    gb_oof_path = Path("模型训练部分/v20_gbdt3_oof_prob.npy")
    gb_test_path = Path("模型训练部分/v20_gbdt3_test_prob.npy")
    cb_oof_path = Path("模型训练部分/v20_cb_native_oof_prob.npy")
    cb_test_path = Path("模型训练部分/v20_cb_native_test_prob.npy")
    if not all(p.exists() for p in [gb_oof_path, gb_test_path, cb_oof_path, cb_test_path]):
        return None

    base_oof = 0.75 * np.load(gb_oof_path) + 0.25 * np.load(cb_oof_path)
    base_test = 0.75 * np.load(gb_test_path) + 0.25 * np.load(cb_test_path)
    rows = []
    best = None
    for w_dl in np.arange(0.0, 0.31, 0.01):
        prob = (1.0 - w_dl) * base_oof + w_dl * dl_oof
        acc = accuracy_score(y, prob >= 0.5)
        rows.append({"w_dl": round(float(w_dl), 2), "oof_acc": float(acc)})
        if best is None or acc > best["oof_acc"]:
            best = rows[-1]
    pd.DataFrame(rows).to_csv(out_dir / "blend_weight_search.csv", index=False)

    for w_dl in [0.05, 0.10, 0.15, 0.20, best["w_dl"]]:
        prob_test = (1.0 - w_dl) * base_test + w_dl * dl_test
        save_submission(test_ids, prob_test, out_dir / f"submission_v20_dl_mlp_w{w_dl:.2f}.csv")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="特征工程部分/train_features_mlp_v2.csv")
    parser.add_argument("--test", default="特征工程部分/test_features_mlp_v2.csv")
    parser.add_argument("--out-dir", default="dl_output")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    y = parse_bool_target(train_df["Transported"])
    test_ids = test_df["PassengerId"].copy()
    groups = passenger_groups(train_df["PassengerId"])

    feature_cols = [c for c in train_df.columns if c not in ["PassengerId", "Transported"]]
    x = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    x_test = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()

    all_oof = []
    all_test = []
    fold_reports = []

    for seed in args.seeds:
        oof = np.zeros(len(y), dtype=np.float32)
        test_accum = np.zeros(len(x_test), dtype=np.float32)
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(x, y, groups))
        for fold, (trn_idx, val_idx) in enumerate(splits):
            val_prob, test_prob, report = train_one_fold(x, y, trn_idx, val_idx, x_test, args, seed, fold, device)
            oof[val_idx] = val_prob
            test_accum += test_prob / args.folds
            report["seed"] = seed
            fold_reports.append(report)
            print(f"seed={seed} fold={fold} acc={report['acc']:.5f} logloss={report['best_logloss']:.5f}", flush=True)
        all_oof.append(oof)
        all_test.append(test_accum)
        print(f"seed={seed} oof_acc={accuracy_score(y, oof >= 0.5):.5f}", flush=True)

    dl_oof = np.mean(all_oof, axis=0)
    dl_test = np.mean(all_test, axis=0)
    np.save(out_dir / "dl_mlp_oof.npy", dl_oof)
    np.save(out_dir / "dl_mlp_test.npy", dl_test)
    pd.DataFrame({"PassengerId": train_df["PassengerId"], "Transported": y, "prob": dl_oof}).to_csv(out_dir / "dl_mlp_oof.csv", index=False)
    pd.DataFrame({"PassengerId": test_ids, "prob": dl_test}).to_csv(out_dir / "dl_mlp_test.csv", index=False)
    save_submission(test_ids, dl_test, out_dir / "submission_dl_mlp.csv")

    report = {
        "device": device,
        "n_features": len(feature_cols),
        "seeds": args.seeds,
        "folds": args.folds,
        "dl_oof_acc": float(accuracy_score(y, dl_oof >= 0.5)),
        "dl_oof_logloss": float(log_loss(y, np.clip(dl_oof, 1e-6, 1 - 1e-6))),
        "dl_test_true_rate": float((dl_test >= 0.5).mean()),
        "fold_reports": fold_reports,
    }
    report["best_blend"] = maybe_blend(out_dir, y, test_ids, dl_oof, dl_test)
    (out_dir / "dl_mlp_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
