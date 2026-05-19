"""
dl_v40_train.py — MLP v40 with new features

Changes from v35:
  1. New feature set: dl_features_v3 (101 features from gbdt_optimize build_features)
  2. v34-style architecture: hidden=[512,256,128], dropout=0.25
  3. 10 seeds (restored from v34), 500 epochs
  4. Blend with v20 baseline (cb_native + gbdt3)
  5. SWA + warmup+cosine + label_smooth + grad_clip retained from v35
"""
import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DL_DIR_FOR_IMPORT = Path(__file__).resolve().parents[2]
if str(DL_DIR_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(DL_DIR_FOR_IMPORT))

from scripts.path_utils import default_rerun_dir

FOLD_SEED = 999


class MLP(nn.Module):
    def __init__(self, n_features, hidden=(512, 256, 128), dropout=0.25):
        super().__init__()
        layers = []
        in_dim = n_features
        for out_dim in hidden:
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
            ])
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


class SWA:
    def __init__(self, model):
        self.model = model
        self.swa_model = copy.deepcopy(model)
        self.n_averaged = 0

    def update(self):
        with torch.no_grad():
            for p_swa, p in zip(self.swa_model.parameters(), self.model.parameters()):
                p_swa.data = (p_swa.data * self.n_averaged + p.data) / (self.n_averaged + 1)
        self.n_averaged += 1

    def apply(self):
        self.model.load_state_dict(self.swa_model.state_dict())


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
    mapped = s.astype(str).str.strip().str.lower().map({
        "true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "yes": 1.0, "no": 0.0
    })
    if mapped.isna().any():
        bad = sorted(s[mapped.isna()].astype(str).unique().tolist())[:10]
        raise ValueError(f"Bad target: {bad}")
    return mapped.astype(np.float32).to_numpy()


def passenger_groups(passenger_ids):
    return pd.Series(passenger_ids).astype(str).str.split("_").str[0].astype(int).to_numpy()


def make_loader(x, y=None, batch_size=256, shuffle=False):
    x_t = torch.tensor(x, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(x_t)
    else:
        ds = TensorDataset(x_t, torch.tensor(y, dtype=torch.float32))
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
    swa = SWA(model) if args.swa else None

    x_train = x[train_idx]
    y_train = y[train_idx]
    x_valid = x[valid_idx]
    y_valid = y[valid_idx]

    y_train_smooth = y_train * (1.0 - args.label_smooth) + 0.5 * args.label_smooth
    train_loader = make_loader(x_train, y_train_smooth, args.batch_size, shuffle=True)

    warmup_epochs = int(args.epochs * 0.10)
    total_epochs = args.epochs
    base_lr = args.lr
    best_loss = float("inf")
    best_state = None
    stale = 0
    swa_start = total_epochs // 3

    for epoch in range(1, total_epochs + 1):
        if epoch <= warmup_epochs:
            lr = base_lr * epoch / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            lr = base_lr * 0.5 * (1.0 + np.cos(np.pi * progress))
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        valid_prob = predict(model, x_valid, device)
        valid_loss = log_loss(y_valid, np.clip(valid_prob, 1e-6, 1 - 1e-6))

        if epoch >= swa_start:
            swa.update() if swa else None

        if valid_loss < best_loss:
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if swa and swa.n_averaged > 0:
        swa.apply()
    else:
        model.load_state_dict(best_state)

    valid_prob = predict(model, x_valid, device)
    test_prob = predict(model, x_test, device)
    valid_acc = accuracy_score(y_valid, valid_prob >= 0.5)
    return valid_prob, test_prob, {"fold": fold, "best_logloss": best_loss, "acc": valid_acc}


def save_submission(passenger_id, probs, out_path, threshold=0.5):
    sub = pd.DataFrame({"PassengerId": passenger_id, "Transported": probs >= threshold})
    sub.to_csv(out_path, index=False)


def default_feature_paths():
    return (
        Path("深度学习/features/dl_features_v3/train_features_dl_v3.csv"),
        Path("深度学习/features/dl_features_v3/test_features_dl_v3.csv"),
    )


def blend_with_v20(out_dir, y, train_ids, test_ids, dl_oof, dl_test):
    """Blend DL with v20 baseline, scan weights on OOF."""
    cb_oof = np.load("模型训练部分/v20_cb_native_oof_prob.npy")
    gb_oof = np.load("模型训练部分/v20_gbdt3_oof_prob.npy")
    cb_test = np.load("模型训练部分/v20_cb_native_test_prob.npy")
    gb_test = np.load("模型训练部分/v20_gbdt3_test_prob.npy")

    base_oof = 0.75 * gb_oof + 0.25 * cb_oof
    base_test = 0.75 * gb_test + 0.25 * cb_test

    rows = []
    best = None
    for w_dl in np.arange(0.0, 0.41, 0.01):
        blend_oof = (1.0 - w_dl) * base_oof + w_dl * dl_oof
        blend_test = (1.0 - w_dl) * base_test + w_dl * dl_test
        acc = accuracy_score(y, blend_oof >= 0.5)
        tt = (blend_test >= 0.5).mean()
        rows.append({"w_dl": round(float(w_dl), 2), "oof_acc": float(acc), "test_tt": float(tt)})
        if best is None or acc > best["oof_acc"]:
            best = rows[-1]
            save_submission(test_ids, blend_test, out_dir / f"submission_v20_dl_v40_w{w_dl:.2f}.csv")

    pd.DataFrame(rows).to_csv(out_dir / "blend_weight_search.csv", index=False)
    print(f"  Best blend: w_dl={best['w_dl']:.2f}  OOF={best['oof_acc']:.5f}  testTrue%={best['test_tt']:.4f}")
    return best


def main():
    parser = argparse.ArgumentParser()
    default_train, default_test = default_feature_paths()
    parser.add_argument("--train", default=str(default_train))
    parser.add_argument("--test", default=str(default_test))
    parser.add_argument("--out-dir", default=str(default_rerun_dir("dl_output_v40")))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 2026, 3407, 123, 777, 999, 2023, 88, 456])
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smooth", type=float, default=0.03)
    parser.add_argument("--swa", action="store_true", default=True)
    parser.add_argument("--no-swa", dest="swa", action="store_false")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: MLP{tuple(args.hidden)} dropout={args.dropout} seeds={args.seeds}")
    print(f"Features: {args.train}")

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
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=FOLD_SEED)
        splits = list(splitter.split(x, y, groups))
        for fold, (trn_idx, val_idx) in enumerate(splits):
            val_prob, test_prob, report = train_one_fold(
                x, y, trn_idx, val_idx, x_test, args, seed, fold, device
            )
            oof[val_idx] = val_prob
            test_accum += test_prob / args.folds
            report["seed"] = seed
            fold_reports.append(report)
            print(f"seed={seed} fold={fold} acc={report['acc']:.5f} logloss={report['best_logloss']:.5f}", flush=True)
        all_oof.append(oof)
        all_test.append(test_accum)
        seed_acc = accuracy_score(y, oof >= 0.5)
        print(f"seed={seed} oof_acc={seed_acc:.5f}", flush=True)

    dl_oof = np.mean(all_oof, axis=0)
    dl_test = np.mean(all_test, axis=0)
    np.save(out_dir / "dl_v40_oof.npy", dl_oof)
    np.save(out_dir / "dl_v40_test.npy", dl_test)
    pd.DataFrame({"PassengerId": train_df["PassengerId"], "Transported": y, "prob": dl_oof}).to_csv(
        out_dir / "dl_v40_oof.csv", index=False
    )
    pd.DataFrame({"PassengerId": test_ids, "prob": dl_test}).to_csv(
        out_dir / "dl_v40_test.csv", index=False
    )
    save_submission(test_ids, dl_test, out_dir / "submission_dl_v40.csv")

    dl_oof_acc = float(accuracy_score(y, dl_oof >= 0.5))
    dl_oof_ll = float(log_loss(y, np.clip(dl_oof, 1e-6, 1 - 1e-6)))
    dl_test_tt = float((dl_test >= 0.5).mean())

    report = {
        "version": "v40",
        "device": device,
        "n_features": len(feature_cols),
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "hidden": args.hidden,
        "dropout": args.dropout,
        "weight_decay": args.weight_decay,
        "fold_seed": FOLD_SEED,
        "folds": args.folds,
        "epochs": args.epochs,
        "dl_oof_acc": dl_oof_acc,
        "dl_oof_logloss": dl_oof_ll,
        "dl_test_true_rate": dl_test_tt,
        "fold_reports": fold_reports,
    }
    print(f"\nDL v40 OOF acc={dl_oof_acc:.5f}  logloss={dl_oof_ll:.5f}  testTrue%={dl_test_tt:.4f}")
    print("\nBlending with v20 baseline (OOF scanning):")
    blend_with_v20(out_dir, y, None, test_ids, dl_oof, dl_test)

    (out_dir / "dl_v40_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nReport saved to {out_dir}/dl_v40_report.json")


if __name__ == "__main__":
    main()
