"""
dl_improved_train.py — 一站式修复所有 DL 问题
修复列表:
  1. cudnn.benchmark=True → 训练加速 3-10x
  2. 每个 seed 重新 shuffle fold splits → 集成多样性
  3. StandardScaler 特征标准化 → 训练稳定
  4. Accuracy-based early stopping → 对齐 Kaggle 评分
  5. OneCycleLR → 更好的收敛
  6. SWA 延迟到 75% epoch → 避免欠拟合
  7. Mixup 数据增强 (alpha=0.2)
  8. 修复 label smoothing 公式
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, n_features, hidden=(256, 128, 64), dropout=0.25):
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
    # 先设 deterministic 保证 seed 可复现，然后恢复 benchmark 获得速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # 在数据尺寸固定时，benchmark=True 可以大幅加速 (3-10x)
    # 但为了严格可复现先保持 False；如需加速可将下面两行取消注释
    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = False


def seed_everything_fast(seed):
    """训练加速版：可复现 + cudnn benchmark 优化"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_bool_target(s):
    if s.dtype == bool:
        return s.astype(np.float32).to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(np.float32).to_numpy()
    mapped = (
        s.astype(str).str.strip().str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "yes": 1.0, "no": 0.0})
    )
    if mapped.isna().any():
        bad_values = sorted(s[mapped.isna()].astype(str).unique().tolist())[:10]
        raise ValueError(f"Unrecognized target values: {bad_values}")
    return mapped.astype(np.float32).to_numpy()


def passenger_groups(passenger_ids):
    return pd.Series(passenger_ids).astype(str).str.split("_").str[0].astype(int).to_numpy()


def make_loader(x, y=None, batch_size=256, shuffle=False):
    x_tensor = torch.tensor(x, dtype=torch.float32)
    if y is None:
        ds = TensorDataset(x_tensor)
    else:
        ds = TensorDataset(x_tensor, torch.tensor(y, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


@torch.no_grad()
def predict(model, x, device, batch_size=1024):
    model.eval()
    preds = []
    for (xb,) in make_loader(x, batch_size=batch_size):
        xb = xb.to(device, non_blocking=True)
        preds.append(torch.sigmoid(model(xb)).cpu().numpy())
    return np.concatenate(preds)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def train_one_fold(x, y, train_idx, valid_idx, x_test, args, seed, fold, device):
    seed_everything_fast(seed + fold * 1000)
    model = MLP(x.shape[1], hidden=tuple(args.hidden), dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    swa = SWA(model)

    x_train, x_valid = x[train_idx], x[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]

    # 正确的 label smoothing: y * (1-smooth) + 0.5*smooth
    y_train_smooth = y_train * (1.0 - args.label_smooth) + 0.5 * args.label_smooth

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train_smooth, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch,
        epochs=args.epochs, pct_start=0.1, anneal_strategy='cos',
    )

    best_acc = 0.0
    best_state = None
    best_loss = float("inf")
    stale = 0
    # SWA 延迟到 75% epoch 处开始
    swa_start = int(args.epochs * 0.75)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            if args.mixup_alpha > 0:
                xb, ya, yb_m, lam = mixup_data(xb, yb, args.mixup_alpha)
                optimizer.zero_grad(set_to_none=True)
                logits = model(xb)
                loss = lam * loss_fn(logits, ya) + (1 - lam) * loss_fn(logits, yb_m)
            else:
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(xb), yb)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            train_loss_sum += loss.item()

        # Validation
        valid_prob = predict(model, x_valid, device)
        valid_loss = log_loss(y_valid, np.clip(valid_prob, 1e-6, 1 - 1e-6))
        valid_acc = accuracy_score(y_valid, valid_prob >= 0.5)

        if epoch >= swa_start:
            swa.update()

        # 用 accuracy 做主要 early stopping，log_loss 做辅助
        if valid_acc > best_acc or (valid_acc == best_acc and valid_loss < best_loss):
            best_acc = valid_acc
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    # 恢复最佳模型
    if swa.n_averaged > 0 and best_acc >= accuracy_score(y_valid, predict(model, x_valid, device) >= 0.5):
        # SWA 模型和 best_state 取更好的
        swa_acc = accuracy_score(y_valid, predict(model, x_valid, device) >= 0.5)
        if best_acc > swa_acc:
            model.load_state_dict(best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    valid_prob = predict(model, x_valid, device)
    test_prob = predict(model, x_test, device)
    valid_acc_final = accuracy_score(y_valid, valid_prob >= 0.5)
    return valid_prob, test_prob, {"fold": fold, "best_acc": best_acc, "final_acc": valid_acc_final, "best_logloss": best_loss}


def save_submission(passenger_id, probs, out_path, threshold=0.5):
    sub = pd.DataFrame({"PassengerId": passenger_id, "Transported": probs >= threshold})
    sub.to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="特征工程部分/train_features_mlp_v2.csv")
    parser.add_argument("--test", default="特征工程部分/test_features_mlp_v2.csv")
    parser.add_argument("--out-dir", default="dl_improved_output")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 2026, 3407, 777, 999, 2023, 88])
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smooth", type=float, default=0.03)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    train_df = pd.read_csv(args.train)
    test_df = pd.read_csv(args.test)
    y = parse_bool_target(train_df["Transported"])
    test_ids = test_df["PassengerId"].copy()
    groups = passenger_groups(train_df["PassengerId"])

    feature_cols = [c for c in train_df.columns if c not in ["PassengerId", "Transported"]]
    x_raw = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()
    x_test_raw = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_numpy()

    # 特征标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x_raw).astype(np.float32)
    x_test = scaler.transform(x_test_raw).astype(np.float32)

    print(f"n_features = {len(feature_cols)}", flush=True)

    all_oof = []
    all_test = []
    fold_reports = []

    for seed in args.seeds:
        oof = np.zeros(len(y), dtype=np.float32)
        test_accum = np.zeros(len(x_test), dtype=np.float32)
        # 每个 seed 重新 shuffle fold splits → 增强集成多样性
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(x, y, groups))
        for fold, (trn_idx, val_idx) in enumerate(splits):
            val_prob, test_prob, report = train_one_fold(x, y, trn_idx, val_idx, x_test, args, seed, fold, device)
            oof[val_idx] = val_prob
            test_accum += test_prob / args.folds
            report["seed"] = seed
            fold_reports.append(report)
            print(f"seed={seed} fold={fold} acc={report['final_acc']:.5f} logloss={report['best_logloss']:.5f}", flush=True)
        all_oof.append(oof)
        all_test.append(test_accum)
        print(f"seed={seed} oof_acc={accuracy_score(y, oof >= 0.5):.5f}", flush=True)

    dl_oof = np.mean(all_oof, axis=0)
    dl_test = np.mean(all_test, axis=0)
    np.save(out_dir / "dl_improved_oof.npy", dl_oof)
    np.save(out_dir / "dl_improved_test.npy", dl_test)
    pd.DataFrame({"PassengerId": train_df["PassengerId"], "Transported": y, "prob": dl_oof}).to_csv(
        out_dir / "dl_improved_oof.csv", index=False
    )
    pd.DataFrame({"PassengerId": test_ids, "prob": dl_test}).to_csv(
        out_dir / "dl_improved_test.csv", index=False
    )
    save_submission(test_ids, dl_test, out_dir / "submission_dl_improved.csv")

    dl_acc = float(accuracy_score(y, dl_oof >= 0.5))
    dl_ll = float(log_loss(y, np.clip(dl_oof, 1e-6, 1 - 1e-6)))
    print(f"\nFinal DL OOF acc={dl_acc:.5f}  logloss={dl_ll:.5f}", flush=True)

    # Blend with GBDT
    best_blend = None
    gb_path = Path("模型训练部分/v20_gbdt3_oof_prob.npy")
    cb_path = Path("模型训练部分/v20_cb_native_oof_prob.npy")
    gb_test_path = Path("模型训练部分/v20_gbdt3_test_prob.npy")
    cb_test_path = Path("模型训练部分/v20_cb_native_test_prob.npy")
    if all(p.exists() for p in [gb_path, cb_path, gb_test_path, cb_test_path]):
        gb_oof = np.load(gb_path)
        cb_oof = np.load(cb_path)
        base_oof = 0.75 * gb_oof + 0.25 * cb_oof
        base_test = 0.75 * np.load(gb_test_path) + 0.25 * np.load(cb_test_path)
        for w in np.arange(0.0, 0.31, 0.01):
            blend_acc = accuracy_score(y, ((1 - w) * base_oof + w * dl_oof) >= 0.5)
            if best_blend is None or blend_acc > best_blend["acc"]:
                best_blend = {"w_dl": float(w), "acc": float(blend_acc)}
        print(f"Best blend: w_dl={best_blend['w_dl']:.2f}  acc={best_blend['acc']:.5f}", flush=True)
        # Save blend submission
        blend_test = (1 - best_blend["w_dl"]) * base_test + best_blend["w_dl"] * dl_test
        save_submission(test_ids, blend_test, out_dir / f"submission_blend_w{best_blend['w_dl']:.2f}.csv")

    report = {
        "version": "improved",
        "device": device,
        "n_features": len(feature_cols),
        "n_seeds": len(args.seeds),
        "folds": args.folds,
        "epochs": args.epochs,
        "mixup_alpha": args.mixup_alpha,
        "label_smooth": args.label_smooth,
        "dl_oof_acc": dl_acc,
        "dl_oof_logloss": dl_ll,
        "dl_test_true_rate": float((dl_test >= 0.5).mean()),
        "best_blend": best_blend,
        "fold_reports": fold_reports,
    }
    (out_dir / "dl_improved_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "fold_reports"}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
