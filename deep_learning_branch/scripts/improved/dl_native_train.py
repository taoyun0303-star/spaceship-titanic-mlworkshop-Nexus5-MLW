"""
dl_native_train.py — DL 专用：从预处理原始数据构建 Embedding+MLP
不使用任何 ML 特征工程（TE/WoE/one-hot/人工交互）
类别特征 → Embedding → 和数值特征拼接 → MLP
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
from sklearn.preprocessing import QuantileTransformer
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ══════════════════════════════════════════
# 模型
# ══════════════════════════════════════════

class EmbeddingMLP(nn.Module):
    """类别 Embedding + 数值特征 → MLP"""

    def __init__(self, cat_cardinalities, n_num, emb_dim=16, hidden=(256, 128, 64), dropout=0.25):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n + 1, emb_dim, padding_idx=0) for n in cat_cardinalities
        ])
        self.n_cat = len(cat_cardinalities)
        n_in = self.n_cat * emb_dim + n_num

        layers = []
        in_dim = n_in
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

    def forward(self, x_num, x_cat):
        embs = []
        for i, emb in enumerate(self.embeddings):
            embs.append(emb(x_cat[:, i]))
        x_cat_emb = torch.cat(embs, dim=1) if embs else torch.empty(x_num.size(0), 0, device=x_num.device)
        x = torch.cat([x_num, x_cat_emb], dim=1)
        return self.net(x).squeeze(1)


class SWA:
    def __init__(self, model):
        self.model = model
        self.swa_model = copy.deepcopy(model)
        self.n_averaged = 0

    def update(self):
        with torch.no_grad():
            for ps, p in zip(self.swa_model.parameters(), self.model.parameters()):
                ps.data = (ps.data * self.n_averaged + p.data) / (self.n_averaged + 1)
        self.n_averaged += 1

    def apply(self):
        self.model.load_state_dict(self.swa_model.state_dict())


# ══════════════════════════════════════════
# 数据加载
# ══════════════════════════════════════════

# 类别特征定义: (列名, 编码映射)
CAT_FEATURES = {
    "HomePlanet": {"Europa": 1, "Earth": 2, "Mars": 3},
    "CryoSleep": {True: 1, False: 2, "True": 1, "False": 2},
    "Destination": {"TRAPPIST-1e": 1, "PSO J318.5-22": 2, "55 Cancri e": 3},
    "VIP": {True: 1, False: 2, "True": 1, "False": 2},
    "Deck": {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8},
    "Side": {"P": 1, "S": 2},
}

# 数值特征 (从预处理数据中选取或派生)
NUM_FEATURES_BASE = [
    "Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck",
    "GroupSize", "CabinNum",
]


def encode_categorical(df, cat_spec):
    """将类别列编码为 int，未知/缺失 → 0"""
    encoded = np.zeros((len(df), len(cat_spec)), dtype=np.int64)
    for i, (col, mapping) in enumerate(cat_spec.items()):
        series = df[col].astype(str).str.strip()
        encoded[:, i] = series.map(mapping).fillna(0).astype(np.int64).values
    cat_cardinalities = [len(m) for m in cat_spec.values()]
    return encoded, cat_cardinalities


def build_numerical(df):
    """从预处理数据构建数值特征矩阵 (含派生特征)"""
    feats = []
    names = []

    for col in NUM_FEATURES_BASE:
        if col in df.columns:
            vals = df[col].fillna(0).astype(np.float32).values
            feats.append(vals)
            names.append(col)

    # 派生: log transform
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    for col in spend_cols:
        if col in df.columns:
            vals = df[col].fillna(0).astype(np.float32).values
            log_vals = np.log1p(np.clip(vals, 0, None))
            feats.append(log_vals)
            names.append(f"Log_{col}")

    # 派生: TotalSpend, LuxurySpend, BasicSpend
    if all(c in df.columns for c in spend_cols):
        s = np.stack([df[c].fillna(0).astype(np.float32).values for c in spend_cols], axis=1)
        total = s.sum(axis=1)
        luxury = s[:, 3] + s[:, 4]  # Spa + VRDeck
        basic = s[:, 0] + s[:, 1] + s[:, 2]  # RoomService + FoodCourt + ShoppingMall
        feats.extend([total, np.log1p(np.clip(total, 0, None))])
        names.extend(["TotalSpend", "Log_TotalSpend"])
        feats.extend([luxury, np.log1p(np.clip(luxury, 0, None))])
        names.extend(["LuxurySpend", "Log_LuxurySpend"])
        feats.extend([basic, np.log1p(np.clip(basic, 0, None))])
        names.extend(["BasicSpend", "Log_BasicSpend"])

    # Spend per age
    age = df["Age"].fillna(df["Age"].median()).astype(np.float32).values
    age_safe = np.clip(age, 1, None)
    feats.append(np.log1p(np.clip(total, 0, None)) / age_safe)
    names.append("LogSpendPerAge")

    # Deck × Side interaction (简单的数值交叉)
    if "CabinNum" in df.columns and "Deck" in df.columns:
        deck_map = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
        deck_num = df["Deck"].astype(str).str.strip().map(deck_map).fillna(0).astype(np.float32).values
        cabin = df["CabinNum"].fillna(0).astype(np.float32).values
        feats.append(deck_num * cabin)
        names.append("Deck_CabinNum")

    x = np.stack(feats, axis=1).astype(np.float32)
    return x, names


def passenger_groups(passenger_ids):
    return pd.Series(passenger_ids).astype(str).str.split("_").str[0].astype(int).to_numpy()


class TwoStreamDataset(Dataset):
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32) if y is not None else None

    def __len__(self):
        return len(self.x_num)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.x_num[idx], self.x_cat[idx], self.y[idx]
        return self.x_num[idx], self.x_cat[idx]


def make_loader(x_num, x_cat, y=None, batch_size=512, shuffle=False):
    ds = TwoStreamDataset(x_num, x_cat, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True)


# ══════════════════════════════════════════
# 训练
# ══════════════════════════════════════════

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False


@torch.no_grad()
def predict(model, x_num, x_cat, device, batch_size=1024):
    model.eval()
    preds = []
    loader = make_loader(x_num, x_cat, batch_size=batch_size)
    for xb_num, xb_cat in loader:
        xb_num, xb_cat = xb_num.to(device, non_blocking=True), xb_cat.to(device, non_blocking=True)
        preds.append(torch.sigmoid(model(xb_num, xb_cat)).cpu().numpy())
    return np.concatenate(preds)


def train_one_fold(x_num, x_cat, y, train_idx, valid_idx, x_test_num, x_test_cat,
                   cat_cardinalities, args, seed, fold, device):
    seed_everything(seed + fold * 1000)
    model = EmbeddingMLP(
        cat_cardinalities=cat_cardinalities,
        n_num=x_num.shape[1],
        emb_dim=args.emb_dim,
        hidden=tuple(args.hidden),
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    swa = SWA(model)

    x_trn_num, x_trn_cat = x_num[train_idx], x_cat[train_idx]
    y_trn = y[train_idx]
    x_val_num, x_val_cat = x_num[valid_idx], x_cat[valid_idx]
    y_val = y[valid_idx]

    y_smooth = y_trn * (1.0 - args.label_smooth) + 0.5 * args.label_smooth
    train_loader = make_loader(x_trn_num, x_trn_cat, y_smooth, args.batch_size, shuffle=True)

    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=steps_per_epoch,
        epochs=args.epochs, pct_start=0.1, anneal_strategy='cos',
    )

    best_acc = 0.0
    best_loss = float("inf")
    best_state = None
    stale = 0
    swa_start = int(args.epochs * 0.75)

    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb_num, xb_cat, yb in train_loader:
            xb_num = xb_num.to(device, non_blocking=True)
            xb_cat = xb_cat.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb_num, xb_cat), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

        valid_prob = predict(model, x_val_num, x_val_cat, device)
        valid_loss = log_loss(y_val, np.clip(valid_prob, 1e-6, 1 - 1e-6))
        valid_acc = accuracy_score(y_val, valid_prob >= 0.5)

        if epoch >= swa_start:
            swa.update()

        if valid_acc > best_acc or (valid_acc == best_acc and valid_loss < best_loss):
            best_acc = valid_acc
            best_loss = valid_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    valid_prob = predict(model, x_val_num, x_val_cat, device)
    test_prob = predict(model, x_test_num, x_test_cat, device)
    valid_acc_final = accuracy_score(y_val, valid_prob >= 0.5)
    return valid_prob, test_prob, {"fold": fold, "best_acc": best_acc, "final_acc": valid_acc_final, "best_logloss": best_loss}


def save_submission(passenger_id, probs, out_path, threshold=0.5):
    sub = pd.DataFrame({"PassengerId": passenger_id, "Transported": probs >= threshold})
    sub.to_csv(out_path, index=False)


# ══════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-raw", default="预处理部分/train_preprocessed.csv")
    parser.add_argument("--test-raw", default="预处理部分/test_preprocessed.csv")
    parser.add_argument("--out-dir", default="dl_native_output")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2024, 2026, 3407, 777, 999, 2023, 88])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--hidden", type=int, nargs="+", default=[256, 128, 64])
    parser.add_argument("--emb-dim", type=int, default=16)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--label-smooth", type=float, default=0.03)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    train_df = pd.read_csv(args.train_raw)
    test_df = pd.read_csv(args.test_raw)

    # 目标
    y = (train_df["Transported"].astype(str).str.strip().str.lower()
         .map({"true": 1.0, "false": 0.0})).values.astype(np.float32)
    test_ids = test_df["PassengerId"].copy()
    groups = passenger_groups(train_df["PassengerId"])

    # 类别特征 → int
    x_cat, cat_cardinalities = encode_categorical(train_df, CAT_FEATURES)
    x_test_cat, _ = encode_categorical(test_df, CAT_FEATURES)

    # 数值特征
    x_num_raw, num_names = build_numerical(train_df)
    x_test_num_raw, _ = build_numerical(test_df)

    # 数值特征标准化 (RobustScaler-like: QuantileTransformer)
    qt = QuantileTransformer(output_distribution='normal', random_state=42, n_quantiles=1000)
    x_num = qt.fit_transform(x_num_raw, ).astype(np.float32)
    x_test_num = qt.transform(x_test_num_raw).astype(np.float32)

    print(f"类别特征: {len(CAT_FEATURES)} 个 (cardinalities: {cat_cardinalities})", flush=True)
    print(f"数值特征: {len(num_names)} 个", flush=True)
    for n in num_names:
        print(f"  {n}", flush=True)

    all_oof = []
    all_test = []
    fold_reports = []

    for seed in args.seeds:
        oof = np.zeros(len(y), dtype=np.float32)
        test_accum = np.zeros(len(test_df), dtype=np.float32)
        splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=seed)
        splits = list(splitter.split(x_num, y, groups))
        for fold, (trn_idx, val_idx) in enumerate(splits):
            val_prob, test_prob, report = train_one_fold(
                x_num, x_cat, y, trn_idx, val_idx,
                x_test_num, x_test_cat, cat_cardinalities,
                args, seed, fold, device,
            )
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
    np.save(out_dir / "dl_native_oof.npy", dl_oof)
    np.save(out_dir / "dl_native_test.npy", dl_test)
    save_submission(test_ids, dl_test, out_dir / "submission_dl_native.csv")

    dl_acc = float(accuracy_score(y, dl_oof >= 0.5))
    dl_ll = float(log_loss(y, np.clip(dl_oof, 1e-6, 1 - 1e-6)))
    print(f"\nFinal DL OOF acc={dl_acc:.5f}  logloss={dl_ll:.5f}", flush=True)

    # Blend
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
        blend_test = (1 - best_blend["w_dl"]) * base_test + best_blend["w_dl"] * dl_test
        save_submission(test_ids, blend_test, out_dir / f"submission_blend_w{best_blend['w_dl']:.2f}.csv")

    report = {
        "version": "native_embedding",
        "cat_cardinalities": cat_cardinalities,
        "n_cat": len(CAT_FEATURES),
        "n_num": len(num_names),
        "num_features": num_names,
        "emb_dim": args.emb_dim,
        "hidden": args.hidden,
        "n_seeds": len(args.seeds),
        "folds": args.folds,
        "epochs": args.epochs,
        "dl_oof_acc": dl_acc,
        "dl_oof_logloss": dl_ll,
        "best_blend": best_blend,
    }
    (out_dir / "dl_native_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in ("num_features",)}, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
