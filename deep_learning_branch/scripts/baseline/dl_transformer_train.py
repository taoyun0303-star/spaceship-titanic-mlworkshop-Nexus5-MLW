import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class FeatureTokenizer(nn.Module):
    """Turn each scalar tabular feature into one token."""

    def __init__(self, n_features, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        nn.init.normal_(self.cls, std=0.02)

    def forward(self, x):
        tokens = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x.size(0), -1, -1)
        return torch.cat([cls, tokens], dim=1)


class SelfAttention(nn.Module):
    """Multi-head self-attention implemented with matmul, not nn.MultiheadAttention."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        bsz, seq_len, d_model = x.shape
        x = x.view(bsz, seq_len, self.n_heads, self.head_dim)
        return x.transpose(1, 2)

    def merge_heads(self, x):
        bsz, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(bsz, seq_len, -1)

    def forward(self, x):
        q = self.split_heads(self.q(x))
        k = self.split_heads(self.k(x))
        v = self.split_heads(self.v(x))
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, v)
        return self.out(self.merge_heads(context))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ffn_mult, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        hidden = int(d_model * ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop1(self.attn(self.norm1(x)))
        x = x + self.drop2(self.ffn(self.norm2(x)))
        return x


class TabTransformerFromScratch(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=3, ffn_mult=2.0, dropout=0.15):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, ffn_mult, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        x = self.tokenizer(x)
        for block in self.blocks:
            x = block(x)
        cls = self.norm(x[:, 0])
        return self.head(cls).squeeze(1)


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
    model = TabTransformerFromScratch(
        n_features=x.shape[1],
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        scheduler.step()

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
    return valid_prob, test_prob, {"seed": seed, "fold": fold, "best_logloss": best_loss, "acc": valid_acc}


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
    pd.DataFrame(rows).to_csv(out_dir / "transformer_blend_weight_search.csv", index=False)

    for w_dl in [0.03, 0.05, 0.10, 0.15, best["w_dl"]]:
        prob_test = (1.0 - w_dl) * base_test + w_dl * dl_test
        save_submission(test_ids, prob_test, out_dir / f"submission_v20_transformer_w{w_dl:.2f}.csv")
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="特征工程部分/train_features_mlp_v2.csv")
    parser.add_argument("--test", default="特征工程部分/test_features_mlp_v2.csv")
    parser.add_argument("--out-dir", default="dl_transformer_output")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026, 3407])
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--patience", type=int, default=35)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=7e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--ffn-mult", type=float, default=2.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
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
            fold_reports.append(report)
            print(
                f"seed={seed} fold={fold} acc={report['acc']:.5f} logloss={report['best_logloss']:.5f}",
                flush=True,
            )
        all_oof.append(oof)
        all_test.append(test_accum)
        print(f"seed={seed} oof_acc={accuracy_score(y, oof >= 0.5):.5f}", flush=True)

    dl_oof = np.mean(all_oof, axis=0)
    dl_test = np.mean(all_test, axis=0)
    np.save(out_dir / "dl_transformer_oof.npy", dl_oof)
    np.save(out_dir / "dl_transformer_test.npy", dl_test)
    pd.DataFrame({"PassengerId": train_df["PassengerId"], "Transported": y, "prob": dl_oof}).to_csv(
        out_dir / "dl_transformer_oof.csv", index=False
    )
    pd.DataFrame({"PassengerId": test_ids, "prob": dl_test}).to_csv(out_dir / "dl_transformer_test.csv", index=False)
    save_submission(test_ids, dl_test, out_dir / "submission_dl_transformer.csv")

    report = {
        "device": device,
        "model": "from_scratch_tabular_transformer",
        "uses_nn_transformer": False,
        "uses_multiheadattention": False,
        "n_features": len(feature_cols),
        "seeds": args.seeds,
        "folds": args.folds,
        "d_model": args.d_model,
        "heads": args.heads,
        "layers": args.layers,
        "dl_oof_acc": float(accuracy_score(y, dl_oof >= 0.5)),
        "dl_oof_logloss": float(log_loss(y, np.clip(dl_oof, 1e-6, 1 - 1e-6))),
        "dl_test_true_rate": float((dl_test >= 0.5).mean()),
        "fold_reports": fold_reports,
    }
    report["best_blend"] = maybe_blend(out_dir, y, test_ids, dl_oof, dl_test)
    (out_dir / "dl_transformer_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
