"""
Round 1 independent local ensemble
==================================
Independent local modeling route for the Spaceship Titanic project.
  - 64 engineered tabular features with cross-validated target/statistical encoding.
  - Local XGBoost, LightGBM, CatBoost-GBDT, and CatBoost-native models.
  - Additional sklearn MLP branch for comparison.
  - Final local submission from a stable 2-of-4 tree/CatBoost vote.
"""
import os, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
warnings.filterwarnings("ignore")

SPEND = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
SEEDS = list(range(10))


def find_prefixed_dir(root: Path, prefix: str) -> Path:
    matches = sorted(path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix))
    if not matches:
        raise FileNotFoundError(f"Missing directory starting with {prefix!r} under {root}")
    return matches[0]


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = find_prefixed_dir(ROOT, "02_")
OUT_ROOT = find_prefixed_dir(ROOT, "04_")
OUT = OUT_ROOT / "round1_independent_local_ensemble_0p81318"
OUT.mkdir(parents=True, exist_ok=True)

LOG = []
def log(m=""):
    LOG.append(str(m))
    try:
        print(m)
    except UnicodeEncodeError:
        print(str(m).encode("ascii", "replace").decode("ascii"))

def save_log():
    with open(OUT / "training_round1_independent_local_ensemble_log.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(LOG))

log("=" * 70)
log("Round 1 independent local ensemble")
log(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
log("=" * 70)

# Local baseline model parameters
LOCAL_XGB = dict(
    n_estimators=357, learning_rate=0.0242, max_depth=10,
    subsample=0.958, colsample_bytree=0.495, min_child_weight=10,
    gamma=2.070, reg_alpha=0.042, reg_lambda=0.039,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", n_jobs=-1, verbosity=0,
)
LOCAL_LGB = dict(
    n_estimators=500, learning_rate=0.025, max_depth=7,
    num_leaves=31, subsample=0.8, colsample_bytree=0.6,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=1.0,
    n_jobs=-1, verbosity=-1,
)
LOCAL_CB_GBDT = dict(
    iterations=500, depth=6, learning_rate=0.03,
    l2_leaf_reg=5.0, loss_function="Logloss",
    verbose=False, allow_writing_files=False,
)
LOCAL_CB_NATIVE = dict(
    depth=6, iterations=1000, learning_rate=0.03,
    l2_leaf_reg=5.0, one_hot_max_size=2,
    loss_function="Logloss", verbose=False, allow_writing_files=False,
)

# ── 数据 ──────────────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")
y     = train["Transported"].astype(int).values
pids  = test["PassengerId"].values
log(f"train={train.shape}  test={test.shape}")

parts  = train["PassengerId"].str.split("_", expand=True)
groups = parts[0].astype(int).values

# Cross-validated target encoding and grouped validation splits
skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
skf_splits = list(skf.split(np.zeros(len(y)), y))
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
sgkf_splits = list(sgkf.split(np.zeros(len(y)), y, groups=groups))


# Feature engineering: 64 local tabular features
def build_features(train_df, test_df, te_splits):
    tr = train_df.copy(); te = test_df.copy()
    tr["_is_train"] = 1; te["_is_train"] = 0
    if "Transported" not in te.columns:
        te["Transported"] = np.nan
    df = pd.concat([tr, te], ignore_index=True)

    p = df["PassengerId"].str.split("_", expand=True)
    df["GroupId"]   = p[0].astype(int)
    df["MemberNum"] = p[1].astype(int)
    df["GroupSize"] = df.groupby("GroupId")["PassengerId"].transform("count")
    df["IsSolo"]    = (df["GroupSize"] == 1).astype(int)

    c = df["Cabin"].str.split("/", expand=True)
    df["Deck"]     = c[0]
    df["CabinNum"] = pd.to_numeric(c[1], errors="coerce")
    df["Side"]     = c[2]
    df["Surname"]  = df["Name"].str.split(" ").str[-1]
    df.loc[df["Name"].isna(), "Surname"] = "Unknown"

    spend_sum = df[SPEND].fillna(0).sum(axis=1)
    df.loc[df["CryoSleep"].isna() & (spend_sum > 0),  "CryoSleep"] = False
    df.loc[df["CryoSleep"].isna() & (spend_sum == 0), "CryoSleep"] = True
    df["CryoSleep"] = df["CryoSleep"].map(
        {True:1,False:0,"True":1,"False":0}).fillna(0).astype(int)
    df["VIP"] = df["VIP"].map(
        {True:1,False:0,"True":1,"False":0}).fillna(0).astype(int)

    def fill_group_mode(col):
        m = df.groupby("GroupId")[col].agg(
            lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode())>0 else np.nan)
        df[col] = df[col].fillna(df["GroupId"].map(m))
    for col in ["HomePlanet","Destination","Deck","Side","Surname"]:
        fill_group_mode(col)

    deck_hp = df.groupby("Deck")["HomePlanet"].agg(
        lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode())>0 else np.nan)
    df["HomePlanet"] = df["HomePlanet"].fillna(df["Deck"].map(deck_hp)).fillna("Earth")
    hp_dest = df.groupby("HomePlanet")["Destination"].agg(
        lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode())>0 else np.nan)
    df["Destination"] = df["Destination"].fillna(df["HomePlanet"].map(hp_dest)).fillna("TRAPPIST-1e")
    hp_deck = df.groupby("HomePlanet")["Deck"].agg(
        lambda x: x.dropna().mode().iloc[0] if len(x.dropna().mode())>0 else np.nan)
    df["Deck"] = df["Deck"].fillna(df["HomePlanet"].map(hp_deck)).fillna("F")
    df["Side"] = df["Side"].fillna("S")
    df["CabinNum"] = df["CabinNum"].fillna(
        df.groupby("GroupId")["CabinNum"].transform("median"))
    df["CabinNum"] = df["CabinNum"].fillna(df["CabinNum"].median())
    df["Age"] = df["Age"].fillna(df.groupby("GroupId")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df.groupby("HomePlanet")["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    for col in SPEND:
        df.loc[df["CryoSleep"]==1, col] = df.loc[df["CryoSleep"]==1, col].fillna(0)
        df[col] = df[col].fillna(
            df.groupby("HomePlanet")[col].transform("median")).fillna(0)
        df.loc[df["CryoSleep"]==1, col] = 0.0
    df["Surname"] = df["Surname"].fillna("Unknown")

    df["TotalSpend"]  = df[SPEND].sum(axis=1)
    df["LuxurySpend"] = df["RoomService"] + df["Spa"] + df["VRDeck"]
    df["BasicSpend"]  = df["FoodCourt"] + df["ShoppingMall"]
    for col in SPEND + ["TotalSpend","LuxurySpend","BasicSpend"]:
        df[f"Log_{col}"] = np.log1p(df[col])
    df["LuxuryRatio"] = df["LuxurySpend"] / (df["TotalSpend"]+1)
    df["BasicRatio"]  = df["BasicSpend"]  / (df["TotalSpend"]+1)
    for col in SPEND:
        df[f"Ratio_{col}"] = df[col] / (df["TotalSpend"]+1)
        df[f"Has_{col}"]   = (df[col]>0).astype(int)
    df["HasAnySpend"]  = (df["TotalSpend"]>0).astype(int)
    df["SpendCount"]   = sum((df[c]>0).astype(int) for c in SPEND)
    df["IsChild"]      = (df["Age"]<=12).astype(int)
    df["IsTeen"]       = ((df["Age"]>12)&(df["Age"]<=17)).astype(int)
    df["IsYoung"]      = ((df["Age"]>17)&(df["Age"]<=25)).astype(int)
    df["AgeBin"]       = pd.cut(df["Age"],bins=[0,12,17,25,45,64,100],
                                labels=[0,1,2,3,4,5],include_lowest=True
                                ).astype(float).fillna(3).astype(int)
    df["SpendPerAge"]  = df["TotalSpend"]/(df["Age"]+1)
    df["Side_S"]       = (df["Side"]=="S").astype(int)
    DECK_ORDER = {"T":0,"A":1,"B":2,"C":3,"D":4,"E":5,"F":6,"G":7}
    df["Deck_Ordinal"]        = df["Deck"].map(DECK_ORDER).fillna(6)
    df["CabinNum_Bin"]        = pd.qcut(df["CabinNum"],q=10,labels=False,duplicates="drop")
    df["IsHighTransportDeck"] = df["Deck"].isin(["B","C"]).astype(int)
    df["DeckSide"]       = df["Deck"].astype(str)+"_"+df["Side"].astype(str)
    df["CryoHomePlanet"] = df["CryoSleep"].astype(str)+"_"+df["HomePlanet"].astype(str)
    df["PlanetDest"]     = df["HomePlanet"].astype(str)+"_"+df["Destination"].astype(str)
    df["DeckHomePlanet"] = df["Deck"].astype(str)+"_"+df["HomePlanet"].astype(str)
    for col,enc in [("DeckSide","DeckSide_Encoded"),
                    ("CryoHomePlanet","CryoHomePlanet_Encoded"),
                    ("PlanetDest","PlanetDest_Encoded")]:
        m = {v:i for i,v in enumerate(sorted(df[col].unique()))}
        df[enc] = df[col].map(m)
    df["HomePlanet_LE"]  = df["HomePlanet"].map({"Earth":0,"Europa":1,"Mars":2}).fillna(0)
    df["Destination_LE"] = df["Destination"].map(
        {"TRAPPIST-1e":0,"PSO J318.5-22":1,"55 Cancri e":2}).fillna(0)
    sn_cnt = df["Surname"].value_counts()
    df["SurnameCount"]   = df["Surname"].map(sn_cnt)
    df.loc[df["Surname"]=="Unknown","SurnameCount"] = \
        df.loc[df["Surname"]=="Unknown","GroupSize"]
    df["IsLargeFamily"]  = (df["SurnameCount"]>=4).astype(int)
    df["GroupSpendMean"] = df.groupby("GroupId")["TotalSpend"].transform("mean")
    df["GroupSpendMax"]  = df.groupby("GroupId")["TotalSpend"].transform("max")
    df["GroupSpendStd"]  = df.groupby("GroupId")["TotalSpend"].transform("std").fillna(0)
    df["SpendDeviation"] = df["TotalSpend"]-df["GroupSpendMean"]
    df["GroupAgeMean"]   = df.groupby("GroupId")["Age"].transform("mean")
    df["GroupAgeStd"]    = df.groupby("GroupId")["Age"].transform("std").fillna(0)
    df["GroupSizeBin"]   = df["GroupSize"].clip(upper=4)
    df["NoCryoNoSpend"]  = ((df["CryoSleep"]==0)&(df["TotalSpend"]==0)).astype(int)
    df["CryoAge"]        = df["CryoSleep"]*df["Age"]
    df["ChildCryo"]      = df["IsChild"]*df["CryoSleep"]
    sn_freq = df["Surname"].value_counts(normalize=True)
    df["Surname_Freq"]   = df["Surname"].map(sn_freq)
    df["Spend_PctInDeck"] = df.groupby("Deck")["TotalSpend"].rank(pct=True)
    dm = df.groupby("Deck")["TotalSpend"].transform("mean")
    ds = df.groupby("Deck")["TotalSpend"].transform("std").clip(lower=1)
    df["SpendZscore_Deck"] = (df["TotalSpend"]-dm)/ds

    # Target/statistical encoding inside stratified folds
    train_mask = df["_is_train"]==1
    test_mask  = df["_is_train"]==0
    tp   = df[train_mask].copy()
    te_p = df[test_mask].copy()
    y_te = tp["Transported"].astype(int)
    gm   = y_te.mean()

    TE_COLS = ["HomePlanet","Destination","Deck","DeckSide","CryoHomePlanet","PlanetDest"]
    for col in TE_COLS:
        tn = f"TE_{col}"
        tp[tn] = 0.0
        for tr_i,va_i in te_splits:
            trd = tp.iloc[tr_i]
            st  = trd.groupby(col)["Transported"].agg(["sum","count"])
            sm  = (st["sum"]+gm*10)/(st["count"]+10)
            tp.iloc[va_i,tp.columns.get_loc(tn)] = \
                tp.iloc[va_i][col].map(sm).fillna(gm).values
        fst = tp.groupby(col)["Transported"].agg(["sum","count"])
        fsm = (fst["sum"]+gm*10)/(fst["count"]+10)
        te_p[tn] = te_p[col].map(fsm).fillna(gm)
    for col in TE_COLS:
        tn = f"TE_{col}"
        df.loc[train_mask,tn] = tp[tn].values
        df.loc[test_mask, tn] = te_p[tn].values

    tp["TE_Surname"] = 0.0
    for tr_i,va_i in te_splits:
        trd = tp.iloc[tr_i]
        st  = trd.groupby("Surname")["Transported"].agg(["sum","count"])
        sm  = (st["sum"]+gm*20)/(st["count"]+20)
        tp.iloc[va_i,tp.columns.get_loc("TE_Surname")] = \
            tp.iloc[va_i]["Surname"].map(sm).fillna(gm).values
    fst = tp.groupby("Surname")["Transported"].agg(["sum","count"])
    fsm = (fst["sum"]+gm*20)/(fst["count"]+20)
    te_p["TE_Surname"] = te_p["Surname"].map(fsm).fillna(gm)
    df.loc[train_mask,"TE_Surname"] = tp["TE_Surname"].values
    df.loc[test_mask, "TE_Surname"] = te_p["TE_Surname"].values

    # Local 64-feature set
    LOCAL_FEATURES = [
        "Age","CabinNum",
        "Log_RoomService","Log_FoodCourt","Log_ShoppingMall","Log_Spa","Log_VRDeck",
        "Log_TotalSpend","Log_LuxurySpend","Log_BasicSpend",
        "LuxuryRatio","BasicRatio",
        "Ratio_RoomService","Ratio_FoodCourt","Ratio_ShoppingMall","Ratio_Spa","Ratio_VRDeck",
        "Has_RoomService","Has_FoodCourt","Has_ShoppingMall","Has_Spa","Has_VRDeck",
        "HasAnySpend","SpendCount",
        "IsChild","IsTeen","IsYoung","AgeBin","SpendPerAge",
        "Side_S","Deck_Ordinal","DeckSide_Encoded","CabinNum_Bin","IsHighTransportDeck",
        "GroupSize","IsSolo","SurnameCount","IsLargeFamily","GroupSizeBin",
        "GroupSpendMean","GroupSpendMax","GroupSpendStd","SpendDeviation",
        "GroupAgeMean","GroupAgeStd",
        "CryoSleep","VIP",
        "CryoHomePlanet_Encoded","PlanetDest_Encoded",
        "NoCryoNoSpend","CryoAge","ChildCryo",
        "HomePlanet_LE","Destination_LE",
        "TE_HomePlanet","TE_Destination","TE_Deck","TE_DeckSide",
        "TE_CryoHomePlanet","TE_PlanetDest",
        "Surname_Freq","TE_Surname",
        "Spend_PctInDeck","SpendZscore_Deck",
    ]
    NATIVE_NUM = [
        "Age","CabinNum",
        "Log_RoomService","Log_FoodCourt","Log_ShoppingMall","Log_Spa","Log_VRDeck",
        "Log_TotalSpend","Log_LuxurySpend","Log_BasicSpend",
        "LuxuryRatio","BasicRatio",
        "Ratio_RoomService","Ratio_FoodCourt","Ratio_ShoppingMall","Ratio_Spa","Ratio_VRDeck",
        "Has_RoomService","Has_FoodCourt","Has_ShoppingMall","Has_Spa","Has_VRDeck",
        "HasAnySpend","SpendCount",
        "IsChild","IsTeen","IsYoung","AgeBin","SpendPerAge",
        "Side_S","Deck_Ordinal","CabinNum_Bin","IsHighTransportDeck",
        "GroupSize","IsSolo","SurnameCount","IsLargeFamily","GroupSizeBin",
        "GroupSpendMean","GroupSpendMax","GroupSpendStd","SpendDeviation",
        "GroupAgeMean","GroupAgeStd",
        "CryoSleep","VIP",
        "NoCryoNoSpend","CryoAge","ChildCryo",
        "Surname_Freq","Spend_PctInDeck","SpendZscore_Deck",
        "Top2SpendGap","CabinPosition","GroupCryoRatio","SpendPattern_Freq",
    ]
    NATIVE_CAT = [
        "HomePlanet","Destination","Deck","Side",
        "DeckSide","CryoHomePlanet","PlanetDest","DeckHomePlanet",
    ]

    # 计算 CB Native 需要的额外列
    sp  = df[SPEND].values
    srt = np.sort(sp,axis=1)[:,::-1]
    df["Top2SpendGap"]   = (srt[:,0]-srt[:,1])/(df["TotalSpend"].values+1)
    df["CabinPosition"]  = df["Deck_Ordinal"]*2000+df["CabinNum"]
    df["GroupCryoRatio"] = df.groupby("GroupId")["CryoSleep"].transform("mean")
    pat = np.zeros(len(df),dtype=int)
    for i,col in enumerate(SPEND):
        pat += (df[col].values>0).astype(int)*(2**i)
    df["SpendPattern_Freq"] = pd.Series(pat).map(
        pd.Series(pat).value_counts(normalize=True)).values

    dtr = df[df["_is_train"]==1].reset_index(drop=True)
    dte = df[df["_is_train"]==0].reset_index(drop=True)
    for col in NATIVE_CAT:
        dtr[col] = dtr[col].astype(str).fillna("Missing")
        dte[col] = dte[col].astype(str).fillna("Missing")
    X_tr = np.nan_to_num(dtr[LOCAL_FEATURES].values.astype(np.float32), nan=0.0)
    X_te = np.nan_to_num(dte[LOCAL_FEATURES].values.astype(np.float32), nan=0.0)
    return dict(df_tr=dtr, df_te=dte, X_tr=X_tr, X_te=X_te,
                LOCAL_FEATURES=LOCAL_FEATURES, NATIVE_NUM=NATIVE_NUM, NATIVE_CAT=NATIVE_CAT)


log("\n[1/4] Building local features with stratified target encoding...")
t0   = time.time()
data = build_features(train, test, skf_splits)
log(f"  done ({time.time()-t0:.1f}s)  features={len(data['LOCAL_FEATURES'])}")
X_tr = data["X_tr"]; X_te = data["X_te"]

dtr_cb = data["df_tr"]; dte_cb = data["df_te"]
ALL_N  = data["NATIVE_NUM"] + data["NATIVE_CAT"]
cb_tr  = dtr_cb[ALL_N].copy(); cb_te_df = dte_cb[ALL_N].copy()
for col in data["NATIVE_NUM"]:
    cb_tr[col]    = pd.to_numeric(cb_tr[col],    errors="coerce").fillna(0).astype(np.float32)
    cb_te_df[col] = pd.to_numeric(cb_te_df[col], errors="coerce").fillna(0).astype(np.float32)
for col in data["NATIVE_CAT"]:
    cb_tr[col] = cb_tr[col].astype(str); cb_te_df[col] = cb_te_df[col].astype(str)
cat_idx = [ALL_N.index(c) for c in data["NATIVE_CAT"]]
save_log()

# ── OOF 评估 (seed=42, SGKF) ──────────────────────────────────────
log("\n[2/4] OOF evaluation...")
t0 = time.time()

xgb_oof = np.zeros(len(y))
lgb_oof = np.zeros(len(y))
cbg_oof = np.zeros(len(y))
cbn_oof = np.zeros(len(y))
mlp_oof = np.zeros(len(y))

for fi, (tr_i, va_i) in enumerate(sgkf_splits):
    # GBDT
    m = xgb.XGBClassifier(**{**LOCAL_XGB, "random_state": 42})
    m.fit(X_tr[tr_i], y[tr_i])
    xgb_oof[va_i] = m.predict_proba(X_tr[va_i])[:,1]

    m = lgb.LGBMClassifier(**{**LOCAL_LGB, "random_state": 42})
    m.fit(X_tr[tr_i], y[tr_i], callbacks=[lgb.log_evaluation(-1)])
    lgb_oof[va_i] = m.predict_proba(X_tr[va_i])[:,1]

    m = CatBoostClassifier(**{**LOCAL_CB_GBDT, "random_seed": 42})
    m.fit(X_tr[tr_i], y[tr_i])
    cbg_oof[va_i] = m.predict_proba(X_tr[va_i])[:,1]

    tp_pool = Pool(cb_tr.iloc[tr_i], y[tr_i], cat_features=cat_idx)
    vp_pool = Pool(cb_tr.iloc[va_i], y[va_i], cat_features=cat_idx)
    m = CatBoostClassifier(**{**LOCAL_CB_NATIVE, "random_seed": 42})
    m.fit(tp_pool)
    cbn_oof[va_i] = m.predict_proba(vp_pool)[:,1]

    # MLP (10 seeds 平均, 折内标准化)
    scaler = StandardScaler()
    Xtr_sc = scaler.fit_transform(X_tr[tr_i])
    Xva_sc = scaler.transform(X_tr[va_i])
    fold_preds = []
    for s in SEEDS:
        mlp = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            max_iter=200, alpha=0.01,
            learning_rate_init=0.001,
            random_state=s, early_stopping=False,
        )
        mlp.fit(Xtr_sc, y[tr_i])
        fold_preds.append(mlp.predict_proba(Xva_sc)[:,1])
    mlp_oof[va_i] = np.mean(fold_preds, axis=0)

log(f"  XGB  OOF={accuracy_score(y,(xgb_oof>=0.5).astype(int)):.5f}")
log(f"  LGB  OOF={accuracy_score(y,(lgb_oof>=0.5).astype(int)):.5f}")
log(f"  CB   OOF={accuracy_score(y,(cbg_oof>=0.5).astype(int)):.5f}")
log(f"  CBN  OOF={accuracy_score(y,(cbn_oof>=0.5).astype(int)):.5f}")
log(f"  MLP  OOF={accuracy_score(y,(mlp_oof>=0.5).astype(int)):.5f}")
log(f"  ({time.time()-t0:.1f}s)")

gbdt3_oof = np.mean([xgb_oof, lgb_oof, cbg_oof], axis=0)

# 三维权重搜索
best_w = (-1, 0.0, 0.0)
for w_cbn in np.arange(0.0, 0.56, 0.05):
    for w_mlp in np.arange(0.0, 0.31, 0.05):
        if w_cbn + w_mlp > 1.0: continue
        w_g3 = 1 - w_cbn - w_mlp
        p = w_cbn*cbn_oof + w_mlp*mlp_oof + w_g3*gbdt3_oof
        a = accuracy_score(y, (p>=0.5).astype(int))
        if a > best_w[0]: best_w = (a, w_cbn, w_mlp)
log(f"  best: w_cbn={best_w[1]:.2f} w_mlp={best_w[2]:.2f}  OOF={best_w[0]:.5f}")

# 无 MLP 对照
best_no_mlp = (-1, 0.0)
for w in np.arange(0.0, 1.01, 0.05):
    p = w*cbn_oof + (1-w)*gbdt3_oof
    a = accuracy_score(y, (p>=0.5).astype(int))
    if a > best_no_mlp[0]: best_no_mlp = (a, w)
log(f"  no-MLP: w_cbn={best_no_mlp[1]:.2f}  OOF={best_no_mlp[0]:.5f}")
save_log()

# ── 全量训练 10 seeds ──────────────────────────────────────────────
log(f"\n[3/4] Full-train {len(SEEDS)} seeds...")

log("  XGB..."); t0 = time.time()
xgb_te = np.mean([xgb.XGBClassifier(**{**LOCAL_XGB,"random_state":s}).fit(X_tr,y
                   ).predict_proba(X_te)[:,1] for s in SEEDS], axis=0)
log(f"  done ({time.time()-t0:.1f}s)")

log("  LGB..."); t0 = time.time()
lgb_te = np.mean([lgb.LGBMClassifier(**{**LOCAL_LGB,"random_state":s}).fit(
                   X_tr, y, callbacks=[lgb.log_evaluation(-1)]
                   ).predict_proba(X_te)[:,1] for s in SEEDS], axis=0)
log(f"  done ({time.time()-t0:.1f}s)")

log("  CB GBDT..."); t0 = time.time()
cbg_te = np.mean([CatBoostClassifier(**{**LOCAL_CB_GBDT,"random_seed":s,
                   "allow_writing_files":False}).fit(X_tr,y
                   ).predict_proba(X_te)[:,1] for s in SEEDS], axis=0)
log(f"  done ({time.time()-t0:.1f}s)")

log("  CB Native..."); t0 = time.time()
full_pool  = Pool(cb_tr, y, cat_features=cat_idx)
cb_te_pool = Pool(cb_te_df, cat_features=cat_idx)
cbn_te = np.mean([CatBoostClassifier(**{**LOCAL_CB_NATIVE,"random_seed":s}).fit(
                   full_pool).predict_proba(cb_te_pool)[:,1] for s in SEEDS], axis=0)
log(f"  done ({time.time()-t0:.1f}s)")

log("  MLP..."); t0 = time.time()
scaler_full = StandardScaler()
X_tr_sc = scaler_full.fit_transform(X_tr)
X_te_sc = scaler_full.transform(X_te)
mlp_te = np.mean([
    MLPClassifier(hidden_layer_sizes=(256,128,64), max_iter=200,
                  alpha=0.01, learning_rate_init=0.001,
                  random_state=s).fit(X_tr_sc, y
                  ).predict_proba(X_te_sc)[:,1]
    for s in SEEDS], axis=0)
log(f"  done ({time.time()-t0:.1f}s)")
save_log()

# ── 生成提交文件 ───────────────────────────────────────────────────
log("\n[4/4] Generating submissions...")

gbdt3_te = np.mean([xgb_te, lgb_te, cbg_te], axis=0)

def save_sub(p, th, name):
    pred = (p>=th).astype(bool)
    pd.DataFrame({"PassengerId":pids,"Transported":pred}).to_csv(
        OUT / f"submission_round1_{name}.csv", index=False)
    log(f"  {name:42s} True%={pred.mean():.4f}")

# 最优 3 维权重 (含 MLP)
wc, wm = best_w[1], best_w[2]
p_best = wc*cbn_te + wm*mlp_te + (1-wc-wm)*gbdt3_te
save_sub(p_best, 0.500, f"mlp_best_wc{int(wc*100):02d}_wm{int(wm*100):02d}_th500")

# 固定 MLP 权重对照
for wm, tag in [(0.05,"m05"),(0.10,"m10"),(0.15,"m15"),(0.20,"m20")]:
    wc = best_no_mlp[1]
    wg = max(0, 1 - wc - wm)
    save_sub(wc*cbn_te + wm*mlp_te + wg*gbdt3_te, 0.500, f"raw_{tag}_wc{int(wc*100):02d}_th500")

# No-MLP local ensemble comparison
wc0 = best_no_mlp[1]
save_sub(wc0*cbn_te + (1-wc0)*gbdt3_te, 0.500, f"no_mlp_wc{int(wc0*100):02d}_th500")

# 2-of-4 voting across local tree/CatBoost models
votes = ((xgb_te>=0.5).astype(int)+(lgb_te>=0.5).astype(int)+
         (cbg_te>=0.5).astype(int)+(cbn_te>=0.5).astype(int))
vote_pred = (votes >= 2)
pd.DataFrame({"PassengerId":pids,"Transported":vote_pred}).to_csv(
    OUT / "submission_round1_independent_local_ensemble_0p81318.csv", index=False)
log(f"  vote_2of4 (no MLP)                         True%={vote_pred.mean():.4f}")

log("\n" + "="*70)
log("SUMMARY")
log("="*70)
log(f"  features: {len(data['LOCAL_FEATURES'])} (local feature set, SKF TE)")
log(f"  XGB  OOF={accuracy_score(y,(xgb_oof>=0.5).astype(int)):.5f}")
log(f"  LGB  OOF={accuracy_score(y,(lgb_oof>=0.5).astype(int)):.5f}")
log(f"  CB   OOF={accuracy_score(y,(cbg_oof>=0.5).astype(int)):.5f}")
log(f"  CBN  OOF={accuracy_score(y,(cbn_oof>=0.5).astype(int)):.5f}")
log(f"  MLP  OOF={accuracy_score(y,(mlp_oof>=0.5).astype(int)):.5f}")
log(f"  best (w/ MLP) OOF={best_w[0]:.5f}  w_cbn={best_w[1]:.2f} w_mlp={best_w[2]:.2f}")
log(f"  best (no MLP) OOF={best_no_mlp[0]:.5f}  w_cbn={best_no_mlp[1]:.2f}")
log(f"  Round 1 independent local ensemble done")
save_log()


