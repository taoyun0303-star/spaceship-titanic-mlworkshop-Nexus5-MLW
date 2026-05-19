# 深度学习模型实验总结报告

## 任务背景

Kaggle Spaceship Titanic 二分类竞赛，训练集 8,693 行，测试集 4,277 行，评估指标 accuracy。

## 对照基线

| 模型 | OOF Accuracy | 说明 |
|------|:---:|------|
| **GBDT3 集成 (XGB+LGB+CB)** | **0.81663** | 树模型基线 |
| 5-model 平均 (XGB/LGB/CB/ET/HGB) | 0.81571 | 全树模型集成 |

---

## DL 实验历程（共 8 个版本）

### v1 — Baseline MLP
- **文件**: `dl_mlp_train.py`
- **架构**: MLP [256→128→64], BatchNorm+SiLU+Dropout(0.25)
- **训练**: 3 seeds × 5 folds × 300 epochs, BCE loss, AdamW(lr=8e-4)
- **结果**: OOF=**0.80824**, Blend(v20 GBDT) w=0.17 → **0.81893**
- **问题**: 无 LR scheduler, 无梯度裁剪, 所有 seed 共享同一套 fold splits

### v2 — Baseline Transformer
- **文件**: `dl_transformer_train.py`
- **架构**: Feature Tokenizer + 3层 Transformer (d=64, 4 heads, FFN×2), GELU
- **训练**: 3 seeds × 5 folds × 250 epochs, CosineAnnealingLR, grad_clip=1.0
- **结果**: OOF=**0.80950**, Blend(v20) w=0.11 → **0.81870**

### v3 — v34 (SWA + Warmup + 大模型)
- **文件**: `dl_v34_train.py`
- **改动**: 10 seeds, 500 epochs, Warmup+Cosine LR, SWA, Label Smooth(0.03), grad_clip=1.0
- **架构**: MLP [512→256→128]
- **结果**: OOF=**0.80755**, Blend w=0.09 → **0.81951**
- **意外**: 投入更大(10 seeds, 500 epochs)却比 baseline 差

### v4 — v35 (ResMLP + SwiGLU)
- **文件**: `dl_v35_train.py`
- **改动**: SwiGLU 激活, LayerNorm 替代 BatchNorm, Residual Block, 更深 [384→384→256→192]
- **结果**: OOF=**0.80317** ← **最差版本**, Blend w=0.07 → 0.81882
- **结论**: 对 8.6k 行小数据, 复杂架构反而严重过拟合

### v5 — v37 (换特征集)
- **文件**: `dl_v37_train.py`
- **改动**: 使用 tree_v2 特征 (56维, 替代 mlp_v2 的 90维), 300 epochs
- **结果**: OOF=**0.80536**, Blend w=0.05 → 0.81859
- **结论**: tree_v2 特征更稀疏, 对 MLP 不友好

### v6 — v38 (MLP+Transformer 集成)
- **结果**: DL 集成 OOF=0.81238, DL+GBDT Blend w=0.07 → 0.81836
- **结论**: 两个 DL 模型集成后仍低于 GBDT 单独

### v7 — Improved (全修复版)
- **文件**: `dl_improved_train.py`
- **改动**: 修复 cudnn.benchmark, 每 seed 重 shuffle folds, StandardScaler, Accuracy Early Stopping, OneCycleLR, SWA 延迟至75%, Mixup(α=0.2), Label Smooth(0.03)
- **架构**: MLP [256→128→64]
- **训练**: 当前归档产物为 8 seeds × 5 folds × 150 epochs；脚本默认可在云 GPU 上用 300 epochs 重跑
- **结果**: OOF=**0.81261** ← **最好 MLP 结果**, Blend w=0.09 → 0.81893

### v8 — Native Embedding (DL 专用特征)
- **文件**: `dl_native_train.py`
- **改动**: 完全抛弃 ML 特征工程, 从预处理原始数据构建:
  - 6个类别特征 → Embedding(dim=16)
  - 21个数值特征 (log、ratio、Deck×CabinNum 等)
  - QuantileTransformer 标准化
- **结果**: OOF=**0.80651** ← **验证失败: ML 特征对 DL 有益无害**
- **结论**: Target Encoding 等 ML 特征帮助 DL, 并非"毒药"

---

## 汇总对比（含 DL+GBDT 混合模型）

```
                                               OOF       类型
─────────────────────────────────────────────────────────────
v34 (SWA)+GBDT blend               0.81951  Blend  ███████████
Baseline MLP+GBDT blend            0.81905  Blend  ███████████
Improved MLP+GBDT blend            0.81893  Blend  ███████████
v35 ResMLP+GBDT blend              0.81882  Blend  ███████████
Native Embed+GBDT blend            0.81870  Blend  ███████████
Transformer+GBDT blend             0.81870  Blend  ███████████
v38 DL ensemble+GBDT               0.81836  Blend  ███████████
─────────────────────────────────────────────────────────────
GBDT3 Solo (对照)                  0.81663  Tree   █████████
─────────────────────────────────────────────────────────────
Improved MLP                       0.81261  DL     ███████
v38 DL Ensemble (MLP+Transformer)  0.81238  DL     ███████
Transformer                        0.80950  DL     █████
Baseline MLP                       0.80824  DL     ████
v34 MLP (SWA)                      0.80755  DL     ████
Native Embedding                   0.80651  DL     ███
v37 (tree_v2 feats)                0.80536  DL     ███
v35 ResMLP+SwiGLU (最差)           0.80317  DL     █
```

**关键观察**:

| 层次 | 分数区间 | 波动 |
|------|:---:|:---:|
| Blend (DL+GBDT) | 0.81836 ~ 0.81951 | ±0.0006 |
| GBDT Solo | **0.81663** | — |
| DL Solo | 0.80317 ~ 0.81261 | ±0.005 |

**三个核心结论**:
1. **Blend 收敛区间**: 所有 7 种 DL+GBDT 混合挤在 0.8184–0.8195 之间，差异仅 0.00115；在当前 OOF 设置下差异较小，仍需 Kaggle LB 或独立验证确认稳定性
2. **DL 对 Blend 区分度有限**: 最好的 Blend (v34, 0.81951) 用的不是最好的 DL (Improved 0.81261)，而是 DL 中等水平 (0.80755) 的 v34 ← 说明 GBDT 在当前集成中占主导
3. **DL 净贡献**: Blend − GBDT Solo = **0.0025** ← 投入 8 个版本迭代，最终换来的只是千分位的提升

## 核心发现

1. **ML 特征工程对 DL 是营养剂, 不是毒药**: 去掉 TE/WoE 后 DL 从 0.81261 跌至 0.80651
2. **复杂架构在小数据上适得其反**: ResMLP+SwiGLU (0.803) < 简单 MLP (0.808/0.813)
3. **训练技巧有边际收益**: OneCycleLR+Accuracy ES+Mixup 提升 +0.005，但无法突破天花板
4. **当前 DL 落后 GBDT ~0.004–0.010**: 数据量(8.6k) + 强规则标签 + 异构特征 → 树模型在本验证设置下更有优势
5. **DL 在集成中几乎无作用**: Blend 分数对 DL 质量完全不敏感，DL 权重仅 0.05–0.17，可以移除
