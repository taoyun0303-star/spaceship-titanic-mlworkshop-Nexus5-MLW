# 深度学习实验详细报告

## 项目信息

| 项目 | 详情 |
|------|------|
| **竞赛名称** | Kaggle Spaceship Titanic |
| **任务类型** | 二分类（预测乘客是否被传送） |
| **评估指标** | Accuracy |
| **训练集规模** | 8,693 行 |
| **测试集规模** | 4,277 行 |
| **特征维度** | 90-102 维（不同版本略有差异） |
| **实验周期** | 共 8 个主要版本迭代 |

---

## 一、目录结构

```
深度学习/
├── scripts/                          # 训练脚本
│   ├── baseline/                     # 基线版本
│   │   ├── dl_mlp_train.py          # v1 基线 MLP
│   │   └── dl_transformer_train.py  # v2 Transformer
│   ├── improved/                     # 改进版本
│   │   ├── dl_improved_train.py     # v7 全修复版（最佳 MLP）⭐
│   │   ├── dl_native_train.py       # v8 EmbeddingMLP
│   │   ├── dl_v34_train.py          # v3 SWA+大模型
│   │   ├── dl_v35_train.py          # v4 ResMLP+SwiGLU
│   │   ├── dl_v36_train.py          # 增强 Transformer
│   │   ├── dl_v37_train.py          # v5 换特征集
│   │   ├── dl_v39_fast.py           # 快速原型测试
│   │   ├── dl_v39_search.py         # 架构搜索
│   │   └── dl_v40_train.py          # v40 新特征 MLP
│   └── path_utils.py                 # 路径定位工具
├── outputs/                          # 精简实验输出
│   └── best/                         # 最佳模型输出
│       ├── dl_improved_output/      # 最佳 MLP (OOF=0.81261)
│       └── dl_output_v34/           # 最高混合分 (0.81951)
├── logs/                             # 训练日志（5个）
├── features/                         # 预处理特征
│   └── dl_features_v3/              # 101维特征集
└── docs/                             # 文档
    ├── DL_Experiment_Summary.md      # 实验总结
    └── 深度学习实验详细报告.md       # 本文档
```

---

## 二、模型架构详解

### 2.1 MLP 架构（主要架构）

```
输入层 (90-102 features)
    ↓
Linear(in, 256) → BatchNorm → SiLU → Dropout(0.25)
    ↓
Linear(256, 128) → BatchNorm → SiLU → Dropout(0.25)
    ↓
Linear(128, 64) → BatchNorm → SiLU → Dropout(0.25)
    ↓
Linear(64, 1) → Sigmoid
    ↓
输出 (概率值)
```

**关键组件说明：**

| 组件 | 作用 | 配置 |
|------|------|------|
| **BatchNorm** | 稳定训练，加速收敛 | 每层后接 |
| **SiLU (Swish)** | 平滑激活函数，优于 ReLU | `x * sigmoid(x)` |
| **Dropout** | 防止过拟合 | p=0.25 |
| **BCEWithLogitsLoss** | 二分类损失函数 | 内置 Sigmoid |

### 2.2 Transformer 架构（v2/v36）

```
输入特征
    ↓
FeatureTokenizer: 将每个特征映射为 d_model 维向量
    ↓
Positional Encoding (可学习)
    ↓
TransformerEncoder × 3 层
├── MultiHeadAttention (4 heads, d=64)
├── LayerNorm + Residual
├── FFN (d_ffn = 2 × d_model)
└── LayerNorm + Residual
    ↓
CLS Token → Linear → Sigmoid
    ↓
输出
```

### 2.3 EmbeddingMLP 架构（v8 Native）

```
类别特征 (6个):
├── HomePlanet → Embedding(4, 16)
├── Destination → Embedding(3, 16)
├── Deck → Embedding(8, 16)
├── Side → Embedding(2, 16)
├── AgeGroup → Embedding(5, 16)
└── CabinRegion → Embedding(4, 16)
    ↓
Concat → [96维]
    ↓
数值特征 (21个): [96 + 21 = 117维]
    ↓
MLP [256, 128, 64] → 输出
```

---

## 三、训练策略详解

### 3.1 基线训练策略（v1）

```python
# 基线配置
optimizer = AdamW(lr=8e-4)
loss = BCEWithLogitsLoss()
epochs = 300
seeds = [42, 2026, 3407]  # 3 seeds
folds = 5  # StratifiedGroupKFold
```

**问题：**
- 无学习率调度器
- 无梯度裁剪
- 所有 seed 共享同一套 fold splits
- 无数据增强

### 3.2 改进训练策略（v7 - 最佳版本）

```python
# 改进配置
optimizer = AdamW(lr=1e-3, weight_decay=1e-3)
scheduler = OneCycleLR(max_lr=1e-3, pct_start=0.1, anneal_strategy='cos')
loss = BCEWithLogitsLoss()
epochs = 150  # 当前归档产物记录；云 GPU 完整重跑可使用脚本默认 300
patience = 40  # Accuracy-based early stopping
seeds = [42, 2024, 2026, 3407, 777, 999, 2023, 88]  # 8 seeds
folds = 5
grad_clip = 1.0
mixup_alpha = 0.2
label_smooth = 0.03
swa_start = 0.75  # 75% epoch 后开始 SWA
```

**8 项关键改进：**

| 改进项 | 问题 | 解决方案 | 效果 |
|--------|------|----------|------|
| 1. cudnn.benchmark | 训练速度慢 | 启用 CUDA benchmark | 加速 3-10x |
| 2. Fold 重 shuffle | 集成多样性不足 | 每 seed 重新生成 fold | 提升集成效果 |
| 3. StandardScaler | 特征尺度不一 | 训练前标准化 | 训练更稳定 |
| 4. Accuracy ES | 与 Kaggle 指标不对齐 | 用 accuracy 做 early stopping | +0.002 |
| 5. OneCycleLR | 收敛不佳 | 10% warmup + cosine decay | 更好收敛 |
| 6. SWA 延迟 | 过早平均导致欠拟合 | 75% epoch 后开始 | +0.001 |
| 7. Mixup | 数据量小易过拟合 | α=0.2 混合样本 | 泛化提升 |
| 8. Label Smooth | 标签噪声 | y * 0.97 + 0.5 * 0.03 | 正则化 |

### 3.3 Mixup 数据增强

```python
def mixup_data(x, y, alpha=0.2):
    """混合两个样本及其标签"""
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam

# 损失计算
loss = lam * loss_fn(logits, y_a) + (1 - lam) * loss_fn(logits, y_b)
```

### 3.4 Label Smoothing

```python
# 正确的 label smoothing 公式
y_smooth = y * (1 - smooth) + 0.5 * smooth
# 例如: smooth=0.03
# y=1 → 0.97 + 0.015 = 0.985
# y=0 → 0 + 0.015 = 0.015
```

### 3.5 SWA (Stochastic Weight Averaging)

```python
class SWA:
    """随机权重平均"""
    def update(self):
        # 指数移动平均
        p_swa = (p_swa * n + p) / (n + 1)

    # 延迟开始：75% epoch 后才开始平均
    swa_start = int(epochs * 0.75)
```

---

## 四、实验结果详细分析

### 4.1 各版本完整对比

| 版本 | 脚本 | 架构 | OOF Acc | 混合 OOF | Seeds | Epochs | 关键改动 |
|------|------|------|:-------:|:--------:|:-----:|:------:|----------|
| v1 | dl_mlp_train.py | MLP [256,128,64] | 0.80824 | 0.81893 | 3 | 300 | 基线 |
| v2 | dl_transformer_train.py | Transformer | 0.80950 | 0.81870 | 3 | 250 | 3层 Attention |
| v3 | dl_v34_train.py | MLP [512,256,128] | 0.80755 | **0.81951** | 10 | 500 | SWA+大模型 |
| v4 | dl_v35_train.py | ResMLP+SwiGLU | **0.80317** | 0.81882 | 5 | 300 | 复杂架构（最差）|
| v5 | dl_v37_train.py | MLP [512,256,128] | 0.80536 | 0.81859 | 3 | 300 | tree_v2 特征 |
| v6 | dl_v38 (集成) | MLP+Transformer | 0.81238 | 0.81836 | - | - | DL 集成 |
| v7 | dl_improved_train.py | MLP [256,128,64] | **0.81261** | 0.81893 | 8 | 150 | 全修复版 ⭐ |
| v8 | dl_native_train.py | EmbeddingMLP | 0.80651 | 0.81870 | 8 | 300 | 无 ML 特征 |
| v40 | dl_v40_train.py | MLP [512,256,128] | 0.80502 | - | 10 | 500 | dl_features_v3 |

### 4.2 最佳模型（v7 Improved）详细日志

```
Device: cuda
n_features = 90

seed=42:   OOF=0.81560 (folds: 0.8056, 0.8235, 0.8200, 0.8130, 0.8159)
seed=2024: OOF=0.81215 (folds: 0.8217, 0.8027, 0.8125, 0.8078, 0.8160)
seed=2026: OOF=0.81295 (folds: 0.8199, 0.8153, 0.8090, 0.8183, 0.8023)
seed=3407: OOF=0.81065 (folds: 0.7976, 0.8108, 0.8153, 0.8119, 0.8177)
seed=777:  OOF=0.81330 (folds: 0.8291, 0.8097, 0.8193, 0.8050, 0.8035)
seed=999:  OOF=0.81456 (folds: 0.8133, 0.8183, 0.8117, 0.8136, 0.8159)
seed=2023: OOF=0.81456 (folds: 0.8114, 0.8160, 0.8269, 0.8061, 0.8124)
seed=88:   OOF=0.81146 (folds: 0.8101, 0.8096, 0.8107, 0.8148, 0.8121)

Final DL OOF acc=0.81261  logloss=0.39463
Best blend: w_dl=0.09  acc=0.81893
```

**统计分析：**
- 8 seeds 的 OOF 均值：0.81316
- 8 seeds 的 OOF 标准差：0.00164
- 最佳 seed：999/2023 (0.81456)
- 最差 seed：3407 (0.81065)
- 40 个 fold 的 acc 范围：0.7976 ~ 0.8291

### 4.3 基线模型（v1 MLP）详细日志

```
seed=42:   OOF=0.80525
seed=2026: OOF=0.80732
seed=3407: OOF=0.80732

Final DL OOF acc=0.80824  logloss=0.39159
Best blend: w_dl=0.17  acc=0.81893
```

### 4.4 性能对比可视化

```
                          OOF Accuracy    混合 OOF
                          ────────────    ────────
v7 Improved MLP           ████████████    0.81261  →  0.81893
v8 Native Embedding       ██████████      0.80651  →  0.81870
v34 SWA+大模型             █████████       0.80755  →  0.81951 ⭐ 最高混合分
v2 Transformer            █████████       0.80950  →  0.81870
v1 Baseline MLP           ████████        0.80824  →  0.81893
v37 tree_v2 特征           ████████        0.80536  →  0.81859
v40 新特征 MLP             ████████        0.80502  →  -
v4 ResMLP+SwiGLU          ██████          0.80317  →  0.81882

GBDT3 对照                █████████████   0.81663
```

---

## 五、关键发现与结论

### 5.1 核心发现

| 发现 | 详细说明 |
|------|----------|
| **1. Blend 收敛区间** | 所有 7 种 DL+GBDT 混合结果集中在 0.8184~0.8195，差异仅 0.00115；在当前 OOF 设置下差异较小，仍需 Kaggle LB 或独立验证确认稳定性 |
| **2. DL 对 Blend 区分度有限** | 最好的 Blend (v34, 0.81951) 用的不是最好的 DL (Improved 0.81261)，说明 GBDT 在当前集成中占主导 |
| **3. DL 净贡献微弱** | Blend − GBDT Solo = 0.81951 − 0.81663 = **0.00288**，仅千分位提升 |
| **4. ML 特征对 DL 有益** | 去掉 TE/WoE 后 DL 从 0.81261 跌至 0.80651，差距 0.0061 |
| **5. 复杂架构适得其反** | ResMLP+SwiGLU (0.80317) < 简单 MLP (0.80824/0.81261) |
| **6. 训练技巧有边际收益** | OneCycleLR+AccuracyES+Mixup 共提升 +0.005 |

### 5.2 为什么当前 DL 结果落后 GBDT？

| 因素 | 分析 |
|------|------|
| **数据量小** | 8,693 样本，DL 需要大数据才能发挥优势 |
| **强规则标签** | 传送与否有明确规则，树模型更擅长捕获规则 |
| **异构特征** | 数值+类别混合，树模型天然处理更好 |
| **特征工程成熟** | TE/WoE 等特征已高度优化，DL 难以超越 |

### 5.3 DL+GBDT 混合权重分析

| 模型 | 最佳权重 w_dl | 混合 OOF | 权重含义 |
|------|:------------:|:--------:|----------|
| v34 SWA | 0.09 | 0.81951 | DL 贡献 9% |
| Baseline MLP | 0.11 | 0.81905 | DL 贡献 11% |
| Improved MLP | 0.09 | 0.81893 | DL 贡献 9% |
| v35 ResMLP | 0.07 | 0.81882 | DL 贡献 7% |
| Native Embed | 0.16 | 0.81870 | DL 贡献 16% |
| Transformer | 0.11 | 0.81870 | DL 贡献 11% |
| v38 集成 | 0.07 | 0.81836 | DL 贡献 7% |

**结论：** DL 权重普遍在 5%~16%，说明 GBDT 主导预测，DL 仅提供微弱扰动。

---

## 六、技术细节附录

### 6.1 超参数配置汇总

| 超参数 | v1 Baseline | v7 Improved | v34 SWA |
|--------|:-----------:|:-----------:|:-------:|
| 学习率 | 8e-4 | 1e-3 | 1e-3 |
| 权重衰减 | - | 1e-3 | 1e-3 |
| Dropout | 0.25 | 0.25 | 0.25 |
| 隐藏层 | [256,128,64] | [256,128,64] | [512,256,128] |
| Batch Size | 256 | 512 | 256 |
| Epochs | 300 | 150（当前归档产物；脚本默认300） | 500 |
| Patience | - | 40 | 40 |
| Seeds | 3 | 8 | 10 |
| Folds | 5 | 5 | 5 |
| LR Scheduler | 无 | OneCycleLR | CosineAnnealing |
| Grad Clip | 无 | 1.0 | 1.0 |
| Mixup | 无 | 0.2 | 无 |
| Label Smooth | 无 | 0.03 | 0.03 |
| SWA | 无 | 75% 开始 | 有 |

### 6.1.1 参数调优搜索空间说明

本方向采用“多版本迭代 + OOF 验证 + 集成权重扫描”的调参方式，而不是一次性大规模网格搜索。主要搜索范围如下：

| 参数类别 | 尝试范围 | 当前最佳/采用值 | 说明 |
|----------|----------|----------------|------|
| 架构 | MLP, Transformer, ResMLP+SwiGLU, EmbeddingMLP | MLP | 小数据表格任务中简单 MLP 更稳定 |
| 隐藏层规模 | [256,128,64], [512,256,128], 更深 ResMLP | [256,128,64] / [512,256,128] | 单模型最佳为 Improved MLP，混合最佳为 v34 |
| 学习率 | 5e-4, 7e-4, 8e-4, 1e-3 | 1e-3 | 配合 OneCycle 或 Cosine 使用 |
| Dropout | 0.15, 0.25, 0.40 | 0.25 | 0.40 在复杂模型中未带来收益 |
| Weight Decay | 1e-3, 5e-3, 1e-2 | 1e-3 | 防止小数据过拟合 |
| Scheduler | 无, CosineAnnealing, OneCycleLR | OneCycleLR / CosineAnnealing | v7 使用 OneCycleLR，v34 使用 Cosine/SWA |
| 正则化 | Label Smooth, Mixup, SWA, Grad Clip | Mixup=0.2, Label Smooth=0.03, Grad Clip=1.0 | 多技巧组合提升明显，但未做完全单因素消融 |
| 集成权重 | w_dl = 0.00 ~ 0.40 | 0.05 ~ 0.17 | 在 OOF 上扫描得到，可能存在轻微选择偏差 |

因此，报告中的“训练技巧提升”应理解为组合策略带来的经验提升，而不是严格单因素因果结论。

### 6.2 特征说明

**v7 使用的特征集（90维）：**
- 来源：`特征工程部分/train_features_mlp_v2.csv`
- 包含：Target Encoding、WoE 编码、数值特征、交互特征

**v8 Native 特征集（27维原始特征）：**
- 6 个类别特征：HomePlanet, Destination, Deck, Side, AgeGroup, CabinRegion
- 21 个数值特征：Age, RoomService, FoodCourt, ShoppingMall, Spa, VRDeck, Log变换, 比率特征等

**dl_features_v3（101维）：**
- 来源：`export_features_for_dl.py` 从 GBDT 特征工程导出
- 包含：标准化后的数值特征 + Target Encoding 特征

### 6.3 输出文件说明

每个输出文件夹通常包含：

| 文件类型 | 文件名模式 | 说明 |
|----------|------------|------|
| OOF 预测 | `*_oof.npy`, `*_oof.csv` | 训练集折外预测概率 |
| 测试预测 | `*_test.npy`, `*_test.csv` | 测试集预测概率（多 seed 平均） |
| 训练报告 | `*_report.json` | 完整配置和每 fold 指标 |
| 纯 DL 提交 | `submission_dl_*.csv` | 纯 DL 模型提交 |
| 混合提交 | `submission_*_w*.csv` | DL+GBDT 混合提交（不同权重） |
| 权重搜索 | `blend_weight_search.csv` | 混合权重扫描结果 |

---

## 七、总结与建议

### 7.1 实验结论

1. **当前 DL 设置仍低于树模型**：经过 8 个版本迭代，DL 单模型 OOF 从 0.80824 提升至 0.81261，但在当前验证设置下仍未突破 GBDT 的 0.81663。

2. **DL 对集成的贡献微弱**：DL+GBDT 混合仅比纯 GBDT 提升 0.00288，且 DL 权重仅 5%~16%。

3. **简单架构更适合小数据**：在 8.6k 样本上，简单 MLP 优于复杂 Transformer 和 ResMLP。

4. **训练技巧比架构更重要**：OneCycleLR、Mixup、Accuracy ES 等技巧共提升 +0.005，超过架构改进。

### 7.2 最佳实践建议

| 场景 | 建议 |
|------|------|
| **小数据 (<10k)** | 使用简单 MLP，避免复杂架构 |
| **特征工程成熟** | 直接使用 ML 特征，无需 DL 专用特征 |
| **追求最高分** | 以 GBDT 为主，DL 仅作微调 |
| **训练稳定性** | 使用 OneCycleLR + Accuracy ES + Mixup |
| **集成策略** | DL 权重控制在 5%~15% |

### 7.3 未来改进方向

1. **更大数据集**：DL 在大数据上才能发挥优势
2. **预训练模型**：使用 TabNet、SAINT 等预训练表格模型
3. **自动特征工程**：使用 AutoML 工具发现新特征
4. **神经架构搜索**：自动化架构设计

---

*报告生成时间：2026-05-09*
*实验环境：CUDA GPU, PyTorch*
