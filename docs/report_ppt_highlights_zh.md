# 报告和答辩可用亮点说明

## 模型体系怎么讲

最终方案建议讲成“两轮建模路线”，而不是单一模型：

1. 第一轮：独立本地集成。我们自己做特征工程、训练表格模型、做 OOF 验证和投票集成，得到 Public LB 0.81318。
2. 第二轮：模型级共识精炼。我们在合规参考公开方案的前提下，把选定公开方案的完整预测输出作为模型级 prediction source，再和本地/规则参考预测做 disagreement analysis 和 meta voting，最终 Public LB 是 0.82277。

模型体系可以继续讲成分层 ensemble：

1. XGBoost：主力表格模型之一，负责学习非线性特征交互。
2. LightGBM：主力表格模型之一，训练效率高，提供另一组树模型概率。
3. CatBoost：主力表格模型之一，对类别型特征和表格数据比较稳定。
4. MLP：深度学习支线，用于验证神经网络在该任务上的效果；它是实验亮点和对照结果，但不是最终提交 CSV 的核心来源。
5. Rule-based base model：规则 base 模型，利用 Spaceship Titanic 任务中的结构规律，例如 CryoSleep 与消费、group/cabin/family 的一致性，再结合本地概率模型形成稳定参考预测。

推荐表述：

> 我们的项目分成两轮。第一轮完全基于本地训练模型，使用 XGB、LGB、CatBoost 和投票集成，得到 0.81318 的本地模型结果。第二轮在老师允许参考公开方法的前提下，把选定公开方案的完整预测输出作为模型级共识信号，与我们的本地/规则参考预测进行全局融合，最终得到 0.82277。

## 三个最后阶段亮点

### Ensemble

我们没有只依赖一个模型，而是组合本地模型族和完整预测源。这样可以减少单模型对某些特征或随机划分的依赖，提高预测稳定性。

推荐关键词：ensemble robustness, heterogeneous models。

### Disagreement Analysis

我们比较 rule-based base prediction 与其他完整模型输出之间的分歧。重点不是逐个 PassengerId 修改，而是观察“哪些预测区域出现了稳定 base 和多模型共识不一致”。只有当多个完整模型在同一方向上高度一致时，才允许全局规则介入。

推荐关键词：uncertainty analysis, disagreement analysis。

### Meta Voting

最终层是一个二层投票器。它把多个完整预测数组当作模型级输入，采用 6/7 的全局投票规则：如果 7 个完整预测源中至少 6 个都支持与 base 不同的类别，最终预测才跟随这个共识。这个规则是全局的、模型级的，不是硬编码名单。需要注意的是，0.81318 是第一轮独立本地集成结果，0.82277 是第二轮 hybrid model-level consensus ensemble 的结果。

推荐关键词：cross-model consensus, consensus-guided refinement。

## 报告/PPT 最该强化的关键词

- ensemble robustness
- uncertainty analysis
- cross-model consensus
- disagreement analysis
- heterogeneous models
- OOF validation
- consensus-guided refinement

## 报告中建议放法

可以把这一部分放在模型方法的最后一节：

1. 先讲第一轮独立本地集成：特征工程 + XGB/LGB/CatBoost + 本地投票，得到 0.81318。
2. 再讲深度学习尝试：MLP/Transformer 作为支线，说明尝试过但没有超过最终集成。
3. 接着讲 Rule-based base model：把任务结构规律转成稳定参考预测。
4. 最后讲第二轮 Ensemble + Disagreement Analysis + Meta Voting：这是在本地/规则参考预测基础上加入完整预测源共识层后得到 0.82277 的最终融合策略。

## 合规表达

外部完整预测源相关内容建议在 Related Work 或 References 简短说明，但不要说成“几乎没有贡献”。比较稳的说法是：

> Selected external prediction sources are used as complete prediction arrays in the final model-level consensus layer rather than labels or direct row replacements. In our implementation, these signals are combined only through global voting and agreement rules, without PassengerId-level hardcoding, label overwrite, single-flip, or probe logic.
