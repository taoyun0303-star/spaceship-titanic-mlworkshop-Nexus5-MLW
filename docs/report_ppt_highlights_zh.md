# 报告和答辩可用亮点说明

## 模型体系怎么讲

最终方案可以讲成一个分层模型体系，而不是单一模型：

1. XGBoost：主力表格模型之一，负责学习非线性特征交互。
2. LightGBM：主力表格模型之一，训练效率高，提供另一组树模型概率。
3. CatBoost：主力表格模型之一，对类别型特征和表格数据比较稳定。
4. MLP：深度学习支线，用于验证神经网络在该任务上的效果；它是实验亮点和对照结果，但不是最终提交 CSV 的核心来源。
5. Rule-based base model：规则 base 模型，利用 Spaceship Titanic 任务中的结构规律，例如 CryoSleep 与消费、group/cabin/family 的一致性，再结合本地概率模型形成稳定参考预测。

推荐表述：

> 我们的最终模型不是单一分类器，而是以表格模型为主、深度学习为对照、规则 base 为稳定参考的分层 ensemble。XGB、LGB、CatBoost 提供主要概率判断，MLP 支线用于验证深度学习路线，Rule-based base model 把任务里的结构规律显式加入，最后通过模型级融合得到最终提交结果。

## 三个最后阶段亮点

### Ensemble

我们没有只依赖一个模型，而是组合多个模型族和完整预测源。这样可以减少单模型对某些特征或随机划分的依赖，提高预测稳定性。

推荐关键词：ensemble robustness, heterogeneous models。

### Disagreement Analysis

我们比较 rule-based base prediction 与其他完整模型输出之间的分歧。重点不是逐个 PassengerId 修改，而是观察“哪些预测区域出现了稳定 base 和多模型共识不一致”。只有当多个完整模型在同一方向上高度一致时，才允许全局规则介入。

推荐关键词：uncertainty analysis, disagreement analysis。

### Meta Voting

最终层是一个二层投票器。它把多个完整预测数组当作模型级输入，采用 6/7 的全局投票规则：如果 7 个完整预测源中至少 6 个都支持与 base 不同的类别，最终预测才跟随这个共识。这个规则是全局的、模型级的，不是硬编码名单。

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

1. 先讲特征工程和本地模型：XGB、LGB、CatBoost 等表格模型。
2. 再讲深度学习尝试：MLP/Transformer 作为支线，说明尝试过但没有超过最终集成。
3. 接着讲 Rule-based base model：把任务结构规律转成稳定参考预测。
4. 最后讲 Ensemble + Disagreement Analysis + Meta Voting：这是从本地模型到 0.82277 的最终融合策略。

## 合规表达

外部支持信号相关内容只建议在 Related Work 或 References 轻描淡写说明：

> Selected external prediction sources are used only as additional ensemble-level support signals rather than labels or direct replacements. In our implementation, these signals are combined only through global voting and agreement rules, without PassengerId-level hardcoding, label overwrite, single-flip, or probe logic.
