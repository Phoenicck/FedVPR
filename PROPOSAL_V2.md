# FedVPR ProposalV2: Two-Stage Boundary Tightening and Pseudo-Unknown Synthesis

相关记录入口：
- [ICLR Proposal 实验规划](/workspace/Phoenic/claude0527/ICLR_PROPOSAL_EXPERIMENT_PLAN.md)
- [FedVPR 第一阶段实验记录](/workspace/Phoenic/claude0527/FedVPR/STAGE1_EXPERIMENT_RECORD.md)
- [FedVPR 方法说明](/workspace/Phoenic/claude0527/FedVPR/FedVPR_METHOD_EXPLAINED.md)

## 1. 核心观点

FedVPR 不应被理解为“一阶段直接学会拒识 unknown，二阶段只是微调”。更合理的两阶段分工是：

```text
Stage-1: 在没有真实 unknown 监督的前提下，收紧 known 边界，并建立稳定的 virtual unknown 坐标系。
Stage-2: 利用边界样本或边界附近特征生成伪未知样本，用这些伪样本真正训练 known-vs-unknown 边界。
```

因此，一阶段只需要证明“边界正在变得更可用”，不需要强求 `UNK Recall` 或 `Open->V` 很高。二阶段才是方法真正开始学习 unknown rejection 的阶段。

一句话版：

```text
一阶段负责把 known space 整理好，并给 unknown 留出方向；二阶段负责把伪未知样本送进这些方向，真正把开放集边界训练出来。
```

## 2. 为什么不能把一阶段目标设成强拒识

一阶段训练只使用 known-class 样本。此时模型没有任何真实 unknown 监督，因此要求它直接做到高 `UNK` 或高 `Open->V` 并不合理。

一阶段更合理的目标是：

- known 类不要散掉
- known 类内部更紧
- known 类之间边界更清楚
- virtual anchors 稳定存在
- open-like 样本相对 known 样本更靠近边界或 virtual space
- 为二阶段的边界样本筛选和伪未知生成提供可用起点

所以，一阶段的成功不应定义为：

```text
Open->V 很高，所以成功。
```

而应定义为：

```text
known 边界更紧，virtual anchors 稳定，open-like 样本出现靠近 reserved space 的趋势，因此具备进入二阶段的条件。
```

## 3. Stage-1: Boundary Tightening

### 3.1 阶段目标

Stage-1 的目标是 `boundary tightening`，也就是让已知类空间更规整，同时预留未知方向。

它要解决的问题是：

- 如果 known 表征本身很散，二阶段生成的伪未知会没有可靠参照
- 如果 virtual anchors 不稳定，二阶段伪未知会被训练到漂移目标上
- 如果 known classifier 过度自信，伪未知很难越过边界

Stage-1 不要求：

- `UNK Recall` 很高
- `Open->V` 很高
- hard unknown 被大量预测成 virtual

Stage-1 要求：

- close-set 性能基本保持
- known 表征更紧
- boundary-like 样本可被筛出
- virtual anchors 稳定且不干扰 known
- open 样本相对 close 样本呈现更靠近边界的趋势

### 3.2 Stage-1 主指标

#### A. Known Performance Preservation

目的：证明一阶段没有破坏基础分类能力。

建议记录：

- `Test-Close ACC`
- `Test-Close Macro-F1`
- `Test-Close Recall`
- `Test-Close Precision`
- 每个 client 的训练 ACC/F1

验收口径：

- 相比 FedOSS 一阶段或 legacy baseline，`ACC/F1` 不应明显下降
- 建议容忍范围为 `0-3%` 绝对下降
- 如果下降超过 `5%`，说明一阶段为了预留 unknown space 付出了过高代价

#### B. Known Boundary Tightening

目的：证明 known space 确实更紧，而不是只靠分类头硬记住。

建议新增或记录：

- intra-class feature variance
- inter-class center distance
- compactness ratio: `intra-class variance / inter-class distance`
- known true-logit margin: `true_known_logit - max(other_known_logits)`
- client disagreement boundary rate

推荐判断：

- `intra-class variance` 下降，说明 known 类内部收紧
- `compactness ratio` 下降，说明类内更紧且类间仍分开
- `true-known margin` 不应持续下降，否则 known 分类边界变脆
- boundary candidate rate 不应为 0，也不应过高

建议阶段性阈值：

- `compactness ratio` 相比初始或 FedOSS baseline 改善 `5-10%` 即可算有趋势
- boundary candidate rate 保持在 `5-40%` 是较健康区间
- 如果 boundary candidate rate 接近 `0%`，二阶段可能没有足够伪未知种子
- 如果 boundary candidate rate 超过 `50%`，说明边界筛选过宽，伪未知质量可能不稳定

#### C. Virtual Anchor Stability

目的：证明 virtual anchors 是可用坐标系，而不是训练噪声。

现有日志已经支持：

- `VNorm`
- `VDriftMax`
- `Vcos(max/mean)`
- `KVcos(max/mean)`
- `CloseK->V`
- `Open->V`
- `OpenVEntropy`

验收口径：

- `VDriftMax` 应接近 `0`
- `VNorm` 应接近 `1`
- `Vcos(max)` 不应持续升高到接近 `1`
- `KVcos(max)` 不应持续异常升高
- `CloseK->V` 应保持很低，理想状态为 `< 1%`

这里 `Open->V` 是观察指标，不是硬性验收指标。尤其在 hard protocol 下，`Open->V` 很低并不直接否定一阶段，只说明二阶段更需要伪未知生成来打开边界。

#### D. Open-Like Trend

目的：证明 open 样本相对 close 样本更接近边界。

建议记录：

- close: `true_known_logit - max_virtual_logit`
- open: `max_known_logit - max_virtual_logit`
- close/open 的 margin gap
- open margin 的 p10/p50/p90
- `AUROC/AUPR/OSCR` 作为排序趋势指标

更合理的判断方式：

```text
不是要求 open margin 立刻小于 0，
而是要求 open margin 相对 close margin 更小，或者随训练出现下降趋势。
```

对于 hard unknown，如果 `OpenKnown-VMax mean` 始终很大，例如长期在 `5+`，说明 virtual branch 仍未参与空间预留。这个结果不是一阶段彻底失败，但它提示二阶段必须依赖更强的 pseudo-unknown generation。

### 3.3 Stage-1 Ready-to-Finetune 条件

满足下面条件即可进入二阶段，不需要等待 `Open->V` 明显升高：

| 维度 | 进入二阶段的最低条件 | 理想状态 |
|---|---|---|
| Known 性能 | Close ACC/F1 相比 baseline 下降不超过 `3-5%` | 与 FedOSS 持平或更高 |
| Known 收紧 | compactness ratio 有下降趋势 | 改善 `5-10%` 以上 |
| Anchor 稳定 | `VDriftMax ~= 0`, `CloseK->V < 1%` | anchors 稳定且分散 |
| 边界种子 | boundary candidate rate 非零 | `5-40%` 稳定区间 |
| Open 趋势 | open margin 比 close 更靠近边界，或 AUROC/OSCR 有排序信号 | `OpenKnown-VMax` 下降，少量 `Open->V` 出现 |

关键点：

```text
Stage-1 的门槛是“能否为 Stage-2 提供稳定边界和伪未知种子”，不是“是否已经完成 unknown rejection”。
```

## 4. Stage-2: Pseudo-Unknown Synthesis and Boundary Learning

### 4.1 阶段目标

Stage-2 才是真正利用伪未知样本训练开放集边界的阶段。

它要解决的问题是：

- 从 known 边界附近找到可用样本
- 通过 i-DUS 或 LUPS 生成伪未知特征
- 把伪未知分配给多个 virtual anchors
- 保持 known 分类能力
- 提升 unknown detection 和 known-vs-unknown ranking

Stage-2 的主目标不是“生成很多样本”，而是：

```text
生成足够可靠、足够多样、足够靠近开放边界的伪未知特征，并让这些特征提升开放集排序与拒识。
```

### 4.2 Stage-2 样本生成指标

#### A. Boundary Sample Mining

目的：证明二阶段真的找到了可用边界样本。

建议记录：

- 每个 client 的 boundary sample count
- boundary sample rate
- 各 known 类的 boundary sample 分布
- peer disagreement score 分布
- boundary 样本与 non-boundary 样本的 feature margin 对比

验收口径：

- boundary count 不能长期为 0
- boundary 样本不能全部来自单个类别
- disagreement 分布应能区分 easy known 和 boundary-like known

#### B. i-DUS Pseudo-Unknown Quality

目的：证明生成的伪未知不是无效扰动。

建议记录：

- i-DUS success rate
- 生成后 `max_known_prob` 是否下降
- 生成后 `max_virtual_prob` 是否上升
- 生成前后 feature distance
- 生成前后 known margin 变化
- 生成样本是否出现 NaN/Inf

验收口径：

- 伪未知生成后 known confidence 应下降
- virtual confidence 应上升
- 生成距离不能过大，否则可能离开有意义的边界区域
- success rate 应稳定非零

#### C. Multi-Virtual Assignment

目的：证明多个 virtual anchors 真的被使用，而不是单锚点塌缩。

建议记录：

- pseudo-unknown virtual target histogram
- virtual assignment entropy
- 每个 virtual anchor 的样本数
- 每个 virtual anchor 的 feature mean/variance

验收口径：

- 至少有 `2` 个 virtual anchors 被稳定使用
- entropy 不应长期接近 `0`
- 如果所有伪未知都落到一个 virtual head，说明多虚拟类别结构没有被充分利用

#### D. LUPS / Global Pseudo-Unknown Quality

目的：证明服务器端统计聚合和采样真的提供了额外信息。

建议记录：

- 每个 virtual anchor 的 uploaded count
- diagonal variance mean/min/max
- sampled pseudo-unknown count
- low-density sampling score
- global pseudo-unknown CE loss
- global pseudo-unknown ranking loss

验收口径：

- 每个 virtual anchor 至少有足够 count 建立分布
- variance 不应过小到退化，也不应过大到采样失控
- global sampled samples 应提升或稳定 OSCR/HOS，而不是只提升 ACC

### 4.3 Stage-2 开放集效果指标

Stage-2 的最终验收应以开放集指标为主，而不是只看 closed-set 分类。

核心指标：

- `UNK Recall`
- `HOS`
- `AUROC`
- `AUPR`
- `OSCR`
- `Close ACC`
- `Macro-F1`

推荐主排序：

```text
OSCR / HOS > AUROC / AUPR > UNK Recall > Close ACC
```

原因是：

- `UNK Recall` 高但 close accuracy 崩掉没有意义
- `ACC/F1` 高但 `OSCR` 下降，说明模型可能只学会了硬分类，没有学会开放集排序
- `OSCR` 同时反映 known 分类和 unknown 排序，更适合作为部署指标

### 4.4 Stage-2 成功标准

建议以 FedOSS 或当前 VPR legacy finetune 作为 baseline。

最低可接受标准：

- `Close ACC` 不显著低于 baseline
- `UNK Recall` 高于 stage-1
- `OSCR` 不低于 baseline
- `HOS` 高于 baseline
- 伪未知生成数量稳定非零

理想标准：

- `OSCR` 提升 `2-5%`
- `HOS` 提升 `5%+`
- `UNK Recall` 明显提升
- `Close ACC` 基本持平
- 多个 virtual anchors 被稳定使用

风险信号：

- `UNK Recall` 提升但 `OSCR` 下降
- `Close ACC` 明显下降
- pseudo-unknown 全部落到单个 virtual anchor
- global pseudo samples 让 known confidence 排序变坏
- LUPS 采样导致 loss 波动或训练不稳定

## 5. 两阶段之间的逻辑关系

### 5.1 正确的因果链

FedVPR 的合理因果链应该写成：

```text
Stage-1:
known 表征收紧
-> 边界样本更可识别
-> virtual anchors 稳定
-> 为伪未知生成提供方向

Stage-2:
边界样本筛选
-> i-DUS / LUPS 生成伪未知
-> 伪未知训练到多个 virtual anchors
-> known-vs-unknown ranking 改善
-> OSCR / HOS / AUROC 提升
```

这条链条比“Stage-1 直接提升 Open->V”更稳，也更符合当前实验现象。

### 5.2 当前实验现象的解释

真实 `OCT hard` 实验里，当前版和 legacy 版都出现：

- `CloseK->V` 很低
- anchors 稳定
- close-set 指标较强
- `Open->V` 很低
- `OpenKnown-VMax` 仍然很大

这说明：

```text
Stage-1 已经能保护 known 并稳定 virtual coordinates，
但 hard unknown 不会自然掉进 virtual space。
```

因此下一步不应只继续调 `vir_weight`，而应把重点转到二阶段：

- 是否能筛到足够边界样本
- i-DUS 是否能把它们推成可用伪未知
- LUPS 是否能聚合出更稳定的全局伪未知分布
- ranking loss 是否能防止 OSCR 被破坏

## 6. 实验记录建议

后续每个实验建议按下面模板记录。

### 6.1 Stage-1 记录模板

| 项目 | 记录内容 |
|---|---|
| Protocol | random / easy / hard |
| Recipe | legacy / margin / boundary-proxy |
| Known performance | Close ACC / F1 / Recall |
| Boundary tightening | compactness ratio / known margin |
| Anchor stability | VDriftMax / Vcos / KVcos |
| Known safety | CloseK->V |
| Open trend | OpenKnown-VMax / Open->V / AUROC / OSCR |
| Decision | 是否进入 Stage-2 |

### 6.2 Stage-2 记录模板

| 项目 | 记录内容 |
|---|---|
| Pretrain checkpoint | 使用哪一个 Stage-1 run |
| Boundary mining | boundary count / rate / class hist |
| i-DUS quality | success rate / known confidence drop / virtual confidence rise |
| LUPS quality | uploaded count / variance / sampled count |
| Multi-virtual usage | virtual hist / entropy |
| Open-set metrics | UNK / HOS / AUROC / AUPR / OSCR |
| Known preservation | Close ACC / Macro-F1 |
| Decision | 是否保留该二阶段策略 |

## 7. ProposalV2 的实验优先级

### Priority 1: 建立真实 hard protocol 的 Stage-1 基线

已经完成初步验证：

- 当前版 `OCT hard`: `Open->V` 很低，但 close-set 和 anchors 稳定
- legacy `OCT hard`: 同样 `Open->V` 很低

结论：

```text
hard protocol 的瓶颈不是新旧一阶段配方差异，而是需要二阶段伪未知来真正打开边界。
```

### Priority 2: 补齐 Stage-1 boundary tightening 诊断

建议下一步补日志：

- intra-class variance
- inter-class center distance
- compactness ratio
- boundary candidate rate
- boundary class histogram

这一步能让“一阶段边界收紧”从口头判断变成可量化证据。

### Priority 3: 做 Stage-2 pseudo-unknown generation 诊断

建议优先补：

- i-DUS success rate
- pseudo-unknown count
- pseudo-unknown confidence shift
- virtual assignment histogram
- local/global pseudo loss

这一步直接决定二阶段是否真的在生成有效伪未知。

### Priority 4: 再优化 Stage-2 ranking 和 LUPS

当伪未知生成稳定后，再系统调：

- `lups_local_weight`
- `lups_global_weight`
- `rank_weight`
- `rank_margin`
- `lups_var_scale`
- `lups_sample_strategy`

此时优化目标应优先看：

```text
OSCR, HOS, AUROC, AUPR
```

而不是只看 ACC。

## 8. 可写进论文的版本

可以把 FedVPR 的核心方法写成：

> We decompose federated open-set learning into two coupled stages. The first stage performs boundary tightening on known classes while constructing stable virtual unknown anchors, without assuming access to real unknown samples. The second stage uses boundary-like samples and federated uncertainty statistics to synthesize pseudo-unknown features, which are assigned to multiple virtual anchors and optimized with ranking-aware objectives. This design separates the formation of a reliable known boundary from the actual learning of unknown rejection, making the framework more robust under heterogeneous medical federated settings.

对应中文表述：

> 我们将联邦开放集学习拆解为两个相互衔接的阶段：第一阶段在没有真实未知样本监督的情况下收紧已知类边界，并构建稳定的虚拟未知锚点；第二阶段利用边界样本和联邦不确定性统计生成伪未知特征，再通过多虚拟锚点和排序保持目标真正学习开放集边界。这样的拆分避免了一阶段被错误地要求直接完成未知拒识，也使二阶段的伪未知生成有了稳定的边界参照。

## 9. 当前最重要的判断

当前不要把 `OCT hard` 上低 `Open->V` 简单理解为方法失败。更准确的判断是：

```text
一阶段已经完成了部分基础工作：known 保持、anchor 稳定。
但 hard unknown 不会仅靠 known-only pretrain 自动进入 virtual space。
因此真正的检验点应转到二阶段：能否从边界样本生成有效伪未知，并用这些伪未知提升 OSCR/HOS。
```

这也是 ProposalV2 相比之前版本最关键的修正。
