# FedVPR 第一阶段实验记录

## 1. 记录目的

本文档用于记录当前 `FedVPR` 第一阶段（`Pretrain`）实验的进展，重点关注以下问题：

- 固定 `easy protocol` 下的表现如何
- 虚拟锚点是否稳定
- 第一阶段是否真的预留出了可用的 unknown space
- 当前问题更像是权重问题，还是目标函数设计问题

当前主要分析数据集：
- `RetinalOCT`

当前固定 `easy protocol`：
- 已知类：`CNV, DME, DR, DRUSEN, NORMAL`
- 未知类：`AMD, CSR, MH`

当前固定 `hard protocol`：
- 已知类：`CSR, DR, DRUSEN, MH, NORMAL`
- 未知类：`AMD, CNV, DME`

## 2. 当前已完成的代码修改

### 2.1 固定 easy protocol

新增了 `--protocol_mode` 参数，支持：
- `random`
- `easy`
- `hard`

并将固定 protocol 接入：
- `FedVPR/data/fed_retinal_oct_relabel.py`
- `FedVPR/data/fed_isic_relabel.py`

其中：
- `RetinalOCT` 的 `hard protocol` 已在本轮补齐
- 之前一段时间里，`RetinalOCT` 的脚本虽然传了 `--protocol_mode='hard'`，但数据代码没有实现该分支，实际会回退到 `random`

### 2.2 虚拟锚点诊断日志

已增加每个 epoch 的诊断信息，包括：
- virtual anchor 的范数
- virtual anchor 相对初始化的漂移
- virtual anchor 两两 cosine
- virtual anchor 与 known classifier 的 cosine
- known 样本被判到 virtual 的比例
- open 样本被判到 virtual 的比例
- virtual 预测直方图与熵
- close / open 的 logit margin 摘要

### 2.3 修复 client 侧 virtual-anchor 冻结 bug

发现的问题：
- 模型构造函数中原本通过 gradient hook 冻结 virtual classifier rows
- 但联邦训练中 `deepcopy(server_model)` 生成 client 模型时，这个 hook 没有被可靠保留
- 导致 client 端仍然更新 virtual anchors，再通过联邦聚合写回 server

修复方式：
- 在 `FedVPR/lib/Pretrain_library.py` 中，在 `loss.backward()` 之后显式执行：
  `net.main_cls.weight.grad[args.known_class:] = 0`

验证结果：
- `virtual_delta = 0.0`
- known classifier rows 仍然正常更新

### 2.4 可配置的 stage-1 virtual loss 权重

新增参数：
- `--vir_weight_warmup`
- `--vir_weight_main`

默认行为保持与原实现一致：
- warmup 阶段：`0.5`
- 主阶段：`0.01`

### 2.5 分项损失日志

当前每个 client 的训练日志已支持打印：
- `CE`
- 原始 `VIR`
- 加权后的 `wVIR`
- `AUX`
- 当前使用的 `vir_weight`

## 3. 关键实验时间线

### 3.1 历史 random protocol 基线

参考日志：
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260602_210254.log`

观察结果：
- close-set 指标正常上升
- 从表面上看，stage-1 在 random protocol 下似乎可用
- 但 `UNK` 始终是 `0`

重要解释：
- random protocol 的结果不能证明 virtual anchors 真正起作用了
- 它更多说明：在那组随机 known/unknown 划分下，仅依赖 known-class ranking 也能得到还不错的开放集排序分数

### 3.2 当前代码对 random protocol 的复现验证

参考日志：
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_190752.log`

实验目的：
- 验证最近的代码改动是否破坏了 stage-1 的主训练逻辑

结果：
- 当前代码可以较好复现旧版 random protocol 的训练轨迹
- 因此可以确认：最近加入的日志和协议接线并没有破坏主训练循环

结论：
- `easy protocol` 下的明显下降，不是因为近期代码改坏了主训练逻辑

### 3.3 easy protocol 下、修复冻结 bug 之前的现象

参考日志：
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_161712.log`

观察结果：
- `Open->V = 0%`
- `UNK = 0`
- virtual anchors 漂移明显
- virtual anchor 两两 cosine 持续增大

解释：
- 这时不仅 virtual 分支没有被使用，virtual anchors 本身也不稳定
- 这直接指向了 client-side freeze bug

### 3.4 修复冻结 bug 后的 short diagnostic run

参考日志：
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_210910.log`

关键观察（epoch 0）：
- `VDriftMax = 0.000000`
- `VNorm ~= 1`
- `Vcos ~= 0`
- `Open->V = 0%`
- `UNK = 0`
- `Val-LogitMargin TrueKnown-VMax mean = 3.0009`
- `Test-LogitMargin OpenKnown-VMax mean = 3.2757`

解释：
- 冻结修复是成功的
- virtual anchors 已经稳定且彼此分散
- 但 open 样本依然明显更靠近 known logits，而不是 virtual logits

### 3.5 easy protocol 当前 10 epoch 基线

参考日志：
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_213239.log`

该 run 的最佳结果：
- 按 OSCR 选出的最佳 epoch：`epoch 9`
- `Test-Close ACC = 89.543`
- `OSCR = 43.105`
- `AUROC = 45.762`
- `AUPR = 46.896`
- `UNK = 0`
- `Open->V = 0%`

趋势总结：
- known 性能恢复得不错
- virtual anchors 始终稳定
- `OpenKnown-VMax` 随训练有下降趋势，但始终为正
- virtual 分支始终没有接收到 open 样本作为预测结果

代表性 logit-margin 变化：
- epoch 0: `OpenKnown-VMax mean = 3.2757`
- epoch 3: `OpenKnown-VMax mean = 6.8679`
- epoch 7: `OpenKnown-VMax mean = 4.4867`
- epoch 9: `OpenKnown-VMax mean = 3.5915`

解释：
- 模型越来越擅长已知类分类
- open 样本在后期可能稍微变得没那么 confidently-known
- 但仍远远没有达到激活 virtual prediction 的程度

### 3.6 提高 post-warmup virtual loss 权重的对照实验

对照设置：
- `vir_weight_warmup = 0.5`
- `vir_weight_main = 0.1`

参考日志：
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_215620.log`

观察结果（训练前中期）：
- `Open->V` 仍然是 `0%`
- known close-set ACC 保持较强
- `OpenKnown-VMax` 没有比 baseline 更快、更明显地缩小

解释：
- 仅仅把 post-warmup 的权重从 `0.01` 提高到 `0.1`，不能解决核心问题
- `vir_weight_main` 偏小不是唯一瓶颈

### 3.7 分项损失诊断实验

参考日志：
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_234021.log`

代表性 epoch-0 client 分项损失：
- Client0: `CE=1.166`, `VIR=0.188`, `wVIR=0.094`, `AUX=1.006`
- Client1: `CE=0.868`, `VIR=0.107`, `wVIR=0.054`, `AUX=0.732`
- Client3: `CE=1.783`, `VIR=0.454`, `wVIR=0.227`, `AUX=1.148`
- Client6: `CE=1.486`, `VIR=0.663`, `wVIR=0.331`, `AUX=0.942`

解释：
- 加权后的 `wVIR` 通常明显小于 `CE` 和 `AUX`
- 因此“权重尺度偏小”确实是问题的一部分
- 但考虑到把 `vir_weight_main` 提高到 `0.1` 后仍未激活 virtual prediction，说明“尺度问题”并不是全部问题

### 3.8 关于 `OCT hard` 的重要更正

需要明确收回之前的一条错误结论：
- 我们之前曾把若干 `RetinalOCT` run 当成了 `hard protocol`
- 但事后检查代码发现，当时 `FedVPR/data/fed_retinal_oct_relabel.py` 实际只实现了 `easy`
- 因此脚本里即使写了 `--protocol_mode='hard'`，运行时也会自动落回 `random`

受影响的代表性日志包括：
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260617_171039.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260618_110540.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260618_115434.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260618_151117.log`

修正后的解释应为：
- 上述 run 只能证明 `OCT random` 下，stage-1 有时能激活一定程度的 virtual prediction
- 它们不能作为 `OCT hard` 已经跑通的证据
- 真正的 `OCT hard` 实验，应从本次补齐 `hard protocol` 代码之后重新开始统计

## 4. 当前基于证据的结论

### 4.1 已经确认有效的部分

1. Stage-1 主训练循环没有坏。
- random protocol 仍能复现历史行为。

2. Virtual-anchor 稳定性问题已经修复。
- 无漂移
- 无塌缩
- 无 virtual-to-virtual cosine 爆炸

3. 在 easy protocol 下，known-class 性能是可以接受的。
- stage-1 没有把 known 分类能力拖垮。

4. 在 `OCT random` 上，virtual branch 曾出现过持续激活信号。
- 代表性 run 中，中后期 `Open->V` 可持续维持在大约 `18% - 27%`。
- 同时 `CloseK->V` 仍然较低。
- 这说明方法方向可能成立，但这还不是 `hard protocol` 证据。

### 4.2 当前仍未解决的问题

1. 当前激活仍然是“部分成功”，不是完整成功。
- 不是所有 open 样本都被 virtual space 接住。
- `OpenKnown-VMax` 的中位数通常仍为正，说明大多数 open 样本整体仍偏向 known logits。

2. 单锚点使用塌缩仍然存在。
- `OpenHist` 常呈现类似 `[0, 217, 10]`、`[0, 223, 20]` 的模式。
- 说明现在塌缩的不是 anchor 几何，而是 anchor assignment / usage。

3. 一阶段训练到 `50 epoch` 时仍未明显进入 plateau。
- `Open->V` 在中后期仍有明显波动和上升空间。
- 因此“训练不充分”仍是一个实际存在的瓶颈。

4. `OCT hard` 仍缺乏真实证据。
- 之前引用的强结果后来证实属于 `random`
- 因此 `hard protocol` 下的 stage-1 行为目前仍需要重新实验确认

### 4.3 当前最可能的诊断

当前 stage-1 的目标函数本质上仍更像是在鼓励：
- `true known logit > virtual logits`

这足以做到：
- 让 known 样本远离 virtual anchors
- 让 reserved directions 稳定存在
- 让一部分更接近边界的 open 样本开始越过边界进入 virtual space

但它仍不足以保证：
- 大多数 open 样本都会自然靠近 virtual anchors
- 多个 virtual anchors 会自动形成良好的分工结构

因此，当前问题不只是 `loss_vir` 数值偏小。
更深层的问题是：

`当前 stage-1 已经能部分激活 virtual space，但仍缺乏让 boundary/open-like samples 更充分、更均衡地占用多个 virtual anchors 的机制。`

## 5. 目前相对于第一阶段验收标准的状态

### 5.1 已满足

- known-class 性能基本保持
- virtual anchors 稳定且分散
- 在 `OCT random` 上，virtual branch 已出现持续激活的积极信号

### 5.2 部分满足但尚未完成

- 一部分 open 样本已经更靠近 virtual space，并被 virtual 预测接住
- 但大多数 open 样本整体仍然更偏 known space
- 尚未形成理想的多锚点 unknown-reserved structure

### 5.3 当前判断

更准确的结论应当是：

`FedVPR stage-1 已经在 RetinalOCT random protocol 上提供了明确的积极信号，说明方法方向可能可行；但当前结果仍属于 positive-yet-incomplete，而且真正的 OCT hard protocol 还需要重新验证。`

## 6. 统一修正版下一步优化表

下面这张表是当前更推荐的统一执行顺序。原则是：
- 先做低风险、信息增益高的训练配方优化
- 再做目标函数增强
- 最后再碰多锚点分工与 pseudo-open 方向

| 优先级 | 方向 | 建议动作 | 预期收益 | 风险判断 | 当前建议 |
|---|---|---|---|---|---|
| 1 | 训练配方优化 | 把 `vir_weight` 从“epoch 4 硬切 `0.5 -> 0.01`”改成平滑衰减；建议衰减到 `0.05` 或 `0.1`，并把 `pretrain` 延长到 `80-100 epoch` | 提高 `Open->V`，验证当前曲线是否还能继续上升 | 低 | 最先做 |
| 2 | `loss_vir` 增强 | 不建议用 margin loss 直接替换 CE；更建议 `CE + margin` 组合，让排序与距离约束同时存在 | 有机会把 `OpenKnown-VMax` 继续往 `0` 拉近 | 中 | 第二步做 |
| 3 | 多锚点利用 | 不建议先做 anchor-geometry 正则；更建议做 `assignment balance`，约束 boundary-like 样本在 virtual anchors 上的使用分布不要塌到单个锚点 | 缓解 `OpenHist` 单锚点占用 | 中 | 第三步做 |
| 4 | 更长远的 pseudo-open 方向 | 不建议直接把高熵 known 样本当 open；若要尝试，应更偏向 boundary-like proxy，并且只施加很弱的引导 | 可能带来结构性突破 | 高 | 暂缓，后做 |

### 6.1 对另一个 agent 提案的统一修正

以下几条判断目前认为是合理的：
- `vir_weight` 平滑衰减是合理的
- `epoch` 提升到 `80-100` 是合理的
- 当前瓶颈里，“信号偏弱”“单锚点使用塌缩”“训练尚未充分”三条判断整体成立

以下几条需要修正后再做：
- 不建议直接把 `loss_vir` 从 CE 全替成 margin loss，而是建议做 `CE + margin`
- 不建议当前优先做 temperature annealing，因为它可能进一步强化“单锚点赢者通吃”
- 不建议当前优先做 `virtue_num = 5`，因为大概率会从“1 个活锚点 + 2 个死锚点”变成“1 个活锚点 + 4 个死锚点”
- 不建议把高熵 known 直接当 pseudo-open，这很容易误伤 hard known

### 6.2 推荐执行顺序

建议后续按下面顺序推进：

1. 先做 `平滑 vir_weight + 更长训练`。
2. 再做 `CE + margin` 的增量式 loss 改造。
3. 再做 `assignment balance` 来缓解单锚点使用塌缩。
4. 最后再决定是否进入 `pseudo-open proxy` 类实验。

## 7. 当前阶段重要日志索引

本阶段主要使用过的日志：
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260602_210254.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_190752.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_161712.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_210910.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_213239.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_215620.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260616_234021.log`
- `/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260617_171039.log`

## 5. 阶段性实验总结

这一阶段的实验，已经足够支持我们对 `FedVPR Stage-1` 做一次重新定性：

- `Stage-1` 不是“直接学会开放集拒识”的阶段。
- `Stage-1` 更像是“known 边界整形 + virtual 坐标系稳定化”的阶段。
- `Stage-2` 才应该承担“利用边界样本 / 伪未知样本真正训练开放集边界”的职责。

换句话说，过去这段时间最重要的进展，不是把 `Open->V` 一路拉高，而是把我们对方法职责的理解校正清楚了。

### 5.1 目前已经可以确认的发现

#### A. 代码逻辑层面：当前主训练逻辑是可信的

- 经过 `random protocol` 复现实验，可以确认近期加入的日志、协议接线和诊断代码没有破坏主训练循环。
- 修复 client-side virtual anchor freeze bug 之后，`VDriftMax` 稳定为 `0`，`VNorm` 维持在 `1` 附近，说明 virtual anchors 的“乱跑”问题已经基本解决。

这意味着：

- 现在日志里看到的现象，大体反映的是方法本身的行为，而不是明显的实现错误。

#### B. 现象层面：known 边界收紧是真实存在的

在 `OCT hard` 当前策略日志中：

- [RetinalOCT_Pretrain_K5_U3_seed0_20260620_205250.log](/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260620_205250.log)

可以看到：

- `Compact` 从 `0.278206` 下降到前中期约 `0.19 - 0.23`
- 后期大多维持在约 `0.26`
- `KnownTrue-Other mean` 从接近 `0` 提升到后期 `10+`
- `CloseTrue-VMax mean` 后期稳定在 `8+`

这说明：

- known 类内部确实被压紧了
- known 类之间的 margin 被明显拉开了
- known 样本与 virtual logits 的间隔很大

因此，`Stage-1` 的“边界收紧”目标不是空的，已经有比较明确的实验支持。

#### C. virtual 分支并不是完全没用，但它的激活高度依赖协议难度

在 `OCT easy` 当前策略日志中：

- [RetinalOCT_Pretrain_K5_U3_seed0_20260620_214513.log](/workspace/Phoenic/claude0527/FedVPR/logs/RetinalOCT_Pretrain_K5_U3_seed0_20260620_214513.log)

可以看到：

- `Compact` 早期从 `0.500270` 下降到 `0.265014`
- 中后期 `Open->V` 可以到 `9% - 15%`
- 代表性 epoch：
  - `35/50`: `Open->V=9.143%`
  - `38/50`: `Open->V=14.762%`
  - `49/50`: `Open->V=12.762%`

这说明：

- 当 unknown 相对容易时，virtual branch 的确可能接住一部分 open 样本
- 所以“virtual reserved space 完全不成立”这个结论并不准确

但同一份日志也同时表明：

- `CloseK->V` 会升到 `1% - 2%`
- `OpenHist` 仍明显偏向单一锚点
- `OpenVEntropy` 总体偏低

所以这更像是“部分激活”，不是“成熟机制”。

#### D. `hard protocol` 与 `easy protocol` 给出了互补信息

这段时间最有价值的一个认识是：

- `easy` 更容易看到 virtual 激活信号
- `hard` 更能检验边界收紧是否扎实

具体地说：

- `easy` 结果更适合回答：“virtual branch 有没有工作起来？”
- `hard` 结果更适合回答：“仅靠 Stage-1，面对困难 unknown 时到底能做到什么程度？”

目前答案是：

- `easy` 下，virtual 有一定激活
- `hard` 下，known 边界更稳，但 `Open->V` 很弱

这说明 `known 收紧` 不会自动推出 `hard unknown 被 virtual 接住`。

### 5.2 目前已经被证伪或被修正的判断

#### A. 之前若干所谓的 `OCT hard` 结论无效

之前曾误以为若干 `RetinalOCT` 训练是 `hard protocol`，但后来检查发现当时数据代码并未真正实现 `hard` 分支，实际会回退到 `random`。

因此，过去一部分“`hard` 也出现不错 virtual 信号”的结论需要收回。那批 run 只能说明：

- `random` 下，方法有时能出现比较积极的 virtual 激活
- 不能说明真实 `hard protocol` 已经跑通

#### B. 单纯调大 `vir_weight_main` 不是根本解法

提高 `vir_weight_main` 后，virtual 分支并没有因此稳定地被激活，说明问题不只是“loss 太弱”，更可能是：

- Stage-1 本身没有真实 unknown 监督
- 仅靠 known-vs-virtual 的排斥，不足以让 hard unknown 自然落入 virtual

#### C. 不能再把 `Open->V` 当成 Stage-1 的唯一成败标准

这段时间最重要的认知修正之一是：

- `Open->V` 低，不代表 `Stage-1` 完全失败
- 尤其在 `hard protocol` 下，`Open->V` 低更可能意味着：Stage-2 必须承担更强的伪未知生成职责

因此后续判断标准应该改成：

- `Stage-1` 是否提供了更紧的 known 边界
- virtual anchors 是否稳定、可控、不塌缩
- 是否存在可供二阶段利用的边界样本和边界趋势

而不是继续要求：

- `Stage-1` 单独完成强开放集拒识

### 5.3 当前最稳妥的方法结论

到目前为止，最稳妥、最符合证据的总结是：

1. `FedVPR` 的整体想法还没有被证伪。
2. 但原先如果把 `Stage-1` 理解为“直接学会 unknown rejection”，这个期待过高了。
3. 当前实验更支持这样一种两阶段分工：
   - `Stage-1`：收紧 known 边界，稳定 virtual 坐标系，筛出边界种子
   - `Stage-2`：利用边界样本或边界附近特征生成伪未知，真正训练开放集边界

也就是说，当前最合理的解释不是“方法错了”，而是：

- 方法的阶段职责需要重新定义
- `Stage-1` 已经完成了它的一部分任务
- 真正决定方法成败的重点，正在逐步转向 `Stage-2`

### 5.4 后续实验的直接指导意义

基于这段时间的结果，后续工作不应再长期停留在“继续死磕把 `Stage-1 Open->V` 拉高”上，而应该转向：

1. 用 `hard protocol` 继续保留 `Stage-1 diagnostics`，但把主要目标改为验证：
   - known 边界是否稳定收紧
   - boundary candidate 是否足够可用
   - virtual anchors 是否保持稳定

2. 尽快进入 `Stage-2` 设计与验证，重点看：
   - 如何筛边界样本
   - 如何生成伪未知特征
   - 生成后的样本是否真的降低 known confidence、提升 virtual / unknown 侧响应

3. `easy` 结果保留为机制佐证：
   - 它说明 virtual branch 不是完全无效
   - 但不应再把它当成最终主结论

## 6. 一句话结论

过去这段时间的实验整体上支持这样一个判断：

```text
FedVPR 的 Stage-1 更像“边界整形器”，而不是“直接开放集识别器”。
它已经证明了 known 边界可以被收紧、virtual anchors 可以被稳定；
但 hard unknown 不会仅凭 Stage-1 自动落入 virtual，因此 Stage-2 的伪未知生成与边界学习是方法成败的关键。
```
