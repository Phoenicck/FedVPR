# FedVPR Method Explained

本文档基于 `/workspace/Phoenic/claude0527/FedVPR` 当前代码实现撰写，目标是把这个方法讲清楚，而不是复述论文摘要。

需要先说明两点：

1. 仓库 `README.md` 仍然写的是 `FedOSS`，没有完整更新到 `FedVPR` 的实现状态。
2. 仓库里没有明确展开 `VPR` 缩写，因此本文不对名称做未经验证的扩写，而是直接按代码行为解释。

## 1. 这个方法在解决什么问题

`FedVPR` 处理的是联邦开放集识别（Federated Open-Set Recognition, FedOSR）：

- 训练时，每个客户端只持有自己的本地数据，不能共享原始样本。
- 训练样本只覆盖已知类别（known classes）。
- 测试时，既有已知类别，也会出现训练时没见过的未知类别（unknown classes）。

核心难点是：

- 联邦场景下本来就数据异构；
- 训练阶段没有真实未知样本；
- 但模型最后必须学会把“未知类”从“已知类”里分出来。

`FedVPR` 的思路不是直接生成像素级未知图像，而是在特征空间里构造、聚合、采样“伪未知特征”，再用这些特征把决策边界撑开。

## 2. 方法总览

按实现看，`FedVPR` 是一个两阶段方法：

1. `Pretrain`：先训练一个带“虚拟类别槽位”的分类器，预留未知空间。
2. `Finetune`：再利用客户端间分歧识别边界样本，合成局部伪未知特征，并在服务器端聚合出全局未知分布，继续强化开放集边界。

如果只看主线，它可以概括成下面这个流程：

1. 从完整类别中划分出 `known_class` 和 `unknown_class`。
2. 训练集只给客户端分发已知类样本，客户端数据再按 Dirichlet 做 non-IID 划分。
3. 预训练阶段引入 `virtue_num` 个虚拟类别，让主分类头输出 `known_class + virtue_num` 个 logit。
4. 微调阶段用客户端之间的不一致性找“边界样本”。
5. 在边界特征上运行 `i-DUS`，把样本推过决策边界，构造局部伪未知特征。
6. 将局部伪未知按虚拟类别聚类统计，服务器聚合成全局未知分布。
7. 从全局未知分布中再采样新的伪未知特征，继续训练虚拟未知分类器。

## 3. 数据组织方式

数据层的关键逻辑在 `FedVPR/data/*.py`：

- 先随机选定哪些原始类别作为 known，哪些作为 unknown。
- 训练集、验证集、测试集都重映射标签。
- 训练阶段客户端只拿到已知类训练样本。
- 测试阶段会同时保留：
  - `closerloader`：已知类测试样本
  - `openloader`：未知类测试样本

客户端内部的数据划分使用 Dirichlet 分布，因此天然支持 non-IID 联邦场景。

这意味着 `FedVPR` 的训练前提非常明确：

- 训练时没有真实 unknown 数据参与客户端学习；
- unknown 只在测试时显式出现；
- 训练阶段用到的“未知”全部来自特征空间合成。

## 4. 模型结构

模型构造在 `FedVPR/lib/common.py` 和 `FedVPR/models/ResNet_FedOSR_*.py`。

### 4.1 主干网络

默认是 `Resnet18`，也支持 `Resnet34` 和 `Resnet18_3D`。

### 4.2 双头结构

模型不是单一分类头，而是两个头：

- `main_cls`：主分类头，输出维度是 `known_class + virtue_num`
- `auxiliary_cls`：辅助分类头，输出维度是 `known_class`

其中：

- 主头负责最终开放集判别；
- 辅助头只做已知类分类，并在微调阶段参与“边界样本识别”。

### 4.3 中间特征接口

模型前向还会返回两组关键特征：

- `boundary_feats`
- `discrete_feats`

它们都来自 `layer3` 后的特征图。在微调阶段：

- `boundary_feats` 用于送给 peer 模型的辅助头，判断样本是否位于不稳定边界附近；
- `discrete_feats` 用于 `i-DUS` 特征扰动，以及后续未知特征合成。

### 4.4 虚拟类别

`virtue_num` 是这个实现非常关键的新增量。它表示主分类头末尾预留多少个“虚拟未知类别槽位”。

这不是把所有未知样本都塞进一个单独的 unknown 类，而是把未知空间拆成多个虚拟锚点。这样做有两个直接作用：

- unknown 不再只是一类，而是多模态表示；
- 服务器可以按虚拟类别分别统计和采样未知分布。

## 5. 第一阶段：Pretrain

入口在 `FedVPR/lib/Pretrain.py`，具体损失在 `FedVPR/lib/Pretrain_library.py`。

### 5.1 目标

预训练阶段不做真正的未知样本合成，它的任务是：

- 学好已知类分类；
- 给后续未知边界预留空间；
- 让主分类器先适应“known + virtual”输出结构。

### 5.2 预训练损失

每个 batch 的损失由三部分组成：

1. 已知类交叉熵 `loss_ce`
2. 虚拟类保留损失 `loss_vir`
3. 辅助头交叉熵

#### 5.2.1 已知类交叉熵

主头输出会被拆成：

- `known_logits = outputs[:, :known_class]`
- `virtual_logits = outputs[:, known_class:]`

然后对已知类部分做普通交叉熵：

`loss_ce = CE(known_logits, targets)`

#### 5.2.2 虚拟类保留损失

代码不是直接把样本标成虚拟类，而是构造：

- 当前样本真实类别的 logit
- 所有虚拟类别的 logits

然后在 `[true_class_logit, all_virtual_logits]` 上做交叉熵，并强制目标是第 0 位，也就是真实类那一位。

这相当于要求：

- 真实类别得分要高于所有虚拟类别；
- 但虚拟类别槽位必须真实存在，并参与判别几何结构；
- 从而为后续 unknown 特征插入决策空间留出位置。

#### 5.2.3 权重调度

预训练前几个 epoch，虚拟类损失权重较大：

- `epoch < 4` 时，系数是 `0.5`
- 之后降到 `0.01`

这说明实现者希望一开始先明显建立“虚拟空间”，后面再弱化其干扰，让主任务回到已知分类本身。

### 5.3 联邦聚合

预训练轮结束后，服务器做一次参数聚合：

- 采用按样本数加权的 FedAvg；
- 但 `auxiliary` 相关参数不会被聚合。

这点很重要：辅助头保持客户端私有，而主干和主头参与联邦同步。

### 5.4 预训练产物

预训练会保存：

- 一个 server checkpoint
- 每个 client 各自的 checkpoint

后续 `Finetune` 阶段不是从随机初始化开始，而是显式加载这些预训练权重。

## 6. 第二阶段：Finetune

入口在 `FedVPR/lib/Finetune.py`，核心逻辑在 `FedVPR/lib/Finetune_library.py`。

这部分才是 `FedVPR` 的主体。

### 6.1 微调阶段的核心目标

它同时做三件事：

1. 继续保持已知类分类性能；
2. 通过本地伪未知样本把边界往外推；
3. 通过服务器聚合得到更稳定、更丰富的全局未知分布。

### 6.2 基础监督项

微调开始时，先保留标准监督损失：

- `CE(outputs, targets)`
- `CE(aux_outputs, targets)`

这保证模型不会因为只顾 unknown 边界而丢掉 closed-set 分类能力。

## 7. 客户端间分歧驱动的边界样本识别

这是 `FedVPR` 延续 `FedOSS` 思路的关键部分。

对于一个客户端当前 batch 的样本：

1. 先用本客户端辅助头预测；
2. 再把 `boundary_feats` 送给若干 peer 客户端的辅助头；
3. 统计这些模型对真实标签的预测一致率；
4. 一致率落在 `(p_lower, p_upper)` 区间内的样本，被视为边界样本。

直觉上：

- 如果所有客户端都很确定，这不是边界；
- 如果所有客户端都完全错乱，也不一定适合拿来构造稳定伪未知；
- 处于“半稳定半不稳定”的样本，更接近决策边界附近。

不同数据集的 `p_upper` 略有差别，例如 BloodMNIST 会更保守一些。

## 8. i-DUS：局部伪未知特征合成

边界样本选出来以后，代码调用 `FedVPR/attack/attack.py` 中的 `i_DUS`。

这里合成的不是输入图像，而是 `layer3` 特征图空间里的特征样本。

### 8.1 做法

`i_DUS` 的逻辑很直接：

1. 把 `discrete_feats` 当成可优化变量；
2. 通过 `net.discrete_forward` 走 `layer4 + avgpool + main_cls`；
3. 对真实标签的交叉熵取负；
4. 沿梯度方向迭代更新特征；
5. 把样本推出原有类别的判别区域。

这实际上是在做“特征空间中的对抗偏移”，让样本越过当前已知类边界，成为局部伪未知。

### 8.2 为什么在特征空间做

这样做有三个现实好处：

1. 比像素空间生成更稳定；
2. 不需要构造视觉上可解释的未知图像；
3. 生成出的样本可以直接服务于最终分类头的边界学习。

## 9. 多虚拟类别监督

局部伪未知样本生成后，`FedVPR` 没有把它们全部标成同一个 unknown。

它做的是：

- `virtual_idx = targets_unknown % virtue_num`
- `virtual_targets = known_class + virtual_idx`

也就是说，局部伪未知会被确定性地分配给多个虚拟类别中的某一个。

这让 unknown 空间变成了多原型结构，而不是单原型结构。

对应损失是：

- 本地伪未知交叉熵，权重为 `lups_local_weight`

## 10. 排序损失

除了伪未知交叉熵，微调阶段还加了一个 `known_unknown_rank_loss`。

它比较的是：

- 已知样本在 known 类上的最大概率
- 伪未知样本在 known 类上的最大概率

目标是让：

- known 样本的 known-score 更高；
- unknown 样本的 known-score 更低。

如果两者间隔不够，就施加 margin 惩罚。

这会直接增强开放集判别的排序能力，而不是只依赖 argmax 分类。

## 11. 本地未知统计量收集

在 `epoch in start_epoch` 时，客户端会把“质量较好的局部伪未知特征”收集起来，用于服务器建模。

这里有两个重要实现点：

### 11.1 筛选标准

代码用一个叫 `PDs` 的量：

- `PD = P(last_virtual) - max(P(other_classes))`

然后用它筛掉不够像 unknown 的伪未知样本。

虽然这个判别式比较硬编码，但它表达的意图很清楚：

- 只有当伪未知在虚拟未知方向上足够突出时，才拿去估计未知分布。

### 11.2 收集的是特征统计，不是样本本身

客户端不会上传原始图像，也不会上传整批特征集合，而是按虚拟类别统计：

- 均值 `mean`
- 方差 `var` 或协方差 `cov`
- 样本数 `count`

这也是整个方法保持联邦约束的关键。

## 12. LUPS：服务器端轻量未知分布聚合

这是 `FedVPR` 相对原始 `FedOSS/FOSS` 的重要演化点。

仓库里有一份设计说明 `FedVPR/LUPS_FOSS_REFACTOR.md`，核心结论是：

- 对高维医学图像特征做 full covariance FOSS，内存和数值稳定性都很差；
- 因此实现里默认改成了轻量的 diagonal 统计聚合。

### 12.1 两种模式

代码支持两种服务器未知分布模式：

- `lups_mode=fullcov`
- `lups_mode=diag`

默认是 `diag`。

### 12.2 统计对象

服务器按虚拟类别分别聚合 unknown 分布。

如果是 `diag` 模式，客户端上传的是：

- `mean: [virtue_num, D]`
- `var: [virtue_num, D]`
- `count: [virtue_num]`

服务器用二阶矩公式合并：

- 全局均值：按样本数加权
- 全局方差：`E[x^2] - E[x]^2`

然后对方差做：

- 下界裁剪 `lups_min_var`
- 缩放 `lups_var_scale`

最终每个虚拟类别得到一个轻量分布：

- `{"mean": ..., "var": ..., "count": ...}`

### 12.3 为什么要这么改

代码里的理由很充分：

- `full covariance` 在高维特征上是 `O(D^2)`；
- RetinalOCT、ISIC 这种数据上 `D` 很大；
- 小样本估计 full covariance 本身就不稳；
- diagonal 近似在工程上更可扩展，也更稳。

## 13. 全局伪未知采样

有了服务器聚合出来的 `unknown_dis` 之后，客户端在后续轮次还能继续从全局未知分布中采样。

### 13.1 采样空间

采样依然发生在特征空间，而不是图像空间。

### 13.2 采样策略

当前默认策略是 `low_density`：

1. 先从对角高斯里采很多候选样本；
2. 用近似 Mahalanobis score 衡量其低密度程度；
3. 选分数高的样本，作为更“远离已知簇中心”的伪未知。

这与原始 FOSS “在开放空间采样”的思想是一致的，但代价小很多。

### 13.3 训练方式

采样得到的全局伪未知特征会被 reshape 回特征图尺寸，再走：

- `net.discrete_forward`

然后仍然用“多虚拟类别目标”做交叉熵监督，权重为：

- `lups_global_weight`

同时继续叠加排序损失。

所以微调阶段的 unknown 学习实际上分成两层：

1. 本地层：i-DUS 生成局部伪未知
2. 全局层：LUPS/FOSS 分布采样生成全局伪未知

## 14. 联邦聚合策略

参数聚合仍然是 FedAvg 风格，按客户端样本数加权。

但实现有一个明确选择：

- `auxiliary` 相关参数不聚合

这代表作者把辅助头当作“客户端本地边界识别器”，而不是全局共享模块。

相比之下：

- 主干网络
- 主分类头

会在每轮后同步到 server，再回写到 client。

## 15. 推理时如何判 unknown

推理阶段并没有额外训练一个单独的 unknown detector。

判断规则很直接：

- 主头输出 `known_class + virtue_num` 个 logit
- 只要预测落到任一虚拟类别槽位上，就视为 unknown

在计算 ACC/F1/Recall/Precision 时，代码会把：

- `pred >= known_class`

统一映射成 unknown 标签。

因此虚拟类别在训练中是“多 unknown 原型”，但在评估中会汇总成一个 unknown 超类。

## 16. 评估指标

测试实现同时报告：

- Closed-set 指标：
  - ACC
  - F1
  - Recall
  - Precision
- Open-set 指标：
  - UNK
  - OS*
  - HOS
  - AUROC
  - AUPR
  - OSCR

其中：

- `UNK`：未知类召回
- `OS*`：已知类宏平均召回
- `HOS`：`OS*` 与 `UNK` 的调和平均
- `AUROC/AUPR`：基于 `1 - max_known_probability`
- `OSCR`：联合衡量已知分类正确率和未知拒识能力

训练过程中模型是按 `OSCR` 选 best checkpoint 的。

## 17. 关键超参数

相对原始 `FedOSS`，`FedVPR` 明显新增了下面这些关键参数：

- `virtue_num`
- `rank_weight`
- `rank_margin`
- `lups_mode`
- `lups_space`
- `lups_pool_size`
- `lups_min_count`
- `lups_min_var`
- `lups_var_scale`
- `lups_candidates`
- `lups_sample_strategy`
- `lups_local_weight`
- `lups_global_weight`

它们分别控制：

- 虚拟未知类别数量
- known/unknown 排序间隔
- 服务器未知分布建模方式
- 特征降维方式
- 全局伪未知采样强度与损失权重

## 18. 这个实现相对 FedOSS 的实质变化

如果把 `FedVPR` 看作 `FedOSS` 的增强版，那么核心变化可以概括为四点：

1. unknown 不再只是一个统一的拒识区域，而是拆成多个虚拟类别原型。
2. 预训练阶段显式做“空间预留”，让主头提前适配 `known + virtual` 输出。
3. 微调阶段同时利用本地 i-DUS 伪未知和服务器全局分布采样伪未知。
4. FOSS 的 full covariance 版本被轻量化为更实用的 LUPS/diag 统计聚合。

## 19. 一句话总结

`FedVPR` 的本质可以概括为：

> 在联邦开放集识别里，先用虚拟类别为未知空间预留结构，再用客户端间分歧找到边界样本，通过特征空间伪未知合成与服务器端未知分布采样，逐步把 known/unknown 决策边界学出来。

如果你更关注工程实现而不是概念名词，那么最重要的理解是：

- 这个方法训练的不是“一个 unknown 类”，
- 而是“多个虚拟 unknown 原型 + 本地/全局两级伪未知特征生成机制”。

## 20. 对应代码入口

建议直接从这几个文件顺着读：

- `FedVPR/main.py`
- `FedVPR/lib/common.py`
- `FedVPR/lib/Pretrain.py`
- `FedVPR/lib/Pretrain_library.py`
- `FedVPR/lib/Finetune.py`
- `FedVPR/lib/Finetune_library.py`
- `FedVPR/lib/communication.py`
- `FedVPR/attack/attack.py`
- `FedVPR/models/ResNet_FedOSR_Pretrain.py`
- `FedVPR/models/ResNet_FedOSR_Finetune.py`

