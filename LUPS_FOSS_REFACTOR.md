# 用不确定度引导的轻量 FOSS 改造方案

## 1. 背景问题

当前 Finetune/FOSS 阶段会在客户端收集伪未知特征，并上传每个类别的均值、完整协方差和样本数。服务器再聚合得到全局高斯分布，用于后续采样伪未知特征。

这个设计在 BloodMNIST 上可以运行，因为 `discrete_feats` 约为：

```text
[B, 256, 2, 2] -> D = 1024
covariance = 1024 x 1024 ~= 4 MB / class
```

但在 RetinalOCT、HyperKvasir、ISIC 上，`discrete_feats` 通常为：

```text
[B, 256, 8, 8] -> D = 16384
covariance = 16384 x 16384 ~= 1 GB / class
```

再乘以客户端数、虚拟类数和 `einsum` 中间张量，显存和内存都会迅速不可控。即使不直接 OOM，高维小样本下的 full covariance 也会病态、低秩、数值不稳定，导致生成的伪未知样本质量差，进一步破坏 OSCR 和 closed-set accuracy。

因此，FOSS 的核心问题不只是工程显存问题，而是：

```text
高维 full covariance 对医学图像特征不可扩展，也不具备稳健统计估计基础。
```

## 2. 改造目标

将原始 full-covariance FOSS 改造为：

```text
Lightweight Uncertainty-Guided Prototype Sampling, LUPS
```

核心思想：

```text
不再上传 D x D 协方差。
客户端只上传每个虚拟未知原型的低维 mean、diagonal variance 和 count。
服务器聚合成轻量不确定分布。
客户端从高不确定或低密度区域采样伪未知特征，用于训练 virtual unknown classifier。
```

这可以同时解决三个问题：

```text
1. 降低内存复杂度：O(D^2) -> O(D)
2. 提高统计稳定性：避免高维小样本 full covariance 病态
3. 缓解隐私风险：上传 diagonal uncertainty 比完整协方差泄露更少结构信息
```

## 3. 方法设计

### 3.1 特征空间选择

建议提供三种模式：

```text
--lups_space fullmap
直接 flatten discrete_feats。
BloodMNIST 可用，大图数据不推荐。

--lups_space pooled
对 feature map 做 adaptive average pooling，例如 8x8 -> 2x2。
所有 2D 数据统一为 D = 256 x 2 x 2 = 1024。

--lups_space gap
global average pooling，D = 256。
最轻量，适合 ISIC、RetinalOCT、HyperKvasir。
```

第一阶段建议优先实现：

```text
pooled + diagonal variance
```

原因是改动最小，并且和 BloodMNIST 的现有维度对齐。

### 3.2 客户端统计量

原始版本：

```python
mean_c = X_c.mean(0)
cov_c = (X_c - mean_c).T @ (X_c - mean_c) / N_c
```

LUPS 改为：

```python
mean_c = X_c.mean(0)
var_c = X_c.var(0, unbiased=False)
count_c = len(X_c)
```

上传内容从：

```text
mean: [C, D]
cov:  [C, D, D]
num:  [C]
```

变成：

```text
mean: [C, D]
var:  [C, D]
num:  [C]
```

其中 `C` 建议只统计 virtual unknown anchors，而不是 `known_class + virtue_num` 的所有槽位。

### 3.3 服务器聚合

对每个虚拟未知类 `c`，设客户端 `k` 上传：

```text
n_k, mu_k, var_k
```

总样本数：

```text
N = sum_k n_k
```

全局均值：

```text
mu = sum_k n_k * mu_k / N
```

全局 diagonal variance：

```text
var = sum_k n_k * (var_k + mu_k^2) / N - mu^2
```

这个公式等价于合并二阶矩：

```text
E[x^2] - E[x]^2
```

最后做数值稳定：

```python
var = torch.clamp(var, min=args.lups_min_var)
var = args.lups_var_scale * var
```

建议不再构造 `torch.distributions.MultivariateNormal`，而是保存轻量结构：

```python
unknown_dis[c] = {
    "mean": global_mean[c],
    "var": global_var[c],
    "count": global_count[c],
}
```

### 3.4 不确定度引导采样

采样候选：

```python
eps = torch.randn(num_candidates, D, device=device)
z = mean + torch.sqrt(var + eps_const) * eps
```

计算 diagonal Mahalanobis distance：

```python
score = ((z - mean) ** 2 / (var + eps_const)).sum(dim=1)
```

采样策略：

```text
random:
直接从 diagonal Gaussian 采样。

low_density:
采样多个候选，保留 Mahalanobis score 较高的样本。

uncertainty:
优先从 variance 较大的 anchor 或维度采样。
```

建议默认：

```text
--lups_sample_strategy low_density
```

因为它和原始 FOSS “从低密度区域采样未知” 的思想一致，但不需要 full covariance。

### 3.5 损失函数

当前代码里 unknown loss 权重存在硬编码风险，建议显式参数化：

```text
--lups_local_weight 0.1
--lups_global_weight 0.01
```

本地 i-DUS 伪未知：

```python
loss += CE(outputs_local_unknown, targets_virtual) * args.lups_local_weight
```

全局 LUPS 采样伪未知：

```python
loss += CE(outputs_global_unknown, targets_virtual) * args.lups_global_weight
```

如果继续使用原始 FedOSS 式“mask 当前虚拟类，然后压到 known_class unknown label”的写法，也需要明确记录为一个 ablation：

```text
target_virtual CE
vs
masked unknown CE
```

## 4. 建议代码参数

在 `main.py` 中新增：

```python
parser.add_argument('--lups_mode', default='diag', choices=['fullcov', 'diag'])
parser.add_argument('--lups_space', default='pooled', choices=['fullmap', 'pooled', 'gap'])
parser.add_argument('--lups_pool_size', default=2, type=int)
parser.add_argument('--lups_min_count', default=10, type=int)
parser.add_argument('--lups_min_var', default=1e-4, type=float)
parser.add_argument('--lups_var_scale', default=1.0, type=float)
parser.add_argument('--lups_candidates', default=100, type=int)
parser.add_argument('--lups_sample_strategy', default='low_density',
                    choices=['random', 'low_density'])
parser.add_argument('--lups_local_weight', default=0.1, type=float)
parser.add_argument('--lups_global_weight', default=0.01, type=float)
```

建议保留：

```text
--lups_mode fullcov
```

作为原始 FOSS baseline，方便 BloodMNIST 上做公平对照。

## 5. 代码改造位置

### 5.1 `lib/Finetune_library.py`

主要改动：

```text
1. 增加 feature projection/pooling 函数
2. 收集 unknown_dict 时存 pooled/gap 后的低维特征
3. 用 var_dict 替代 cov_dict
4. 从 unknown_dis 采样时支持 diag Gaussian
5. unknown loss 权重改为参数
```

建议封装：

```python
def prepare_lups_feature(args, feats):
    if args.lups_space == 'fullmap':
        return feats
    if args.lups_space == 'pooled':
        return F.adaptive_avg_pool2d(feats, (args.lups_pool_size, args.lups_pool_size))
    if args.lups_space == 'gap':
        return F.adaptive_avg_pool2d(feats, (1, 1))
```

### 5.2 `lib/communication.py`

新增：

```python
compute_global_diag_statistic(...)
sample_diag_unknown(...)
```

原始 `compute_global_statistic` 保留给 fullcov baseline。

### 5.3 `models/ResNet_FedOSR_Finetune.py`

如果采用 `pooled=2`，通常 `discrete_forward()` 仍可吃 `[B, 256, 2, 2]`。

如果采用 `gap`，建议新增轻量 unknown head：

```python
embedding_forward(z)
```

第一版可以先不动模型，用 pooled=2 做最小可行版本。

## 6. BloodMNIST 验证实验

BloodMNIST 是最适合验证 LUPS 猜想的第一站，因为 fullcov 可以跑通，方便做等条件对照。

实验设置：

```text
dataset = BloodMNIST
known_class = 5
unknown_class = 3
client_num = 8
dirichlet = 0.5
worker_steps = 1
seed = 1, 42, 66
Pretrain checkpoint 固定
Finetune epoch = 30
start_epoch = [5,10,15,20,25]
sample_from = 8
eps = 0.1
num_steps = 1
```

实验组：

```text
A. Pretrain only
不跑 Finetune，作为 VSR 基线。

B. Original FOSS fullcov
当前 full covariance 版本。

C. LUPS-Diag-Fullmap
不池化，flatten 后用 diagonal variance。

D. LUPS-Diag-Pooled
pooled=2 后用 diagonal variance。
BloodMNIST 上等价于 fullmap，主要用于和大图数据统一代码路径。

E. LUPS-Diag-Pooled + low_density
验证不确定度引导采样是否优于随机采样。
```

报告指标：

```text
Closed-set ACC
Macro-F1
UNK Recall
OS*
HOS
AUROC
AUPR-Out
OSCR
峰值显存
训练时间
mean ± std over seeds
```

核心假设：

```text
如果 LUPS-Diag 在 BloodMNIST 上接近或超过 fullcov，
说明完整协方差不是 FOSS 有效性的必要条件。

如果 LUPS-Diag 的 OSCR/HOS 更稳定，
说明轻量 diagonal uncertainty 可能比 full covariance 更适合医学开放集联邦场景。
```

## 7. RetinalOCT / ISIC / HyperKvasir 扩展实验

在 BloodMNIST 验证后，迁移到大图数据：

```text
RetinalOCT
ISIC 2019
HyperKvasir
```

对比：

```text
Pretrain only
Original FOSS fullcov，如果能跑则记录，否则报告 OOM/不可扩展
LUPS-Diag-Pooled
LUPS-Diag-GAP
```

重点说明：

```text
fullcov 在大图特征上不是公平可扩展方案；
LUPS 在相同训练流程下保持可运行，并且降低内存占用。
```

## 8. 论文叙事

可以将第二阶段从原来的 PGS/FOSS 重写为：

```text
Uncertainty-Guided Lightweight Open-Space Sampling
```

贡献表达：

```text
We observe that full-covariance feature synthesis is computationally prohibitive
and statistically unstable for high-dimensional medical image features.
To address this, we propose a lightweight uncertainty-guided prototype sampling
strategy that models virtual unknown anchors with diagonal uncertainty.
This reduces client-server statistics from O(D^2) to O(D), improves scalability,
and mitigates privacy leakage from transmitting full covariance matrices.
```

对应审稿意见：

```text
1. 方法与 FedOSS 太像：
从 full covariance FOSS 改成 lightweight uncertainty-guided sampling。

2. 实验不够复杂：
支持 RetinalOCT、ISIC、HyperKvasir 等高维医学图像。

3. 隐私问题：
不上传完整协方差，只上传低维 diagonal uncertainty。

4. PGS 降低 OSCR：
通过 uncertainty scale、low-density sampling、loss weight 做系统 ablation。
```

## 9. 推荐实施顺序

```text
Step 1:
保留当前 fullcov 代码作为 baseline。

Step 2:
新增 diag statistic 聚合分支。

Step 3:
BloodMNIST seed=1 跑通 LUPS-Diag。

Step 4:
BloodMNIST seed=1/42/66 做 fullcov vs diag 对照。

Step 5:
迁移 RetinalOCT 和 ISIC，优先用 pooled=2。

Step 6:
加入 gap 版本，观察是否进一步降低显存并保持性能。

Step 7:
整理 ablation：space、sample strategy、variance scale、loss weight。
```

第一版目标不是立刻追求所有指标最高，而是证明：

```text
LUPS 比 fullcov 更可扩展、更稳定，并且不会牺牲核心开放集性能。
```
