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

## 9. 修订后的核心适配原则

当前 VPR 第一阶段已经不是 FedOSS 的单 unknown 预留，而是 `known_class + virtue_num` 的多虚拟类输出。因此第二阶段必须先完成多虚拟类适配，再实现 LUPS。

关键约束：

```text
1. Finetune 阶段不能再退回 K + 1 输出。
2. Finetune 模型必须保持 K + virtue_num 输出，完整继承 Pretrain 的 virtual heads。
3. 所有虚拟类 pred >= known_class 在测试指标中仍统一映射为 unknown。
4. 本地 i-DUS 和全局 LUPS 采样的伪未知样本，应训练到具体 virtual target: known_class + v。
5. AUROC/AUPR/OSCR 仍基于 max known probability，因此第二阶段需要 ranking/margin loss 防止 known-confidence 排序被破坏。
```

第二阶段目标应从“把伪未知统一压到单个 unknown 类”改为：

```text
把伪未知分配到多个 virtual unknown anchors，并保持已知/未知 confidence ranking。
```

## 10. Agent 执行清单

下面步骤按依赖顺序排列。另一个 agent 应每完成一阶段先跑 smoke test，再进入下一阶段。

### Step 0. 建立基线和检查点

目标：确认当前 Pretrain 多虚拟类已经可运行，并记录现有 Finetune 的行为。

执行：

```text
1. 检查 main.py 是否已有 --virtue_num。
2. 检查 models/ResNet_FedOSR_Pretrain.py 的 main head 是否为 num_classes + num_virtual。
3. 跑一次 BloodMNIST 或 RetinalOCT Pretrain smoke test，确认 checkpoint 能生成。
4. 记录当前 Finetune 是否因为 head shape 不一致导致 virtual weights 没有完整加载。
```

验收：

```text
Pretrain log 中 args 包含 virtue_num。
Pretrain checkpoint 的 main_cls.weight 第一维应为 known_class + virtue_num。
```

### Step 1. 修复 Finetune 输出维度

目标：让 Finetune 继续使用 K + virtue_num 输出，而不是 K + 1。

改动文件：

```text
models/ResNet_FedOSR_Finetune.py
lib/common.py
```

具体改动：

```text
1. 在 ResNet_FedOSR_Finetune.ResNet.__init__ 中新增 num_virtual 参数。
2. 将 self.main_cls = nn.Linear(512, num_classes + 1) 改为 num_classes + num_virtual。
3. 将 resnet18/resnet34/resnet50 签名同步增加 num_virtual=0。
4. 在 lib/common.py Finetune setup 中构建 server/client model 时传入 num_virtual=args.virtue_num。
5. 确认 checkpoint 加载时 main_cls.weight shape 能匹配 Pretrain。
```

验收：

```text
Finetune 构建模型后，outputs.shape[1] == args.known_class + args.virtue_num。
加载 Pretrain checkpoint 时，main_cls.weight 不应因为 shape mismatch 被跳过。
```

### Step 2. 统一多虚拟类测试逻辑

目标：测试时保留多个 virtual logits，但指标仍把所有 virtual predictions 当 unknown。

改动文件：

```text
lib/Pretrain_library.py
lib/Finetune_library.py
```

具体改动：

```text
1. 保留 pred_list_mapped[pred_list_mapped >= args.known_class] = args.known_class。
2. 确认 UNK 仍为 np.mean(pred_list[unknown_mask] >= args.known_class)。
3. 确认 AUROC/AUPR/OSCR 仍使用 prob[:, :args.known_class].max(1)[0]。
4. 可选：额外记录 virtual prediction histogram，方便判断未知样本落在哪些 virtual heads。
```

验收：

```text
K + M 输出不会破坏现有 OSR 指标。
OSR ACC/F1/UNK/HOS 的 unknown label 仍统一为 known_class。
```

### Step 3. 改造本地 i-DUS unknown loss

目标：本地生成的 boundary unknown 不再统一训练到 known_class，而是分配到某个 virtual class。

改动文件：

```text
lib/Finetune_library.py
```

推荐新增函数：

```python
def assign_virtual_targets(args, outputs_unknown, targets_unknown=None):
    if targets_unknown is not None:
        virtual_idx = targets_unknown % args.virtue_num
    else:
        virtual_logits = outputs_unknown[:, args.known_class:]
        virtual_idx = virtual_logits.argmax(dim=1)
    return args.known_class + virtual_idx
```

将原逻辑：

```python
gt_unknown = torch.ones(outputs_unknown.shape[0]).long().to(device) * args.known_class
loss += criterion(outputs_unknown, gt_unknown) * args.unknown_weight
```

替换为：

```python
virtual_targets = assign_virtual_targets(args, outputs_unknown.detach(), targets_unknown)
loss += criterion(outputs_unknown, virtual_targets) * args.lups_local_weight
```

建议第一版使用 deterministic assignment，也就是 `targets_unknown % virtue_num`，方便复现；第二版再尝试 argmax assignment 或 uncertainty assignment。

验收：

```text
训练 loss 正常下降。
unknown 样本预测可以落到多个 virtual heads，而不是只落到一个槽。
```

### Step 4. 加入 confidence ranking loss

目标：解决 Finetune 后 F1/ACC/UNK 上升但 AUROC/AUPR/OSCR 下降的问题。

改动文件：

```text
main.py
lib/Finetune_library.py
```

新增参数：

```python
parser.add_argument('--rank_weight', default=0.05, type=float)
parser.add_argument('--rank_margin', default=0.2, type=float)
```

推荐实现：

```python
def known_unknown_rank_loss(args, outputs_known, outputs_unknown):
    prob_known = torch.softmax(outputs_known, dim=-1)
    prob_unknown = torch.softmax(outputs_unknown, dim=-1)
    known_score = prob_known[:, :args.known_class].max(dim=1)[0]
    unknown_known_score = prob_unknown[:, :args.known_class].max(dim=1)[0]
    return torch.relu(args.rank_margin - known_score.mean() + unknown_known_score.mean())
```

在 local unknown 和 global unknown loss 后追加：

```python
loss += args.rank_weight * known_unknown_rank_loss(args, outputs, outputs_unknown)
```

验收：

```text
Finetune 后 AUROC/AUPR/OSCR 不应明显低于无 ranking loss 版本。
如果 closed-set ACC 下降，降低 rank_weight 或 unknown/local/global weight。
```

### Step 5. 实现 LUPS diagonal statistic

目标：用 mean + diagonal variance 替代 full covariance，避免 RetinalOCT/ISIC/HyperKvasir OOM。

改动文件：

```text
main.py
lib/Finetune_library.py
lib/communication.py
```

新增参数：

```python
parser.add_argument('--lups_mode', default='diag', choices=['fullcov', 'diag'])
parser.add_argument('--lups_space', default='pooled', choices=['fullmap', 'pooled', 'gap'])
parser.add_argument('--lups_pool_size', default=2, type=int)
parser.add_argument('--lups_min_count', default=10, type=int)
parser.add_argument('--lups_min_var', default=1e-4, type=float)
parser.add_argument('--lups_var_scale', default=1.0, type=float)
parser.add_argument('--lups_candidates', default=100, type=int)
parser.add_argument('--lups_sample_strategy', default='low_density', choices=['random', 'low_density'])
parser.add_argument('--lups_local_weight', default=0.1, type=float)
parser.add_argument('--lups_global_weight', default=0.01, type=float)
```

新增 feature projection：

```python
def prepare_lups_feature(args, feats):
    if args.lups_space == 'fullmap':
        return feats
    if args.lups_space == 'pooled':
        return F.adaptive_avg_pool2d(feats, (args.lups_pool_size, args.lups_pool_size))
    if args.lups_space == 'gap':
        return F.adaptive_avg_pool2d(feats, (1, 1))
    raise ValueError(args.lups_space)
```

客户端统计从：

```text
unknown_dict: [known_class]
cov_dict: full covariance
```

改为：

```text
unknown_dict: [virtue_num]
mean_dict: [virtue_num, D]
var_dict: [virtue_num, D]
number_dict: [virtue_num]
```

验收：

```text
RetinalOCT/ISIC 使用 pooled=2 时不再构造 D x D covariance。
日志中可打印 LUPS feature shape，例如 D=1024 或 D=256。
```

### Step 6. 实现服务器端 diagonal 聚合和采样

目标：服务器聚合客户端上传的 virtual-anchor 统计，并返回轻量 unknown_dis。

改动文件：

```text
lib/communication.py
```

新增函数：

```python
def compute_global_diag_statistic(args, mean_clients, var_clients, number_clients):
    # mean_clients: [client_num, virtue_num, D]
    # var_clients: [client_num, virtue_num, D]
    # number_clients: [client_num, virtue_num]
```

聚合公式：

```python
N = number_clients.sum(0)
mu = (number_clients[..., None] * mean_clients).sum(0) / N.clamp_min(1)[..., None]
second = (number_clients[..., None] * (var_clients + mean_clients ** 2)).sum(0) / N.clamp_min(1)[..., None]
var = second - mu ** 2
var = torch.clamp(var, min=args.lups_min_var) * args.lups_var_scale
```

返回：

```python
unknown_dis[v] = {
    'mean': mu[v],
    'var': var[v],
    'count': N[v],
}
```

验收：

```text
unknown_dis 长度等于 args.virtue_num。
count 小于 lups_min_count 的 virtual anchor 返回 None。
```

### Step 7. 实现 global LUPS sampling loss

目标：从每个 virtual anchor 的 diagonal Gaussian 中采样伪未知特征，并训练对应 virtual class。

改动文件：

```text
lib/Finetune_library.py
```

采样策略：

```text
random: z = mean + sqrt(var) * eps
low_density: 每个 virtual anchor 先采 lups_candidates 个候选，计算 diagonal Mahalanobis distance，取分数最高的 sample_num[v] 个样本。
```

训练目标：

```python
virtual_targets = torch.ones(num_samples).long().to(device) * (args.known_class + v)
loss += criterion(outputs_global_unknown, virtual_targets) * args.lups_global_weight
loss += args.rank_weight * known_unknown_rank_loss(args, outputs, outputs_global_unknown)
```

注意：

```text
如果 lups_space=pooled 且 pooled size 与模型 layer4 输入兼容，可以继续用 net.discrete_forward(z)。
如果 lups_space=gap，则需要新增 embedding-level head，第一版先不要做 gap。
```

验收：

```text
BloodMNIST pooled=2 可跑通。
RetinalOCT pooled=2 不 OOM。
global unknown loss 开始生效后，UNK/HOS 有提升，同时 AUROC/AUPR/OSCR 不应剧烈下降。
```

### Step 8. 保留 fullcov baseline 分支

目标：BloodMNIST 上仍可复现原始 fullcov FOSS，作为论文对照。

执行：

```text
1. 当 args.lups_mode == 'fullcov' 时，沿用原 cov_dict 和 compute_global_statistic。
2. 当 args.lups_mode == 'diag' 时，使用 LUPS。
3. 大图数据默认禁止或不推荐 fullcov，只作为 OOM/不可扩展证据。
```

验收：

```text
BloodMNIST fullcov 和 diag 都能跑。
RetinalOCT/ISIC 默认使用 diag pooled。
```

### Step 9. 更新训练脚本

目标：让实验脚本显式记录多虚拟类、LUPS 和 ranking 参数。

改动文件：

```text
scripts/train_bloodmnist.sh
scripts/train_OCT.sh
scripts/train_ISIC.sh
```

Finetune 推荐初始参数：

```bash
--virtue_num=3 \
--lups_mode='diag' \
--lups_space='pooled' \
--lups_pool_size=2 \
--lups_local_weight=0.1 \
--lups_global_weight=0.01 \
--rank_weight=0.05 \
--rank_margin=0.2 \
--lups_sample_strategy='low_density' \
--lups_candidates=100
```

调参规则：

```text
如果 closed-set ACC/F1 下降，先降 lups_local_weight 和 lups_global_weight，再降 rank_weight，最后调小 lups_var_scale。
如果 UNK/HOS 低，先升 lups_local_weight，再升 lups_global_weight，最后尝试增大 lups_var_scale。
```

### Step 10. 验证顺序

推荐按下面顺序跑，避免一次性改动后难以定位问题：

```text
1. BloodMNIST seed=1, virtue_num=3, local virtual loss only。
2. BloodMNIST seed=1, local virtual loss + ranking loss。
3. BloodMNIST seed=1, LUPS diag pooled + random sampling。
4. BloodMNIST seed=1, LUPS diag pooled + low_density sampling。
5. BloodMNIST seed=1/42/66，完整对照。
6. RetinalOCT seed=0，diag pooled smoke test。
7. ISIC/HyperKvasir，diag pooled 扩展。
```

每个实验记录：

```text
Closed-set ACC/F1
OSR ACC/F1
UNK
OS*
HOS
AUROC
AUPR
OSCR
peak memory
training time
virtual prediction histogram
```

### Step 11. 预期结果和判断标准

第一版不要求所有指标都最高，优先满足：

```text
1. Finetune 正确继承 Pretrain 的多个 virtual heads。
2. RetinalOCT/ISIC 不再因为 full covariance OOM。
3. LUPS-Diag 的 HOS/OSCR 至少不显著低于 fullcov 或原 Finetune。
4. AUROC/AUPR/OSCR 不再出现明显因 Finetune 导致的大幅下降。
5. 多 seed 下结果方差低于 full covariance 版本。
```

如果出现 ACC/F1/UNK 上升但 AUROC/AUPR/OSCR 下降，优先检查：

```text
1. rank_loss 是否启用。
2. known samples 的 max_known_prob 是否被整体压低。
3. lups_local_weight/lups_global_weight 是否过大。
4. virtual targets 是否集中落到一个 virtual head。
5. LUPS var_scale 是否过大导致采样太远。
```

## 11. 最小可交付版本

如果时间有限，另一个 agent 应优先交付下面四项：

```text
MVP-1: Finetune 模型改为 K + virtue_num，并能完整加载 Pretrain checkpoint。
MVP-2: local i-DUS unknown loss 改为多 virtual target。
MVP-3: 加入 rank_loss，缓解 AUROC/AUPR/OSCR 下降。
MVP-4: 实现 LUPS diag pooled，替代 full covariance，RetinalOCT smoke test 不 OOM。
```

MVP 完成后，再做：

```text
1. low_density sampling ablation。
2. gap space ablation。
3. seed=1/42/66 完整实验。
4. fullcov vs diag 论文表格。
```
