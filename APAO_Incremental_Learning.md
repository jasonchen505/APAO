# APAO 复现过程增量学习笔记

> 本文档记录在复现过程中对比前两轮（代码分析、面试准备）新学习到的点

---

## 第一阶段：环境搭建 & 单实验验证

### 新学习点 1: 模型规模与资源匹配

**前两轮认知**：
- 知道 APAO 使用 T5 和 Llama 作为 backbone
- 知道模型配置参数（d_model=128, num_layers=4）

**本轮新认知**：
- 实际计算参数量：T5 ~5M, Llama ~10M
- 单卡 3090 (24GB) 完全够用，甚至可以跑更大 batch
- 8 卡的价值在于**并行实验**，而不是单实验加速

**实践意义**：
```
资源规划思路：
├── 小模型 + 多卡 → 并行超参搜索
├── 大模型 + 多卡 → 数据并行 / 模型并行
└── 本项目：小模型，所以用并行实验策略
```

---

### 新学习点 2: 确定性训练的工程细节

**前两轮认知**：
- 知道 `set_seed()` 函数设置随机种子
- 知道要保证可复现性

**本轮新认知**：
```python
# utils/utils.py 中的确定性设置
torch.backends.cudnn.benchmark = False      # 禁用 cuDNN 自动调优
torch.backends.cudnn.deterministic = True   # 使用确定性算法
torch.backends.cudnn.enabled = False        # 完全禁用 cuDNN
torch.use_deterministic_algorithms(True, warn_only=False)  # PyTorch 级别确定性
```

**为什么需要这些**：
- `benchmark=True` 会根据输入大小自动选择最快算法，但可能不确定
- `deterministic=True` 强制使用确定性算法，可能慢一点但可复现
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` 控制 cuBLAS 工作空间，保证确定性

**实践意义**：
- 学术复现：必须开启确定性
- 工程部署：可以关闭，追求速度

---

### 新学习点 3: 数据处理的完整流程

**前两轮认知**：
- 知道物品被编码为 4 个 token（如 `<a_159><b_42><c_1><d_0>`）
- 知道使用 RQ-KMeans 进行编码

**本轮新认知**：
完整数据流：
```
原始数据 (item_id) 
  → RQ-KMeans 编码 (4个token) 
  → Tokenizer 编码 (token_ids) 
  → 拼接成序列 
  → 模型输入
```

关键细节：
```python
# utils/data.py 中的 _remap_items
def _remap_items(self):
    self.remapped_inters = dict()
    for uid, items in self.inters.items():
        # 将 item_id 映射为 token 字符串
        new_items = ["".join(self.indices[str(i)]) for i in items]
        self.remapped_inters[uid] = new_items

# 例如：item_id=0 → "<a_159><b_42><c_1><d_0>"
```

**训练数据构造**：
```python
# _process_train_data
for i in range(1, len(items)):
    one_data["item"] = items[i]           # 目标物品
    one_data["inters"] = "".join(history)  # 历史序列拼接
```

---

## 第二阶段：Baseline 复现

### 新学习点 4: CE vs MSL 的实现差异

**前两轮认知**：
- CE 是标准交叉熵
- MSL 是 Masked Softmax Loss
- 知道 MSL 在训练时对无效 token 进行 mask

**本轮新认知**：

CE 实现（`src/models/T5_CE.py`）：
```python
# 标准交叉熵，无特殊处理
loss_fct = CrossEntropyLoss(ignore_index=-100)
loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
```

MSL 实现（`src/models/T5_MSL.py`）：
```python
# 训练时：使用 constrain_mask 过滤无效 token
if constrain_mask is not None:
    lm_logits[constrain_mask == 0] = -float("inf")

# 推理时：使用 temperature scaling
else:
    lm_logits = lm_logits / self.ce_tau
```

**关键区别**：
- CE：训练和推理都用标准 softmax
- MSL：训练时 mask 无效 token，推理时用温度缩放
- APAO：在 CE 基础上加前缀损失

---

### 新学习点 5: 约束 Beam Search 的 Trie 实现细节

**前两轮认知**：
- 知道使用 Trie 结构约束解码
- 知道 `prefix_allowed_tokens_fn` 的作用

**本轮新认知**：

Trie 构建（`utils/generation_trie.py`）：
```python
class Trie:
    def __init__(self, sequences):
        self.trie_dict = {}
        for sequence in sequences:
            # 递归构建字典树
            Trie._add_to_trie(sequence, self.trie_dict)
    
    def get(self, prefix_sequence):
        # 返回当前前缀下所有可能的下一个 token
        return Trie._get_from_trie(prefix_sequence, self.trie_dict)
```

约束函数（`utils/utils.py`）：
```python
def prefix_allowed_tokens_fn(candidate_trie):
    def prefix_allowed_tokens(batch_id, sentence):
        sentence = sentence.tolist()
        trie_out = candidate_trie.get(sentence)
        return trie_out
    return prefix_allowed_tokens
```

**在 generate 中的使用**：
```python
output = model.generate(
    input_ids=inputs["input_ids"],
    prefix_allowed_tokens_fn=prefix_allowed_tokens,  # 约束函数
    num_beams=20,
    ...
)
```

**工程细节**：
- Trie 在测试前预构建，包含所有有效物品编码
- 每步解码时，查询 Trie 获取允许的 token 列表
- 这保证生成的序列一定是有效物品

---

## 第三阶段：APAO Pointwise 深入

### 新学习点 6: 前缀损失的完整计算流程

**前两轮认知**：
- 知道公式：`prefix_loss_t = -1/t * Σ log P(y_i | y_{0:i-1}, x)`
- 知道要除以 t 进行归一化

**本轮新认知**：

完整代码（`src/models/T5_APAO_Pointwise.py`）：
```python
T = labels.shape[1]  # 物品 token 长度（这里是 4）

# 计算每个 token 的对数概率
log_probs = F.log_softmax(t_logits, dim=-1)  # (B, T, V)
token_logp = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)  # (B, T)

# 计算每个前缀长度的损失
prefix_losses = []
for t in range(1, T):  # t = 1, 2, 3
    prefix_logp = (token_logp[:, :t]).sum(dim=1) / t  # (B,)
    prefix_loss_t = (-prefix_logp).mean()  # (B,)
    prefix_losses.append(prefix_loss_t)
```

**关键洞察**：
- T=4，所以有 3 个前缀长度（t=1,2,3）
- 每个前缀的损失是独立计算的
- 最终损失是加权求和

**为什么是 t in range(1, T)**：
- t=0 没有意义（空前缀）
- t=T 是完整的物品，已经包含在 CE 损失中
- 所以只优化中间的前缀

---

### 新学习点 7: 自适应权重的实现细节

**前两轮认知**：
- 知道使用 EMA 更新权重
- 知道公式：`w_new = w_old * exp(η * normalized_loss)`

**本轮新认知**：

完整代码：
```python
# 收集所有前缀损失
all_prefix_losses = torch.stack(prefix_losses)  # (T-1,)
all_prefix_losses = torch.nan_to_num(all_prefix_losses, nan=0.0, posinf=0.0, neginf=0.0)
base_losses = all_prefix_losses.detach()  # 不参与梯度计算

# 自适应权重更新
if self.last_w is None:
    # 第一步：根据损失初始化权重
    w = base_losses / base_losses.sum()
else:
    # 后续步骤：EMA 更新
    normed_base_losses = base_losses / base_losses.sum()
    weighted_w = self.last_w * torch.exp(self.eta_adapt * normed_base_losses)
    w = weighted_w / weighted_w.sum()

self.last_w = w  # 保存当前权重

# 最终损失
weighted_prefix_loss = (w * all_prefix_losses).sum()
loss = softmax_ce_loss + self.beta * weighted_prefix_loss
```

**关键细节**：
1. `base_losses.detach()`：权重更新不参与梯度计算
2. `nan_to_num`：防止 NaN 传播
3. `self.last_w`：跨 batch 保存权重状态

**为什么 detach**：
- 权重是超参数级别的，不应该影响梯度
- 如果不 detach，权重更新会引入额外的梯度

---

### 新学习点 8: Llama 版本的特殊处理

**前两轮认知**：
- 知道 Llama 是 Decoder-Only 架构
- 知道需要特殊处理输入输出

**本轮新认知**：

Llama 的前向传播（`src/models/Llama_APAO_Pointwise.py`）：
```python
# 1. 编码历史序列（使用 KV Cache）
history_outputs = self.model(
    input_ids=input_ids,
    use_cache=True,  # 启用 KV Cache
    ...
)
history_past_key_values = history_outputs.past_key_values

# 2. 编码目标物品（复用历史的 KV Cache）
outputs = self.model(
    input_ids=labels,
    past_key_values=history_past_key_values,  # 复用
    ...
)

# 3. 拼接隐藏状态
hidden_states = torch.cat([
    history_hidden_states[:, -1:, :],  # 历史最后一步
    outputs.last_hidden_state[:, -T:-1, :]  # 目标前 T-1 步
], dim=1)
```

**为什么这样处理**：
- Decoder-Only 需要自回归生成
- 历史序列只需要编码一次
- 目标物品的每个位置需要看到之前的 token

**与 T5 的区别**：
- T5：Encoder 编码历史，Decoder 生成目标
- Llama：统一用 Decoder，通过 KV Cache 复用历史

---

## 第四阶段：APAO Pairwise 深入

### 新学习点 9: 负采样的工程实现

**前两轮认知**：
- 知道需要采样负样本
- 知道使用均匀采样

**本轮新认知**：

负采样数据结构（`utils/data.py`）：
```python
class NegSampleOnEpochDataset(SeqRecDataset):
    def __init__(self, ...):
        # 预分配负样本矩阵
        self.inter_negs = self._rng.integers(
            low=0, high=self.n_items,
            size=(self.N_inters, self.neg_k),  # (交互数, 负样本数)
            dtype=np.int64
        )
    
    def resample_negs(self, generator=None, seed=None):
        # 每个 epoch 重新采样
        self.inter_negs = self._rng.integers(...)
```

负样本重采样回调（`utils/callbacks.py`）：
```python
class ResampleNegsCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, **kwargs):
        # 每个 epoch 开始时重新采样负样本
        self.train_dataset.resample_negs(generator=train_generator)
        self.valid_dataset.resample_negs(generator=valid_generator)
```

**为什么每 epoch 重采样**：
- 增加训练多样性
- 避免模型过拟合到固定的负样本
- 提供更丰富的对比信号

---

### 新学习点 10: 前缀掩码的实现细节

**前两轮认知**：
- 知道需要 mask 掉与正样本共享前缀的负样本
- 知道目的是减少噪声

**本轮新认知**：

前缀掩码构建（`utils/collator.py`）：
```python
# 为每个前缀长度构建掩码
labels_with_negs_prefix_mask = []
for t in range(1, T + 1):
    t_negs_prefix_mask = torch.zeros((B, neg_k + 1), dtype=torch.bool)
    for ii in range(B):
        seen_prefixes = set()
        pos_prefix = inputs['labels'][ii, :t].cpu().tolist()
        seen_prefixes.add(tuple(pos_prefix))
        for j in range(1, neg_k + 1):
            neg_prefix = tuple(all_input_ids[ii][j, :t].cpu().tolist())
            if neg_prefix in seen_prefixes:
                t_negs_prefix_mask[ii, j] = True  # 标记为需要 mask
            else:
                seen_prefixes.add(tuple(neg_prefix))
    labels_with_negs_prefix_mask.append(t_negs_prefix_mask)
```

在损失计算中的使用：
```python
if labels_with_negs_prefix_mask is not None:
    t_prefix_mask = labels_with_negs_prefix_mask[t-1].to(score_matrix.device)
    score_matrix = score_matrix.masked_fill(t_prefix_mask, float('-inf'))
```

**关键细节**：
- 掩码是逐前缀长度构建的
- 使用 `float('-inf')` 而不是 0，因为后面要 exp
- 被 mask 的负样本不参与 softmax 计算

---

### 新学习点 11: 负样本缩放的作用

**前两轮认知**：
- 知道有负样本缩放机制
- 知道目的是考虑有效前缀数量差异

**本轮新认知**：

缩放因子计算：
```python
if self.valid_prefix_num is not None:
    # valid_prefix_num[t-1] 是第 t 个前缀位置的有效前缀总数
    # (~t_prefix_mask).sum(dim=-1) 是当前 batch 中未被 mask 的负样本数
    neg_scale_factor = ((self.valid_prefix_num[t-1]-1) / (~t_prefix_mask).sum(dim=-1))
    neg_probs = neg_probs * neg_scale_factor.unsqueeze(-1)
```

**为什么需要缩放**：
- 不同前缀位置的有效负样本数量不同
- 如果不缩放，有效负样本少的位置会被低估
- 缩放后，损失的量级更一致

**valid_prefix_num 的计算**：
```python
# 在 train_apao.py 中
all_items_texts = train_data.get_all_items()
all_items = [tokenizer.encode(candidate) for candidate in all_items_texts]
T = all_items[0].__len__()
prefix_set = [set() for _ in range(T-1)]
for _item in all_items:
    for t in range(1, T):
        prefix = tuple(_item[:t])
        prefix_set[t-1].add(prefix)
valid_prefix_num = [len(prefix_set[t]) for t in range(T-1)]
```

---

## 第五阶段：最终复现 & 总结

### 新学习点 12: 超参搜索的并行策略

**前两轮认知**：
- 知道需要搜索 β 和 η
- 知道有 30 种组合

**本轮新认知**：

实际并行策略：
```
8 卡并行，每卡跑一组超参
├── Batch 1: GPU 0-7，跑 8 组
├── Batch 2: GPU 0-7，跑 8 组
├── Batch 3: GPU 0-7，跑 8 组
└── Batch 4: GPU 0-5，跑 6 组
总计：30 组，4 批完成
```

资源利用：
- 每组实验只用 1 卡，显存占用 < 2GB
- 8 卡可以同时跑 8 组
- 总时间从 30 x 3小时 = 90小时 缩短到 4 x 3小时 = 12小时

---

### 新学习点 13: 结果收集与分析

**前两轮认知**：
- 知道要收集测试结果
- 知道指标是 Hit@K 和 NDCG@K

**本轮新认知**：

结果文件结构：
```
results/
└── Office_T5_APAO_Pointwise/
    └── T5_APAO_Pointwise_seed42_beta0.3_eta1e-4/
        └── final_beam_search_results.json
```

结果内容：
```json
{
    "final_results": {
        "hit@10": 0.07,
        "hit@20": 0.12,
        "ndcg@10": 0.05,
        "ndcg@20": 0.06
    }
}
```

分析脚本：
```python
import json
import glob

results = {}
for f in glob.glob("./results/*/final_beam_search_results.json"):
    with open(f) as fp:
        data = json.load(fp)
    name = f.split("/")[2]  # 提取实验名
    results[name] = data["final_results"]

# 按 NDCG@10 排序
sorted_results = sorted(results.items(), key=lambda x: x[1]["ndcg@10"], reverse=True)
```

---

### 新学习点 14: 训练监控的工程实践

**前两轮认知**：
- 知道要监控训练过程
- 知道使用 wandb

**本轮新认知**：

自定义 Trainer 的日志（`utils/custom_trainer.py`）：
```python
def _maybe_log_save_evaluate(self, ...):
    logs["loss"] = round(tr_loss_scalar / steps, 4)
    
    # 记录 APAO 特有的指标
    if hasattr(model, "softmax_ce_loss"):
        logs["softmax_ce_loss"] = model.softmax_ce_loss.item()
    if hasattr(model, "weighted_prefix_loss"):
        logs["weighted_prefix_loss"] = model.weighted_prefix_loss.item()
    if hasattr(model, "prefix_losses"):
        for i, l in enumerate(model.prefix_losses):
            logs[f"prefix_loss_{i+1}"] = l.item()
    if hasattr(model, "adaptive_weights"):
        for i, w in enumerate(model.adaptive_weights):
            logs[f"adaptive_weight_{i+1}"] = w
```

**监控要点**：
- 总损失是否下降
- CE 损失和前缀损失的比例
- 自适应权重的变化趋势
- 验证集指标是否提升

---

## 总结：三轮学习的认知递进

| 阶段 | 认知层次 | 重点 |
|------|----------|------|
| 第一轮：代码分析 | 理解"是什么" | 代码结构、算法流程 |
| 第二轮：面试准备 | 理解"为什么" | 设计原因、局限性、改进方向 |
| 第三轮：复现实践 | 理解"怎么做" | 工程细节、调试技巧、资源管理 |

### 关键收获

1. **理论到实践的鸿沟**：
   - 理论上简单的东西，工程实现有很多细节
   - 例如：前缀掩码需要逐位置构建，负采样需要每 epoch 重采样

2. **资源管理的重要性**：
   - 8 卡不是用来加速单个实验，而是并行多个实验
   - 超参搜索的时间成本需要提前规划

3. **可复现性的代价**：
   - 确定性训练会牺牲一些速度
   - 需要记录所有随机种子和超参数

4. **监控的价值**：
   - 不仅要看最终指标，还要看训练过程
   - 自适应权重的变化可以反映模型学习状态

---

## 待补充（后续复现过程中继续更新）

- [ ] 实际训练时间记录
- [ ] 遇到的具体问题和解决方案
- [ ] 与论文结果的对比分析
- [ ] 改进尝试的结果
