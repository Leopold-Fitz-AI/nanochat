# 第四课：注意力机制详解 - 完整总结

## 📚 课程概览

第四课深入讲解了 Attention 机制的数学原理、Multi-Head 架构和 Causal Mask 实现。

**学习目标**：
- ✅ 理解 Attention 的数学推导
- ✅ 掌握 Multi-Head Attention 的原理
- ✅ 理解 Causal Mask 如何实现自回归生成

---

## Part 1: Attention 机制的数学推导

### 核心公式

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

### 四个关键步骤

**1. 计算相似度：Q @ K^T**
- 点积衡量 Query 和 Key 的相似度
- 矩阵乘法高效，GPU 友好
- 结果：(seq_len, seq_len) 相似度矩阵

**2. 缩放：/ sqrt(d_k)**
- 原因：点积方差随维度增长 Var[score] = d_k
- 效果：归一化方差到 1
- 好处：稳定 Softmax 梯度，防止梯度消失

**3. 归一化：softmax(scores)**
- 转换为概率分布（非负，总和为1）
- 可解释为"注意力权重"
- 可微分，梯度友好

**4. 加权求和：weights @ V**
- 根据注意力权重聚合 Value
- 融合所有相关位置的信息
- 输出每个位置的新表示

### 实验验证：缩放效果

```
维度 |  原始点积方差 | 缩放后方差
-----|--------------|----------
   4 |         3.68 |      0.92
  16 |        15.57 |      0.97
  64 |        62.31 |      0.97
 256 |       278.14 |      1.09
```

→ 缩放后方差稳定在 1 附近！

### 关键洞察

**为什么用点积？**
- 计算高效（矩阵乘法）
- 批量并行
- 数学性质好（可微）

**为什么除以 sqrt(d_k)？**
- 归一化方差到 1
- 稳定 Softmax 梯度
- 防止数值不稳定

**为什么用 Softmax？**
- 转换为概率分布
- 可解释性
- 可微分

---

## Part 2: Multi-Head Attention 详解

### 核心思想

**将高维空间分割成多个低维子空间**：
```
d_model = 768
n_head = 6
d_k = d_model / n_head = 128
```

每个头在自己的子空间中独立学习不同的模式！

### 完整公式

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
```

### 张量形状变化（nanochat 示例）

```
输入 X:         (B, T, 768)
  ↓ 投影
Q, K, V:        (B, T, 768)
  ↓ Reshape
                (B, T, 6, 128)  # 分成 6 个头
  ↓ Transpose
                (B, 6, T, 128)  # Head 维度移到 Batch 旁
  ↓ Attention
                (B, 6, T, 128)  # 每个头独立计算
  ↓ Transpose
                (B, T, 6, 128)
  ↓ Reshape
                (B, T, 768)     # 拼接所有头
  ↓ Linear
输出:           (B, T, 768)
```

### 每个头学到了什么？

真实观察：

**Head 1: 局部依赖**
- 关注相邻词
- 权重集中在对角线附近
- 捕捉短距离语法关系

**Head 2: 长距离依赖**
- 关注远距离词
- 权重分散更均匀
- 捕捉主谓关系、指代关系

**Head 3: 位置敏感**
- 关注特定位置（句首、句尾）
- 捕捉特殊标记信息

**Head 4: 罕见词关注**
- 对低频词更多注意力
- 帮助处理 OOV 和罕见词

### 为什么拼接而不是求平均？

**拼接 + Linear 的优势**：
- ✅ 保留所有信息
- ✅ 可学习的组合（W_O 学习最优组合）
- ✅ 更灵活（不同头可以有不同权重）

**求平均的劣势**：
- ❌ 信息丢失
- ❌ 固定权重（每个头 1/h）
- ❌ 表达能力弱

### Single-Head vs Multi-Head

| 特性 | Single-Head | Multi-Head |
|------|-------------|------------|
| 表达能力 | 有限 | 更强 |
| 参数量 | 较少 | 相同（分配方式不同）|
| 关注模式 | 单一 | 多样化 |
| 计算效率 | 略快 | 高度并行 |
| 可解释性 | 简单 | 可分析不同头 |

### 最佳实践

**头数选择**：
- 小模型: 4-8 头
- 中等模型: 8-12 头
- 大模型: 12-16 头

**每个头的维度**：
- 通常 d_k = 64 或 128
- 太小: 表达能力不足
- 太大: 失去专业化

**nanochat 的选择**：
- n_head = 6
- head_dim = 128
- 平衡效率和表达能力

---

## Part 3: Causal Attention 和 Masked Attention

### 什么是自回归生成？

**生成过程**：
```
时刻 1: "The cat" → 预测 "sat"
时刻 2: "The cat sat" → 预测 "on"
时刻 3: "The cat sat on" → 预测 "the"
时刻 4: "The cat sat on the" → 预测 "mat"
```

**关键约束**：
- ❗ 预测位置 t 时，只能用位置 0 到 t-1 的信息
- ❗ 不能"偷看"未来！

### Causal Mask 实现

**核心思想**：
```
将未来位置的 score 设为 -inf
→ Softmax 后未来位置权重为 0
```

**Causal Mask 矩阵**（下三角）：
```
     t0  t1  t2  t3  t4
t0   ✓   ✗   ✗   ✗   ✗
t1   ✓   ✓   ✗   ✗   ✗
t2   ✓   ✓   ✓   ✗   ✗
t3   ✓   ✓   ✓   ✓   ✗
t4   ✓   ✓   ✓   ✓   ✓
```

### 手工计算示例

**原始相似度**：
```
     t0     t1     t2     t3
t0  0.884  0.707  0.247  0.707
t1  0.707  0.884  0.283  0.566
t2  0.247  0.283  0.092  0.198
t3  0.707  0.566  0.198  0.566
```

**应用 Mask 后**：
```
     t0     t1     t2     t3
t0  0.884   -inf   -inf   -inf
t1  0.707  0.884   -inf   -inf
t2  0.247  0.283  0.092   -inf
t3  0.707  0.566  0.198  0.566
```

**Softmax 归一化**：
```
     t0     t1     t2     t3
t0  1.000  0.000  0.000  0.000
t1  0.456  0.544  0.000  0.000
t2  0.346  0.358  0.296  0.000
t3  0.300  0.260  0.180  0.260
```

→ 未来位置权重自动为 0！

### 不同类型的 Mask

**1. Causal Mask**：
- 用于 GPT 等自回归模型
- 规则：只能看到过去和当前
- 形状：下三角矩阵

**2. Padding Mask**：
- 用于处理变长序列
- 规则：屏蔽 padding 位置
- 例子：["hello", "world", <PAD>, <PAD>]

**3. Combined Mask**：
- Causal + Padding 的组合
- 用于复杂场景

### nanochat 实现

**训练时（序列完整）**：
```python
y = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,  # 自动应用 Causal Mask
    enable_gqa=enable_gqa
)
```

**推理时（单 token）**：
```python
y = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=False,  # 已生成的都可见
    enable_gqa=enable_gqa
)
```

**推理时（多 token + KV Cache）**：
```python
# 手动构造 attn_mask
attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool)
prefix_len = Tk - Tq
if prefix_len > 0:
    attn_mask[:, :prefix_len] = True  # Prefix 可见
# Chunk 内部使用 Causal Mask
attn_mask[:, prefix_len:] = torch.tril(
    torch.ones((Tq, Tq), dtype=torch.bool)
)
```

### 为什么 Causal Mask 有效？

**信息论视角**：
- 强制模型学习 P(x_t | x_0, ..., x_{t-1})
- 而不是 P(x_t | x_0, ..., x_n)（包含未来）
- 符合语言生成的因果性

**防止信息泄漏**：
- 训练时不能"作弊"
- 学习真正的序列建模能力
- 训练和推理分布一致

**梯度流动**：
- 梯度只来自未来位置
- 符合序列生成的因果关系
- 帮助学习正确的依赖结构

---

## 🎯 第四课总结

### 核心要点

**Part 1: Attention 数学原理**
- ✅ Attention 是"软查询"机制
- ✅ 点积 → 缩放 → Softmax → 加权和
- ✅ 每一步都有明确的数学动机

**Part 2: Multi-Head Attention**
- ✅ 多个头在不同子空间并行学习
- ✅ 每个头专注于不同模式
- ✅ 拼接 + Linear 提供可学习的组合

**Part 3: Causal Attention**
- ✅ Mask 通过 -inf 屏蔽未来信息
- ✅ 保证训练和推理分布一致
- ✅ 自回归生成的核心机制

### 数学本质

```
# 完整的 Causal Multi-Head Attention
MultiHead(Q, K, V, Mask) = Concat(head_1, ..., head_h) @ W_O

head_i = softmax((Q @ W_Q^i @ (K @ W_K^i)^T + Mask) / sqrt(d_k)) @ V @ W_V^i

Mask = {
    Causal: 下三角 = 0, 上三角 = -inf
    Padding: valid = 0, padding = -inf
    None: 全部 = 0
}
```

### 关键技术点

**1. 缩放的重要性**
- Var[score] = d_k（未缩放）
- Var[score / sqrt(d_k)] = 1（缩放后）
- 稳定 Softmax 梯度

**2. 多头的优势**
- 并行学习不同模式
- 不增加参数量（重新分配）
- 提升表达能力

**3. Causal 的必要性**
- 符合语言生成过程
- 防止 Exposure Bias
- 训练/推理一致性

**4. 高效实现**
- 批量投影（所有头一起）
- 并行计算（GPU 加速）
- Flash Attention 优化

### 工程实践

**nanochat 的选择**：
```python
n_head = 6          # 6 个头
head_dim = 128      # 每个头 128 维
d_model = 768       # 总维度 768
```

**PyTorch 优化**：
```python
F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,
    enable_gqa=True
)
```

### 学习成果检查

- [ ] 能够推导 Attention 的完整公式
- [ ] 理解为什么要除以 sqrt(d_k)
- [ ] 解释 Multi-Head 的工作原理
- [ ] 手工计算 Masked Attention 的例子
- [ ] 理解不同类型的 Mask 及其应用
- [ ] 能够实现简单的 Attention 层

---

## 📖 参考资源

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化教程
- [Visualizing Attention in Transformer Models](https://www.youtube.com/watch?v=_UVfwBqcnbM) - 注意力可视化
- [Flash Attention](https://arxiv.org/abs/2205.14135) - 高效实现

---

## 🚀 下一步

**第五课：位置编码与 RoPE**

我们将学习：
1. 为什么 Attention 需要位置信息？
2. 绝对位置编码 vs 相对位置编码
3. RoPE 的数学推导
4. RoPE 的外推性能和长度泛化

继续深入！🎓
