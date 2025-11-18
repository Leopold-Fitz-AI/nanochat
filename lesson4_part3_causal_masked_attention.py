"""
第四课 Part 3: Causal Attention 和 Masked Attention
理解自回归生成的核心机制
"""

print("=" * 70)
print("第四课 Part 3: Causal Attention 和 Masked Attention")
print("=" * 70)
print()

print("""
📖 本节目标：

到目前为止，我们学习的 Attention 允许每个位置看到所有其他位置。
但对于自回归生成（如 GPT），我们需要：
1. 只能看到过去，不能看到未来（Causal）
2. 如何实现这种"因果性"？（Mask）
3. 不同类型的 Mask 及其应用

这是理解 GPT 生成机制的关键！
""")
print()

# ============================================================================
# 1. 什么是自回归生成？
# ============================================================================

print("=" * 70)
print("1️⃣  什么是自回归生成（Autoregressive Generation）")
print("=" * 70)
print()

print("""
【场景】GPT 生成文本

输入: "The cat"
期望输出: "sat on the mat"

生成过程：
  时刻 1: "The cat" → 预测 "sat"
  时刻 2: "The cat sat" → 预测 "on"
  时刻 3: "The cat sat on" → 预测 "the"
  时刻 4: "The cat sat on the" → 预测 "mat"

关键约束：
❗在预测位置 t 的词时，只能使用位置 0 到 t-1 的信息
❗不能"偷看"未来的词！

这就是"因果性"（Causality）要求。
""")
print()

print("【为什么需要因果性？】")
print("-" * 70)
print("""
如果允许看到未来：

训练时：
  输入: "The cat [MASK] on the mat"
  模型可以看到 "on the mat"
  很容易预测 [MASK] 是 "sat"

推理时：
  输入: "The cat"
  没有未来的词可看
  模型表现会很差

训练和推理的分布不一致！
这叫做 Exposure Bias（暴露偏差）

Causal Attention 解决这个问题：
✅ 训练时也不让模型看到未来
✅ 训练和推理保持一致
✅ 学习真正的序列建模能力
""")
print()

# ============================================================================
# 2. Causal Mask：如何屏蔽未来信息
# ============================================================================

print("=" * 70)
print("2️⃣  Causal Mask：如何屏蔽未来信息")
print("=" * 70)
print()

print("""
【回忆 Attention 计算】

scores = Q @ K^T / sqrt(d_k)  # (seq_len, seq_len)
weights = softmax(scores)      # (seq_len, seq_len)
output = weights @ V           # (seq_len, d_v)

关键：scores 矩阵
  scores[i, j] = 位置 i 对位置 j 的相似度

如果不加限制：
  scores[2, 3] 有值 → 位置 2 可以看到位置 3（未来）❌

Causal Mask 的做法：
  将未来位置的 scores 设为 -inf
  scores[i, j] = -inf  (当 j > i 时)

Softmax 后：
  exp(-inf) = 0
  → 未来位置的权重为 0 ✅
""")
print()

print("【Causal Mask 矩阵可视化】")
print("-" * 70)

seq_len = 5
print(f"序列长度: {seq_len}")
print()

# 创建 Causal Mask
print("Causal Mask (下三角矩阵，True=保留，False=屏蔽):")
print("-" * 70)
print("      ", end="")
for j in range(seq_len):
    print(f"  t{j}  ", end="")
print()
print("-" * (8 + seq_len * 7))

for i in range(seq_len):
    print(f"  t{i}  ", end="")
    for j in range(seq_len):
        if j <= i:
            print("  ✓   ", end="")
        else:
            print("  ✗   ", end="")
    print()
print()

print("💡 解读:")
print("  - 对角线及以下: ✓ (可以看到)")
print("  - 对角线以上: ✗ (屏蔽未来)")
print("  - t2 只能看到 t0, t1, t2，看不到 t3, t4")
print()

# ============================================================================
# 3. 手工计算 Masked Attention
# ============================================================================

print("=" * 70)
print("3️⃣  手工计算 Masked Attention")
print("=" * 70)
print()

import random
random.seed(42)

# 简化数据
seq_len = 4
d_k = 2

Q = [
    [1.0, 0.5],
    [0.5, 1.0],
    [0.2, 0.3],
    [0.8, 0.4],
]

K = [
    [1.0, 0.5],
    [0.5, 1.0],
    [0.2, 0.3],
    [0.8, 0.4],
]

V = [
    [1.0, 0.0],
    [0.0, 1.0],
    [0.5, 0.5],
    [0.3, 0.7],
]

def dot_product(a, b):
    return sum(ai * bi for ai, bi in zip(a, b))

def softmax(scores):
    max_score = max(scores)
    exp_scores = [2.71828 ** (s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    return [e / sum_exp for e in exp_scores]

print("【步骤 1: 计算相似度矩阵】")
print("-" * 70)

# 计算 Q @ K^T
scores = []
for q in Q:
    row = []
    for k in K:
        score = dot_product(q, k) / (d_k ** 0.5)
        row.append(score)
    scores.append(row)

print("Q @ K^T / sqrt(d_k):")
print("      ", end="")
for j in range(seq_len):
    print(f"   t{j}   ", end="")
print()
print("-" * (8 + seq_len * 9))

for i, row in enumerate(scores):
    print(f"  t{i}  ", end="")
    for score in row:
        print(f" {score:6.3f}  ", end="")
    print()
print()

print("【步骤 2: 应用 Causal Mask】")
print("-" * 70)

# 应用 mask: 将未来位置设为 -inf
masked_scores = []
for i, row in enumerate(scores):
    masked_row = []
    for j, score in enumerate(row):
        if j <= i:
            masked_row.append(score)
        else:
            masked_row.append(float('-inf'))  # 屏蔽未来
    masked_scores.append(masked_row)

print("Masked scores (未来位置 = -inf):")
print("      ", end="")
for j in range(seq_len):
    print(f"   t{j}   ", end="")
print()
print("-" * (8 + seq_len * 9))

for i, row in enumerate(masked_scores):
    print(f"  t{i}  ", end="")
    for score in row:
        if score == float('-inf'):
            print("  -inf  ", end="")
        else:
            print(f" {score:6.3f}  ", end="")
    print()
print()

print("【步骤 3: Softmax 归一化】")
print("-" * 70)

# Softmax（会自动处理 -inf）
attention_weights = []
for i, row in enumerate(masked_scores):
    # 只对非 -inf 的部分做 softmax
    valid_scores = [s for s in row if s != float('-inf')]
    valid_weights = softmax(valid_scores)

    # 构造完整的权重（未来位置权重为 0）
    weights_row = []
    valid_idx = 0
    for j in range(seq_len):
        if j <= i:
            weights_row.append(valid_weights[valid_idx])
            valid_idx += 1
        else:
            weights_row.append(0.0)  # 未来位置权重 = 0
    attention_weights.append(weights_row)

print("Attention weights (未来位置 = 0):")
print("      ", end="")
for j in range(seq_len):
    print(f"   t{j}   ", end="")
print()
print("-" * (8 + seq_len * 9))

for i, row in enumerate(attention_weights):
    print(f"  t{i}  ", end="")
    for weight in row:
        print(f" {weight:6.4f} ", end="")
    print(f"  (sum={sum(row):.4f})")
print()

print("💡 观察:")
print("  - 每一行的权重总和 = 1.0")
print("  - 对角线以上的权重 = 0（未来被屏蔽）")
print("  - t0 只能看到自己")
print("  - t3 可以看到 t0, t1, t2, t3")
print()

print("【步骤 4: 加权求和】")
print("-" * 70)

# weights @ V
output = []
for weights_row in attention_weights:
    out_vec = [0.0] * d_k
    for w, v in zip(weights_row, V):
        for i in range(d_k):
            out_vec[i] += w * v[i]
    output.append(out_vec)

print("Output (加权求和):")
for i, vec in enumerate(output):
    print(f"  t{i}: {[f'{x:.4f}' for x in vec]}")
print()

# ============================================================================
# 4. 不同类型的 Attention Mask
# ============================================================================

print("=" * 70)
print("4️⃣  不同类型的 Attention Mask")
print("=" * 70)
print()

print("""
除了 Causal Mask，还有其他类型的 Mask：

1. Causal Mask（因果掩码）
   - 用于: GPT 等自回归模型
   - 规则: 只能看到过去和当前
   - 形状: 下三角矩阵

2. Padding Mask（填充掩码）
   - 用于: 处理变长序列
   - 规则: 屏蔽 padding 位置
   - 例子: ["hello", "world", <PAD>, <PAD>]
           → 屏蔽后两个位置

3. Look-ahead Mask（前瞻掩码）
   - 用于: Transformer 解码器
   - 规则: Causal Mask + Padding Mask 的组合

4. Cross-Attention Mask
   - 用于: 编码器-解码器架构
   - 规则: 解码器可以看到编码器的所有位置

【可视化对比】

假设序列: ["The", "cat", "sat", <PAD>]

Causal Mask:
    The  cat  sat  PAD
The  ✓   ✗   ✗   ✗
cat  ✓   ✓   ✗   ✗
sat  ✓   ✓   ✓   ✗
PAD  ✓   ✓   ✓   ✓

Padding Mask:
    The  cat  sat  PAD
The  ✓   ✓   ✓   ✗
cat  ✓   ✓   ✓   ✗
sat  ✓   ✓   ✓   ✗
PAD  ✗   ✗   ✗   ✗

Combined (Causal + Padding):
    The  cat  sat  PAD
The  ✓   ✗   ✗   ✗
cat  ✓   ✓   ✗   ✗
sat  ✓   ✓   ✓   ✗
PAD  ✗   ✗   ✗   ✗
""")
print()

# ============================================================================
# 5. nanochat 中的实现
# ============================================================================

print("=" * 70)
print("5️⃣  nanochat 中的 Causal Attention 实现")
print("=" * 70)
print()

print("""
【PyTorch 的高效实现】

在 nanochat/gpt.py 中：

# 训练时（序列完整）
y = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,  # 自动应用 Causal Mask
    enable_gqa=enable_gqa
)

# 推理时（KV Cache，单个 query）
y = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=False,  # 已经生成的都可以看到
    enable_gqa=enable_gqa
)

# 推理时（KV Cache，多个 query）
attn_mask = torch.zeros((Tq, Tk), dtype=torch.bool)
prefix_len = Tk - Tq
if prefix_len > 0:
    attn_mask[:, :prefix_len] = True  # 可以看到 prefix
# Causal mask 在 chunk 内部
attn_mask[:, prefix_len:] = torch.tril(
    torch.ones((Tq, Tq), dtype=torch.bool)
)
y = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attn_mask,
    enable_gqa=enable_gqa
)

【三种情况详解】

情况 1: 训练时 (Tq == Tk)
  - 完整序列，使用标准的 Causal Mask
  - is_causal=True 自动处理

情况 2: 推理时单token (Tq == 1)
  - 当前 query 可以看到所有已生成的 token
  - is_causal=False（不需要 mask）

情况 3: 推理时多token (1 < Tq < Tk)
  - Prefix 部分：全部可见
  - Chunk 内部：Causal Mask
  - 需要手动构造 attn_mask

示例：

假设 KV Cache 中已有 5 个 token，现在输入 3 个新 token：

Tk = 8 (总共 8 个 key)
Tq = 3 (3 个新 query)
prefix_len = 5

Attention Mask:
         k0  k1  k2  k3  k4  k5  k6  k7
    q5   ✓   ✓   ✓   ✓   ✓   ✓   ✗   ✗   (看到所有 prefix + 自己)
    q6   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✗   (prefix + q5, q6)
    q7   ✓   ✓   ✓   ✓   ✓   ✓   ✓   ✓   (prefix + 所有)
         └─────────────┘   └─────────┘
            Prefix          Causal
""")
print()

# ============================================================================
# 6. 为什么 Causal Mask 有效？
# ============================================================================

print("=" * 70)
print("6️⃣  为什么 Causal Mask 有效？")
print("=" * 70)
print()

print("""
【信息论视角】

Causal Mask 强制模型学习：
  P(x_t | x_0, x_1, ..., x_{t-1})

而不是：
  P(x_t | x_0, x_1, ..., x_{t-1}, x_{t+1}, ..., x_n)

这与语言的生成过程一致：
✅ 人类写作时也是从左到右
✅ 下一个词只依赖于已经写的词
✅ 训练和推理分布一致

【防止信息泄漏】

如果允许看到未来：
  Input:  "The cat ? on the mat"
  Model sees: "mat" → easily predicts "sat"

这是作弊！模型学到的是：
  "看到 mat → 填入 sat"

而不是真正理解语言结构。

Causal Mask 防止这种捷径：
  Model only sees: "The cat"
  Must learn: 什么动词合理？
  → 学到真正的语言模型

【梯度流动】

Causal Mask 还影响梯度传播：
- 每个位置的梯度只来自未来位置
- 这符合序列生成的因果关系
- 帮助模型学习正确的依赖结构
""")
print()

# ============================================================================
# 7. 实战：Causal vs Non-Causal 的区别
# ============================================================================

print("=" * 70)
print("7️⃣  对比实验：Causal vs Non-Causal Attention")
print("=" * 70)
print()

print("""
【实验设置】

序列: ["The", "cat", "sat"]
任务: 在位置 1 预测 "cat"

Non-Causal Attention (可以看到未来):
  位置 1 看到: ["The", "cat", "sat"]
  注意力权重: [0.2, 0.5, 0.3]
  → 模型知道后面是 "sat"
  → 容易推断出 "cat"

Causal Attention (不能看到未来):
  位置 1 只看到: ["The", "cat"]
  注意力权重: [0.4, 0.6, 0.0]  # "sat" 被屏蔽
  → 模型必须从 "The" 推断
  → 更难，但学到的是真实的生成能力

训练效果:
- Non-Causal: 训练快，但推理差
- Causal: 训练稍慢，但推理好

GPT 选择 Causal，因为：
✅ 推理性能才是最终目标
✅ 训练和推理保持一致
""")
print()

# ============================================================================
# 8. 总结
# ============================================================================

print("=" * 70)
print("8️⃣  总结")
print("=" * 70)
print()

print("""
【关键要点】

1. Causal Attention 的核心
   ✅ 只能看到过去，不能看到未来
   ✅ 通过 Mask 将未来位置的 score 设为 -inf
   ✅ Softmax 后未来位置的权重自动为 0

2. 为什么需要 Causal
   ✅ 符合语言生成的因果性
   ✅ 训练和推理分布一致
   ✅ 防止信息泄漏
   ✅ 学习真正的序列建模能力

3. 不同类型的 Mask
   - Causal Mask: 自回归生成
   - Padding Mask: 变长序列
   - Combined Mask: 复杂场景

4. 实现要点
   - 训练: is_causal=True
   - 推理(单token): is_causal=False
   - 推理(多token): 手动构造 attn_mask

5. PyTorch 的优化
   - F.scaled_dot_product_attention
   - Flash Attention 优化
   - 自动处理 mask 的高效实现

【数学本质】

Attention(Q, K, V, Mask) = softmax((Q @ K^T + Mask) / sqrt(d_k)) @ V

其中 Mask:
  - Causal: 下三角 = 0, 上三角 = -inf
  - Padding: valid = 0, padding = -inf
  - None: 全部 = 0
""")
print()

print("=" * 70)
print("✅ Part 3 完成！")
print("=" * 70)
print()

print("""
你现在完全理解了 Causal Attention！

关键要点：
✅ Causal Mask 通过 -inf 屏蔽未来信息
✅ 保证训练和推理的分布一致
✅ 是自回归生成的核心机制
✅ 不同场景需要不同的 Mask 策略

第四课完成！你已经掌握了 Attention 机制的：
- 数学原理（Part 1）
- Multi-Head 架构（Part 2）
- Causal Mask 实现（Part 3）

下一课：位置编码与 RoPE 详解
我们将深入理解如何给 Attention 注入位置信息！
""")
