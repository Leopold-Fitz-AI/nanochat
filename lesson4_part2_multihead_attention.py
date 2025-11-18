"""
第四课 Part 2: Multi-Head Attention 详解
理解为什么需要多个"头"以及它们如何协作
"""

print("=" * 70)
print("第四课 Part 2: Multi-Head Attention 详解")
print("=" * 70)
print()

print("""
📖 本节目标：

在 Part 1 中我们理解了单头 Attention 的数学原理。
现在我们要理解：
1. 为什么需要多个 Attention 头？
2. 每个头学习什么不同的模式？
3. Multi-Head Attention 的完整计算流程
4. nanochat 中的实际实现

从直觉到数学，从理论到代码！
""")
print()

# ============================================================================
# 1. 问题：单头 Attention 的局限性
# ============================================================================

print("=" * 70)
print("1️⃣  单头 Attention 的局限性")
print("=" * 70)
print()

print("""
【场景】分析句子："The cat sat on the mat because it was tired."

一个 Attention 头需要同时处理多种关系：
1. 句法关系: "sat" 的主语是 "cat"
2. 语义关系: "it" 指代 "cat"
3. 位置关系: "on the mat" 描述位置
4. 因果关系: "because" 引导原因

问题：
❌ 单个头很难同时学习所有这些模式
❌ 不同类型的关系可能需要不同的表示空间
❌ 一个头的表达能力有限

解决方案：Multi-Head Attention
✅ 多个头并行工作，各自专注于不同的模式
✅ 增加模型的表达能力
✅ 类似于 CNN 的多个卷积核
""")
print()

print("【类比：多个专家协作】")
print("-" * 70)
print("""
想象一个团队分析文本：

专家 1（语法专家）:
  - 关注主谓宾关系
  - "sat" → "cat" (主语)
  - "sat" → "mat" (宾语)

专家 2（语义专家）:
  - 关注词义关联
  - "it" → "cat" (指代)
  - "tired" → "cat" (状态)

专家 3（位置专家）:
  - 关注空间关系
  - "sat" → "on" (位置介词)
  - "on" → "mat" (位置对象)

每个专家给出自己的分析（一个 Attention 头）
最后综合所有专家的意见（Concat + Linear）

这就是 Multi-Head Attention 的思想！
""")
print()

# ============================================================================
# 2. Multi-Head Attention 的数学推导
# ============================================================================

print("=" * 70)
print("2️⃣  Multi-Head Attention 的数学推导")
print("=" * 70)
print()

print("""
【核心思想】

将高维空间分割成多个低维子空间：
- 原始维度: d_model = 768
- 头数: n_head = 12
- 每个头的维度: d_k = d_model / n_head = 64

每个头在自己的子空间中做 Attention！

【完整公式】

MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

其中每个 head_i 计算：
  head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)

参数矩阵：
  W_Q^i, W_K^i: (d_model, d_k) - 将输入投影到子空间
  W_V^i:        (d_model, d_v)
  W_O:          (h × d_v, d_model) - 将所有头合并
""")
print()

print("【张量形状变化】")
print("-" * 70)
print("""
以 nanochat 为例 (d_model=768, n_head=6, d_k=128):

输入 X:         (B, T, 768)
  ↓
投影 Q, K, V:   (B, T, 768) @ (768, 768) = (B, T, 768)
  ↓
Reshape:        (B, T, 768) → (B, T, 6, 128)
                将 768 维分成 6 个 128 维子空间
  ↓
Transpose:      (B, T, 6, 128) → (B, 6, T, 128)
                把 Head 维度移到 Batch 维度旁边
  ↓
Attention:      每个头独立计算
                Q @ K^T: (B, 6, T, T)
                weights @ V: (B, 6, T, 128)
  ↓
Transpose:      (B, 6, T, 128) → (B, T, 6, 128)
  ↓
Reshape:        (B, T, 6, 128) → (B, T, 768)
                重新拼接所有头
  ↓
Linear:         (B, T, 768) @ (768, 768) = (B, T, 768)
                输出投影
""")
print()

# ============================================================================
# 3. 手工计算示例
# ============================================================================

print("=" * 70)
print("3️⃣  手工计算 Multi-Head Attention")
print("=" * 70)
print()

print("【简化示例：2 个头，4 维向量】")
print("-" * 70)

import random
random.seed(42)

# 简化参数
seq_len = 3  # 3 个词
d_model = 4  # 模型维度
n_head = 2   # 2 个头
d_k = d_model // n_head  # 每个头 2 维

print(f"序列长度: {seq_len} (三个词: cat, sat, mat)")
print(f"模型维度: {d_model}")
print(f"头数: {n_head}")
print(f"每个头维度: {d_k}")
print()

# 输入（简化为单个样本）
X = [
    [1.0, 0.5, 0.2, 0.1],  # cat
    [0.5, 1.0, 0.3, 0.2],  # sat
    [0.2, 0.3, 1.0, 0.5],  # mat
]

print("输入 X (seq_len=3, d_model=4):")
for i, row in enumerate(X):
    print(f"  Token {i}: {row}")
print()

def split_heads(x, n_head):
    """将向量分割成多个头"""
    seq_len = len(x)
    d_model = len(x[0])
    d_k = d_model // n_head

    # 重组为 (seq_len, n_head, d_k)
    result = []
    for row in x:
        heads = []
        for h in range(n_head):
            head_vec = row[h * d_k : (h + 1) * d_k]
            heads.append(head_vec)
        result.append(heads)
    return result

# 分割成多头
X_heads = split_heads(X, n_head)

print("分割成 2 个头:")
print("-" * 70)
for i, token_heads in enumerate(X_heads):
    print(f"Token {i}:")
    for h, head_vec in enumerate(token_heads):
        print(f"  Head {h}: {head_vec}")
print()

def dot_product(a, b):
    """点积"""
    return sum(ai * bi for ai, bi in zip(a, b))

def softmax(scores):
    """Softmax"""
    max_score = max(scores)
    exp_scores = [2.71828 ** (s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    return [e / sum_exp for e in exp_scores]

def attention_single_head(X_head):
    """单头 Attention（简化：Q=K=V）"""
    seq_len = len(X_head)
    d_k = len(X_head[0])

    # 计算相似度矩阵
    scores = []
    for q in X_head:
        row_scores = []
        for k in X_head:
            score = dot_product(q, k) / (d_k ** 0.5)
            row_scores.append(score)
        scores.append(row_scores)

    # Softmax
    weights = [softmax(row) for row in scores]

    # 加权求和
    output = []
    for weight_row in weights:
        out_vec = [0.0] * d_k
        for w, v in zip(weight_row, X_head):
            for i in range(d_k):
                out_vec[i] += w * v[i]
        output.append(out_vec)

    return output, weights

# 对每个头独立计算 Attention
print("每个头独立计算 Attention:")
print("=" * 70)

all_head_outputs = []
all_head_weights = []

for h in range(n_head):
    print(f"\nHead {h}:")
    print("-" * 70)

    # 提取这个头的所有 token
    head_data = [X_heads[i][h] for i in range(seq_len)]

    print(f"输入 (seq_len={seq_len}, d_k={d_k}):")
    for i, vec in enumerate(head_data):
        print(f"  Token {i}: {vec}")
    print()

    # 计算 Attention
    output, weights = attention_single_head(head_data)

    print("注意力权重矩阵:")
    print(f"{'':8} | {'Token 0':>10} {'Token 1':>10} {'Token 2':>10}")
    print("-" * 50)
    for i, row in enumerate(weights):
        print(f"Token {i:1} | {row[0]:10.4f} {row[1]:10.4f} {row[2]:10.4f}")
    print()

    print("输出:")
    for i, vec in enumerate(output):
        print(f"  Token {i}: {[f'{x:.4f}' for x in vec]}")

    all_head_outputs.append(output)
    all_head_weights.append(weights)

# 拼接所有头
print("\n" + "=" * 70)
print("拼接所有头的输出:")
print("=" * 70)

concat_output = []
for i in range(seq_len):
    # 每个 token 拼接所有头的输出
    token_out = []
    for h in range(n_head):
        token_out.extend(all_head_outputs[h][i])
    concat_output.append(token_out)

print(f"拼接后的输出 (seq_len={seq_len}, d_model={d_model}):")
for i, vec in enumerate(concat_output):
    print(f"  Token {i}: {[f'{x:.4f}' for x in vec]}")
print()

print("💡 解读:")
print("-" * 70)
print("""
每个头关注不同的模式：
- Head 0: 可能关注某种语法关系
- Head 1: 可能关注另一种语义关系

拼接后，每个 token 的表示融合了多个头的信息！
""")
print()

# ============================================================================
# 4. 每个头学到了什么？
# ============================================================================

print("=" * 70)
print("4️⃣  每个头学到了什么？")
print("=" * 70)
print()

print("""
【实际观察】来自论文和实验

在真实的 Transformer 中，不同的头学习到不同的模式：

Head 1: 局部依赖
  - 主要关注相邻的词
  - 权重集中在对角线附近
  - 捕捉短距离的语法关系

Head 2: 长距离依赖
  - 关注远距离的词
  - 权重分散更均匀
  - 捕捉主谓关系、指代关系

Head 3: 位置敏感
  - 关注特定位置的词（如句首、句尾）
  - 捕捉特殊标记的信息

Head 4: 罕见词关注
  - 对低频词给予更多注意力
  - 帮助处理 OOV 和罕见词

...等等

【可视化示例】

假设有一个句子："The cat sat on the mat"

Head 1 的注意力权重（关注相邻词）:
         The  cat  sat  on   the  mat
    The [0.8  0.2  0.0  0.0  0.0  0.0]
    cat [0.3  0.6  0.1  0.0  0.0  0.0]
    sat [0.0  0.3  0.5  0.2  0.0  0.0]
    ...

Head 2 的注意力权重（关注主谓关系）:
         The  cat  sat  on   the  mat
    The [0.2  0.2  0.2  0.2  0.2  0.0]
    cat [0.1  0.5  0.3  0.0  0.0  0.1]
    sat [0.0  0.6  0.2  0.1  0.0  0.1]  # sat 强烈关注 cat
    ...

每个头看到的是不同的"视角"！
""")
print()

# ============================================================================
# 5. 为什么要拼接而不是求平均？
# ============================================================================

print("=" * 70)
print("5️⃣  为什么拼接（Concat）而不是求平均？")
print("=" * 70)
print()

print("""
【两种合并方式】

方案 A: 求平均
  output = (head_1 + head_2 + ... + head_h) / h

方案 B: 拼接 + Linear
  output = Concat(head_1, head_2, ..., head_h) @ W_O

Transformer 选择了方案 B，为什么？

拼接的优势：
✅ 保留所有信息 - 每个头的输出都完整保留
✅ 可学习的组合 - W_O 学习如何最优地组合各个头
✅ 更灵活 - 模型可以学习给不同的头不同的权重

求平均的劣势：
❌ 信息丢失 - 不同头的模式被混合稀释
❌ 固定权重 - 每个头的贡献都是 1/h，无法调整
❌ 表达能力弱 - 无法学习复杂的组合方式

【数学视角】

拼接 + Linear:
  y = [head_1; head_2; ...; head_h] @ W_O

这相当于：
  y = head_1 @ W_O^1 + head_2 @ W_O^2 + ... + head_h @ W_O^h

每个头有自己的权重矩阵 W_O^i！
模型可以学习：
- 给某些头更大的权重
- 对某些头做非线性变换
- 让某些头在特定任务中更重要
""")
print()

# ============================================================================
# 6. nanochat 的实现
# ============================================================================

print("=" * 70)
print("6️⃣  nanochat 中的 Multi-Head Attention 实现")
print("=" * 70)
print()

print("""
【回顾 nanochat/gpt.py 的实现】

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        self.n_head = config.n_head          # 6
        self.head_dim = config.n_embd // self.n_head  # 128

        # Q, K, V 投影（一次性投影所有头）
        self.c_q = nn.Linear(768, 6 × 128, bias=False)
        self.c_k = nn.Linear(768, 6 × 128, bias=False)
        self.c_v = nn.Linear(768, 6 × 128, bias=False)

        # 输出投影
        self.c_proj = nn.Linear(768, 768, bias=False)

    def forward(self, x, cos_sin, kv_cache):
        B, T, C = x.size()  # (batch, seq_len, 768)

        # 步骤 1: 投影
        q = self.c_q(x).view(B, T, 6, 128)
        k = self.c_k(x).view(B, T, 6, 128)
        v = self.c_v(x).view(B, T, 6, 128)

        # 步骤 2: RoPE + QK Norm
        q = apply_rotary_emb(q, cos_sin)
        k = apply_rotary_emb(k, cos_sin)
        q, k = norm(q), norm(k)

        # 步骤 3: 调整维度 (B, T, H, D) -> (B, H, T, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 步骤 4: Attention（所有头并行计算）
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        # 输出: (B, H, T, D) = (B, 6, T, 128)

        # 步骤 5: 重组 (B, H, T, D) -> (B, T, H×D)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        # (B, 6, T, 128) -> (B, T, 6, 128) -> (B, T, 768)

        # 步骤 6: 输出投影
        y = self.c_proj(y)

        return y

关键实现细节：

1. 批量投影
   - 一次性为所有头生成 Q, K, V
   - 然后 reshape 分割成多个头
   - 比单独投影每个头更高效

2. 并行计算
   - transpose 后，Head 维度在 Batch 旁边
   - 所有头同时计算，GPU 并行加速

3. 高效拼接
   - 不需要显式 concat
   - 直接 view reshape 即可

4. 输出投影
   - c_proj 相当于我们说的 W_O
   - 学习如何最优地组合各个头
""")
print()

# ============================================================================
# 7. 总结与对比
# ============================================================================

print("=" * 70)
print("7️⃣  总结：Single-Head vs Multi-Head")
print("=" * 70)
print()

print("""
┌──────────────────┬────────────────────┬─────────────────────┐
│ 特性             │ Single-Head        │ Multi-Head          │
├──────────────────┼────────────────────┼─────────────────────┤
│ 表达能力         │ 有限               │ 更强                │
│ 参数量           │ 较少               │ 相同 (分配方式不同) │
│ 关注模式         │ 单一               │ 多样化              │
│ 计算效率         │ 略快               │ 高度并行            │
│ 可解释性         │ 简单               │ 可分析不同头        │
└──────────────────┴────────────────────┴─────────────────────┘

关键洞察：

1. Multi-Head 不是简单的集成
   - 不是多个独立模型的平均
   - 而是在多个子空间并行学习

2. 参数量实际上相同
   - Single-Head: d_model × d_model
   - Multi-Head:  h × (d_k × d_k) = h × (d_model/h)² ≈ d_model × d_model
   - 只是参数的组织方式不同！

3. 分而治之的思想
   - 将复杂问题分解为多个简单问题
   - 每个头专注于一种模式
   - 综合所有头的决策

4. 类似 CNN 的多通道
   - CNN: 多个卷积核提取不同特征
   - Multi-Head: 多个 Attention 头捕捉不同关系
   - 都是增加模型的表达能力

【最佳实践】

头数选择：
- 小模型: 4-8 头
- 中等模型: 8-12 头
- 大模型: 12-16 头

每个头的维度：
- 通常 d_k = 64 或 128
- 太小: 表达能力不足
- 太大: 每个头过于复杂，失去专业化

nanochat 的选择：
- n_head = 6
- head_dim = 128
- 平衡了效率和表达能力
""")
print()

print("=" * 70)
print("✅ Part 2 完成！")
print("=" * 70)
print()

print("""
你现在完全理解了 Multi-Head Attention！

关键要点：
✅ 多头让模型在多个子空间并行学习不同模式
✅ 每个头可以专注于不同类型的关系
✅ 拼接 + Linear 提供可学习的组合方式
✅ 高效的实现通过批量投影和并行计算

下一步：Part 3 - Causal Attention 和 Masked Attention
我们将理解如何实现自回归生成和序列掩码！
""")
