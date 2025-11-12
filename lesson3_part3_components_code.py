"""
第三课 Part 3: 核心组件代码实现
手写 Transformer 的关键组件，理解每个细节
"""

print("=" * 70)
print("第三课 Part 3: 手写 Transformer 核心组件")
print("=" * 70)
print()

# ============================================================================
# 0. 前置说明
# ============================================================================

print("📚 学习目标")
print("-" * 70)
print("""
在这一部分，我们将从零实现：
1. RMSNorm - 归一化层
2. Self-Attention - 注意力机制
3. MLP - 前馈网络
4. TransformerBlock - 完整的 Transformer 块

我们将使用纯 Python + 简单的数学运算来实现核心逻辑，
以便更好地理解每个组件的工作原理。
""")
print()

# ============================================================================
# 1. RMSNorm - 均方根归一化
# ============================================================================

print("=" * 70)
print("1️⃣  RMSNorm - 均方根归一化")
print("=" * 70)
print()

print("【什么是 RMSNorm？】")
print("-" * 70)
print("""
RMSNorm 是 LayerNorm 的简化版本：
- LayerNorm: y = (x - mean) / std * gamma + beta
- RMSNorm:  y = x / rms(x) * gamma  (无 mean 和 beta)

公式:
  rms(x) = sqrt(mean(x²))
  y = x / rms(x) * gamma

优势:
✅ 计算更简单（不需要计算均值）
✅ 更少的参数（没有 beta）
✅ 性能几乎相同
""")
print()

print("【手写实现】")
print("-" * 70)

def rms_norm(x, weight, eps=1e-6):
    """
    RMSNorm 的纯 Python 实现

    参数:
        x: list[float] - 输入向量 (长度 d)
        weight: list[float] - 可学习的缩放参数 (长度 d)
        eps: float - 数值稳定性项

    返回:
        list[float] - 归一化后的向量
    """
    # 步骤 1: 计算均方根 (RMS)
    square_sum = sum(xi * xi for xi in x)
    rms = (square_sum / len(x)) ** 0.5

    # 步骤 2: 归一化
    norm_x = [xi / (rms + eps) for xi in x]

    # 步骤 3: 缩放
    output = [norm_xi * wi for norm_xi, wi in zip(norm_x, weight)]

    return output

# 测试 RMSNorm
print("\n【测试 RMSNorm】")
print("-" * 70)

x = [1.0, 2.0, 3.0, 4.0]
weight = [1.0, 1.0, 1.0, 1.0]

print(f"输入 x:    {x}")
print(f"权重 w:    {weight}")
print()

# 计算过程
square_sum = sum(xi * xi for xi in x)
print(f"平方和:     {square_sum}")
print(f"均方:       {square_sum / len(x)}")
print(f"均方根:     {(square_sum / len(x)) ** 0.5:.4f}")
print()

output = rms_norm(x, weight)
print(f"输出 y:    {[f'{yi:.4f}' for yi in output]}")
print()

print("验证: 归一化后的向量应该有单位范数（近似）")
output_rms = (sum(yi * yi for yi in output) / len(output)) ** 0.5
print(f"输出的 RMS: {output_rms:.4f} (应该接近 1.0)")
print()

# ============================================================================
# 2. Self-Attention - 自注意力机制
# ============================================================================

print("=" * 70)
print("2️⃣  Self-Attention - 自注意力机制")
print("=" * 70)
print()

print("【Self-Attention 的核心步骤】")
print("-" * 70)
print("""
Self-Attention 让每个 token 都能"看到"所有其他 token。

步骤:
1. 线性变换: Q = X @ W_q, K = X @ W_k, V = X @ W_v
2. 计算相似度: scores = Q @ K^T / sqrt(d_k)
3. Softmax: weights = softmax(scores)
4. 加权求和: output = weights @ V

关键参数:
- d_model: 模型维度 (如 768)
- n_head: 头数 (如 12)
- d_k = d_model / n_head: 每个头的维度 (如 64)
""")
print()

print("【简化版实现 - 单头注意力】")
print("-" * 70)

def softmax(x):
    """Softmax 函数"""
    # 数值稳定性：减去最大值
    x_max = max(x)
    exp_x = [2.71828 ** (xi - x_max) for xi in x]
    sum_exp = sum(exp_x)
    return [ei / sum_exp for ei in exp_x]

def matmul(A, B):
    """矩阵乘法: A (m×n) @ B (n×p) -> C (m×p)"""
    m, n = len(A), len(A[0])
    p = len(B[0])
    C = [[0.0 for _ in range(p)] for _ in range(m)]
    for i in range(m):
        for j in range(p):
            C[i][j] = sum(A[i][k] * B[k][j] for k in range(n))
    return C

def transpose(A):
    """矩阵转置"""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def scaled_dot_product_attention(Q, K, V):
    """
    缩放点积注意力

    参数:
        Q: list[list[float]] - Query 矩阵 (seq_len × d_k)
        K: list[list[float]] - Key 矩阵 (seq_len × d_k)
        V: list[list[float]] - Value 矩阵 (seq_len × d_k)

    返回:
        list[list[float]] - 输出矩阵 (seq_len × d_k)
    """
    seq_len = len(Q)
    d_k = len(Q[0])

    # 步骤 1: Q @ K^T
    K_T = transpose(K)
    scores = matmul(Q, K_T)  # (seq_len × seq_len)

    # 步骤 2: 缩放
    scale = d_k ** 0.5
    scores = [[s / scale for s in row] for row in scores]

    # 步骤 3: Softmax (对每一行)
    weights = [softmax(row) for row in scores]

    # 步骤 4: weights @ V
    output = matmul(weights, V)  # (seq_len × d_k)

    return output, weights

# 测试 Self-Attention
print("\n【测试 Self-Attention】")
print("-" * 70)

# 简化的例子: 3 个 token，每个 4 维
print("假设输入序列: ['I', 'love', 'AI']")
print("每个 token 的向量维度: d_k = 4")
print()

# 为了演示，我们手工构造 Q, K, V
Q = [
    [1.0, 0.0, 0.0, 0.0],  # "I" 的 query
    [0.0, 1.0, 0.0, 0.0],  # "love" 的 query
    [0.0, 0.0, 1.0, 0.0],  # "AI" 的 query
]

K = [
    [1.0, 0.5, 0.0, 0.0],  # "I" 的 key
    [0.5, 1.0, 0.5, 0.0],  # "love" 的 key
    [0.0, 0.5, 1.0, 0.0],  # "AI" 的 key
]

V = [
    [1.0, 2.0, 3.0, 4.0],  # "I" 的 value
    [2.0, 3.0, 4.0, 5.0],  # "love" 的 value
    [3.0, 4.0, 5.0, 6.0],  # "AI" 的 value
]

print("Q (Query) 矩阵:")
for i, row in enumerate(Q):
    print(f"  Token {i}: {[f'{x:.2f}' for x in row]}")
print()

print("K (Key) 矩阵:")
for i, row in enumerate(K):
    print(f"  Token {i}: {[f'{x:.2f}' for x in row]}")
print()

print("V (Value) 矩阵:")
for i, row in enumerate(V):
    print(f"  Token {i}: {[f'{x:.2f}' for x in row]}")
print()

output, weights = scaled_dot_product_attention(Q, K, V)

print("注意力权重矩阵 (每行表示一个 token 对所有 token 的注意力分布):")
print("-" * 70)
tokens = ['I', 'love', 'AI']
print(f"{'':10} | {' '.join(f'{t:>8}' for t in tokens)}")
print("-" * 70)
for i, row in enumerate(weights):
    print(f"{tokens[i]:10} | {' '.join(f'{w:8.4f}' for w in row)}")
print()

print("输出矩阵 (每个 token 的新表示):")
for i, row in enumerate(output):
    print(f"  Token {i} ({tokens[i]:5}): {[f'{x:.4f}' for x in row]}")
print()

print("💡 解读:")
print("-" * 70)
print("""
注意力权重告诉我们每个 token 关注其他 token 的程度：
- 对角线上的值通常较大（token 关注自己）
- 相邻 token 之间的注意力通常较高
- 输出是所有 token 的 value 的加权平均
""")
print()

# ============================================================================
# 3. MLP - 前馈神经网络
# ============================================================================

print("=" * 70)
print("3️⃣  MLP - 前馈神经网络")
print("=" * 70)
print()

print("【MLP 的结构】")
print("-" * 70)
print("""
Transformer 中的 MLP 是一个两层的前馈网络：

结构:
  x (d_model)
    -> Linear1: d_model → 4 × d_model
    -> Activation: ReLU² 或 GELU
    -> Linear2: 4 × d_model → d_model
    -> output (d_model)

为什么是 4 倍？
- 给模型更多的表达能力
- MLP 被认为是存储"知识"的地方
- 4 倍是经验法则（Transformer 论文）
""")
print()

print("【手写实现】")
print("-" * 70)

def relu_squared(x):
    """ReLU² 激活函数: max(0, x)²"""
    return max(0, x) ** 2

def mlp(x, W1, b1, W2, b2):
    """
    两层 MLP

    参数:
        x: list[float] - 输入向量 (d_model)
        W1: list[list[float]] - 第一层权重 (d_model × hidden_dim)
        b1: list[float] - 第一层偏置 (hidden_dim)
        W2: list[list[float]] - 第二层权重 (hidden_dim × d_model)
        b2: list[float] - 第二层偏置 (d_model)

    返回:
        list[float] - 输出向量 (d_model)
    """
    # 第一层: x @ W1 + b1
    hidden = [sum(x[i] * W1[i][j] for i in range(len(x))) + b1[j]
              for j in range(len(b1))]

    # 激活函数
    hidden = [relu_squared(h) for h in hidden]

    # 第二层: hidden @ W2 + b2
    output = [sum(hidden[i] * W2[i][j] for i in range(len(hidden))) + b2[j]
              for j in range(len(b2))]

    return output

# 测试 MLP
print("\n【测试 MLP】")
print("-" * 70)

d_model = 4
hidden_dim = 8  # 2倍扩展（为了演示，实际是4倍）

x = [1.0, 2.0, 3.0, 4.0]

# 简化的权重（实际训练中会学习）
W1 = [[0.1] * hidden_dim for _ in range(d_model)]
b1 = [0.0] * hidden_dim
W2 = [[0.1] * d_model for _ in range(hidden_dim)]
b2 = [0.0] * d_model

print(f"输入维度: {d_model}")
print(f"隐藏层维度: {hidden_dim} (扩展 {hidden_dim / d_model}x)")
print(f"输出维度: {d_model}")
print()

print(f"输入 x: {x}")
output = mlp(x, W1, b1, W2, b2)
print(f"输出 y: {[f'{yi:.4f}' for yi in output]}")
print()

# ============================================================================
# 4. Transformer Block - 完整的 Transformer 块
# ============================================================================

print("=" * 70)
print("4️⃣  Transformer Block - 组合所有组件")
print("=" * 70)
print()

print("【Transformer Block 的结构】")
print("-" * 70)
print("""
一个完整的 Transformer Block 包含：

x_input
  |
  ├─> RMSNorm ─> Attention ─> Residual ─> x_attn
  |                                |
  └────────────────────────────────┘

x_attn
  |
  ├─> RMSNorm ─> MLP ─> Residual ─> x_output
  |                          |
  └──────────────────────────┘

关键点:
✅ Pre-Norm: 先归一化，再应用 Attention/MLP
✅ Residual: 输入直接加到输出上 (x + f(x))
""")
print()

print("【伪代码实现】")
print("-" * 70)
print("""
def transformer_block(x):
    # 1. Attention 子层
    x_norm1 = rms_norm(x, weight1)
    attn_out = self_attention(x_norm1)
    x = x + attn_out  # 残差连接

    # 2. MLP 子层
    x_norm2 = rms_norm(x, weight2)
    mlp_out = mlp(x_norm2)
    x = x + mlp_out  # 残差连接

    return x
""")
print()

def vector_add(a, b):
    """向量加法"""
    return [ai + bi for ai, bi in zip(a, b)]

def transformer_block_single_token(x, norm_weight1, norm_weight2, W1, b1, W2, b2):
    """
    简化版 Transformer Block（处理单个 token）

    参数:
        x: list[float] - 输入向量
        norm_weight1, norm_weight2: RMSNorm 权重
        W1, b1, W2, b2: MLP 权重

    返回:
        list[float] - 输出向量
    """
    # 1. Attention 子层（这里简化，直接跳过）
    x_norm1 = rms_norm(x, norm_weight1)
    # attn_out = self_attention(x_norm1)  # 简化：假设 attention 不改变 x
    attn_out = [0.0] * len(x)  # 零输出（演示用）
    x = vector_add(x, attn_out)  # 残差连接

    # 2. MLP 子层
    x_norm2 = rms_norm(x, norm_weight2)
    mlp_out = mlp(x_norm2, W1, b1, W2, b2)
    x = vector_add(x, mlp_out)  # 残差连接

    return x

# 测试 Transformer Block
print("【测试 Transformer Block】")
print("-" * 70)

x_input = [1.0, 2.0, 3.0, 4.0]
norm_weight1 = [1.0, 1.0, 1.0, 1.0]
norm_weight2 = [1.0, 1.0, 1.0, 1.0]

print(f"输入 x: {x_input}")
x_output = transformer_block_single_token(
    x_input, norm_weight1, norm_weight2, W1, b1, W2, b2
)
print(f"输出 x: {[f'{xi:.4f}' for xi in x_output]}")
print()

# ============================================================================
# 5. 总结与对比
# ============================================================================

print("=" * 70)
print("5️⃣  与 nanochat GPT 的对比")
print("=" * 70)
print()

print("""
我们实现的组件 vs nanochat 的实际实现：

| 组件        | 我们的实现           | nanochat GPT                      |
|------------|---------------------|-----------------------------------|
| RMSNorm    | ✅ 纯 Python        | PyTorch (torch.nn functional)     |
| Attention  | ✅ 单头，无 mask    | 多头 + Causal Mask + RoPE         |
| MLP        | ✅ 简化版           | ReLU² + 无 bias                   |
| Block      | ✅ 概念正确         | 完整的 GPT Block                  |

关键区别：
1. 我们用纯 Python，nanochat 用 PyTorch（GPU 加速）
2. 我们是单头 Attention，nanochat 是多头
3. 我们没有 Causal Mask，nanochat 有（用于自回归生成）
4. 我们没有位置编码（RoPE），nanochat 有

但核心思想是一样的！🎉
""")
print()

# ============================================================================
# 6. 下一步
# ============================================================================

print("=" * 70)
print("6️⃣  下一步：Part 4 - 阅读 nanochat GPT 源码")
print("=" * 70)
print()

print("""
现在你已经理解了核心组件的原理，准备好阅读真实代码了！

在 Part 4 中，我们将逐行阅读 `nanochat/gpt.py`：
1. GPTConfig - 配置类
2. CausalSelfAttention - 完整的多头注意力实现
3. MLP - nanochat 的 MLP 实现
4. Block - Transformer Block
5. GPT - 完整的 GPT 模型
6. 前向传播 - 从输入到输出的完整流程

你学到的所有概念都会在真实代码中看到！
""")
print()

print("=" * 70)
print("✅ Part 3 完成！你已经手写了 Transformer 的核心组件！")
print("=" * 70)
