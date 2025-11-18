"""
第四课 Part 1: Attention 机制的数学推导
从第一性原理理解 Attention 的每一步
"""

print("=" * 70)
print("第四课 Part 1: Attention 机制的数学推导")
print("=" * 70)
print()

print("""
📖 本节目标：

我们将从零推导 Attention 机制，理解：
1. 为什么需要 Attention？
2. 为什么用点积计算相似度？
3. 为什么要除以 sqrt(d_k)？
4. Softmax 的作用是什么？
5. 完整的数学推导过程

不是记公式，而是理解每一步的动机！
""")
print()

# ============================================================================
# 1. 问题起源：如何让序列中的每个元素"看到"其他元素？
# ============================================================================

print("=" * 70)
print("1️⃣  问题起源：序列中的信息聚合")
print("=" * 70)
print()

print("""
【场景】翻译句子："The cat sat on the mat"

传统 RNN 的问题：
- "sat" 需要知道主语是 "cat"（距离较远）
- RNN 逐步传递信息，远距离依赖容易丢失

我们需要一种机制：
✅ 让 "sat" 能直接"看到" "cat"
✅ 让每个词都能访问所有其他词
✅ 能够学习哪些词之间关系更重要

这就是 Attention 的核心思想！
""")
print()

print("【直觉类比：数据库查询】")
print("-" * 70)
print("""
假设我们有一个信息库（数据库）：

┌─────────────┬──────────────┐
│ Key (键)    │ Value (值)   │
├─────────────┼──────────────┤
│ "cat"       │ [动物, 主语] │
│ "sat"       │ [动作, 过去] │
│ "mat"       │ [物体, 地点] │
└─────────────┴──────────────┘

当我们处理 "sat" 时，我们想问：
Query: "sat 需要什么上下文信息？"

Attention 做的就是：
1. 用 Query 去匹配所有 Key（计算相似度）
2. 根据相似度加权获取 Value
3. 得到聚合后的上下文表示

这是一种"软查询"（soft lookup）！
""")
print()

# ============================================================================
# 2. 第一步：如何计算相似度？
# ============================================================================

print("=" * 70)
print("2️⃣  如何计算相似度？为什么用点积？")
print("=" * 70)
print()

print("""
【问题】如何度量两个向量的相似度？

常见方法：
1. 欧氏距离: ||q - k||²
   - 衡量"距离"
   - 越小越相似

2. 余弦相似度: (q · k) / (||q|| × ||k||)
   - 衡量"方向"
   - 范围 [-1, 1]

3. 点积: q · k
   - 衡量"方向 + 幅度"
   - 计算最简单

Transformer 选择了点积，原因：
✅ 计算效率高（矩阵乘法，GPU 友好）
✅ 可以批量并行计算
✅ 数学性质好（可微，易优化）
""")
print()

print("【手工计算示例】")
print("-" * 70)

# 定义三个词的向量（简化为 4 维）
vectors = {
    'cat': [1.0, 0.5, 0.2, 0.1],
    'sat': [0.5, 1.0, 0.3, 0.2],
    'mat': [0.2, 0.3, 1.0, 0.5],
}

def dot_product(a, b):
    """计算点积"""
    return sum(ai * bi for ai, bi in zip(a, b))

print("\n假设我们有三个词的向量表示（4 维）：")
for word, vec in vectors.items():
    print(f"  {word:5}: {vec}")
print()

print("计算 'sat' 与所有词的点积相似度：")
print("-" * 70)
query = vectors['sat']
print(f"Query (sat): {query}\n")

similarities = {}
for word, key in vectors.items():
    score = dot_product(query, key)
    similarities[word] = score
    print(f"sat · {word} = {score:.4f}")
print()

print("💡 解读：")
print("  - sat 与自己最相似 (1.56)")
print("  - sat 与 cat 有一定相似度 (1.00)")
print("  - sat 与 mat 相似度较低 (0.56)")
print()

# ============================================================================
# 3. 第二步：为什么要缩放？除以 sqrt(d_k)
# ============================================================================

print("=" * 70)
print("3️⃣  为什么要缩放？除以 sqrt(d_k) 的秘密")
print("=" * 70)
print()

print("""
【问题】点积的值随着向量维度增加而增大

假设向量是随机初始化的（均值 0，方差 1）：
- 维度 d = 4:   点积的期望方差 ≈ 4
- 维度 d = 64:  点积的期望方差 ≈ 64
- 维度 d = 512: 点积的期望方差 ≈ 512

随着维度增加，点积的值会越来越大！

这会导致什么问题？
❌ Softmax 的梯度消失
❌ 数值不稳定

解决方案：除以 sqrt(d_k)
✅ 归一化方差：Var(score / sqrt(d_k)) ≈ 1
✅ 稳定 Softmax 的输入范围
""")
print()

print("【数学推导】")
print("-" * 70)
print("""
假设 q, k 的每个分量独立同分布：q_i, k_i ~ N(0, 1)

点积: score = q · k = Σ(q_i × k_i)

期望: E[score] = Σ E[q_i × k_i] = 0 (因为 E[q_i] = E[k_i] = 0)

方差: Var[score] = Σ Var[q_i × k_i]
               = Σ E[q_i²] × E[k_i²]  (独立变量的方差)
               = Σ 1 × 1
               = d_k

因此：Var[score] = d_k

缩放后：Var[score / sqrt(d_k)] = Var[score] / d_k = 1

✅ 方差归一化到 1！
""")
print()

print("【实验验证】")
print("-" * 70)

import random
random.seed(42)

def random_vector(d):
    """生成均值0方差1的随机向量"""
    return [random.gauss(0, 1) for _ in range(d)]

dimensions = [4, 16, 64, 256]
n_samples = 1000

print(f"{'维度':>6} | {'原始点积方差':>15} | {'缩放后方差':>15}")
print("-" * 50)

for d in dimensions:
    # 生成多组随机向量，计算点积
    scores = []
    scaled_scores = []

    for _ in range(n_samples):
        q = random_vector(d)
        k = random_vector(d)
        score = dot_product(q, k)
        scores.append(score)
        scaled_scores.append(score / (d ** 0.5))

    # 计算方差
    mean_score = sum(scores) / len(scores)
    var_score = sum((s - mean_score) ** 2 for s in scores) / len(scores)

    mean_scaled = sum(scaled_scores) / len(scaled_scores)
    var_scaled = sum((s - mean_scaled) ** 2 for s in scaled_scores) / len(scaled_scores)

    print(f"{d:6} | {var_score:15.2f} | {var_scaled:15.2f}")

print()
print("💡 观察：")
print("  - 原始点积方差 ≈ 维度 d")
print("  - 缩放后方差 ≈ 1（归一化成功！）")
print()

# ============================================================================
# 4. 第三步：Softmax 归一化
# ============================================================================

print("=" * 70)
print("4️⃣  Softmax：从分数到概率分布")
print("=" * 70)
print()

print("""
【为什么需要 Softmax？】

点积给出相似度分数，但：
❌ 分数可能是负数
❌ 分数没有归一化（总和不为1）
❌ 难以解释为"权重"

Softmax 的作用：
✅ 将分数转换为概率分布（非负，总和为1）
✅ 保持相对大小关系
✅ 可微分（梯度友好）

公式：
  softmax(x_i) = exp(x_i) / Σ exp(x_j)
""")
print()

print("【手工计算示例】")
print("-" * 70)

def softmax(scores):
    """Softmax 函数"""
    # 数值稳定性：减去最大值
    max_score = max(scores)
    exp_scores = [2.71828 ** (s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    return [e / sum_exp for e in exp_scores]

# 使用之前计算的相似度
scores = [similarities[word] for word in ['cat', 'sat', 'mat']]
scaled_scores = [s / (4 ** 0.5) for s in scores]  # d_k = 4

print("原始点积分数：")
for word, score in zip(['cat', 'sat', 'mat'], scores):
    print(f"  {word}: {score:.4f}")
print()

print("缩放后 (/ sqrt(4))：")
for word, score in zip(['cat', 'sat', 'mat'], scaled_scores):
    print(f"  {word}: {score:.4f}")
print()

attention_weights = softmax(scaled_scores)
print("Softmax 归一化后（注意力权重）：")
for word, weight in zip(['cat', 'sat', 'mat'], attention_weights):
    print(f"  {word}: {weight:.4f}")
print()

print(f"权重总和: {sum(attention_weights):.6f} (应该等于 1.0)")
print()

print("💡 解读：")
print("  - 'sat' 最关注自己 (0.5279)")
print("  - 也会关注 'cat' (0.3399)")
print("  - 对 'mat' 关注较少 (0.1322)")
print()

# ============================================================================
# 5. 第四步：加权求和 Value
# ============================================================================

print("=" * 70)
print("5️⃣  加权求和：获取上下文表示")
print("=" * 70)
print()

print("""
【最后一步】用注意力权重对 Value 加权求和

回到数据库类比：
- Query 找到了相关的 Key（通过相似度）
- 现在要获取对应的 Value（信息内容）
- 按权重加权求和

公式：
  output = Σ (attention_weight_i × value_i)
""")
print()

print("【完整计算示例】")
print("-" * 70)

# 假设 Value 向量（可以与 Key 不同）
values = {
    'cat': [1.0, 0.0, 0.0, 0.0],  # "主语"信息
    'sat': [0.0, 1.0, 0.0, 0.0],  # "动作"信息
    'mat': [0.0, 0.0, 1.0, 0.0],  # "宾语"信息
}

print("Value 向量（语义信息）：")
for word, val in values.items():
    print(f"  {word}: {val}")
print()

print("注意力权重：")
for word, weight in zip(['cat', 'sat', 'mat'], attention_weights):
    print(f"  {word}: {weight:.4f}")
print()

# 计算加权和
output = [0.0] * 4
for word, weight in zip(['cat', 'sat', 'mat'], attention_weights):
    value = values[word]
    for i in range(4):
        output[i] += weight * value[i]

print("输出（加权求和）：")
print(f"  output = {[f'{x:.4f}' for x in output]}")
print()

print("💡 解读：")
print("  - 位置 0 (cat/主语): 0.3399 - 来自 'cat' 的贡献")
print("  - 位置 1 (sat/动作): 0.5279 - 来自 'sat' 的贡献")
print("  - 位置 2 (mat/宾语): 0.1322 - 来自 'mat' 的贡献")
print()
print("  这个向量融合了所有相关词的信息，权重由注意力决定！")
print()

# ============================================================================
# 6. 完整的 Attention 公式
# ============================================================================

print("=" * 70)
print("6️⃣  完整的 Attention 公式推导")
print("=" * 70)
print()

print("""
【从直觉到公式】

步骤回顾：
1. 计算相似度: scores = Q @ K^T
2. 缩放:       scores = scores / sqrt(d_k)
3. 归一化:     weights = softmax(scores)
4. 加权求和:   output = weights @ V

完整公式：
┌─────────────────────────────────────────┐
│ Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V │
└─────────────────────────────────────────┘

矩阵形状：
  Q: (seq_len, d_k)  - Query 矩阵
  K: (seq_len, d_k)  - Key 矩阵
  V: (seq_len, d_v)  - Value 矩阵

  Q @ K^T: (seq_len, seq_len)  - 相似度矩阵
  softmax: (seq_len, seq_len)  - 注意力权重矩阵
  output:  (seq_len, d_v)      - 输出矩阵

每一行对应一个 Query 的输出！
""")
print()

print("【矩阵视角】")
print("-" * 70)
print("""
假设序列长度 T = 3, 维度 d = 4:

Q @ K^T 得到相似度矩阵:
         cat   sat   mat
    cat [0.5  0.3  0.1]
    sat [0.3  0.6  0.2]
    mat [0.1  0.2  0.5]

Softmax 归一化（每行独立）:
         cat   sat   mat
    cat [0.4  0.35 0.25]
    sat [0.3  0.5  0.2]
    mat [0.25 0.3  0.45]

weights @ V 得到输出:
每个词都得到了融合所有其他词信息的新表示！
""")
print()

# ============================================================================
# 7. 关键洞察与总结
# ============================================================================

print("=" * 70)
print("7️⃣  关键洞察与设计选择")
print("=" * 70)
print()

print("""
【为什么 Attention 如此强大？】

1. 并行计算
   - RNN: 必须顺序处理，t_i 依赖 t_{i-1}
   - Attention: 所有位置可以同时计算
   ✅ GPU 友好，训练快

2. 长距离依赖
   - RNN: 远距离信息通过多步传递，容易衰减
   - Attention: 任意两个位置的"距离"都是 1
   ✅ 直接建立连接

3. 可解释性
   - 注意力权重可视化
   - 看到模型"关注"哪些信息
   ✅ 黑盒更透明

4. 灵活性
   - Q, K, V 可以来自不同来源
   - Self-Attention: Q=K=V (同一序列)
   - Cross-Attention: Q≠K=V (不同序列)
   ✅ 适用多种场景

【三个关键设计选择总结】

1. 为什么用点积？
   答: 计算高效，GPU 友好，批量并行

2. 为什么除以 sqrt(d_k)？
   答: 归一化方差到 1，稳定 Softmax 梯度

3. 为什么用 Softmax？
   答: 转换为概率分布，可解释，可微分
""")
print()

print("=" * 70)
print("🎓 数学推导完成！")
print("=" * 70)
print()

print("""
你现在完全理解了 Attention 的数学原理！

关键要点：
✅ Attention 是一种"软查询"机制
✅ 点积衡量相似度（方向 + 幅度）
✅ 缩放避免梯度消失
✅ Softmax 转换为概率分布
✅ 加权求和聚合信息

下一步：Part 2 - Multi-Head Attention
我们将理解为什么需要多个"头"以及它们如何协作！
""")
print()

print("=" * 70)
print("✅ Part 1 完成！")
print("=" * 70)
