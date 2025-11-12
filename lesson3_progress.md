# 第三课进度：Transformer 基础架构（Part 1-2）🏗️

## ✅ 已完成内容

### Part 1: 为什么需要 Transformer？

**核心收获**：
- ✅ 理解了语言模型的本质任务（预测下一个词）
- ✅ 了解了历史方法的局限（N-gram、RNN）
- ✅ 掌握了 Transformer 的核心创新：Self-Attention
- ✅ 建立了直觉：Attention 是"软"数据库查询

**关键概念**：
```
Attention 的核心思想：
  query = "当前词要找什么"
  keys  = "所有词提供的索引"
  values = "所有词的内容"

  scores = similarity(query, keys)
  weights = softmax(scores)
  output = weighted_sum(weights, values)

  → 每个词都能"看到"所有其他词
  → 直接捕获长距离依赖
  → 可以并行计算
```

---

### Part 2: Transformer 架构鸟瞰

**核心收获**：
- ✅ 掌握了完整的 GPT 架构流程
- ✅ 理解了张量形状的变化（重要！）
- ✅ 明确了每个组件的职责
- ✅ 了解了 nanochat 的架构创新

**完整架构**：
```
Token IDs
  ↓
Token Embedding (离散 → 连续)
  ↓
RMSNorm (归一化)
  ↓
┌─────────────────────────────────────┐
│  Transformer Block × N              │
│  ┌──────────────────────────────┐  │
│  │ Norm → Attention → Residual  │  │
│  │ Norm → MLP → Residual         │  │
│  └──────────────────────────────┘  │
└─────────────────────────────────────┘
  ↓
Final Norm
  ↓
LM Head (词汇表投影)
  ↓
Logits (概率分布)
```

**张量形状追踪**：
```
输入:    (B, T)           # batch × seq_len
Embedding: (B, T, d)      # 每个 token 变成 d 维向量
Attention: (B, T, d)      # 形状不变，内容变化
MLP:       (B, T, d)      # 形状不变
LM Head:   (B, T, V)      # 投影到词汇表
```

**nanochat 的 4 大创新**：
1. RoPE - 旋转位置编码
2. RMSNorm - 简化的归一化
3. No Bias - 所有 Linear 层无偏置
4. ReLU² - 更简单的激活函数

---

## 📊 知识点总结

### 1. Transformer 的核心组件

| 组件 | 输入形状 | 输出形状 | 主要作用 |
|-----|---------|---------|---------|
| Embedding | (B, T) | (B, T, d) | 离散→连续 |
| RMSNorm | (B, T, d) | (B, T, d) | 归一化 |
| Attention | (B, T, d) | (B, T, d) | token 交互 |
| MLP | (B, T, d) | (B, T, d) | 非线性变换 |
| LM Head | (B, T, d) | (B, T, V) | 词汇表投影 |

### 2. Self-Attention 的直觉理解

**类比：图书馆查询**
```
你（query）：想找关于 "机器学习" 的书
图书馆（keys + values）：所有书籍

步骤：
1. 计算相关性：哪些书和 "机器学习" 相关？
   - 《深度学习》相关度 0.9
   - 《Python 编程》相关度 0.6
   - 《烹饪大全》相关度 0.1

2. Softmax 归一化：
   - 《深度学习》权重 0.55
   - 《Python 编程》权重 0.35
   - 《烹饪大全》权重 0.10

3. 加权组合：
   result = 0.55 × 《深度学习》内容
          + 0.35 × 《Python 编程》内容
          + 0.10 × 《烹饪大全》内容

→ 你得到了一个融合了多本相关书籍的知识总结
```

### 3. 为什么需要多层 Transformer？

**层次化特征学习**：
- **浅层（Layer 1-5）**：语法、词性、简单模式
  - "is" 后面跟形容词或名词
  - 名词的单复数

- **中层（Layer 6-15）**：语义、关系、中距离依赖
  - 主谓一致
  - 代词指代
  - 局部逻辑推理

- **深层（Layer 16-20+）**：抽象概念、长距离推理
  - 段落级理解
  - 常识推理
  - 复杂逻辑

### 4. 残差连接的重要性

**问题**：深层网络梯度消失
```
没有残差连接：
  x → Layer1 → Layer2 → ... → Layer20 → out
  梯度: ∂L/∂x = ∂L/∂out × ∂out/∂Layer20 × ... × ∂Layer2/∂x
         ↑ 经过 20 次乘法，梯度可能趋近于 0
```

**解决方案**：残差连接
```
有残差连接：
  x → (x + Layer1(x)) → (x + Layer2(x)) → ... → out
  梯度: 可以直接从 out 传到 x（高速公路）

效果：
✅ 可以训练 100+ 层的网络
✅ 收敛更快
✅ 性能更好
```

---

## 🎯 重要概念检查清单

自测：你是否理解了这些概念？

### 基础概念
- [ ] 什么是自回归生成？
- [ ] 为什么 RNN 无法并行化？
- [ ] Self-Attention 如何捕获长距离依赖？

### 架构理解
- [ ] Token Embedding 的作用是什么？
- [ ] 为什么需要 Layer Normalization？
- [ ] Attention 和 MLP 各自的职责是什么？
- [ ] 残差连接如何帮助梯度流动？

### 张量形状
- [ ] (B, T, d) 中每个维度代表什么？
- [ ] Attention 内部如何重塑张量？
- [ ] 为什么 MLP 要先扩展 4 倍再压缩？

### nanochat 特点
- [ ] RMSNorm 和 LayerNorm 的区别？
- [ ] RoPE 的优势是什么？
- [ ] 为什么 nanochat 不用 bias？

---

## 📝 思考题

### 问题 1: 张量形状计算

假设：
- batch_size = 8
- seq_len = 64
- dim = 512
- n_head = 8
- vocab_size = 32000

请计算：
1. Embedding 的参数量？
2. LM Head 的参数量？
3. Attention 中 Q @ K^T 的输出形状？

<details>
<summary>点击查看答案</summary>

1. Embedding 参数量：
   vocab_size × dim = 32000 × 512 = 16,384,000 (约 16M)

2. LM Head 参数量：
   dim × vocab_size = 512 × 32000 = 16,384,000 (约 16M)
   （注意：Embedding 和 LM Head 参数量相同，但 nanochat 不共享权重）

3. Q @ K^T 的输出形状：
   Q shape: (8, 8, 64, 64)  # (B, n_head, T, head_dim)
   K^T shape: (8, 8, 64, 64)  # transpose 最后两维
   Q @ K^T: (8, 8, 64, 64)
</details>

---

### 问题 2: 为什么 MLP 要扩展 4 倍？

<details>
<summary>点击查看答案</summary>

原因：
1. **增加模型容量**：更多参数 = 更强的表达能力
2. **知识存储**：MLP 被认为是模型"记忆"事实知识的地方
3. **信息瓶颈**：dim → 4×dim → dim 形成"沙漏"形状
   - 中间层有更大的空间来处理信息
   - 类似于压缩-解压缩的过程
4. **经验法则**：4 倍是实验得出的最佳比例
   - 太小：表达能力不足
   - 太大：参数过多，过拟合风险

Transformer 论文原文建议 4 倍，后续研究证明这是个好选择。
</details>

---

### 问题 3: 并行化的优势

为什么 Transformer 比 RNN 快得多？请从计算图的角度解释。

<details>
<summary>点击查看答案</summary>

**RNN 的计算图（串行）**：
```
h1 = f(x1, h0)      ← 必须等 h0
h2 = f(x2, h1)      ← 必须等 h1 算完
h3 = f(x3, h2)      ← 必须等 h2 算完
...
```
→ 完全串行，无法并行

**Transformer 的计算图（并行）**：
```
Q = Linear(x)       ← 所有位置同时计算
K = Linear(x)       ← 所有位置同时计算
V = Linear(x)       ← 所有位置同时计算
scores = Q @ K^T    ← 一次矩阵乘法，GPU 高度并行
output = softmax(scores) @ V  ← 一次矩阵乘法
```
→ 所有位置同时处理，充分利用 GPU 并行能力

**速度对比**：
- RNN: 处理 1000 tokens 需要 1000 步
- Transformer: 处理 1000 tokens 只需要常数步（矩阵运算）
- 实际加速比：10-100 倍（取决于硬件和序列长度）
</details>

---

---

### Part 3: 核心组件代码实现

**核心收获**：
- ✅ 手写了 RMSNorm 归一化
- ✅ 实现了 Self-Attention 机制
- ✅ 实现了 MLP 前馈网络
- ✅ 组合成完整的 Transformer Block

**关键实现**：
```python
# RMSNorm: y = x / rms(x) * gamma
def rms_norm(x, weight, eps=1e-6):
    rms = (sum(xi * xi for xi in x) / len(x)) ** 0.5
    norm_x = [xi / (rms + eps) for xi in x]
    output = [norm_xi * wi for norm_xi, wi in zip(norm_x, weight)]
    return output

# Self-Attention: output = softmax(Q @ K^T / sqrt(d_k)) @ V
def scaled_dot_product_attention(Q, K, V):
    scores = Q @ K^T / sqrt(d_k)
    weights = softmax(scores)
    output = weights @ V
    return output

# MLP: d → 4d → d
def mlp(x, W1, b1, W2, b2):
    hidden = relu_squared(x @ W1 + b1)
    output = hidden @ W2 + b2
    return output
```

**注意力权重可视化**：
```
         |        I     love       AI
--------------------------------------
I        |   0.4192   0.3265   0.2543
love     |   0.3045   0.3910   0.3045
AI       |   0.2543   0.3265   0.4192
```
→ 每个 token 关注所有其他 token，对角线权重最大

---

### Part 4: nanochat GPT 源码精读

**核心收获**：
- ✅ 理解了 nanochat 的 7 大创新特性
- ✅ 掌握了 GPTConfig 配置细节
- ✅ 深入理解 RoPE 旋转位置编码
- ✅ 完整阅读 CausalSelfAttention 实现
- ✅ 理解 MQA/GQA 优化技术
- ✅ 掌握训练和推理的完整流程

**关键代码片段**：

1. **RoPE 旋转位置编码**：
```python
def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)
```

2. **Pre-Norm + 残差连接**：
```python
def forward(self, x, cos_sin, kv_cache):
    x = x + self.attn(norm(x), cos_sin, kv_cache)
    x = x + self.mlp(norm(x))
    return x
```

3. **自回归生成**：
```python
for _ in range(max_tokens):
    logits = self.forward(ids)[:, -1, :]
    next_id = sample(logits / temperature)
    ids = torch.cat((ids, next_id), dim=1)
    yield next_id.item()
```

**完整架构流程**：
```
Token IDs (B, T)
  ↓ Embedding
(B, T, 768)
  ↓ Norm
(B, T, 768)
  ↓ Block 1-12 (Attention + MLP)
(B, T, 768)
  ↓ Final Norm
(B, T, 768)
  ↓ LM Head
(B, T, 50304)
  ↓ Softcap
Logits (B, T, 50304)
```

**nanochat 的 7 大特性总结**：
1. ✅ RoPE - 旋转位置编码
2. ✅ QK Norm - Q 和 K 归一化
3. ✅ Untied Weights - Embedding 和 LM Head 独立
4. ✅ ReLU² - 简化的激活函数
5. ✅ Norm After Embedding - 稳定输入
6. ✅ No Learnable RMSNorm - 纯函数式归一化
7. ✅ No Bias - 所有 Linear 层无偏置

---

## 🚀 第三课总结

### 完整学习路径回顾

**Part 1: 为什么需要 Transformer？**
- 理解了 N-gram 和 RNN 的局限性
- 引入了 Self-Attention 的核心思想
- 建立了 Attention 是"软数据库查询"的直觉

**Part 2: Transformer 架构鸟瞰**
- 掌握了完整的 GPT 架构流程
- 理解了张量形状变化 (B, T) → (B, T, d) → (B, T, V)
- 明确了每个组件的职责

**Part 3: 核心组件代码实现**
- 手写了 RMSNorm 归一化
- 实现了 Self-Attention 机制
- 实现了 MLP 前馈网络
- 组合成完整的 Transformer Block

**Part 4: nanochat GPT 源码精读**
- 逐行阅读了 nanochat/gpt.py (308行)
- 理解了所有工程细节和优化
- 掌握了训练和推理的完整流程

### 关键概念检查清单

**基础架构** ✓
- [x] Token Embedding 的作用
- [x] RMSNorm vs LayerNorm
- [x] 残差连接的重要性
- [x] Pre-Norm vs Post-Norm

**注意力机制** ✓
- [x] Q, K, V 的含义
- [x] Scaled Dot-Product Attention
- [x] Multi-Head Attention
- [x] Causal Mask 的作用

**位置编码** ✓
- [x] 为什么需要位置信息
- [x] RoPE 的工作原理
- [x] RoPE 支持外推的原因

**高级优化** ✓
- [x] MQA/GQA 的原理
- [x] KV Cache 的作用
- [x] QK Norm 的好处
- [x] Softcap 的意义

### 参数规模计算

以 nanochat 默认配置为例：
```
vocab_size = 50304
n_embd = 768
n_layer = 12
n_head = 6

Embedding:    50304 × 768 = 38.6M
Block (单个):
  - Attention: 768 × 768 × 4 = 2.4M
  - MLP:       768 × 3072 × 2 = 4.7M
  - 小计:      7.1M
Total Blocks: 7.1M × 12 = 85.2M
LM Head:      768 × 50304 = 38.6M

总参数量: 38.6M + 85.2M + 38.6M ≈ 162M
```

---

## 🚀 下一步学习

---

## 💡 学习建议

1. **复习 Part 1-2**：
   - 重新运行代码，理解每个输出
   - 画出架构图，标注张量形状
   - 用自己的话解释每个组件

2. **做思考题**：
   - 尝试自己计算，不要马上看答案
   - 理解为什么是这个答案

3. **准备 Part 3**：
   - 复习 PyTorch 基础（Tensor, nn.Module, nn.Linear）
   - 理解矩阵乘法的形状变化
   - 准备好写代码！

---

## 📖 参考资源

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - 原始 Transformer 论文
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) - 可视化教程
- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT 架构
- [RoFormer Paper](https://arxiv.org/abs/2104.09864) - RoPE 位置编码

---

🎉 恭喜完成第三课前半部分！你已经建立了对 Transformer 的完整认知框架。

准备好继续 Part 3 了吗？或者需要先复习巩固？告诉我你的想法！
