"""
第三课 Part 4: nanochat GPT 源码精读
逐行解读 nanochat/gpt.py 的每个细节
"""

print("=" * 70)
print("第三课 Part 4: nanochat GPT 源码精读")
print("=" * 70)
print()

print("""
📖 本节目标：

我们将逐行阅读 nanochat/gpt.py (308 行代码)，理解：
1. GPTConfig - 模型配置
2. RMSNorm 和 RoPE 的实际实现
3. CausalSelfAttention - 支持 MQA/GQA 的注意力机制
4. MLP - ReLU² 激活函数
5. Block 和 GPT - 模型主体
6. Forward 和 Generate - 训练与推理

你将看到我们在 Part 3 中手写的所有概念，如何在真实代码中实现！
""")
print()

# ============================================================================
# 1. 文件头注释和特性列表
# ============================================================================

print("=" * 70)
print("1️⃣  nanochat GPT 的 7 大特性")
print("=" * 70)
print()

print("""
nanochat/gpt.py 开头的注释列出了 7 大创新特性：

```python
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
```

让我们逐一解读：

1. Rotary Embeddings (RoPE)
   ❓ 问题: 传统的位置编码是固定的或可学习的向量，加到 token embedding 上
   ✅ RoPE: 通过旋转操作注入位置信息，更灵活，支持外推

2. QK Norm
   ❓ 问题: Attention 计算时，Q 和 K 的范数可能不稳定
   ✅ QK Norm: 对 Q 和 K 做归一化，提升训练稳定性

3. Untied Weights
   ❓ 传统: Token Embedding 和 LM Head 共享权重（transpose）
   ✅ nanochat: 两者独立，给模型更多灵活性

4. ReLU² Activation
   ❓ 传统: GELU 或 SiLU（Swish）
   ✅ nanochat: ReLU²（max(0, x)²），更简单高效

5. Norm After Token Embedding
   ❓ 传统: 只在 Transformer Block 内部做 norm
   ✅ nanochat: Token Embedding 后立即 norm，稳定输入

6. No Learnable Params in RMSNorm
   ❓ 传统: RMSNorm 有可学习的 gamma 参数
   ✅ nanochat: 纯函数式 RMSNorm，无参数

7. No Bias in Linear Layers
   ❓ 传统: Linear(x) = x @ W + b
   ✅ nanochat: Linear(x) = x @ W (无 bias)

这些设计让模型更简单、更高效、更易训练！
""")
print()

# ============================================================================
# 2. GPTConfig - 配置类
# ============================================================================

print("=" * 70)
print("2️⃣  GPTConfig - 模型配置类")
print("=" * 70)
print()

print("""
【代码：nanochat/gpt.py:26-34】
----------------------------------------------------------------------
@dataclass
class GPTConfig:
    sequence_len: int = 1024      # 最大序列长度
    vocab_size: int = 50304       # 词汇表大小
    n_layer: int = 12             # Transformer 层数
    n_head: int = 6               # Query 头数
    n_kv_head: int = 6            # Key/Value 头数 (MQA)
    n_embd: int = 768             # 模型维度

关键点：

1. sequence_len = 1024
   - 训练时的最大上下文长度
   - 推理时可以超过（RoPE 支持外推）

2. vocab_size = 50304
   - 为什么不是整数（如 50000）？
   - 为了对齐：50304 = 256 × 197（对 GPU 友好）

3. n_layer = 12
   - 12 层 Transformer Block
   - GPT-2 Small 也是 12 层

4. n_head = 6, n_kv_head = 6
   - Multi-Head Attention：6 个 Query 头
   - Multi-Query Attention (MQA)：当 n_kv_head < n_head 时
   - 这里相等，表示标准的 Multi-Head Attention

5. n_embd = 768
   - 模型的隐藏层维度
   - head_dim = n_embd / n_head = 768 / 6 = 128

示例计算：
----------
假设 batch_size = 4, seq_len = 512:
- 输入 shape: (4, 512)
- Embedding: (4, 512, 768)
- 每个 Attention 头: (4, 6, 512, 128)
- MLP hidden: (4, 512, 3072)  # 4 × 768
- 输出 logits: (4, 512, 50304)
""")
print()

# ============================================================================
# 3. RMSNorm - 纯函数式归一化
# ============================================================================

print("=" * 70)
print("3️⃣  RMSNorm - 纯函数式归一化")
print("=" * 70)
print()

print("""
【代码：nanochat/gpt.py:36-38】
----------------------------------------------------------------------
def norm(x):
    # Purely functional rmsnorm with no learnable params
    return F.rms_norm(x, (x.size(-1),))

解读：

1. 纯函数式：无可学习参数
   - 不同于标准的 RMSNorm，这里没有 gamma 权重
   - 直接调用 PyTorch 的 F.rms_norm

2. F.rms_norm(x, normalized_shape)
   - 对最后一个维度做归一化
   - x.size(-1) 是最后一维的大小（d_model）

3. 数学公式：
   rms = sqrt(mean(x²))
   y = x / rms

4. 为什么无参数也能工作？
   - RMSNorm 的主要作用是稳定梯度
   - gamma 只是额外的缩放，不是必需的

示例：
-----
x = torch.randn(4, 512, 768)  # (B, T, d)
y = norm(x)  # (4, 512, 768)
# 每个位置的 768 维向量被归一化
""")
print()

# ============================================================================
# 4. RoPE - 旋转位置编码
# ============================================================================

print("=" * 70)
print("4️⃣  RoPE - 旋转位置编码")
print("=" * 70)
print()

print("""
【代码：nanochat/gpt.py:41-49】
----------------------------------------------------------------------
def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention (B, H, T, D)
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # 分成两半
    y1 = x1 * cos + x2 * sin         # 旋转操作
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)     # 重新拼接
    out = out.to(x.dtype)
    return out

RoPE 的核心思想：

1. 将向量分成两半 (d/2 对)
   x = [x1, x2, x3, x4, ..., xd]
   → [(x1, x2), (x3, x4), ..., (xd-1, xd)]

2. 每一对做 2D 旋转
   [x1']   [cos θ   sin θ ] [x1]
   [x2'] = [-sin θ  cos θ ] [x2]

   展开：
   x1' = x1 * cos θ + x2 * sin θ
   x2' = x1 * (-sin θ) + x2 * cos θ

3. 旋转角度 θ 随位置变化
   θ_t = t / base^(2i/d)
   - t: 位置索引 (0, 1, 2, ...)
   - i: 维度索引 (0, 1, 2, ...)
   - base: 基础频率 (默认 10000)

4. 为什么有效？
   - 相对位置信息被编码到向量旋转中
   - Q @ K^T 时，旋转差异反映相对距离
   - 支持外推：训练时 seq_len=1024，推理时可以超过

示例形状变化：
-------------
x:    (B=4, H=6, T=512, D=128)
d:    128 // 2 = 64
x1:   (4, 6, 512, 64)  # 前半部分
x2:   (4, 6, 512, 64)  # 后半部分
y1:   x1 * cos + x2 * sin
y2:   x1 * (-sin) + x2 * cos
out:  cat([y1, y2], dim=-1) -> (4, 6, 512, 128)

cos 和 sin 的形状：
------------------
cos: (1, seq_len, 1, d/2) -> 广播到 (B, H, T, d/2)
sin: (1, seq_len, 1, d/2)
""")
print()

# ============================================================================
# 5. CausalSelfAttention - 注意力机制
# ============================================================================

print("=" * 70)
print("5️⃣  CausalSelfAttention - 核心注意力机制")
print("=" * 70)
print()

print("""
【类初始化：nanochat/gpt.py:51-64】
----------------------------------------------------------------------
class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head          # 6
        self.n_kv_head = config.n_kv_head    # 6
        self.n_embd = config.n_embd          # 768
        self.head_dim = self.n_embd // self.n_head  # 128

        # 确保维度可以整除
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head

        # Q, K, V 投影层（无 bias）
        self.c_q = nn.Linear(768, 6 × 128, bias=False)
        self.c_k = nn.Linear(768, 6 × 128, bias=False)
        self.c_v = nn.Linear(768, 6 × 128, bias=False)

        # 输出投影层
        self.c_proj = nn.Linear(768, 768, bias=False)

关键点：

1. layer_idx: 层索引（用于 KV Cache）

2. head_dim = n_embd / n_head
   - 每个头的维度
   - 768 / 6 = 128

3. Multi-Query Attention (MQA) 支持
   - 标准 MHA: n_head = n_kv_head = 6
   - MQA: n_kv_head = 1，所有 query 头共享一个 KV
   - GQA: 1 < n_kv_head < n_head，分组共享

4. 无 bias
   - 所有 Linear 层都设置 bias=False
   - 简化模型，减少参数
""")
print()

print("""
【前向传播：nanochat/gpt.py:66-110】
----------------------------------------------------------------------

def forward(self, x, cos_sin, kv_cache):
    B, T, C = x.size()  # (batch, seq_len, n_embd)

    # 步骤 1: 投影到 Q, K, V
    q = self.c_q(x).view(B, T, n_head, head_dim)
    k = self.c_k(x).view(B, T, n_kv_head, head_dim)
    v = self.c_v(x).view(B, T, n_kv_head, head_dim)

    # 步骤 2: 应用 RoPE
    cos, sin = cos_sin
    q = apply_rotary_emb(q, cos, sin)
    k = apply_rotary_emb(k, cos, sin)

    # 步骤 3: QK Norm
    q = norm(q)
    k = norm(k)

    # 步骤 4: 调整维度顺序（B, T, H, D -> B, H, T, D）
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # 步骤 5: KV Cache（推理加速）
    if kv_cache is not None:
        k, v = kv_cache.insert_kv(self.layer_idx, k, v)

    # 步骤 6: Scaled Dot-Product Attention
    y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    # 步骤 7: 重组多头（B, H, T, D -> B, T, H×D）
    y = y.transpose(1, 2).contiguous().view(B, T, C)

    # 步骤 8: 输出投影
    y = self.c_proj(y)

    return y

详细解读每个步骤：

步骤 1: Q, K, V 投影
--------------------
输入: x = (4, 512, 768)
c_q(x): (4, 512, 768) @ (768, 768) = (4, 512, 768)
view: (4, 512, 768) -> (4, 512, 6, 128)
      ↑   ↑    ↑         ↑   ↑   ↑   ↑
      B   T   C          B   T   H  D_head

步骤 2: RoPE
------------
为 Q 和 K 添加位置信息：
q: (4, 512, 6, 128) + RoPE -> (4, 512, 6, 128)
k: (4, 512, 6, 128) + RoPE -> (4, 512, 6, 128)
v: 不需要 RoPE（只有 Q 和 K 需要位置信息）

步骤 3: QK Norm
---------------
归一化 Q 和 K（提升稳定性）：
q = norm(q)  # 对最后一个维度（128）做归一化
k = norm(k)

步骤 4: 维度调整
---------------
从 (B, T, H, D) 调整为 (B, H, T, D)：
- 让 Head 成为 batch 维度
- 方便并行计算每个头的 attention

步骤 5: KV Cache
---------------
训练时: kv_cache = None，跳过
推理时: 缓存历史的 K, V，避免重复计算

步骤 6: Attention 计算
---------------------
F.scaled_dot_product_attention:
  scores = Q @ K^T / sqrt(D_head)
  weights = softmax(scores) with causal_mask
  output = weights @ V

is_causal=True: 自动应用因果掩码
- token 只能看到它之前的 token
- 实现自回归生成

步骤 7-8: 输出处理
-----------------
transpose: (B, H, T, D) -> (B, T, H, D)
view: (B, T, H, D) -> (B, T, H×D)
c_proj: (B, T, 768) -> (B, T, 768)

完整形状变化链：
--------------
输入:         (4, 512, 768)
Q, K, V:      (4, 512, 6, 128)
RoPE:         (4, 512, 6, 128)
Transpose:    (4, 6, 512, 128)
Attention:    (4, 6, 512, 128)
Re-assemble:  (4, 512, 768)
输出:         (4, 512, 768)

形状始终不变：(B, T, d_model)
但内容已经通过 attention 更新！
""")
print()

# ============================================================================
# 6. MLP - 前馈网络
# ============================================================================

print("=" * 70)
print("6️⃣  MLP - 前馈神经网络")
print("=" * 70)
print()

print("""
【代码：nanochat/gpt.py:113-123】
----------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=False)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)          # (B, T, 768) -> (B, T, 3072)
        x = F.relu(x).square()    # ReLU²
        x = self.c_proj(x)        # (B, T, 3072) -> (B, T, 768)
        return x

解读：

1. 两层 Linear 网络
   - 扩展: 768 → 3072 (4 倍)
   - 压缩: 3072 → 768

2. ReLU² 激活函数
   - 传统: GELU(x) 或 SiLU(x)
   - nanochat: ReLU(x)² = max(0, x)²

   为什么用 ReLU²？
   ✅ 更简单：无复杂的指数计算
   ✅ 更快：GPU 友好
   ✅ 效果相当：实验表明性能相近

3. 形状变化
   输入: (4, 512, 768)
   c_fc: (4, 512, 3072)  # 扩展 4 倍
   relu²: (4, 512, 3072)  # 激活
   c_proj: (4, 512, 768)  # 压缩回原维度

4. 为什么要 4 倍扩展？
   - 给模型更多的表达空间
   - MLP 被认为是存储"知识"的地方
   - 4 倍是经验法则（Transformer 论文）

5. 无 bias
   - 两个 Linear 层都没有 bias
   - 简化模型结构
""")
print()

# ============================================================================
# 7. Block - Transformer 块
# ============================================================================

print("=" * 70)
print("7️⃣  Block - Transformer 块")
print("=" * 70)
print()

print("""
【代码：nanochat/gpt.py:126-135】
----------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, cos_sin, kv_cache):
        x = x + self.attn(norm(x), cos_sin, kv_cache)  # 残差连接
        x = x + self.mlp(norm(x))                      # 残差连接
        return x

解读：

这就是我们在 Part 3 中手写的 Transformer Block！

1. Pre-Norm 架构
   - 先 norm，再 attn/mlp
   - 相比 Post-Norm 更稳定

2. 残差连接
   x = x + f(x)
   - f(x) 是 attention 或 MLP 的输出
   - 直接相加，保证梯度流畅

3. 完整流程
   输入 x: (B, T, d)
   ↓
   norm(x) -> Attention -> + x  (残差)
   ↓
   norm(x) -> MLP -> + x  (残差)
   ↓
   输出 x: (B, T, d)

4. 对比我们的手写版本
   我们写的:
   ```python
   x_norm1 = rms_norm(x, weight1)
   attn_out = self_attention(x_norm1)
   x = x + attn_out

   x_norm2 = rms_norm(x, weight2)
   mlp_out = mlp(x_norm2)
   x = x + mlp_out
   ```

   nanochat 的:
   ```python
   x = x + self.attn(norm(x), cos_sin, kv_cache)
   x = x + self.mlp(norm(x))
   ```

   完全一致！🎉

5. 为什么要两次 norm？
   - 每个子层之前都 norm
   - 确保输入的分布稳定
   - 有助于深层网络的训练
""")
print()

# ============================================================================
# 8. GPT - 完整模型
# ============================================================================

print("=" * 70)
print("8️⃣  GPT - 完整的模型")
print("=" * 70)
print()

print("""
【模型初始化：nanochat/gpt.py:138-156】
----------------------------------------------------------------------
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(vocab_size, n_embd),
            "h": nn.ModuleList([Block(config, i)
                               for i in range(n_layer)]),
        })
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

        # 预计算 RoPE embeddings
        self.rotary_seq_len = config.sequence_len * 10
        cos, sin = self._precompute_rotary_embeddings(...)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

结构分解：

1. Token Embedding (wte)
   - 输入: token IDs (B, T)
   - 输出: embeddings (B, T, 768)
   - 参数量: vocab_size × n_embd = 50304 × 768 ≈ 38M

2. Transformer Blocks (h)
   - n_layer = 12 个 Block
   - 每个 Block 包含 Attention + MLP
   - 堆叠起来形成深层网络

3. LM Head
   - 输入: (B, T, 768)
   - 输出: (B, T, 50304)  # 词汇表概率
   - 参数量: 768 × 50304 ≈ 38M

4. RoPE Embeddings
   - 预计算 cos 和 sin 表
   - rotary_seq_len = 1024 × 10 = 10240
   - persistent=False: 不保存到 checkpoint

5. 参数统计
   Embedding:   38M
   12 × Block:  ~70M (每个 Block 约 6M)
   LM Head:     38M
   Total:       ~146M 参数
""")
print()

print("""
【前向传播：nanochat/gpt.py:244-276】
----------------------------------------------------------------------
def forward(self, idx, targets=None, kv_cache=None):
    B, T = idx.size()

    # 1. 获取 RoPE embeddings
    T0 = 0 if kv_cache is None else kv_cache.get_pos()
    cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T]

    # 2. Token Embedding + Norm
    x = self.transformer.wte(idx)
    x = norm(x)

    # 3. 通过所有 Transformer Blocks
    for block in self.transformer.h:
        x = block(x, cos_sin, kv_cache)

    # 4. Final Norm
    x = norm(x)

    # 5. LM Head
    logits = self.lm_head(x)
    softcap = 15
    logits = softcap * torch.tanh(logits / softcap)

    # 6. 计算 loss（如果有 targets）
    if targets is not None:
        loss = F.cross_entropy(logits.view(-1, vocab_size),
                               targets.view(-1))
        return loss
    else:
        return logits

完整数据流：

输入:           (B, T) = (4, 512)
              ↓
Embedding:     (4, 512, 768)
              ↓
Norm:          (4, 512, 768)
              ↓
Block 1:       (4, 512, 768)
Block 2:       (4, 512, 768)
...
Block 12:      (4, 512, 768)
              ↓
Final Norm:    (4, 512, 768)
              ↓
LM Head:       (4, 512, 50304)
              ↓
Softcap:       (4, 512, 50304)

关键技术：

1. Softcap (logits 软限制)
   logits = 15 * tanh(logits / 15)
   - 限制 logits 的范围在 (-15, 15)
   - 防止过大的 logits 导致数值不稳定
   - 类似于梯度裁剪的思想

2. 训练模式 vs 推理模式
   训练: targets != None, 返回 loss
   推理: targets == None, 返回 logits

3. KV Cache
   训练: kv_cache = None
   推理: kv_cache 缓存历史 K, V
""")
print()

# ============================================================================
# 9. Generate - 自回归生成
# ============================================================================

print("=" * 70)
print("9️⃣  Generate - 自回归生成")
print("=" * 70)
print()

print("""
【代码：nanochat/gpt.py:278-308】
----------------------------------------------------------------------
@torch.inference_mode()
def generate(self, tokens, max_tokens, temperature=1.0,
             top_k=None, seed=42):
    # tokens: 初始 token 序列（Python list）
    device = self.get_device()
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    ids = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_tokens):
        # 1. 前向传播
        logits = self.forward(ids)  # (1, T, vocab_size)
        logits = logits[:, -1, :]   # (1, vocab_size) 只取最后一个位置

        # 2. Top-k 采样
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, vocab_size))
            logits[logits < v[:, [-1]]] = -float('Inf')

        # 3. 温度采样
        if temperature > 0:
            logits = logits / temperature
            probs = F.softmax(logits, dim=-1)
            next_ids = torch.multinomial(probs, num_samples=1,
                                        generator=rng)
        else:
            # 贪婪解码
            next_ids = torch.argmax(logits, dim=-1, keepdim=True)

        # 4. 拼接新 token
        ids = torch.cat((ids, next_ids), dim=1)

        # 5. 返回新 token
        token = next_ids.item()
        yield token

生成过程详解：

初始状态:
---------
tokens = [1, 2, 3]  # 初始 prompt
max_tokens = 10

迭代 1:
-------
ids = [1, 2, 3]
logits = forward(ids)  # (1, 3, 50304)
logits = logits[:, -1, :]  # (1, 50304) 只取位置 2
probs = softmax(logits / temperature)
next_id = sample(probs)  # 假设采样到 4
ids = [1, 2, 3, 4]
yield 4

迭代 2:
-------
ids = [1, 2, 3, 4]
logits = forward(ids)  # (1, 4, 50304)
logits = logits[:, -1, :]  # (1, 50304) 只取位置 3
next_id = sample(...)  # 假设采样到 5
ids = [1, 2, 3, 4, 5]
yield 5

...重复 max_tokens 次

关键技术：

1. @torch.inference_mode()
   - 禁用梯度计算
   - 推理加速

2. Top-k 采样
   - 只保留概率最高的 k 个 token
   - 避免采样到低质量的 token

3. 温度采样
   - temperature = 0: 贪婪解码（确定性）
   - temperature = 1: 标准采样
   - temperature > 1: 更随机（创造性）
   - temperature < 1: 更确定（保守）

4. 自回归生成
   - 每次只生成一个 token
   - 新 token 拼接到输入序列
   - 重复这个过程

5. yield 生成器
   - 逐个返回生成的 token
   - 支持流式输出
   - 用户可以实时看到生成过程
""")
print()

# ============================================================================
# 10. 总结与对比
# ============================================================================

print("=" * 70)
print("🎓 总结：从手写到真实代码")
print("=" * 70)
print()

print("""
恭喜！你已经完整阅读了 nanochat 的 GPT 实现。

对比我们的学习路径：

Part 1: 为什么需要 Transformer？
  ✅ 理解了动机：RNN 的局限
  ✅ 引入了 Self-Attention 概念

Part 2: Transformer 架构鸟瞰
  ✅ 掌握了完整的数据流
  ✅ 理解了张量形状变化
  ✅ 了解了 nanochat 的创新

Part 3: 手写核心组件
  ✅ 实现了 RMSNorm
  ✅ 实现了 Self-Attention
  ✅ 实现了 MLP
  ✅ 组合成 Transformer Block

Part 4: 阅读真实代码
  ✅ 看到了所有概念的实际应用
  ✅ 理解了工程细节（RoPE, QK Norm, KV Cache）
  ✅ 掌握了训练和推理流程

你现在已经完全理解了：
---------------------
1. Transformer 的数学原理
2. nanochat 的实现细节
3. 如何从零构建一个 GPT 模型
4. 训练和推理的完整流程

下一步建议：
-----------
1. 运行 nanochat 的训练脚本
   python train.py

2. 修改超参数，观察影响
   - 改变 n_layer, n_head, n_embd
   - 尝试不同的学习率

3. 阅读训练代码
   - nanochat/train.py
   - nanochat/engine.py
   - nanochat/optimizer.py

4. 继续学习第四课
   - 深入注意力机制的数学
   - 理解 RoPE 的理论基础
   - 学习优化器原理

🎉 第三课完成！你已经掌握了 Transformer 的核心！
""")
print()

print("=" * 70)
print("✅ Part 4 完成！第三课全部结束！")
print("=" * 70)
