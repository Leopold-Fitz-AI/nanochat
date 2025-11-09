# 第二课总结：深入 BPE 算法和实现 🧬

## 🎯 学习目标回顾

通过今天的学习，你应该掌握了：

1. ✅ **BPE 算法的核心思想**
   - 迭代合并：反复找到最频繁的 token 对并合并
   - 频率驱动：越常见的组合越早被学习
   - 层次结构：字符 → 字符对 → 词根 → 完整词

2. ✅ **完整的 BPE 训练流程**
   - 初始化：每个字符都是一个 token
   - 统计：计算所有相邻 token 对的频率
   - 合并：选择最频繁的 pair 并合并
   - 重复：直到达到目标词汇表大小

3. ✅ **nanochat RustBPE 的实现细节**
   - 并行化：使用 rayon 多线程加速统计
   - 堆优化：用最大堆高效找到最频繁 pair
   - Lazy Refresh：延迟刷新策略避免不必要的更新
   - 增量更新：只更新受影响的词

4. ✅ **编码算法的原理**
   - 贪婪匹配：优先匹配最早学到的 merge
   - merge ID 越小越优先（训练时越早学到）
   - 保证编码与训练一致

---

## 📚 课程内容回顾

### Part 1: BPE 核心思想 (lesson2_part1_bpe_core.py)

**手工演示**：
```
训练语料: ["low", "lower", "newest", "widest"]

初始: l o w   l o w e r   n e w e s t   w i d e s t

第1次合并: 'l' + 'o' -> 'lo' (频率: 2)
结果: lo w   lo w e r   n e w e s t   w i d e s t

第2次合并: 'lo' + 'w' -> 'low' (频率: 2)
结果: low   low e r   n e w e s t   w i d e s t
```

**关键洞察**：
- BPE 自动学习常见的字符组合
- 形成层次化的词汇表
- 完全数据驱动，无需人工规则

---

### Part 2: Python 实现 (lesson2_part2_bpe_implementation.py)

**核心代码结构**：
```python
class SimpleBPETrainer:
    def train(self, corpus):
        # 1. 统计词频
        word_freqs = Counter(corpus)

        # 2. 初始化词汇表（所有字符）
        vocab = self._get_initial_vocab(word_freqs)

        # 3. 初始分解
        splits = {word: (list(word), freq) for word, freq in word_freqs.items()}

        # 4. 迭代合并
        for _ in range(num_merges):
            # 4.1 统计 pair 频率
            pair_freqs = self._get_pair_frequencies(splits)

            # 4.2 找到最频繁的 pair
            best_pair = pair_freqs.most_common(1)[0][0]

            # 4.3 合并
            splits = self._merge_pair(best_pair, splits)

            # 4.4 记录
            self.merges.append(best_pair)
            self.vocab[best_pair[0] + best_pair[1]] = next_id

    def encode(self, text):
        # 从字符开始
        tokens = list(text)

        # 应用所有 merges
        for pair in self.merges:
            tokens = apply_merge(tokens, pair)

        return tokens
```

**测试结果**：
```
the       -> ['the']                  (完整学习)
runner    -> ['runn', 'er']           (共享词根)
queenly   -> ['qu', 'e', 'e', 'n', 'l', 'y']  (分解为已知部分)
xyz       -> ['x', 'y', 'z']          (退回字符级)
```

---

### Part 3: RustBPE 源码精读 (lesson2_part3_rustbpe_explained.py)

**核心数据结构**：

1. **Word** - 词的表示
   ```rust
   struct Word {
       ids: Vec<u32>,  // token ID 序列
   }
   ```

2. **MergeJob** - 堆中的元素
   ```rust
   struct MergeJob {
       pair: Pair,           // (token_a, token_b)
       count: u64,           // 全局频率
       pos: AHashSet<usize>, // 包含此 pair 的词索引
   }
   ```

**关键优化**：

1. **并行化统计**
   ```rust
   let (pair_counts, where_to_update) = words
       .par_iter()  // 并行迭代
       .map(|word| count_pairs_in_word(word))
       .reduce(|a, b| merge_counts(a, b));
   ```

2. **堆优化**
   - 用最大堆维护 pair 频率
   - 从 O(n) 降到 O(log n)
   - 使用 8-叉堆提升 cache 命中率

3. **Lazy Refresh**
   ```rust
   let current = pair_counts.get(&top.pair);
   if top.count != current {
       top.count = current;
       heap.push(top);  // 重新入堆
       continue;
   }
   ```

4. **增量更新**
   - 只更新包含该 pair 的词（通过 pos 集合）
   - 不遍历所有词

**性能对比**：

| 特性 | Python | RustBPE |
|-----|--------|---------|
| 训练速度 | 1x | 20-50x |
| 编码速度 | 1x | 100-200x |
| 并行化 | ❌ | ✅ |
| Lazy Refresh | ❌ | ✅ |

---

## 🔑 核心概念

### 1. BPE 训练的本质

BPE 训练就是在构建一个**最优压缩码本**：
- 把常见的字节序列编码为单个 token
- 减少序列长度，提升模型效率
- 同时保持无损可逆性（任何文本都能编码）

### 2. 为什么选择最小 merge ID？

训练时的合并顺序代表频率：
```
merge 1: (t, h) -> 256   (最频繁)
merge 2: (256, e) -> 257  (the)
merge 3: (a, n) -> 258
...
```

编码时选择 ID 最小的，等价于选择**最早学到的**，即**最频繁的**：
```
"the" 可以匹配:
  (t, h) -> 256
  (256, e) -> 257

选择 256 (ID 更小) -> [256, e]
继续匹配:
  (256, e) -> 257 -> [257]
```

### 3. Lazy Refresh 的精妙之处

**问题**：合并 (a, b) 会影响其他 pair 的频率

**朴素方案**：立即更新堆中所有受影响的 pair
- 需要频繁的堆操作
- 大量 pair 的频率变化很小

**Lazy Refresh**：等到 pop 时再检查
- 大部分 pair 不会被 pop 到
- 避免了不必要的堆操作
- 只有真正需要的才更新

### 4. 增量更新的威力

**关键思想**：维护 `pos: AHashSet<usize>`

合并 (a, b) 时：
- 不遍历所有词
- 只遍历 `pos` 中记录的词（包含 (a, b) 的词）
- 通常只占总词数的很小一部分

**效果**：
- 朴素方法：O(num_words × num_merges)
- 增量方法：O(affected_words × num_merges)
- affected_words << num_words

---

## 💡 实战技巧

### 1. 如何调试 BPE 训练

```python
# 打印前 10 次合并
for i, (pair, freq) in enumerate(merge_history[:10]):
    print(f"Merge {i}: {pair} (freq: {freq})")

# 检查特定词的编码
word = "hello"
tokens = tokenizer.encode(word)
print(f"{word} -> {tokens}")
```

### 2. 如何选择词汇表大小

| 词汇表大小 | 适用场景 | 特点 |
|-----------|---------|-----|
| 8K-16K | 小模型/资源受限 | 序列稍长，词汇表小 |
| 32K-50K | 中型模型 | ✅ 平衡点 |
| 100K+ | 大型模型 | 序列短，词汇表大 |

**经验法则**：
- GPT-2: 50K
- GPT-3/4: 50K-100K
- nanochat: 65K (2^16)

### 3. 如何处理多语言

BPE 天然支持多语言（基于字节）：
```python
# 中英文混合
text = "Hello 你好 world 世界"
tokens = tokenizer.encode(text)
# UTF-8 编码 -> 字节序列 -> BPE tokens
```

**优势**：
- 无需为每种语言单独训练
- 自动学习常见的跨语言模式
- 统一的词汇表

---

## 🎓 练习作业

### 练习 1: 手工 BPE (基础)

给定语料：
```
["cat", "cats", "dog", "dogs"]
```

目标词汇表大小：15

请手工执行 BPE 训练：
1. 列出初始词汇表（所有字符）
2. 进行前 3 次合并（写出每次的 pair 和频率）
3. 编码 "catalog" 和 "doggy"

<details>
<summary>点击查看答案</summary>

初始词汇表：{a, c, d, g, o, s, t}  (7 个)

合并历史：
1. ('c', 'a') -> 'ca' (频率: 2)
   cat -> ca t,  cats -> ca ts
2. ('ca', 't') -> 'cat' (频率: 2)
   cat -> cat,  cats -> cat s
3. ('d', 'o') -> 'do' (频率: 2)
   dog -> do g,  dogs -> do gs

编码结果：
- catalog -> ['cat', 'a', 'l', 'o', 'g']
- doggy -> ['do', 'g', 'g', 'y']
</details>

---

### 练习 2: 理解增量更新 (进阶)

假设我们有 1000 个唯一词，现在要合并 pair (t, h)。

问题：
1. 朴素方法需要检查多少个词？
2. 如果只有 50 个词包含 (t, h)，增量方法需要检查多少个词？
3. 加速比是多少？

<details>
<summary>点击查看答案</summary>

1. 朴素方法：检查所有 1000 个词
2. 增量方法：只检查 50 个词
3. 加速比：1000 / 50 = 20x
</details>

---

### 练习 3: 阅读 RustBPE 源码 (高级)

阅读 `rustbpe/src/lib.rs` 中的 `merge_pair` 函数（line 51-89），回答：

1. 为什么要计算 `left` 和 `right`？
2. deltas 数组记录了什么信息？
3. 为什么要返回 deltas 而不是直接更新全局统计？

<details>
<summary>点击查看提示</summary>

1. left 和 right 是为了更新**受影响的 pair**：
   - 合并 (a, b) -> c 后
   - 左边的 (x, a) 变成 (x, c)
   - 右边的 (b, y) 变成 (c, y)

2. deltas 记录了局部的 pair 频率变化：
   - -1 表示删除了一个 pair
   - +1 表示创建了一个新 pair

3. 返回 deltas 的原因：
   - merge_pair 只处理单个词
   - 全局统计需要乘以这个词的频率（counts[i]）
   - 分离关注点，提升代码清晰度
</details>

---

## 🚀 下节预告

**第三课：Transformer 基础架构**

我们将学习：
1. Transformer 的整体架构
2. Self-Attention 机制的原理
3. Feed-Forward Network 的作用
4. Layer Normalization 的细节
5. nanochat 的 GPT 模型逐行解读

---

## 💡 学习建议

1. **循序渐进**：
   - 先理解 Part 1 的手工演示
   - 再看 Part 2 的 Python 实现
   - 最后对比 Part 3 的 Rust 优化

2. **动手实践**：
   - 运行所有示例代码
   - 修改参数观察效果
   - 尝试在不同语料上训练

3. **深入源码**：
   - 对照 Python 和 Rust 版本
   - 理解每个优化的原理
   - 思考如何应用到其他场景

4. **准备下一课**：
   - 预习 `nanochat/gpt.py`
   - 复习线性代数（矩阵乘法）
   - 了解 Transformer 的基本概念

---

## 📖 参考资源

- [BPE 原论文](https://arxiv.org/abs/1508.07909) - Neural Machine Translation of Rare Words with Subword Units
- [GPT-2 Tokenizer](https://github.com/openai/gpt-2) - OpenAI 的实现
- [tiktoken](https://github.com/openai/tiktoken) - OpenAI 的高效 BPE 库
- [Hugging Face Tokenizers](https://github.com/huggingface/tokenizers) - 功能完整的 tokenizer 库

---

🎉 恭喜你完成第二课！你现在已经深入理解了 BPE 算法的每个细节。

准备好开始第三课了吗？让我们一起探索 Transformer 的奥秘！
