"""
第二课 Part 1: BPE 算法核心思想
通过手工演示理解 BPE 的工作原理
"""

print("=" * 70)
print("第二课 Part 1: BPE (Byte Pair Encoding) 算法核心")
print("=" * 70)
print()

# ============================================================================
# 1. BPE 的起源和核心思想
# ============================================================================

print("1️⃣  BPE 的起源")
print("-" * 70)
print("""
BPE 最初是一个数据压缩算法（1994年）：
- 目标: 找到文本中最常见的字节对（byte pair）
- 方法: 反复合并最频繁的字节对，直到达到目标词汇表大小
- 应用: 2016 年被引入到 NLP 领域用于 subword tokenization
""")
print()

print("2️⃣  核心思想：迭代合并")
print("-" * 70)
print("""
BPE 算法的核心是一个简单的贪婪算法：

1. 初始化：每个字符都是一个 token
2. 统计：找到最频繁出现的 token 对
3. 合并：将这个 token 对合并成一个新 token
4. 重复：回到步骤 2，直到达到目标词汇表大小
""")
print()

# ============================================================================
# 2. 手工演示 BPE 训练过程
# ============================================================================

print("=" * 70)
print("3️⃣  手工演示：一步步训练 BPE")
print("=" * 70)
print()

# 训练语料
corpus = [
    "low",
    "lower",
    "newest",
    "widest"
]

print("训练语料:")
print("-" * 70)
for i, word in enumerate(corpus, 1):
    print(f"{i}. {word}")
print()

# 步骤 0: 初始化 - 每个字符都是一个 token
print("=" * 70)
print("步骤 0: 初始化")
print("=" * 70)
print()
print("将每个单词分解为字符序列:")
print("-" * 70)

# 在词尾添加特殊标记 </w> 表示词的结束
initial_tokens = []
for word in corpus:
    # 每个字符后面加空格，词尾加 </w>
    tokens = list(word) + ['</w>']
    initial_tokens.append(tokens)
    print(f"{word:10} -> {' '.join(tokens)}")

print()
print("初始词汇表（基础字符）:")
vocab = set()
for tokens in initial_tokens:
    vocab.update(tokens)
print(sorted(vocab))
print(f"词汇表大小: {len(vocab)}")
print()

# 步骤 1: 第一次合并
print("=" * 70)
print("步骤 1: 第一次合并")
print("=" * 70)
print()

# 手工统计所有 token 对的频率
print("统计所有相邻 token 对的出现频率:")
print("-" * 70)

# 当前的词表示
current_words = [
    ['l', 'o', 'w', '</w>'],      # low
    ['l', 'o', 'w', 'e', 'r', '</w>'],  # lower
    ['n', 'e', 'w', 'e', 's', 't', '</w>'],  # newest
    ['w', 'i', 'd', 'e', 's', 't', '</w>']   # widest
]

# 统计 token 对
from collections import Counter

def count_pairs(words):
    """统计所有相邻 token 对的频率"""
    pairs = Counter()
    for word in words:
        for i in range(len(word) - 1):
            pair = (word[i], word[i+1])
            pairs[pair] += 1
    return pairs

pairs = count_pairs(current_words)
for pair, count in pairs.most_common():
    print(f"  {pair[0]:5} + {pair[1]:5} -> 出现 {count} 次")
print()

# 找到最频繁的 pair
most_frequent = pairs.most_common(1)[0]
print(f"✅ 最频繁的 pair: ('{most_frequent[0][0]}', '{most_frequent[0][1]}') 出现 {most_frequent[1]} 次")
print()

# 合并这个 pair
print("合并操作:")
print("-" * 70)
merge_pair = most_frequent[0]
print(f"将所有 '{merge_pair[0]}' + '{merge_pair[1]}' 合并为 '{merge_pair[0] + merge_pair[1]}'")
print()

def merge(words, pair):
    """合并指定的 token 对"""
    new_words = []
    for word in words:
        new_word = []
        i = 0
        while i < len(word):
            # 如果找到匹配的 pair，合并
            if i < len(word) - 1 and word[i] == pair[0] and word[i+1] == pair[1]:
                new_word.append(pair[0] + pair[1])
                i += 2
            else:
                new_word.append(word[i])
                i += 1
        new_words.append(new_word)
    return new_words

current_words = merge(current_words, merge_pair)

print("合并后的词表示:")
for i, (orig, tokens) in enumerate(zip(corpus, current_words)):
    print(f"{orig:10} -> {' '.join(tokens)}")
print()

# 更新词汇表
vocab.add(merge_pair[0] + merge_pair[1])
print(f"新词汇表: {sorted(vocab)}")
print(f"词汇表大小: {len(vocab)}")
print()

# 步骤 2: 第二次合并
print("=" * 70)
print("步骤 2: 第二次合并")
print("=" * 70)
print()

pairs = count_pairs(current_words)
print("当前所有 token 对的频率:")
for pair, count in pairs.most_common():
    print(f"  {pair[0]:5} + {pair[1]:5} -> 出现 {count} 次")
print()

most_frequent = pairs.most_common(1)[0]
print(f"✅ 最频繁的 pair: ('{most_frequent[0][0]}', '{most_frequent[0][1]}') 出现 {most_frequent[1]} 次")
print()

merge_pair = most_frequent[0]
current_words = merge(current_words, merge_pair)

print("合并后的词表示:")
for i, (orig, tokens) in enumerate(zip(corpus, current_words)):
    print(f"{orig:10} -> {' '.join(tokens)}")
print()

vocab.add(merge_pair[0] + merge_pair[1])
print(f"新词汇表: {sorted(vocab)}")
print(f"词汇表大小: {len(vocab)}")
print()

# ============================================================================
# 4. BPE 的关键洞察
# ============================================================================

print("=" * 70)
print("4️⃣  BPE 的关键洞察")
print("=" * 70)
print()

print("通过上面的演示，我们可以看到：")
print("-" * 70)
print("""
1. 🎯 频率驱动：
   - BPE 总是合并最频繁的 token 对
   - 这意味着常见的字符组合会被优先学习

2. 🔄 迭代优化：
   - 每次合并后，词汇表增加 1 个 token
   - 重复这个过程，直到达到目标大小（如 32K, 50K）

3. 📊 层次结构：
   - 字符 -> 常见字符对 -> 常见词根 -> 完整的词
   - 例如: 'e' + 's' -> 'es', 'est' -> 'est', 'low' + 'est' -> 'lowest'

4. ✅ 无损压缩：
   - 任何文本都能分解回基础字符
   - 即使是训练时没见过的词也能处理
""")

# ============================================================================
# 5. BPE vs 传统方法对比
# ============================================================================

print("=" * 70)
print("5️⃣  BPE 的优势总结")
print("=" * 70)
print()

comparison = """
| 特性            | 字符级    | 词级      | BPE (Subword) |
|----------------|----------|----------|---------------|
| 词汇表大小      | ~256     | 50K-1M   | 32K-100K      |
| 序列长度        | 很长     | 短       | 适中          |
| OOV 处理        | 无 OOV   | 严重     | 无 OOV        |
| 形态变化        | 无共享   | 无共享   | ✅ 共享词根   |
| 训练成本        | 无需训练 | 简单     | 需要统计      |
| 推理速度        | 慢       | 快       | 中等          |
| 语义表达        | 弱       | 强       | ✅ 平衡       |
"""
print(comparison)

print()
print("=" * 70)
print("下一步：Part 2 - 从零实现完整的 BPE 训练算法")
print("=" * 70)
