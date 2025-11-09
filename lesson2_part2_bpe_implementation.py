"""
第二课 Part 2: 从零实现 BPE 训练
完整的 BPE tokenizer 实现（教学版）
"""

from collections import Counter, defaultdict
import re

print("=" * 70)
print("第二课 Part 2: 从零实现 BPE 训练器")
print("=" * 70)
print()

# ============================================================================
# 1. BPE Trainer 类
# ============================================================================

class SimpleBPETrainer:
    """
    简化版的 BPE 训练器
    这个实现与 nanochat 的 RustBPE 逻辑相同，但用 Python 写成便于理解
    """

    def __init__(self, vocab_size=300):
        """
        参数:
            vocab_size: 目标词汇表大小
        """
        self.vocab_size = vocab_size
        self.merges = []  # 存储所有合并操作的顺序
        self.vocab = {}   # 最终的词汇表 {token: id}

    def _get_initial_vocab(self, word_freqs):
        """
        步骤 1: 构建初始词汇表（所有出现过的字符）

        参数:
            word_freqs: {word: frequency} 词频字典
        返回:
            vocab: {char: id} 字符到 ID 的映射
        """
        vocab = set()
        for word in word_freqs.keys():
            vocab.update(word)

        # 按字母顺序排序，分配 ID
        vocab = sorted(list(vocab))
        vocab_dict = {char: idx for idx, char in enumerate(vocab)}

        print(f"📊 初始词汇表大小: {len(vocab_dict)}")
        print(f"   包含字符: {vocab[:20]}..." if len(vocab) > 20 else f"   包含字符: {vocab}")
        print()

        return vocab_dict

    def _get_pair_frequencies(self, splits):
        """
        步骤 2: 统计所有相邻 token 对的频率

        参数:
            splits: {word: (tokens, freq)} 词的分解和频率
        返回:
            pair_freqs: Counter 对象，统计每个 pair 的出现次数
        """
        pair_freqs = Counter()

        for word, (tokens, freq) in splits.items():
            # 遍历这个词中的所有相邻 token 对
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pair_freqs[pair] += freq

        return pair_freqs

    def _merge_pair(self, pair, splits):
        """
        步骤 3: 合并指定的 token 对

        参数:
            pair: (token1, token2) 要合并的 token 对
            splits: {word: (tokens, freq)} 当前的词分解
        返回:
            new_splits: 合并后的新分解
        """
        new_splits = {}
        new_token = pair[0] + pair[1]

        for word, (tokens, freq) in splits.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                # 如果找到匹配的 pair，合并
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            new_splits[word] = (new_tokens, freq)

        return new_splits

    def train(self, corpus):
        """
        完整的 BPE 训练流程

        参数:
            corpus: list[str] 训练语料（字符串列表）
        """
        print("🚀 开始 BPE 训练")
        print("=" * 70)
        print()

        # 1. 统计词频
        print("步骤 1: 统计词频")
        print("-" * 70)
        word_freqs = Counter(corpus)
        print(f"总共 {len(corpus)} 个 token，{len(word_freqs)} 个不同的词")
        print(f"示例: {list(word_freqs.items())[:5]}")
        print()

        # 2. 初始化：每个字符都是一个 token
        print("步骤 2: 初始化词汇表")
        print("-" * 70)
        self.vocab = self._get_initial_vocab(word_freqs)
        base_vocab_size = len(self.vocab)

        # 3. 初始分解：每个词分解为字符序列
        splits = {}
        for word, freq in word_freqs.items():
            splits[word] = (list(word), freq)

        # 4. 迭代合并
        print("步骤 3: 开始迭代合并")
        print("-" * 70)
        num_merges = self.vocab_size - base_vocab_size
        print(f"目标词汇表大小: {self.vocab_size}")
        print(f"需要进行 {num_merges} 次合并")
        print()

        for merge_idx in range(num_merges):
            # 统计 pair 频率
            pair_freqs = self._get_pair_frequencies(splits)

            if not pair_freqs:
                print(f"⚠️  没有更多的 pair 可以合并了")
                break

            # 找到最频繁的 pair
            best_pair = pair_freqs.most_common(1)[0][0]
            best_freq = pair_freqs[best_pair]

            # 合并这个 pair
            splits = self._merge_pair(best_pair, splits)

            # 记录这次合并
            self.merges.append(best_pair)
            new_token = best_pair[0] + best_pair[1]
            self.vocab[new_token] = base_vocab_size + merge_idx

            # 每 50 次打印一次进度
            if (merge_idx + 1) % 50 == 0 or merge_idx < 5:
                print(f"  合并 {merge_idx + 1:3d}: ('{best_pair[0]}', '{best_pair[1]}') "
                      f"-> '{new_token}' (频率: {best_freq})")

        print()
        print(f"✅ 训练完成！最终词汇表大小: {len(self.vocab)}")
        print()

    def encode(self, text):
        """
        使用训练好的 BPE 编码文本

        参数:
            text: str 要编码的文本
        返回:
            tokens: list[str] token 序列
        """
        # 从字符开始
        tokens = list(text)

        # 应用所有学习到的合并操作
        for pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def get_token_ids(self, text):
        """
        编码文本并返回 token ID 序列
        """
        tokens = self.encode(text)
        return [self.vocab.get(t, 0) for t in tokens]  # 0 是 unknown token


# ============================================================================
# 2. 测试 BPE 训练器
# ============================================================================

print("=" * 70)
print("实战演示：在真实语料上训练 BPE")
print("=" * 70)
print()

# 准备训练语料（一些简单的英文单词）
training_corpus = [
    "the", "the", "the", "the", "the", "the",  # the 出现 6 次
    "quick", "quick", "quick",  # quick 出现 3 次
    "brown", "brown",  # brown 出现 2 次
    "fox", "fox",
    "jumped", "jumped",
    "over",
    "lazy",
    "dog",
    "running", "running", "running",
    "jumper",
    "quicker",
    "quickly",
]

print(f"训练语料: {len(training_corpus)} 个 token")
print(f"示例: {training_corpus[:10]}")
print()

# 训练 BPE
trainer = SimpleBPETrainer(vocab_size=100)
trainer.train(training_corpus)

# ============================================================================
# 3. 测试编码效果
# ============================================================================

print("=" * 70)
print("测试编码效果")
print("=" * 70)
print()

test_words = [
    "the",           # 训练时见过
    "quick",         # 训练时见过
    "runner",        # 没见过，但有共同词根 "run"
    "quickly",       # 训练时见过
    "queenly",       # 没见过，但有共同部分 "qu"
    "xyz",           # 完全没见过
]

for word in test_words:
    tokens = trainer.encode(word)
    token_ids = trainer.get_token_ids(word)
    print(f"{word:15} -> {str(tokens):30} -> IDs: {token_ids}")

print()

# ============================================================================
# 4. 关键洞察
# ============================================================================

print("=" * 70)
print("🔑 关键洞察")
print("=" * 70)
print()

print("""
1. 训练过程：
   ✅ 从字符开始，逐步学习常见的字符组合
   ✅ 频率驱动：越常见的组合越早被学习
   ✅ 层次结构：字符 -> 字符对 -> 词根 -> 完整词

2. 编码过程：
   ✅ 贪婪匹配：总是优先匹配最长的 token
   ✅ 顺序重要：按训练时的合并顺序应用
   ✅ 无 OOV：任何文本都能分解为已知 token

3. 与 nanochat 的联系：
   ✅ nanochat 的 RustBPE 用 Rust 实现了相同的算法
   ✅ tiktoken 用 C++ 优化了编码速度
   ✅ 本质算法是一样的！

4. 下一步：
   我们将深入 nanochat 的 RustBPE 源码
   理解它如何在 Rust 中高效实现这个算法
""")

print()
print("=" * 70)
print("下一步：Part 3 - 深入 nanochat 的 RustBPE 实现")
print("=" * 70)
