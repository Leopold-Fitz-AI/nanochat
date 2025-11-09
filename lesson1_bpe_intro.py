"""
第一课：BPE (Byte Pair Encoding) 简介
Subword tokenization 的核心思想
"""

def simple_bpe_demo():
    """
    简化版 BPE 演示（手工构造的例子）
    """
    # 假设我们已经训练好的 BPE 词汇表
    # 包含：单个字符 + 常见的字符对 + 常见的词
    vocab = {
        # 单字符 (总是在词汇表中，作为后备)
        'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7,
        'i': 8, 'j': 9, 'k': 10, 'l': 11, 'm': 12, 'n': 13, 'o': 14,
        'p': 15, 'q': 16, 'r': 17, 's': 18, 't': 19, 'u': 20, 'v': 21,
        'w': 22, 'x': 23, 'y': 24, 'z': 25,
        ' ': 26,

        # 常见字符对
        'th': 27, 'he': 28, 'in': 29, 'er': 30, 'an': 31, 'en': 32,
        'ing': 33, 'ed': 34, 'ly': 35,

        # 常见单词
        'the': 36, 'and': 37, 'run': 38, 'quick': 39,

        # 常见词根
        'runn': 40,  # run + n
        'running': 41,  # 完整的词
        'runs': 42,     # run + s
    }

    # 反向映射
    id_to_token = {v: k for k, v in vocab.items()}

    print("=" * 70)
    print("BPE 词汇表示例 (部分):")
    print("=" * 70)
    print("单字符:", [k for k in vocab.keys() if len(k) == 1][:10], "...")
    print("字符对:", [k for k in vocab.keys() if len(k) == 2])
    print("常见词:", [k for k in vocab.keys() if len(k) > 2])
    print()

    # 简化的 BPE 分词函数
    def tokenize_bpe(text, vocab):
        """
        贪婪地匹配最长的 subword
        """
        tokens = []
        i = 0
        text = text.lower()

        while i < len(text):
            # 尝试匹配最长的 subword
            matched = False
            for length in range(min(10, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in vocab:
                    tokens.append(vocab[substr])
                    i += length
                    matched = True
                    break

            if not matched:
                # 如果没有匹配，跳过这个字符
                i += 1

        return tokens

    def decode_bpe(ids, id_to_token):
        """
        解码：拼接所有 subword
        """
        return ''.join([id_to_token[i] for i in ids])

    # 示例 1: 基本分词
    print("=" * 70)
    print("示例 1: 基本分词")
    print("=" * 70)
    examples = ["the", "quick", "brown", "fox", "running", "runs"]
    for word in examples:
        ids = tokenize_bpe(word, vocab)
        tokens = [id_to_token[i] for i in ids]
        print(f"{word:12} -> {str(ids):20} -> {tokens}")
    print()

    # 示例 2: 处理未登录词
    print("=" * 70)
    print("示例 2: 处理未登录词 (优势)")
    print("=" * 70)
    oov_words = ["purple", "unbelievable", "supercalifragilistic"]
    for word in oov_words:
        ids = tokenize_bpe(word, vocab)
        tokens = [id_to_token[i] for i in ids]
        print(f"{word:20} -> {ids}")
        print(f"{'':20}    分解为: {tokens}")
    print()
    print("✅ 优势: 即使词不在词汇表中，也能分解为更小的单元")
    print()

    # 示例 3: 词形变化的优雅处理
    print("=" * 70)
    print("示例 3: 词形变化 (共享词根)")
    print("=" * 70)
    morphs = ["run", "runs", "running"]
    for word in morphs:
        ids = tokenize_bpe(word, vocab)
        tokens = [id_to_token[i] for i in ids]
        print(f"{word:12} -> {tokens}")
    print()
    print("✅ 优势: 'run', 'runs', 'running' 共享词根，模型能学到它们的关系")
    print()

    # BPE 的优势总结
    print("=" * 70)
    print("BPE 的三大优势:")
    print("=" * 70)
    print("1. ✅ 固定词汇表大小: 通常 32K-100K 个 token")
    print("2. ✅ 无未登录词: 任何文本都能分解为已知 subword")
    print("3. ✅ 语义共享: 相似的词共享 subword，模型能学到词根、词缀的关系")
    print()

    # 与其他方法对比
    print("=" * 70)
    print("三种编码方法对比:")
    print("=" * 70)
    comparison = """
    | 方法         | 词汇表大小 | 序列长度 | OOV 问题 | 语义单元 |
    |-------------|-----------|---------|---------|---------|
    | 字符级       | ~100      | 很长     | 无      | 弱      |
    | 词级         | 50K-1M    | 短      | 严重     | 强      |
    | BPE (subword)| 32K-100K  | 适中     | 无      | 平衡    |
    """
    print(comparison)


if __name__ == "__main__":
    simple_bpe_demo()
