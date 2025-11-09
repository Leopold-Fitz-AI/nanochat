"""
第一课：词级编码示例
使用单词作为基本单位
"""

def build_vocab(texts):
    """
    从文本中构建词汇表
    """
    vocab = set()
    for text in texts:
        words = text.lower().split()
        vocab.update(words)

    # 添加特殊 token
    vocab_list = ["<PAD>", "<UNK>"] + sorted(list(vocab))
    word_to_id = {word: i for i, word in enumerate(vocab_list)}
    id_to_word = {i: word for word, i in word_to_id.items()}

    return word_to_id, id_to_word


def word_to_id(text, vocab):
    """
    将文本编码为 ID 序列
    """
    words = text.lower().split()
    return [vocab.get(word, vocab["<UNK>"]) for word in words]


def id_to_word(ids, id_to_word_dict):
    """
    将 ID 序列解码为文本
    """
    return ' '.join([id_to_word_dict[i] for i in ids])


# 示例：构建词汇表
training_texts = [
    "Hello, world!",
    "The quick brown fox",
    "jumps over the lazy dog"
]

vocab, id_to_word_dict = build_vocab(training_texts)
print("=" * 60)
print("词汇表 (前 20 个):")
print("=" * 60)
for i, (word, idx) in enumerate(sorted(vocab.items())[:20]):
    print(f"{idx}: {word}")
print(f"... (总共 {len(vocab)} 个词)")
print()

# 编码示例
text = "The quick brown fox"
ids = word_to_id(text, vocab)
print(f"原文: {text}")
print(f"编码: {ids}")
print(f"解码: {id_to_word(ids, id_to_word_dict)}")
print()

# 问题演示
print("=" * 60)
print("词级编码的问题:")
print("=" * 60)

# 问题 1: 未登录词 (OOV - Out of Vocabulary)
oov_text = "The quick PURPLE fox"
oov_ids = word_to_id(oov_text, vocab)
print(f"1. 未登录词问题:")
print(f"   原文: {oov_text}")
print(f"   编码: {oov_ids}")
print(f"   解码: {id_to_word(oov_ids, id_to_word_dict)}")
print(f"   问题: 'PURPLE' 不在词汇表中，被替换为 <UNK>")
print()

# 问题 2: 词汇表爆炸
print(f"2. 词汇表大小问题:")
print(f"   当前只有 {len(training_texts)} 个句子，词汇表已有 {len(vocab)} 个词")
print(f"   英语常用词汇约 50,000 个")
print(f"   考虑变形（running, runs, ran）词汇表会更大")
print()

# 问题 3: 形态变化
print(f"3. 形态变化问题:")
morph_texts = ["run", "runs", "running", "ran"]
for word in morph_texts:
    if word in vocab:
        print(f"   '{word}' -> ID {vocab[word]} (在词汇表中)")
    else:
        print(f"   '{word}' -> <UNK> (不在词汇表中)")
print(f"   问题: 相似的词被当作完全不同的 token")
