"""
第一课：字符级编码示例
这是最简单的文本->数字转换方法
"""

def char_to_id_simple(text):
    """
    最简单的编码：直接使用 ASCII 码
    """
    return [ord(c) for c in text]

def id_to_char_simple(ids):
    """
    解码：将数字转回字符
    """
    return ''.join([chr(i) for i in ids])


# 示例 1: 编码一个简单的句子
text = "Hello, world!"
ids = char_to_id_simple(text)
print(f"原文: {text}")
print(f"编码: {ids}")
print(f"解码: {id_to_char_simple(ids)}")
print()

# 问题分析
print("=" * 60)
print("字符级编码的问题:")
print("=" * 60)
print(f"1. 词汇表大小: ASCII 有 128 个字符，Unicode 有 100,000+ 个字符")
print(f"2. 序列长度: 'Hello, world!' 需要 {len(ids)} 个 token")
print(f"3. 语义单位: 每个字符没有独立的语义")
print()

# 示例 2: 长文本的问题
long_text = "The quick brown fox jumps over the lazy dog"
long_ids = char_to_id_simple(long_text)
print(f"长文本: {long_text}")
print(f"需要 {len(long_ids)} 个 token")
print("这对于模型来说太长了！")
print()

# 示例 3: 中文的问题
chinese_text = "你好，世界！"
chinese_ids = char_to_id_simple(chinese_text)
print(f"中文文本: {chinese_text}")
print(f"编码: {chinese_ids}")
print(f"问题: Unicode 编码超出了 ASCII 范围，需要更大的词汇表")
