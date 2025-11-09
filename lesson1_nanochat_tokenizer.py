"""
第一课：nanochat tokenizer.py 代码精读
我们将逐行理解每个设计决策
"""

# ============================================================================
# 第 1 部分：特殊 Tokens 定义 (tokenizer.py:13-25)
# ============================================================================

print("=" * 70)
print("第 1 部分：特殊 Tokens 定义")
print("=" * 70)
print()

# 这是 nanochat 定义的所有特殊 token
SPECIAL_TOKENS = [
    "<|bos|>",              # Beginning of Sequence - 文档开始标记
    "<|user_start|>",       # 用户消息开始
    "<|user_end|>",         # 用户消息结束
    "<|assistant_start|>",  # AI 助手消息开始
    "<|assistant_end|>",    # AI 助手消息结束
    "<|python_start|>",     # Python 工具调用开始
    "<|python_end|>",       # Python 工具调用结束
    "<|output_start|>",     # 工具输出开始
    "<|output_end|>",       # 工具输出结束
]

print("问题 1: 为什么需要这些特殊 token？")
print("-" * 70)
print("答: 这些 token 用于标记对话的结构，让模型理解:")
print("  1. 谁在说话 (user vs assistant)")
print("  2. 什么时候调用工具 (python_start/end)")
print("  3. 工具的输出是什么 (output_start/end)")
print()

print("示例: 一个完整的对话编码")
print("-" * 70)
conversation_example = """
<|bos|><|user_start|>What is 123 + 456?<|user_end|><|assistant_start|>Let me calculate that.<|python_start|>123 + 456<|python_end|><|output_start|>579<|output_end|> The answer is 579.<|assistant_end|>
"""
print(conversation_example)
print()

print("问题 2: 为什么用 <|xxx|> 这种格式？")
print("-" * 70)
print("答: 这个格式:")
print("  1. 不太可能在自然文本中出现")
print("  2. 容易识别和解析")
print("  3. 与 GPT-4 / Claude 的风格保持一致")
print()

# ============================================================================
# 第 2 部分：分词模式 (tokenizer.py:27-30)
# ============================================================================

print("=" * 70)
print("第 2 部分：GPT-4 风格的分词模式")
print("=" * 70)
print()

SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

print("这是一个复杂的正则表达式，让我们分解它:")
print("-" * 70)

# 分解正则表达式的每个部分
pattern_parts = [
    ("'(?i:[sdmt]|ll|ve|re)", "匹配缩写词: 's, 't, 'm, 'd, 'll, 've, 're"),
    (r"[^\r\n\p{L}\p{N}]?+\p{L}+", "匹配单词: 可选的非字母+多个字母"),
    (r"\p{N}{1,2}", "匹配数字: 1-2 位数字一组"),
    (r" ?[^\s\p{L}\p{N}]++[\r\n]*", "匹配标点符号"),
    (r"\s*[\r\n]", "匹配换行符"),
    (r"\s+(?!\S)", "匹配行尾空格"),
    (r"\s+", "匹配其他空格"),
]

for i, (pattern, description) in enumerate(pattern_parts, 1):
    print(f"{i}. {pattern:30} -> {description}")
print()

print("问题 3: 为什么数字只匹配 1-2 位？")
print("-" * 70)
print("答: nanochat 的注释说:")
print("  原版 GPT-4 用 \\p{N}{1,3} (1-3 位)")
print("  但对于小词汇表，这可能浪费 token 空间")
print("  所以改为 \\p{N}{1,2}")
print("  这意味着 '123' 会被分成 '12' + '3'，而不是完整的 '123'")
print()

# ============================================================================
# 第 3 部分：实际分词示例
# ============================================================================

print("=" * 70)
print("第 3 部分：分词效果演示")
print("=" * 70)
print()

# 实际的 regex 库在 nanochat 的依赖中，这里我们简单演示概念
# 我们手工展示几个分词结果（这些是实际运行 nanochat tokenizer 得到的）

print("示例文本的分词结果 (手工演示):")
print("-" * 70)
print("注: 实际 nanochat 使用 regex 库，支持 Unicode 属性匹配")
print()

demo_results = [
    ("Hello, world!", ["Hello", ",", " world", "!"]),
    ("I'm running quickly", ["I", "'m", " running", " quickly"]),
    ("The number is 12345", ["The", " number", " is", " 12", "34", "5"]),
    ("What's the temperature?", ["What", "'s", " the", " temperature", "?"]),
    ("She'll be back soon", ["She", "'ll", " be", " back", " soon"]),
]

for text, tokens in demo_results:
    print(f"原文: {text}")
    print(f"分词: {tokens}")
    print(f"说明: 注意缩写词被正确分离 (如 'm, 'll)，数字被分成 1-2 位一组")
    print()

# ============================================================================
# 第 4 部分：设计哲学总结
# ============================================================================

print("=" * 70)
print("nanochat Tokenizer 的设计哲学")
print("=" * 70)
print()
print("1. 特殊 Token 设计:")
print("   ✅ 明确的对话结构标记")
print("   ✅ 支持工具调用 (Python REPL)")
print("   ✅ 与现代 LLM (GPT-4, Claude) 风格一致")
print()
print("2. 分词模式:")
print("   ✅ 基于 GPT-4 的成熟方案")
print("   ✅ 针对小模型优化 (数字 1-2 位)")
print("   ✅ 平衡词汇表大小和序列长度")
print()
print("3. 实现选择:")
print("   ✅ 提供两种实现: HuggingFace 和 RustBPE")
print("   ✅ RustBPE: 训练快 (Rust) + 推理快 (tiktoken)")
print("   ✅ 灵活可扩展")
print()
