# 第一课总结：Tokenization 基础

## 🎯 学习目标回顾

通过今天的学习，你应该理解了：

1. ✅ **为什么需要 Tokenization**
   - 计算机只认识数字，需要把文本转换为数字序列
   - 字符级编码序列太长，词级编码词汇表太大
   - BPE (Subword) 是最佳平衡方案

2. ✅ **BPE 的三大优势**
   - 固定词汇表大小 (32K-100K)
   - 无未登录词问题
   - 相似词共享 subword，模型能学到语义关系

3. ✅ **nanochat 的特殊 Token 设计**
   - `<|bos|>`: 文档开始
   - `<|user_start|>` / `<|user_end|>`: 用户消息
   - `<|assistant_start|>` / `<|assistant_end|>`: AI 消息
   - `<|python_start|>` / `<|python_end|>`: 工具调用
   - `<|output_start|>` / `<|output_end|>`: 工具输出

4. ✅ **GPT-4 风格的分词模式**
   - 缩写词分离 ('m, 'll, 've)
   - 数字按 1-2 位分组
   - Unicode 友好

## 🔑 关键概念

### 1. Tokenization 的层次结构

```
文本 "I'm running"
    ↓ (Split Pattern)
["I", "'m", " running"]
    ↓ (BPE Merge)
[34, 1245, 678]  # token IDs
```

### 2. 特殊 Token 的作用

特殊 token 不是通过 BPE 训练得到的，而是**人为插入**到词汇表中的。
它们的作用是：
- **结构化信息**: 标记对话的边界和角色
- **控制信号**: 告诉模型何时调用工具
- **训练信号**: 可以只在某些 token 上计算 loss

### 3. 为什么 nanochat 有两种实现？

| 实现               | 训练速度 | 推理速度 | 依赖      | 优势              |
|-------------------|---------|---------|----------|------------------|
| HuggingFaceTokenizer | 慢      | 中等     | tokenizers | 功能完整，易用     |
| RustBPETokenizer  | 快 (Rust) | 快 (tiktoken) | rustbpe + tiktoken | 性能优化，生产级 |

## 📚 重要代码片段

### 特殊 Token 定义
```python
# tokenizer.py:13-25
SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", "<|user_end|>",
    "<|assistant_start|>", "<|assistant_end|>",
    "<|python_start|>", "<|python_end|>",
    "<|output_start|>", "<|output_end|>",
]
```

### 分词模式
```python
# tokenizer.py:30
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""
```

## 🤔 深度思考题

1. **为什么不直接用 UTF-8 编码？**
   - UTF-8 编码空间太大 (100K+ 字符)
   - 每个字符没有语义
   - 序列会非常长

2. **为什么数字要分成 1-2 位一组？**
   - 平衡词汇表大小和常见数字的表示效率
   - "12345" → ["12", "34", "5"] (3 tokens)
   - 如果是 1-3 位: ["123", "45"] (2 tokens)，但词汇表要为 000-999 预留空间

3. **特殊 token 的 ID 分配策略？**
   ```python
   # RustBPETokenizer:172-175
   tokens_offset = len(mergeable_ranks)  # BPE 词汇表大小
   special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
   ```
   - 特殊 token 的 ID 在 BPE 词汇表**之后**
   - 确保不会和 BPE 训练的 token 冲突

## 📝 练习作业

### 练习 1: 手工分词 (基础)

给定文本："She'll visit in 2024"

请根据 nanochat 的分词模式，手工分解成 chunks：
```
答案：["She", "'ll", " visit", " in", " 20", "24"]
```

### 练习 2: 对话编码 (进阶)

将以下对话编码为 token 序列（使用特殊 token）：
```
用户: "What is 5 * 7?"
助手: "Let me calculate. <python>5 * 7</python> <output>35</output> The answer is 35."
```

答案格式：
```
<|bos|><|user_start|>What is 5 * 7?<|user_end|><|assistant_start|>Let me calculate.<|python_start|>5 * 7<|python_end|><|output_start|>35<|output_end|> The answer is 35.<|assistant_end|>
```

### 练习 3: 分析 tokenizer.py (高级)

阅读 `nanochat/tokenizer.py:258-342` 中的 `render_conversation` 函数，回答：

1. 为什么需要 `mask` 数组？
2. `mask=0` 和 `mask=1` 分别代表什么？
3. 为什么用户的消息都是 `mask=0`？

<details>
<summary>点击查看答案</summary>

1. **为什么需要 mask？**
   - 在训练时，我们只想让模型学习生成 **助手的回复**
   - 用户输入、特殊 token、工具输出都不应该计算 loss

2. **mask 的含义：**
   - `mask=0`: 不计算 loss (ignore_index = -1)
   - `mask=1`: 计算 loss，模型需要学习预测这些 token

3. **用户消息为什么是 mask=0？**
   - 用户消息是输入，不是模型要学习生成的
   - 模型只需要学习如何**根据用户输入生成助手回复**
   - 这样可以节省计算，加快训练
</details>

## 🎓 下节预告

**第二课：深入 BPE 算法和实现**

我们将学习：
1. BPE 训练算法的详细步骤
2. 如何构建 mergeable_ranks
3. RustBPE 的 Rust 代码逐行解读
4. tiktoken 的高效推理原理

---

## 💡 学习建议

1. **动手实践**: 运行今天的所有示例代码
2. **完成练习**: 特别是练习 3，需要阅读源码
3. **提出疑问**: 任何不理解的地方都可以问我
4. **准备下一课**: 预习 `rustbpe/src/lib.rs`

## 📖 参考资源

- [BPE 原论文](https://arxiv.org/abs/1508.07909)
- [GPT-4 Technical Report](https://arxiv.org/abs/2303.08774)
- [tiktoken GitHub](https://github.com/openai/tiktoken)
- [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)

---

🎉 恭喜你完成了第一课！记得完成练习，我们下节课见！
