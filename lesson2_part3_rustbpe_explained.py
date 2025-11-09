"""
第二课 Part 3: 深入 nanochat 的 RustBPE 实现
逐行解读 rustbpe/src/lib.rs 的关键代码
"""

print("=" * 70)
print("第二课 Part 3: nanochat RustBPE 源码精读")
print("=" * 70)
print()

# ============================================================================
# 1. 数据结构设计
# ============================================================================

print("1️⃣  核心数据结构")
print("=" * 70)
print()

print("""
【数据结构 1: Word - 词的表示】
----------------------------------------------------------------------
// lib.rs:30-44
struct Word {
    ids: Vec<u32>,  // token ID 序列
}

作用：
- 表示一个词（或text chunk）的 token 序列
- 例如: "hello" -> [104, 101, 108, 108, 111] (字节)
- 合并后可能变成: [25678, 111] (假设 "hell" 被合并为 token 25678)

关键方法：
1. pairs(): 返回所有相邻 token 对的迭代器
   例如: [1, 2, 3, 4] -> (1,2), (2,3), (3,4)

2. merge_pair(): 合并指定的 pair
   - 输入: pair=(a, b), new_id=c
   - 输出: 更新后的 ids + pair 频率的变化量(deltas)

   示例:
   Before: [1, 2, 3, 2, 3, 4]
   Merge:  (2, 3) -> 5
   After:  [1, 5, 5, 4]
   Deltas:
     (2, 3): -2  (删除了 2 个)
     (1, 2): -1  (左边的 pair 被破坏)
     (1, 5): +1  (创建了新的 pair)
     (5, 5): +1  (两个合并结果相邻)
     (5, 4): +1  (右边的 pair 被创建)
     (3, 4): -1  (右边的 pair 被破坏)
""")

print("""
【数据结构 2: MergeJob - 堆中的元素】
----------------------------------------------------------------------
// lib.rs:92-122
struct MergeJob {
    pair: Pair,           // (token_a, token_b)
    count: u64,           // 这个 pair 的全局频率
    pos: AHashSet<usize>, // 包含这个 pair 的词的索引集合
}

作用：
- 堆中的一个元素，代表一个待合并的 pair
- 按 count 降序排列（最大堆）
- 当 count 相同时，按 pair 的字典序排列（保证确定性）

为什么需要 pos（位置集合）？
- 当合并一个 pair 时，我们只需要更新包含这个 pair 的词
- 而不是遍历所有词（效率优化）
- 例如: pair (2, 3) 只出现在词 #5, #17, #23 中
  那么 pos = {5, 17, 23}，我们只更新这 3 个词
""")

print()

# ============================================================================
# 2. 训练算法核心逻辑
# ============================================================================

print("2️⃣  训练算法核心 (train_core_incremental)")
print("=" * 70)
print()

print("""
【算法流程 - lib.rs:164-256】
----------------------------------------------------------------------

输入:
- words: Vec<Word>  // 所有唯一的词（chunk）
- counts: Vec<i32>  // 每个词的出现频率
- vocab_size: u32   // 目标词汇表大小（如 32K, 50K）

输出:
- self.merges: HashMap<Pair, u32>  // 记录所有的合并操作

步骤 0: 初始化
----------------------------------------------------------------------
let num_merges = vocab_size - 256;  // 需要进行的合并次数
// 为什么是 256？因为初始词汇表是所有字节 (0-255)

步骤 1: 统计初始 pair 频率（并行化）
----------------------------------------------------------------------
// lib.rs:172
let (pair_counts, where_to_update) = count_pairs_parallel(&words, &counts);

并行化策略（lib.rs:125-155）：
1. 使用 rayon 的 par_iter() 并行遍历所有词
2. 每个线程统计自己负责的词中的 pair 频率
3. 最后 reduce 合并所有线程的结果

伪代码：
words.par_iter()
    .map(|word| {
        // 每个线程的局部统计
        local_counts = {}
        for pair in word.pairs():
            local_counts[pair] += word.count
        return local_counts
    })
    .reduce(|a, b| {
        // 合并两个线程的结果
        for (pair, count) in b:
            a[pair] += count
        return a
    })

为什么并行化很重要？
- 训练语料可能有数百万个唯一词
- 统计 pair 是 CPU 密集型操作
- 多线程可以大幅加速（8 核 CPU 可以提速 6-7 倍）

步骤 2: 构建最大堆
----------------------------------------------------------------------
// lib.rs:176-186
let mut heap = OctonaryHeap::with_capacity(pair_counts.len());
for (pair, pos) in where_to_update.drain() {
    heap.push(MergeJob { pair, count, pos });
}

为什么用堆（Heap）？
- 需要反复找到频率最高的 pair
- 朴素方法：每次遍历所有 pair，O(n) 时间
- 堆方法：维护一个最大堆，O(log n) 时间
- 总复杂度：O(vocab_size * log n) vs O(vocab_size * n)

什么是 OctonaryHeap？
- 8-叉堆（每个节点有 8 个子节点）
- 相比二叉堆，cache 友好性更好
- 适合大规模数据

步骤 3: 主循环 - 迭代合并
----------------------------------------------------------------------
// lib.rs:193-253
while merges_done < num_merges {
    // 3.1 从堆中取出频率最高的 pair
    let mut top = heap.pop();

    // 3.2 Lazy Refresh（延迟刷新）策略
    //     堆中的 count 可能过时了，检查是否需要更新
    let current = pair_counts.get(&top.pair);
    if top.count != current {
        top.count = current;
        heap.push(top);  // 重新入堆
        continue;
    }

    // 3.3 记录这次合并
    let new_id = 256 + merges_done;
    self.merges.insert(top.pair, new_id);

    // 3.4 在所有包含这个 pair 的词中执行合并
    for word_idx in top.pos {
        let changes = words[word_idx].merge_pair(top.pair, new_id);

        // 3.5 根据 deltas 更新全局 pair_counts
        for (pair, delta) in changes {
            pair_counts[pair] += delta * counts[word_idx];
            // 如果 delta > 0，说明创建了新的 pair，记录位置
            if delta > 0:
                local_pos_updates[pair].insert(word_idx);
        }
    }

    // 3.6 将更新后的 pair 重新加入堆
    for (pair, pos) in local_pos_updates {
        heap.push(MergeJob { pair, count: pair_counts[pair], pos });
    }

    merges_done += 1;
}

关键优化 - Lazy Refresh（延迟刷新）:
----------------------------------------------------------------------
问题：
- 当我们合并 pair (a, b) 时，会影响其他 pair 的频率
- 例如: 合并 (a, b) 后，(x, a) 和 (b, y) 的频率会减少
- 但堆中已经存在这些 pair 的旧频率

朴素解决：
- 每次合并后，更新堆中所有受影响的 pair
- 需要频繁的堆操作，很慢

Lazy Refresh 策略：
- 不立即更新堆中的旧值
- 当 pop 出来时，检查频率是否过时
- 如果过时，更新后重新入堆
- 如果是最新的，才真正处理

为什么有效？
- 大部分 pair 的频率变化不大
- 只有真正被 pop 出来的 pair 才需要检查
- 避免了大量不必要的堆操作
""")

print()

# ============================================================================
# 3. 编码算法（推理）
# ============================================================================

print("3️⃣  编码算法 (encode)")
print("=" * 70)
print()

print("""
【编码流程 - lib.rs:429-467】
----------------------------------------------------------------------

输入: text: &str  // 要编码的文本
输出: Vec<u32>    // token ID 序列

步骤 1: 用正则表达式分割文本
----------------------------------------------------------------------
// lib.rs:433
for chunk in self.compiled_pattern.find_iter(text) {
    // 例如: "Hello, world!"
    //   -> ["Hello", ",", " world", "!"]
}

步骤 2: 将 chunk 转换为字节序列
----------------------------------------------------------------------
let mut ids: Vec<u32> = chunk.bytes().map(|b| b as u32).collect();
// 例如: "Hello" -> [72, 101, 108, 108, 111]

步骤 3: 贪婪地应用 merges
----------------------------------------------------------------------
while ids.len() >= 2 {
    // 找到优先级最高的 pair（merge ID 最小的）
    let mut best_pair = None;
    for i in 0..ids.len() - 1 {
        let pair = (ids[i], ids[i + 1]);
        if let Some(new_id) = self.merges.get(&pair) {
            // 选择 new_id 最小的（训练时最早学到的）
            if best_pair.is_none() || new_id < best_pair.unwrap().2 {
                best_pair = Some((i, pair, new_id));
            }
        }
    }

    // 应用合并
    if let Some((idx, _, new_id)) = best_pair {
        ids[idx] = new_id;
        ids.remove(idx + 1);  // 删除第二个 token
    } else {
        break;  // 没有更多合并
    }
}

为什么选择 new_id 最小的？
----------------------------------------------------------------------
- merge ID 是按训练顺序分配的：256, 257, 258, ...
- ID 越小，说明越早学到（频率越高）
- 这保证了编码与训练时的优先级一致

示例：
假设训练时的合并顺序：
  1. (h, e) -> 256  (最频繁)
  2. (256, l) -> 257  (即 "hel")
  3. (257, l) -> 258  (即 "hell")

编码 "hello":
  Initial: [h, e, l, l, o]
  Step 1:  找到 (h, e)，合并为 256 -> [256, l, l, o]
  Step 2:  找到 (256, l)，合并为 257 -> [257, l, o]
  Step 3:  找到 (257, l)，合并为 258 -> [258, o]
  Final:   [258, o]  // "hell" + "o"
""")

print()

# ============================================================================
# 4. 关键优化总结
# ============================================================================

print("4️⃣  RustBPE 的关键优化")
print("=" * 70)
print()

optimizations = """
1. 并行化统计 (Parallelization)
   ✅ 使用 rayon 并行处理词频统计
   ✅ 多线程加速 6-8 倍

2. 堆优化 (Heap-based)
   ✅ 用最大堆维护 pair 频率
   ✅ 从 O(n) 降到 O(log n)
   ✅ 使用 8-叉堆提升 cache 命中率

3. 延迟刷新 (Lazy Refresh)
   ✅ 不立即更新过时的堆元素
   ✅ pop 时检查是否需要刷新
   ✅ 减少不必要的堆操作

4. 增量更新 (Incremental Update)
   ✅ 只更新受影响的词（通过 pos 集合）
   ✅ 不遍历所有词
   ✅ 大幅减少计算量

5. 内存优化 (Memory Efficiency)
   ✅ 使用 CompactString 减少小字符串的内存占用
   ✅ 使用 AHashMap（ahash）提升哈希性能
   ✅ 避免不必要的内存分配

6. 编码优化 (Encoding Optimization)
   ✅ 预编译正则表达式
   ✅ 贪婪算法一次遍历
   ✅ 选择最早学到的 merge（ID 最小）
"""

print(optimizations)
print()

# ============================================================================
# 5. 与我们的 Python 实现对比
# ============================================================================

print("5️⃣  RustBPE vs 我们的 Python 实现")
print("=" * 70)
print()

comparison = """
| 特性                | Python 实现     | RustBPE              |
|--------------------|----------------|---------------------|
| 核心算法            | ✅ 相同         | ✅ 相同              |
| 并行化              | ❌ 无          | ✅ rayon 多线程      |
| 堆优化              | ✅ 简单堆       | ✅ 8-叉堆（cache友好）|
| Lazy Refresh       | ❌ 无          | ✅ 有                |
| 增量更新            | ❌ 全遍历       | ✅ 只更新受影响的词   |
| 内存优化            | ❌ 标准类型     | ✅ CompactString等   |
| 训练速度            | 1x (基准)      | 20-50x              |
| 编码速度            | 1x (基准)      | 100-200x            |
| 代码复杂度          | ⭐ 简单         | ⭐⭐⭐ 复杂         |

结论：
- Python 版本：适合学习和理解算法
- Rust 版本：适合生产环境，极致性能
- 核心思想相同，只是工程优化的差异
"""

print(comparison)
print()

# ============================================================================
# 6. 学习建议
# ============================================================================

print("6️⃣  学习建议")
print("=" * 70)
print()

print("""
如何深入学习 RustBPE：

1. 对比阅读
   - 先理解 Python 版本的逻辑
   - 再看 Rust 版本如何优化相同的逻辑
   - 关注数据结构和算法的映射关系

2. 关键函数
   - count_pairs_parallel(): 并行统计的实现
   - merge_pair(): 增量更新的核心
   - train_core_incremental(): 主循环逻辑
   - encode(): 编码算法

3. 性能分析
   - 思考每个优化为什么能提升性能
   - 理解并行化的瓶颈在哪里
   - 学习 Rust 的零成本抽象

4. 实验验证
   - 运行 nanochat 的 tokenizer 训练
   - 观察训练过程的日志
   - 对比不同参数的影响

下一步：Part 4 - tiktoken 的高效推理原理
""")

print()
print("=" * 70)
print("下一步：Part 4 - tiktoken 如何实现极速编码")
print("=" * 70)
