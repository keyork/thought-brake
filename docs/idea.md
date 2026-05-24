# 未来工作方向：从可靠基线到压缩信号早停

本文档记录 `thought-brake` 后续研发路线。当前判断是：**客户端黑盒 LLM API 的流式早停方向合理，但必须先把实验基线和 Phase 2 输出质量做扎实，再推进压缩信号和 BOCPD/MDL。**

## 1. 当前定位

`thought-brake` 面向只能访问 LLM API 流式文本输出的用户：

- 不能访问 hidden states / logits
- 不能修改采样器
- 不能训练或微调模型
- 需要在客户端侧低成本部署
- 希望在 reasoning model 过度推理时提前停止

因此，我们的价值不在于打败内部信号方法，而在于填补这个部署场景：

> 黑盒 LLM API + 客户端流式监控 + 可复用早停 detector。

## 2. 已有实验暴露的问题

第一轮 hard-budget 实验给了两个重要结论：

1. GSM8K 上小 budget 并不一定伤害质量，`budget=200` 的离线修正评估结果最好。
2. 现有结果还不能证明 token savings，因为旧 baseline 使用 passthrough，`reasoning_chars=0`，没有记录真实 baseline reasoning 长度。
3. Phase 2 输出经常泄漏推理草稿，answer 长度达到 baseline 的 3-7 倍，这会影响体验，也可能抵消收益。

所以后续不能直接跳到 BOCPD。正确顺序是：

1. 可靠测量 baseline reasoning
2. 修 Phase 2 final-answer-only 行为
3. 建 detector 插件体系
4. 再比较 hard budget / compression / BOCPD

## 3. 代码架构方向

当前代码已经开始按以下模块拆分：

| 模块 | 责任 |
|---|---|
| `src/thought_brake/client.py` | 对外 client、Phase 1/Phase 2 编排 |
| `src/thought_brake/_monitor.py` | 流式读取 reasoning/content |
| `src/thought_brake/detectors.py` | 可插拔早停 detector |
| `src/thought_brake/_prefill.py` | Phase 2 prefill 构造和收集 |
| `experiments/runner.py` | 并发实验、resume、schema 化 JSONL |
| `experiments/analysis.py` | 结果分析，自动过滤最新 schema |

detector 接口是后续复用的关键：

```python
class ReasoningDetector(Protocol):
    name: DetectorName

    def update(self, piece: str, total_chars: int) -> StopDecision:
        ...
```

已支持：

- `none`：baseline 监控，不截断
- `budget`：soft/hard budget
- `compression`：Layer 1 压缩信号原型

## 4. Roadmap

### Phase A：可靠实验基线

目标：让结果文件能真实回答“省了多少 reasoning”。

工作项：

- baseline 使用 `detector="none"` 走 stream monitor，不再 passthrough
- 结果写入 `schema_version`
- 结果记录 `detector`、`phase2_used`、`phase2_failed`、`answer_chars`
- analysis 只分析最新 schema，避免旧结果污染
- 修 exact-match 评测对 `70,000` 这类数字的误判

验收：

- baseline 记录真实 `reasoning_chars`
- 每个 `(question_id, budget, detector)` 可 resume
- 能计算 `avg_savings_rate`

### Phase B：Phase 2 输出收敛 ✅ 已完成

目标：早停后只输出最终答案，不泄漏 reasoning 草稿。

**已完成方案**：

实现了 `direct` 模式替代原有 `prefill` 模式：

1. 截断后发一条新 user 消息（含原始问题 + reasoning 摘要），不再用 assistant prefill
2. 通过 `enable_thinking=False` 从根本上禁用 Phase 2 推理阶段
3. 推理摘要使用 head-only 策略（短推理时避免 head+tail 重叠噪音）
4. `clean_final_answer()` 后处理去除元推理包装和尾部重复

**实验验证**（riddles 数据集，direct 模式）：

| Budget | Easy | Medium | Hard | 整体 |
|---|---|---|---|---|
| 100 | 61.1% | 92.9% | 100% | 81.2% |
| 200 | 77.8% | 78.6% | 75.0% | 77.3% |
| **300** | **88.9%** | **100%** | **100%** | **95.0%** |
| 500 | 83.3% | 100% | 87.5% | 90.6% |

**关键发现**：

- Prefill 模式在实践中失败：模型不尊重注入的 `boxed` 标签，继续推理
- `enable_thinking=False` 是关键突破 — 彻底消除 reasoning 泄漏
- Head+tail 摘要在短推理（<400 chars）时有害，"……" 分隔符引入噪音
- Budget=300 是 riddles 数据集的最优值（95% 质量保持率）

验收：

- ✅ `answer_chars` 不显著高于 baseline（direct 模式下 < 0.4x baseline）
- ✅ LLM judge 质量保持率 95%（budget=300）
- ✅ `phase2_failed` 维持低位

### Phase C：Detector 对照实验 ✅ 已完成

目标：把 detector 从实现细节变成可实验变量，对比不同早停信号的效果。

**已实现 5 种 detector**：

| detector | 原理 | 参数 |
|---|---|---|
| `budget` | soft/hard 字符预算截断 | `soft_budget`, `hard_limit` |
| `compression` | CRD + LZ-rate 压缩信号 | `crd_threshold`, `lz_threshold` |
| `ngram` | n-gram literal overlap | `ngram_size`, `ngram_window_chars`, `ngram_threshold` |
| `keyword` | 犹豫短语密度（结论后触发） | `keyword_window_chars`, `keyword_trigger_threshold` |
| `semantic` | 内容词 Jaccard 相似度 | `semantic_window_chars`, `semantic_jaccard_threshold` |

**实验结果**（riddles 数据集，20 题，3 难度，budget=300，direct 模式）：

| Detector | Easy | Medium | Hard | 整体 | Token Save | Latency |
|---|---|---|---|---|---|---|
| **budget** | **88.9%** | **100%** | **100%** | **95.0%** | 66.2% | 35s |
| compression | 77.8% | 85.7% | 75.0% | 80.0% | 37.3% | 25s |
| ngram | 77.8% | 85.7% | 75.0% | 80.0% | 33.3% | 21s |
| keyword | 88.9% | 92.9% | 75.0% | 87.5% | 34.2% | 21s |
| semantic | 83.3% | 100% | 75.0% | 87.5% | 36.5% | 17s |

**关键发现**：

1. **Budget 仍然最优（95%）**，但它是 magic number，没有语义理解
2. **所有信号 detector 在 hard 题上都是 75%** — 系统性地在复杂推理上过早截断
3. **Compression/ngram 只检测 literal 重复**，但模型 overthinking 主要是语义重复（换词重述同一论点），不是字面重复。实测 CRD > 1.0（新文本反而更难压缩）
4. **Keyword detector（基于 SelfDoubt HVR）和 semantic detector（Jaccard）** 在 easy/medium 上接近 budget（87.5%），但 hard=75% 说明合法复杂推理中也会出现犹豫短语和语义重叠
5. **Overthinking 的 5 种结构模式**：
   - A. Arrive-Rethink-Reconfirm（~60%）— "但真的会这样吗？"
   - B. Enumerate-then-Select（~25%）— "还有其他可能吗？"
   - C. Explain-the-Explainer（~15%）— "组织回复..."
   - D. Literal loops（<5%）— 唯一被 compression/ngram 捕获的模式
   - E. Multi-angle re-argument（~30%）— 同一观点换词重复

**核心结论**：

> 黑盒文本信号的天花板在于无法区分"复杂推理中的合理自我质疑"和"真正的过度思考"。Budget 绕过了这个问题（直接给足够空间），但 signal-based detector 需要更深的语义理解才能突破 75% hard 上限。

验收：

- ✅ 5 种 detector 实现，接口统一
- ✅ 63 个测试全部通过
- ✅ Budget=95%, keyword/semantic=87.5%, compression/ngram=80%
- ✅ 定位了信号 detector 的瓶颈（语义重复 vs literal 重复、hard 题误截断）

### Phase D：研究增强（规划中）

Phase C 的结论指向两个方向：

#### 方向 1：组合信号 Composite Detector

单一信号的天花板是 87.5%（hard=75%）。组合多个信号可能突破：
- keyword + semantic 双重确认：只在两个 detector 同时触发时才截断
- Signal → confidence score：每个 detector 输出 [0,1] 置信度，加权组合
- 问题难度自适应：根据前 N 个字符判断难度，调整阈值

#### 方向 2：Embedding-based 语义冗余检测（PUMA 路线）

PUMA（arXiv:2605.17672）用 embedding 相似度检测语义冗余，实现 26.2% token 减少：
- 用轻量 embedding 模型（如 text2vec）对滑动窗口做 embedding
- cosine similarity > θ → 语义重复
- 优势：能检测换词重述（Pattern E），不受 literal overlap 限制
- 代价：需要额外 embedding 模型调用或本地推理

#### 方向 3：BOCPD on 文本信号

BOCPD（Bayesian Online Changepoint Detection）仍是合理方向：
- 不消除参数，但给参数概率意义
- 把 CRD/Jaccard/hedge-density 作为观测序列，检测 changepoint
- 适合在 Phase D 确认信号组合有效后推进

#### 方向 4：Answer Oscillation 检测

文献报告 answer oscillation 与 overthinking 的 r=0.78 相关：
- 从 reasoning 中实时提取 candidate answer
- 检测 answer 是否在多个选项间摇摆
- 摇摆 = 还在推理，稳定 = 可以停止
- 问题：需要问题类型感知（选择题 vs 开放问答）

**推荐优先级**：方向 1（组合信号）> 方向 2（embedding）> 方向 4（oscillation）> 方向 3（BOCPD）

## 5. Compression Detector Layer 1

Layer 1 用两个相对信号：

- CRD：当前窗口 gzip 压缩比相对起始窗口的衰减
- LZ-rate：当前窗口 LZ factor rate 相对起始窗口的衰减

停止规则：

```text
STOP if CRD < theta_crd OR LZ_ratio < theta_lz
持续 k 个窗口
```

这是 MVP，不应包装成“已验证的最终算法”。它的价值是：

- 无模型依赖
- 无额外 API 调用
- 可离线回放和在线使用
- 方便与 budget detector 做 A/B

## 6. 不适用边界

以下场景默认不应激进早停：

- Agent 多步工作流
- 数学证明、复杂推导
- 代码调试
- 用户要求展示完整推理过程
- LLM API 按 max tokens 而非实际生成计费

## 7. 当前优先级

Phase A ✅ → Phase B ✅ → Phase C ✅ → Phase D 规划中

下一步：

1. **组合信号实验** — keyword + semantic composite detector，验证是否突破 hard=75%
2. **Embedding baseline** — 用 text2vec 做 embedding 相似度检测，作为语义检测的 upper bound
3. **跨数据集验证** — 在 GSM8K 上重跑 Phase C 对照，确认 riddles 结论是否泛化
4. **参数调优** — 当前 keyword/semantic 参数是首次猜测，hard 题误截断可能通过更大 `min_history` 缓解
5. **文献 claims 逐条核验**，再决定是否写论文/报告
