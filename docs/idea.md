# 研发路线：从可靠基线到任务自适应早停

本文档记录 `thought-brake` 研发路线。当前核心判断是：**没有"最优 detector"，只有"某个质量/成本目标下的最优策略"。** 最新主线 token 实验把默认策略收敛到 `compression@1000`，`compression@300` 则成为更激进的 balanced-aggressive 策略。接下来先收敛 v0.1 发布叙事，再把 BOCPD 作为 v0.2 明确推进；具体执行计划见 [plan.md](plan.md)。

最新主线实验结论：

| 策略 | 定位 | 质量 | Total token 节省 |
|---|---|---:|---:|
| `keyword@1000` | 保守高质量策略 | 94.3% | 14.3% |
| `compression@1000` | 当前默认策略 | 93.5% | 19.0% |
| `compression@300` | balanced-aggressive 策略 | 92.3% | 29.1% |
| `budget@300` | 激进省 token 策略 | 88.6% | 42.1% |

注意：当前 total token 统计按行优先使用 API `total_tokens`，流式 usage 不可用时回退到本地估算。最新 full run 的 treated rows 中，400/1800 是 API 实测，1400/1800 是估算值。Phase 1 被早停时会主动关闭 stream，因此拿不到 provider 最后的 streaming usage chunk；这不是报告脚本问题，而是早停机制和 usage 返回时机之间的冲突。

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
- `compression`：CRD + LZ-rate 压缩信号
- `ngram`：n-gram literal overlap
- `keyword`：犹豫短语密度（结论后触发）
- `semantic`：内容词 Jaccard 相似度

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
2. 通过 LLM API 支持的禁用推理参数，从根本上禁用 Phase 2 推理阶段
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

- Prefill 模式在实践中失败：模型不尊重注入的结束标签，继续推理
- 配置化禁用推理参数是关键突破 — 显著减少 reasoning 泄漏
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

#### 早期跨数据集探索结果

以下是早期小样本探索：三个数据集，各 20 题，4 种 detector × 3 个 budget（300/500/1000），direct 模式。它用于发现方向，不作为当前最终结论。

**Detector@300 对照（最优 budget 点）**：

| Detector | Riddles | GSM8K | MMLU | 推理长度分布 |
|---|---|---|---|---|
| budget | **95.0%** (66%) | 70.0% (80%) | 95.0% (76%) | 短且均匀时好使 |
| compression | 80.0% (37%) | **80.0%** (65%) | **100%** (69%) | 长且分散时碾压 |
| keyword | 87.5% (34%) | 70.0% (64%) | **100%** (69%) | 长且分散时碾压 |
| semantic | 87.5% (37%) | 75.0% (65%) | 95.0% (70%) | 稳定 |

*括号内为推理节省率。Baseline 全部 100%。*

**MMLU 完整对照（推理长度 615 - 63806 chars，avg 7176）**：

| Detector@Budget | 正确率 | 节省 | Short(<2k) | Med(2k-5k) | Long(>=5k) |
|---|---|---|---|---|---|
| budget@300 | 95.0% | 75.7% | 7/7 | 6/6 | 6/7 |
| budget@1000 | **100%** | 49.3% | 7/7 | 6/6 | 7/7 |
| **compression@300** | **100%** | **69.0%** | 7/7 | 6/6 | 7/7 |
| **keyword@300** | **100%** | **68.9%** | 7/7 | 6/6 | 7/7 |

#### 关键发现

**1. "最优 detector" 不存在，取决于任务类型**：

| 数据集 | 最优 Detector@Budget | 原因 |
|---|---|---|
| Riddles (avg 2077) | budget@300 (95%) | 推理短且均匀，hard budget 足够 |
| GSM8K (avg 1979) | signal@1000 (90%) | 数学推理中等长度，需等到重复 |
| MMLU (avg 7176) | signal@300 (100%) | 长度极宽(615-63806)，必须自适应 |

**2. Budget 的稳定性依赖任务推理长度分布**：
- Riddles 和 MMLU 上 budget@300 都到 95%——但 MMLU 上 compression@300 达到 100%
- GSM8K 上 budget 反常：@500 (65%) < @300 (70%)，不单调递增
- Budget 本质上是"猜测推理需要多长"，猜错就伤质量

**3. 信号 detector 的真正价值是"自适应"**：
- 不需要预判推理长度——短的自然结束，长的等到重复才截
- MMLU 上 compression@300 和 budget@1000 都是 100%，但 compression 省 69% vs budget 只省 49%
- 同样 100% 正确率，信号 detector 多省 20%

**4. Overthinking 的 5 种结构模式**（来自 riddles 分析）：
- A. Arrive-Rethink-Reconfirm（~60%）— "但真的会这样吗？"
- B. Enumerate-then-Select（~25%）— "还有其他可能吗？"
- C. Explain-the-Explainer（~15%）— "组织回复..."
- D. Literal loops（<5%）— 唯一被 compression/ngram 捕获的模式
- E. Multi-angle re-argument（~30%）— 同一观点换词重复

**5. "数学推理不适合早停"被推翻**：
- GSM8K 上信号 detector@1000 达到 90%，还省 6.8%-25.9%
- 数学推理的重复模式更规律（重复计算、反复列式），比脑筋急转弯更适合信号检测

验收：

- ✅ 5 种 detector 实现，接口统一
- ✅ 73 个测试全部通过
- ✅ 早期 3 个数据集 × 4 种 detector × 3 个 budget = 36 组探索实验
- ✅ 主线 token 实验：math=100、mmlu=100、riddle=100；budget/compression/keyword × 0/300/1000
- ✅ riddle 数据集已扩充到 100 条，`--n` 已支持限制 riddles 小批量运行
- ✅ 当前默认策略：`compression@1000`
- ✅ 核心发现：策略选择需要同时看质量损失、reasoning 节省和 total token 节省

### Phase D：任务自适应路由（规划中）

Phase C 实验表明：**没有万能最优 detector，任务类型和质量/成本目标共同决定策略。** 下一步不是继续无序增加 detector，而是分成两个明确阶段：

1. v0.1：发布 Layer 1，强调 client-side + black-box API + visible reasoning text 的工程 niche。
2. v0.2：实现 BOCPD / change-point detector，回应最初“数学上更美、减少 magic threshold”的目标。

任务路由仍然重要，但应排在 v0.1 叙事收敛和 v0.2 BOCPD 之后。

#### 方向 1：v0.1 发布与叙事收敛（当前最高优先级）

把当前已经跑通的 Layer 1 结果变成可解释、可复现、可发布的版本：

- 重写 README 的 public-facing narrative
- 升级 focused report，使其更像研究报告而不是工程日志
- 做一张 strategy map 图，解释 default / aggressive / conservative 三档策略
- 明确和 EAT、内部信号方法、proxy-model 方法的差异
- 明确 limitations：token estimate uncertainty、single-vendor evidence

#### 方向 2：BOCPD / Layer 2（数学美感主线）

BOCPD 是当前最值得保留的研究增量。它不是“再加一个 detector”，而是尝试把停止判据从手调阈值推进到在线变化点检测。

目标：

- 减少 `@300`、压缩阈值、连续窗口数等 magic parameters 的主导地位
- 用 posterior change probability 描述过度推理阶段切换
- 和 `compression@300` / `keyword@300` / `budget@300` 做直接对比

#### 方向 3：任务分类路由

根据问题特征自动选择最优策略：
- 短推理/均匀分布 → budget（简单快速）
- 长推理/分布未知 → signal detector（自适应）
- 实现方式：前 N 个字符判断问题类型，或用户显式指定任务类型
- `EarlyStopConfig.for_task()` 已有预设，但当前只区分 chat/qa/math/complex
- 需要根据实验数据更新预设值
- 但不要在 v0.1 里把它包装成已验证的智能路由

#### 方向 4：Hybrid Detector — signal guard + budget fallback

信号 detector 做主力，budget 做兜底：
- 信号 detector 在 soft_budget 前不触发（warmup 期）
- 超过 hard_limit 时强制截断（和现在一样）
- 关键参数：soft_budget 和 hard_limit 的选择——实验表明这仍然依赖任务类型
- 本质上是把"任务自适应"问题下推到了"参数选择"

#### 方向 5：Embedding-based 语义冗余检测（PUMA 路线）

PUMA（arXiv:2605.17672）用 embedding 相似度检测语义冗余：
- 用轻量 embedding 模型对滑动窗口做 embedding
- cosine similarity > θ → 语义重复
- 优势：能检测换词重述（Pattern E），不受 literal overlap 限制
- 代价：需要额外 embedding 模型调用或本地推理
- 当前优先级降低——信号 detector 已在 MMLU 上达到 100%，增量收益不确定

#### 方向 6：Answer Oscillation 检测

文献报告 answer oscillation 与 overthinking 的 r=0.78 相关：
- 从 reasoning 中实时提取 candidate answer
- 检测 answer 是否在多个选项间摇摆
- 对多选题（MMLU）特别适用——可以直接监控 A/B/C/D 的出现频率
- MMLU 上 keyword@300 已经 100%，但 oscillation 可能提供更优雅的信号

**推荐优先级**：方向 1（v0.1 发布）> 方向 2（BOCPD / Layer 2）> 方向 3（cost calibration + cross-vendor）> 方向 4（任务路由）> 方向 5（hybrid）> 方向 6（oscillation / embedding）

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
- 数学证明、定理推导
- 代码调试
- 用户要求展示完整推理过程
- LLM API 按 max tokens 而非实际生成计费

注意：GSM8K 实验表明数学题的常规推理仍然适合早停（信号 detector@1000 达到 90%）。"数学推理不适合早停"只适用于完整推理链是输出本身的场景（如证明题）。

## 7. 当前优先级

Phase A ✅ → Phase B ✅ → Phase C ✅（主线 token 实验完成）→ v0.1 发布准备中

下一步：

1. **v0.1 发布叙事** — README、final report、strategy map、limitations、quickstart。
2. **文献 claims 逐条核验** — 尤其是 EAT、内部信号方法、proxy-model 方法。
3. **BOCPD 设计文档** — 明确 v0.2 的 signal、posterior update、stop criterion 和实验计划。
4. **Cost calibration** — 用 calibration run 或更贴近 provider 的 tokenizer 降低当前约 15% MAE。
5. **Cross-vendor sanity check** — 至少再用一个 LLM API 做小规模验证。
