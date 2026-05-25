# v0.1 Milestone Status

日期：2026-05-25

这份文档记录 `thought-brake` 在 v0.1 阶段的研究定位、当前结论、已实现内容、未实现内容和下一步优先级。它不是最终论文，也不是完整 README；它是 v0.1 发布前的 milestone 状态快照。

当前阶段定义：

- v0.1：Layer 1 literal text detectors，默认策略 `compression@1000`
- v0.2：BOCPD / change-point detector，目标是减少 magic threshold
- v0.3：token calibration + cross-vendor sanity check

## 1. 当前一句话定位

`thought-brake` 的核心目标是：

> 在黑盒 LLM API 场景下，仅依靠客户端可见的 streaming reasoning 文本，在线监控过度推理信号，主动截断 thinking，并通过第二阶段答案收束保持最终输出质量。

更短的英文表述：

> A client-side streaming monitor-and-interrupt method for black-box reasoning LLM APIs, operating purely on visible reasoning text, with two-phase final-answer recovery and cost-aware evaluation.

这里的重点不是“做了一个 wrapper”，也不是“做了一个 fixed budget baseline”，而是：

- 在线监控 reasoning dynamics
- 基于文本信号判断 overthinking / redundancy / hesitation
- 在过度推理继续扩大前中断 generation
- 用 Phase 2 recovery 得到可用最终答案
- 同时报告 reasoning savings 和 total-token savings

## 2. 我认为当前最合理的 Contribution

### Contribution 1：Client-side Streaming Monitor + Interrupt 方法

这是 main contribution。

我们提出一种面向黑盒 reasoning LLM API 用户的 client-side streaming monitor-and-interrupt 方法。它不访问 logits、hidden states、模型权重或采样器，也不修改服务端 decoding；只依赖 LLM API 暴露出来的 visible reasoning text，在生成过程中实时监控文本信号，并在检测到过度推理时由客户端主动中断。

这个定位要比“streaming monitor + interrupt”本身更精确。相关工作中已经存在在线停止、内部信号或代理模型思路；我们的 niche 是三者交集：

- client-side
- black-box API
- visible reasoning text / literal text signals

这和简单 fixed budget 不同：

- fixed budget 是“猜这道题需要多长推理”
- monitor-and-interrupt 是“观察推理是否已经进入低收益/重复/犹豫阶段”

当前实现支持的在线信号包括：

- `compression`：压缩率变化，捕捉重复和低信息密度
- `keyword`：犹豫、反复确认、结论后继续推理等短语
- `ngram`：literal n-gram overlap
- `semantic`：内容词 Jaccard 重合
- `budget`：固定预算 baseline 和安全兜底

与最接近的 black-box prior work 也要区分清楚。当前 survey 里需要继续补文献核验，尤其是 EAT (arXiv:2510.08146)。按现有调研笔记，EAT 使用 sequence-level entropy 和 proxy model；我们的不同点是：

- 使用 compression / keyword / n-gram 等字面文本信号，不需要 proxy model
- 系统比较多个 detector family，而不是只验证一种停止信号
- 把中断后的 two-phase recovery 和 total-token cost 评估纳入同一实验闭环

### Contribution 2：Two-phase Final-answer Recovery

单纯中断 stream 会导致没有最终答案，因此我们把 early stopping 变成两阶段系统：

```text
Phase 1: stream reasoning + monitor + interrupt
Phase 2: direct final-answer recovery
```

Phase 2 的关键不是 prefill，而是 direct mode：

- 保留 system/developer 控制消息
- 重新构造 user prompt
- 带上原始问题和简短 reasoning 摘要
- 通过配置化 `phase2_extra_body` 禁用 Phase 2 thinking

早期 prefill 方案容易让模型继续沿着 partial reasoning 推理，导致最终答案泄漏推理草稿。direct mode 明显更稳定。

### Contribution 3：Plug-in Detector Design and Empirical Comparison

我们把 detector 做成可插拔接口，而不是把某个 heuristic 写死。

当前实现并比较了：

- `budget`
- `compression`
- `keyword`
- `ngram`
- `semantic`

当前主线实验结论是：没有万能最优 detector。策略选择取决于任务类型、质量目标和成本目标。但在当前 full run 上，`compression@1000` 是最合理的默认策略，`compression@300` 是更激进的 balanced-aggressive 策略。

这里需要公平对待 fixed budget：`budget@300` 不是“坏 baseline”，它是 Pareto frontier 上的 aggressive policy。monitor-and-interrupt 方法的价值不是在所有点上压倒 fixed budget，而是在较高质量区域提供 fixed budget 难以达到的折中点。

### Contribution 4：Cost-aware Evaluation

仅报告 reasoning 长度减少是不够的，因为实际 API 成本还包括：

- prompt tokens
- visible completion tokens
- reasoning tokens
- Phase 2 recovery tokens

因此当前 schema v3 同时记录：

- `phase1_*_tokens`
- `phase2_*_tokens`
- `total_*_tokens`
- `estimated_*_tokens`
- `token_usage_source`

这个 contribution 的价值在于避免夸大收益。当前结果显示：

- 当前默认策略的 reasoning chars savings 是 28.4%
- 当前默认策略的 total-token savings 是 19.0%
- 更激进的 `compression@300` 可以达到 62.2% reasoning chars savings 和 29.1% total-token savings，但质量更低

这比只说“省了 60%+ reasoning”更诚实，也更适合做论文或正式报告。

脚注口径很重要：

- ReasoningSavings 只衡量 Phase 1 reasoning 文本长度减少
- TokenSavings 包括 Phase 1、Phase 2 recovery 和 prompt tokens
- 两者差距反映了 Phase 2 recovery 成本和 prompt 成本

## 3. 当前实验结论

当前 focused report 只读取以下三个主线结果文件：

```text
experiments/results/full_budget.jsonl
experiments/results/full_compression.jsonl
experiments/results/full_keyword.jsonl
```

当前已完成主线实验覆盖：

- `math=100`
- `mmlu=100`
- `riddle=100`
- detector：`budget` / `compression` / `keyword`
- budgets：`0,300,1000`
- Phase 2：`direct`
- token schema：v3

主要结果：

| Config | Quality | Baseline | ReasoningSavings | TokenSavings | Lost | Phase2Fail |
|---|---:|---:|---:|---:|---:|---:|
| budget@300 | 88.6% | 97.3% | 77.4% | 42.1% | 31 | 0.0% |
| budget@1000 | 91.0% | 97.3% | 40.1% | 20.6% | 26 | 0.0% |
| compression@300 | 92.3% | 97.7% | 62.2% | 29.1% | 20 | 0.0% |
| compression@1000 | 93.5% | 97.7% | 28.4% | 19.0% | 18 | 0.0% |
| keyword@300 | 91.0% | 96.7% | 61.2% | 28.9% | 23 | 0.0% |
| keyword@1000 | 94.3% | 96.7% | 22.1% | 14.3% | 11 | 0.0% |

当前推荐：

- 默认策略：`compression@1000`
- 保守高质量策略：`keyword@1000`
- balanced-aggressive 策略：`compression@300`
- 激进省 token 策略：`budget@300`

为什么默认不是 `compression@300`：

- `compression@300` 的 TokenSavings 是 29.1%，明显高于 `compression@1000` 的 19.0%
- 但 `compression@300` 的 Quality 是 92.3%，低于 `compression@1000` 的 93.5%
- `compression@300` 更适合作为 balanced-aggressive 策略；默认策略更偏向质量稳定

为什么默认不是 `keyword@1000`：

- `keyword@1000` 的 Quality 是 94.3%，比 `compression@1000` 高 0.8pp
- 但它的 TokenSavings 只有 14.3%，低于 `compression@1000` 的 19.0%
- 在当前实验规模下，0.8pp quality 差异不应过度解释；默认策略偏向更高成本收益

为什么默认不是 `budget@300`：

- `budget@300` 最省 token，TokenSavings 42.1%
- 但质量只有 88.6%，Lost=31
- 它适合作为 aggressive policy，不适合作为 default policy

`@300` / `@1000` 的含义也要说清楚：这里的数字是 detector 的 safety budget / hard-limit 相关配置，单位是 reasoning chars。对 `compression` / `keyword` 这类 signal detector 来说，真正触发点由 detector signal 决定；budget 仍作为安全网，避免没有信号时无限推理。

## 4. Token 统计的当前判断

当前 token 统计不是完美 API 实测。

在 treated rows 中：

- 400/1800 使用 API `total_tokens`
- 1400/1800 使用 `estimated_total_tokens`

原因已经确认：早停 Phase 1 会主动关闭 stream，而 provider 的 final usage chunk 通常只在 stream 自然结束时返回。因此：

- natural finish 可以拿到 Phase 1 API usage
- early-stop Phase 1 通常拿不到 Phase 1 API usage
- Phase 2 基本可以拿到 API usage

当前 report 里已经明确写了这个限制，并计算了估算误差：

- 在有 API usage 可校准的 Phase 1 样本上
- `estimated_phase1_total_tokens` mean absolute error 约 14.4%

这意味着精确的 TokenSavings 数字应带不确定性。以 `compression@1000` 的 19.0% 为例，可以粗略理解为区间级结论，而不是精确到 0.1pp 的成本测量。这个不确定性不影响“确实节省 total tokens”的定性结论，但会影响 detector 之间小幅 TokenSavings 差异的比较。

当前判断：

- total-token savings 可以指导方向
- 但不能包装成 billing-grade 精确成本结论
- 后续若要写论文，需要更严谨的 tokenizer 校准或单独 calibration run

## 5. 已实现内容

### 核心库

- `ThoughtBrakeClient`
- Phase 1 streaming monitor
- Phase 2 direct / prefill recovery
- fallback on Phase 2 failure
- `EarlyStopConfig`
- env 配置
- usage extraction
- local token estimate

### Detector

已实现：

- `none`
- `budget`
- `compression`
- `ngram`
- `keyword`
- `semantic`

接口：

```python
class ReasoningDetector(Protocol):
    name: DetectorName

    def update(self, piece: str, total_chars: int) -> StopDecision:
        ...
```

### 实验系统

- 并发 runner
- resume
- schema v3 JSONL
- `--track-usage`
- `--phase2 direct`
- `--workers`
- `--n`
- focused report
- 结论导向图：
  - `overall_tradeoff.png`
  - `dataset_decision_matrix.png`
  - `loss_vs_savings.png`

### 数据集

当前支持：

- riddles，本地 JSONL，已扩展到 100 条
- GSM8K，经 HuggingFace `datasets`
- MMLU，经 HuggingFace `datasets`

注意：

- 当前主线 full report 已包含 riddle 100 条
- 早期 riddle 小样本旧结论已被新的 full run 覆盖

### 测试

当前测试数：73 passed。

覆盖：

- detector 行为
- monitor 行为
- Phase 2 prompt / direct / fallback
- token usage aggregation
- exact match
- MMLU stable id
- riddles load / n limit

## 6. 未实现内容

### 1. Task Router

idea.md 里写了 task-aware routing，但当前没有真正实现 runtime router。

已有的只是：

- report 里的 per-dataset 推荐
- `EarlyStopConfig.for_task()` 这类简单 preset

还没有：

- 输入问题分类
- 根据任务类型选择 detector/budget
- conservative/default/aggressive policy selector
- router 实验验证

### 2. Hybrid Detector

当前每次只能使用一个 detector。虽然每个 detector 内部都有 hard limit 兜底，但这还不是完整 hybrid。

未实现：

```text
signal detector as primary guard
budget detector as fallback
task-specific hard limit
```

### 3. Answer Oscillation

未实现。

这个方向适合 MMLU / multiple-choice：

- 实时提取 candidate answer
- 监控 A/B/C/D 摇摆
- 如果答案稳定但 reasoning 继续展开，可以截断

### 4. Embedding-based Redundancy

未实现。

当前 `semantic` 是内容词 Jaccard，不是 embedding。还没有：

- embedding window
- cosine similarity
- local embedding model
- embedding API

### 5. BOCPD

未实现。

目前看 v0.1 优先级较低，因为 compression/keyword 已经能支撑当前 main claim。

但 BOCPD 仍然值得作为 v0.2 明确推进，因为它对应最初“数学上更美、减少 magic threshold”的目标。当前 `compression@1000` / `compression@300` 仍然有 `@1000` / `@300`、压缩阈值、连续窗口数等参数；BOCPD 的价值在于把停止决策推进到变化点检测和 posterior odds，而不是继续手调阈值。

### 6. Provider-specific Tokenizer / Billing-grade Cost

未实现。

当前本地 token estimate 是粗估。若要让 total-token savings 更硬，需要：

- 接入目标 LLM API 对应 tokenizer，或
- 用自然结束样本做系统校准，或
- 设计独立 calibration experiment

## 7. 我对当前项目的判断

### 现在不要无序堆 detector

已有 detector 足够支撑 v0.1 main contribution。继续无序增加 embedding / oscillation 等功能会让项目发散。

但 BOCPD 不应和普通 detector expansion 混为一谈。它对应的是 Layer 2：减少 magic threshold，提高方法的统计解释性。按最初目标“真实工程问题 + 数学上美丽 + 黑盒 API 可用”来看，BOCPD 是 v0.2 的核心 milestone。

### 现在最该做的是收敛叙事

项目已经有：

- 方法
- 实现
- 实验
- 指标
- report

但还缺一个清晰 narrative：

1. 我们的问题是什么
2. 为什么 fixed budget 不够
3. streaming monitor + interrupt 的方法是什么
4. direct recovery 为什么必要
5. 实验支持什么结论
6. 局限是什么

### Main claim 应该谨慎但明确

建议 main claim 写成：

> In black-box LLM API settings, streaming reasoning traces provide enough online signal to interrupt overthinking before excessive reasoning accumulates. With a two-phase final-answer recovery mechanism, this can preserve answer quality while reducing both reasoning length and total token cost.

中文：

> 在黑盒 LLM API 场景下，流式 reasoning 文本本身已经包含足够的在线信号，可以用于在过度推理扩大前主动截断；结合第二阶段答案收束，可以在保持回答质量的同时减少 reasoning 长度和 total token 成本。

这个 claim 比“我们省了多少 token”更稳，也比“我们做了一个工具”更有研究贡献。

## 8. 建议下一步

我建议下一步按这个顺序：

### Step 1：收敛 v0.1 发布叙事

把 `README.md`、`docs/idea.md` 和 final report 收敛到同一条主线：

1. client-side + black-box API + visible reasoning text
2. streaming monitor + interrupt
3. two-phase final-answer recovery
4. cost-aware evaluation
5. limitations and no-overclaim

### Step 2：升级最终 report

当前 `report.md` 是工程报告。建议生成更像最终研究报告的版本：

- research question
- setup
- metrics
- main findings
- limitations
- strategy recommendation

### Step 3：做 strategy map 图

需要一张最服务结论的图：

```text
quality
  ^
  |          keyword@1000
  |      compression@1000  <- default
  |   compression@300
  |
  | keyword@300
  | budget@300
  +----------------------> total-token savings
```

这张图用于解释：

- 为什么 `compression@1000` 是 default
- 为什么 `keyword@300` 不是 default
- 为什么 `budget@300` 只是 aggressive

### Step 4：写 v0.1 发布材料

包括：

- GitHub README 的 public-facing 版本
- blog post 大纲
- installation / quickstart
- 当前实验结论和 limitations
- 和 EAT / LZ Penalty / Adaptive CoT / internal-signal methods 的定位对比

### Step 5：把 BOCPD 作为 v0.2 设计文档落地

不是 vague future work，而是明确：

```text
v0.1: Layer 1 literal text detectors, compression@1000 default
v0.2: BOCPD threshold-reduced detector
v0.3: token calibration + cross-vendor validation
```

### Step 6：再考虑 router

router 应该在叙事稳定后做。第一版可以很简单：

```text
default: compression@1000
conservative: keyword@1000
aggressive: budget@300
balanced-aggressive: compression@300
math: compression@300
mmlu: keyword@300
riddle: keyword@1000
```

### Step 7：最后再考虑其他高级 detector

优先级：

1. hybrid detector
2. answer oscillation
3. embedding redundancy

这些都不是当前 main contribution 的必要条件。

## 9. 当前可以直接引用的结论

可以引用：

- `compression@1000` is the current default policy.
- It achieves 93.5% answer quality, 28.4% reasoning-char savings, and about 19.0% total-token savings under the current mixed API/estimated token accounting.
- `compression@300` is a balanced-aggressive policy with 92.3% answer quality and about 29.1% total-token savings.
- `budget@300` is more aggressive but loses too much quality.
- `keyword@1000` is more conservative but saves fewer total tokens.
- Phase 2 API/recovery failure rate is 0.0% in the current full run. Cases where Phase 2 answers are wrong or hard to parse are reflected in Quality, not in `phase2_failed`.
- Early-stop Phase 1 token usage cannot usually be read from API usage chunks because the stream is intentionally interrupted before the final usage event.

不要过度 claim：

- 不要说 total-token savings 是完全 API 实测
- 不要说 task router 已实现
- 不要说 embedding / BOCPD 已验证
- 不要说 `compression@1000` 是所有任务的理论最优
- 不要用 0.x pp 级别的 TokenSavings 微差做精确排序；default 选择应同时看 quality、Lost 和 token-savings 区间

## 10. 当前工作树提示

截至这份文档写入时，相关代码已经推到远端，当前主线 commit 是：

```text
020ee2e Expand riddle dataset and clarify token metrics
```

如果后续继续修改，建议先检查：

```bash
git status --short --branch
uv run pytest
uv run ruff check src tests experiments
uv run mypy src
```
