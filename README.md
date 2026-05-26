# thought-brake

**Client-side streaming monitor-and-interrupt for black-box reasoning LLM APIs.**

Reasoning model 在处理简单问题时经常产生过度推理（overthinking）：反复自我验证、横跳怀疑、已经得到答案后继续展开。对只使用 LLM API 的用户来说，这很难从服务端解决，因为通常拿不到 hidden states、logits、采样器或模型权重。

`thought-brake` 是一个纯客户端库：它只依赖 API streaming 暴露出来的 visible reasoning text，在线监控重复、犹豫、低信息密度等字面信号；当信号触发时主动中断 thinking，并通过 Phase 2 direct recovery 收束最终答案。

v0.1 的核心定位：

- **Client-side**：不修改服务端，不训练模型
- **Black-box API**：不访问 logits / hidden states / sampler
- **Visible reasoning text**：只看流式 reasoning 文本
- **Two-phase recovery**：中断后重新请求 final answer
- **Cost-aware evaluation**：同时报告 reasoning savings 和 total-token savings

正式 v0.1 研究报告见 [docs/report_v0_1.md](docs/report_v0_1.md)。

## 实验结果

### 主线 token 实验

当前 v0.1 主线实验覆盖 `math=100`、`mmlu=100`、`riddle=100`，每个策略配置共 300 条样本，使用 `direct` Phase 2，并记录 schema v3 token 字段。v0.2 新运行会写 schema v4，并额外记录 detector `stop_detail`。运行主线实验并生成报告：

```bash
./experiments/run_token_main.sh
```

当前推荐默认策略是 **`compression@1000`**：在质量接近高位的前提下，拿到可用的 total-token savings。`compression@300` 是更激进的 balanced-aggressive 策略，`keyword@1000` 是保守高质量策略，`budget@300` 是 aggressive savings 策略。

| Config | Quality | Baseline | ReasoningSavings | TokenSavings | Lost | Phase2Fail |
|---|---:|---:|---:|---:|---:|---:|
| budget@300 | 88.6% | 97.3% | 77.4% | **42.1%** | 31 | 0.0% |
| budget@1000 | 91.0% | 97.3% | 40.1% | 20.6% | 26 | 0.0% |
| compression@300 | 92.3% | 97.7% | **62.2%** | **29.1%** | 20 | 0.0% |
| **compression@1000** | **93.5%** | 97.7% | 28.4% | 19.0% | 18 | 0.0% |
| keyword@300 | 91.0% | 96.7% | 61.2% | 28.9% | 23 | 0.0% |
| keyword@1000 | 94.3% | 96.7% | 22.1% | 14.3% | 11 | 0.0% |

分数据集的第一版路由建议：

| Dataset | Recommended | Quality | TokenSavings | 说明 |
|---|---|---:|---:|---|
| math | compression@300 | 93.0% | 24.9% | 在 3pp 质量容忍内，比保守策略多省 token |
| mmlu | keyword@300 | 96.0% | 32.4% | 在高质量区间内 token 节省最高 |
| riddle | keyword@1000 | 94.0% | 12.7% | 扩展到 100 题后，保守策略最稳 |

`TokenSavings` 使用 API `total_tokens`；流式 usage 不可用时按行回退到 `estimated_total_tokens`。本次 treated rows 中，400/1800 使用 API `total_tokens`，1400/1800 使用估算值。当前估算器在可校准 Phase 1 样本上的 mean absolute error 约 14.4%，所以 total-token savings 应按区间级结论理解，不应精确比较 0.x pp 的差异。

## 工作原理

`thought-brake` 不是单一的固定 budget 截断器，而是一套“流式观测 → 早停决策 → 答案收束”的客户端控制框架。当前实现已经支持 budget、compression、ngram、keyword、semantic 多种 detector，并已加入实验性的 `bocpd` detector；v0.1 聚焦可解释、可复现的 Layer 1 literal text detectors，v0.2 会推进 BOCPD / change-point detector 来减少 magic threshold。

### 1. 两阶段执行框架

每次请求分为两个阶段：

```
┌──────────────────────────────────────────────────────────┐
│                    Client Wrapper                         │
│                                                           │
│  Phase 1: Stream + Monitor                               │
│  ┌──────────────────────┐                                │
│  │ 流式接收 reasoning    │                                │
│  │ 提取文本信号          │                                │
│  │ detector 判断是否截断  │──────── 触发早停 ────────┐    │
│  └──────────┬───────────┘                          │      │
│             │ 模型自然结束                           │      │
│             ↓                                        ↓      │
│  Phase 2: 引导最终答案                                │      │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Direct 模式（默认）:                               │    │
│  │   构造 user 消息（原始对话 + 问题 + 推理摘要）       │    │
│  │   通过配置化 API 参数禁用推理                       │    │
│  │                                                    │    │
│  │ Prefill 模式（备选）:                               │    │
│  │   assistant 前缀续写（用于不支持禁用推理的 API）      │    │
│  └──────────────────────┬───────────────────────────┘    │
│                          ↓                                │
│                  Final Answer                            │
└──────────────────────────────────────────────────────────┘
```

**Phase 1 — Stream + Monitor**：流式调用模型，收集 `reasoning_content`，并持续把新增片段交给 detector。detector 可以只看长度，也可以看压缩率、重复 n-gram、犹豫短语、内容词重合度等文本信号。

**Phase 2 — Direct 模式**（核心突破）：截断后保留 system/developer 控制消息，并发一条新 user 消息，包含原始对话、当前问题和推理摘要；如果 LLM API 支持禁用推理，可以通过 `phase2_extra_body` 配置对应参数。相比 prefill 模式，能显著减少推理泄漏和元评论问题。

### 2. 早停信号不是只有 budget

最早的 baseline 是 `budget`：超过 `soft_budget` 后在句末截断，超过 `hard_limit` 强制截断。它简单、可控，在推理长度分布稳定的数据集上很好用，但本质是在“猜这类问题需要多长推理”。

后续方案把早停决策改成可插拔 detector。这里 `@300` / `@1000` 是 safety budget / fallback hard-limit 相关配置，单位是 reasoning chars；对 `compression` / `keyword` 这类 signal detector 来说，真正停止点仍由文本信号决定。

| 层次 | 方案 | 作用 |
|---|---|---|
| Baseline | `budget` | 用固定字符预算控制上限，作为最稳的对照组 |
| Signal detector | `compression` / `ngram` / `keyword` / `semantic` | 不预判推理长度，等模型出现重复、犹豫、语义重述等 overthinking 信号再截断 |
| Hybrid | signal guard + budget fallback | 先等信号，超过硬上限仍兜底截断，避免无限推理 |
| Router | task-aware detector selection | 根据问题类型自动选择 detector 和预算 |
| Research extensions | answer oscillation / embedding / BOCPD | 面向多选摇摆、换词重述、阶段变化等更细粒度信号 |

跨数据集实验已经验证：没有万能最优 detector。Riddles 这类短且均匀的任务适合 budget；MMLU 这类推理长度分布极宽的任务，compression/keyword 在同等或更低预算下更稳。

### 3. 后续自适应方案

下一步不是继续手调一个全局 budget，而是把“选择策略”也纳入系统：

```text
question
  → task router
  → choose(detector, soft_budget, hard_limit, phase2_mode)
  → stream monitor
  → detector decision
  → final-answer phase
```

计划中的优先级：

1. **v0.1 release**：发布当前 Layer 1 方法、实验结果和 limitations。
2. **BOCPD / change-point detector**：减少 magic threshold，这是 v0.2 的核心。
3. **Token calibration / cross-vendor sanity check**：让 total-token savings 更硬。
4. **任务路由 / hybrid detector**：在叙事和成本测量稳定后再做。

### 4. 核心洞察

Reasoning 是 autoregressive sampling 过程，每一步 token 生成依赖前缀上下文。这意味着在任意 token 边界中断 sampling 是合法操作——外部强制终止 thinking 阶段在数学上是无损的。

### 5. 为什么 Direct 模式优于 Prefill

在大量实验中发现，prefill 模式（把 partial reasoning 作为 assistant 前缀注入，让模型续写）在实践中失败：模型不尊重注入的 thinking 结束标签，经常从 prefill 内容继续推理，导致答案中泄漏大量推理草稿。

Direct 模式通过 LLM API 支持的禁用推理参数阻止 Phase 2 中继续推理，使答案更直接、简洁。

## 研究背景

Reasoning model overthinking 的缓解方法可以分为四类：

| 类别 | 信号 | 黑盒 API 可用 | 代表方法 |
|---|---|---|---|
| 训练阶段 | 奖励 / stop token | 否 | RL 训练、SFT |
| 内部信号 | hidden states / logits / entropy | 通常否 | 探针、熵检测 |
| 探测调用 | trial answer / answer stability | 是，但贵 | 多次采样对比 |
| **文本信号** | **n-gram / compression / repetition** | **是** | **thought-brake** |

`thought-brake` 聚焦最后一类——客户端流式文本信号。不依赖模型权重、hidden states 或额外 API 调用，适合任何 Chat Completions 兼容的 LLM API 用户。

> 这一定位不试图打败内部信号方法（它们通常更强），而是填补"黑盒 API + 客户端部署"这个空白场景。

详细调研见 [docs/survey.md](docs/survey.md)。

## Detector

早停决策由可插拔的 detector 负责：

| detector | 说明 |
|---|---|
| `none` | 只监控不截断，用于 baseline 测量真实 reasoning 长度 |
| `budget` | soft/hard 字符预算截断，当前作为 fixed-budget baseline 和 aggressive policy |
| `compression` | CRD + LZ-rate 压缩信号，当前默认策略使用 `compression@1000` |
| `ngram` | n-gram literal overlap，用于捕捉字面重复 |
| `keyword` | 犹豫短语密度、结论后继续推理等信号，当前保守策略使用 `keyword@1000` |
| `semantic` | 内容词 Jaccard 相似度，用于捕捉粗粒度重述 |
| `bocpd` | 实验性 BOCPD / change-point detector，用低维文本特征检测推理阶段变化 |

Detector 接口：

```python
class ReasoningDetector(Protocol):
    name: DetectorName
    def update(self, piece: str, total_chars: int) -> StopDecision: ...
```

## 快速开始

### 安装

```bash
uv sync --dev
```

如果要运行实验：

```bash
uv sync --dev --group experiments
```

### 配置

```bash
cp .env.example .env
```

至少填写 API 信息：

```bash
THOUGHT_BRAKE_API_KEY=your-api-key
THOUGHT_BRAKE_BASE_URL=https://your-llm-api.example/v1
THOUGHT_BRAKE_MODEL=your-reasoning-model
```

### 使用

```python
from thought_brake import EarlyStopConfig, ThoughtBrakeClient

client = ThoughtBrakeClient()

resp = client.chat(
    messages=[{"role": "user", "content": "盲人买剪刀还是聋哑人买锤子，谁先买到？"}],
    config=EarlyStopConfig.for_task("chat"),
)

print(resp.content)
print(resp.metrics)
```

`resp.metrics`：

| 字段 | 含义 |
|---|---|
| `reasoning_chars` | 收集到的 reasoning 字符数 |
| `stop_reason` | `natural` / `soft` / `hard` / `interrupted` |
| `phase2_used` | 是否触发 Phase 2 |
| `phase2_failed` | Phase 2 是否失败走 fallback |

### 任务预设

| preset | soft_budget | hard_limit | 适用场景 |
|---|---:|---:|---|
| `chat` | 200 | 400 | 闲聊、脑筋急转弯 |
| `qa` | 500 | 1000 | 一般问答 |
| `math` | 1500 | 3000 | 数学、结构化推理 |
| `complex` | 3000 | 6000 | 复杂推理任务 |

手动配置：

```python
resp = client.chat(
    messages=[{"role": "user", "content": "问题"}],
    config=EarlyStopConfig(
        soft_budget=500,
        hard_limit=1000,
        phase2_mode="direct",
    ),
)
```

禁用早停：

```python
resp = client.chat(messages, config=EarlyStopConfig(enable=False))
```

## API 兼容性

Direct 模式是否能完全关闭 Phase 2 推理，取决于 LLM API 是否提供对应参数。默认配置为：

```bash
THOUGHT_BRAKE_PHASE2_EXTRA_BODY='{"enable_thinking": false}'
```

如果你的 LLM API 使用其他字段，可以在 `.env` 中覆盖这个 JSON；如果不支持禁用推理，可以设置为空并改用 `phase2_mode="prefill"`。

## Roadmap

### Phase A：可靠基线 ✅

- Schema 化结果文件，记录 detector、phase2 状态、reasoning 长度
- `detector="none"` baseline 测量真实 reasoning 长度
- 断点续跑、exact-match 评测修复

### Phase B：Phase 2 输出收敛 ✅

- Direct 模式替代 prefill，禁用推理参数配置化以减少推理泄漏
- Threshold-based 推理摘要：短推理 head-only，长推理 head+tail
- `clean_final_answer()` 后处理
- Budget=300 达到 95% 质量保持率 + 72.5% 推理节省

### Phase C：Detector 对照实验 ✅

- 5 种 detector 对照：budget / compression / ngram / keyword / semantic
- 跨 3 个数据集（riddles / GSM8K / MMLU）× 3 个 budget（300/500/1000）
- 核心发现：没有万能最优，任务类型决定策略。Budget 适合推理短且均匀的任务，信号 detector 适合推理长度分布宽的任务

### Phase D：任务自适应路由（规划中）

- 任务分类路由：根据问题特征自动选择 detector + budget
- Hybrid detector：signal guard + budget fallback
- Answer oscillation 检测（对多选题特别适用）
- BOCPD / embedding 作为研究增强

详细研发路线见 [docs/idea.md](docs/idea.md)。

## 运行实验

详见 [docs/experiments.md](docs/experiments.md)。

```bash
# 25 并发运行主线 token 实验，并生成 focused report
./experiments/run_token_main.sh
```

## 测试

```bash
uv run pytest              # 73 tests
uv run ruff check src tests
uv run mypy src            # --strict
```

## 不适用场景

以下场景默认不应激进早停：

- **Agent 多步工作流** — 每步推理可能依赖前序完整推理
- **数学证明、定理推导** — 完整推理链不可截断
- **多步代码调试** — reasoning 即是解题过程本身
- **用户要求展示完整推理过程** — 截断违背需求
- **API 按请求 max_tokens 而非实际生成计费** — 早停无成本收益

## 项目结构

```
src/thought_brake/        核心库
  client.py               对外 Client，Phase 1/2 编排
  config.py               EarlyStopConfig dataclass
  detectors.py            可插拔 detector（none/budget/compression/ngram/keyword/semantic/bocpd）
  _monitor.py             流式 reasoning 监控
  _prefill.py             Phase 2 direct + prefill 模式
  types.py                类型定义
  _utils.py               工具函数
tests/                    73 个测试
experiments/              实验 runner、数据集、评测和分析
  datasets/               riddles、GSM8K、MMLU
  evaluate/               exact-match、LLM judge
  runner.py               并发实验（断点续跑）
  analysis.py             结果分析 + 可视化
docs/                     技术方案、调研、实验说明
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/technical_proposal.md](docs/technical_proposal.md) | 完整技术方案（架构、算法、实验数据） |
| [docs/report_v0_1.md](docs/report_v0_1.md) | v0.1 正式研究报告 |
| [docs/release_v0_1.md](docs/release_v0_1.md) | v0.1 release notes 和 blog 大纲 |
| [docs/plan.md](docs/plan.md) | milestone 执行计划 |
| [docs/v0_1_milestone_status.md](docs/v0_1_milestone_status.md) | v0.1 状态快照 |
| [docs/bocpd_design.md](docs/bocpd_design.md) | v0.2 BOCPD / change-point detector 设计草案 |
| [docs/bocpd_probe_20_report.md](docs/bocpd_probe_20_report.md) | BOCPD 20 题 probe 结果与下一步诊断 |
| [docs/survey.md](docs/survey.md) | Reasoning model overthinking 缓解方向调研 |
| [docs/idea.md](docs/idea.md) | 研发路线和 roadmap |
| [docs/experiments.md](docs/experiments.md) | 实验运行指南 |
| [docs/testing.md](docs/testing.md) | 测试说明 |

## License

MIT
