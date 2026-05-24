# thought-brake

**Reasoning model 推理早停工具 — 在客户端截断过度思考，降低 token 成本和延迟。**

Reasoning model（如 DeepSeek-R1、Qwen-QWQ、GLM-5 等）在处理简单问题时经常产生过度推理（overthinking）：反复自我验证、横跳怀疑、沉没成本式继续推理。实测中，一道脑筋急转弯的 reasoning tokens 可达 prompt 的 32 倍，其中 70%+ 是无效的自我验证。

`thought-brake` 是一个纯客户端库，通过流式监控 reasoning 输出来检测过度推理，提前截断 thinking 阶段，再引导模型直接输出最终答案。不需要模型权重、不需要额外 API 调用、不需要训练。

## 实验结果

Riddles 数据集（20 题，3 个难度），budget=300，direct 模式：

| 指标 | 数值 |
|---|---|
| **质量保持率** | **95.0%**（easy 88.9% / medium 100% / hard 100%） |
| **推理节省** | **72.5%**（avg 2077 chars → 317 chars） |
| **延迟** | **0.90x** baseline（略低） |
| **答案长度** | **0.19x** baseline（无推理泄漏，更简洁） |

Budget sweep 对比：

| Budget | Easy | Medium | Hard | 整体 | 推理节省 |
|---|---|---|---|---|---|
| 100 | 61.1% | 92.9% | 100% | 81.2% | 90.4% |
| 200 | 77.8% | 78.6% | 75.0% | 77.3% | 81.2% |
| **300** | **88.9%** | **100%** | **100%** | **95.0%** | **72.5%** |
| 500 | 83.3% | 100% | 87.5% | 90.6% | 55.4% |

## 工作原理

系统分为两个阶段：

```
┌──────────────────────────────────────────────────────────┐
│                    Client Wrapper                         │
│                                                           │
│  Phase 1: Stream + Monitor                               │
│  ┌──────────────────────┐                                │
│  │ 流式接收 reasoning    │                                │
│  │ 字符计数              │                                │
│  │ detector 判断是否截断  │──────── 超出预算 ────────┐    │
│  └──────────┬───────────┘                          │      │
│             │ 模型自然结束                           │      │
│             ↓                                        ↓      │
│  Phase 2: 引导最终答案                                │      │
│  ┌──────────────────────────────────────────────────┐    │
│  │ Direct 模式（默认）:                               │    │
│  │   构造 user 消息（问题 + 推理摘要）                 │    │
│  │   enable_thinking=False 禁用推理                   │    │
│  │                                                    │    │
│  │ Prefill 模式（备选）:                               │    │
│  │   assistant 前缀续写（用于不支持禁用推理的模型）      │    │
│  └──────────────────────┬───────────────────────────┘    │
│                          ↓                                │
│                  Final Answer                            │
└──────────────────────────────────────────────────────────┘
```

**Phase 1 — Stream + Monitor**：流式调用模型，收集 `reasoning_content` 并计数。当超过 `soft_budget` 时在句末标点处软截断，超过 `hard_limit` 时强制截断。

**Phase 2 — Direct 模式**（核心突破）：截断后发一条新 user 消息，包含原始问题和推理摘要，通过 `enable_thinking=False` 从根本上禁用模型的推理阶段。相比 prefill 模式，彻底消除了推理泄漏和元评论问题。

### 核心洞察

Reasoning 是 autoregressive sampling 过程，每一步 token 生成依赖前缀上下文。这意味着在任意 token 边界中断 sampling 是合法操作——外部强制终止 thinking 阶段在数学上是无损的。

### 为什么 Direct 模式优于 Prefill

在大量实验中发现，prefill 模式（把 partial reasoning 作为 assistant 前缀注入，让模型续写）在实践中失败：模型不尊重注入的 thinking 结束标签，经常从 prefill 内容继续推理，导致答案中泄漏大量推理草稿。

Direct 模式的 `enable_thinking=False` 从根本上阻止了 Phase 2 中的推理行为，使答案直接、简洁、无泄漏。

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
| `budget` | soft/hard 字符预算截断，当前最优（95%） |
| `compression` | CRD + LZ-rate 压缩信号（80%） |
| `ngram` | n-gram literal overlap（80%） |
| `keyword` | 犹豫短语密度，结论后触发（87.5%） |
| `semantic` | 内容词 Jaccard 相似度（87.5%） |

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

Direct 模式需要 API 支持禁用推理：

| 模型 | 禁用推理参数 | 支持情况 |
|---|---|---|
| GLM-5.x | `extra_body={"enable_thinking": False}` | ✅ |
| DeepSeek V4 | `extra_body={"thinking": {"type": "disabled"}}` | ✅ |
| Qwen hybrid | `extra_body={"enable_thinking": False}` | ✅ |
| DeepSeek-R1 | 不支持 | ❌ 使用 prefill 模式 |
| Qwen-QWQ | 不支持 | ❌ 使用 prefill 模式 |

## Roadmap

### Phase A：可靠基线 ✅

- Schema 化结果文件，记录 detector、phase2 状态、reasoning 长度
- `detector="none"` baseline 测量真实 reasoning 长度
- 断点续跑、exact-match 评测修复

### Phase B：Phase 2 输出收敛 ✅

- Direct 模式替代 prefill，`enable_thinking=False` 消除推理泄漏
- Threshold-based 推理摘要：短推理 head-only，长推理 head+tail
- `clean_final_answer()` 后处理
- Budget=300 达到 95% 质量保持率 + 72.5% 推理节省

### Phase C：Detector 对照实验 ✅

- 5 种 detector 对照：budget / compression / ngram / keyword / semantic
- Budget 95%，keyword/semantic 87.5%，compression/ngram 80%
- 关键发现：黑盒文本信号无法区分"合法复杂推理"和"过度思考"（hard 题全部 75%）
- Overthinking 主要是语义重复（换词重述），不是 literal 重复

### Phase D：研究增强（规划中）

- 组合信号：keyword + semantic composite，验证是否突破 hard=75%
- Embedding-based 语义冗余检测（PUMA 路线）
- Answer oscillation 检测（r=0.78 相关）
- BOCPD on 文本信号序列

详细研发路线见 [docs/idea.md](docs/idea.md)。

## 运行实验

详见 [docs/experiments.md](docs/experiments.md)。

```bash
# 10 并发运行 riddles 实验
uv run python experiments/runner.py \
  --dataset riddles \
  --budgets 0,100,200,300,500 \
  --detector budget \
  --phase2 direct \
  --workers 10

# 生成分析报告
uv run python experiments/analysis.py \
  --input experiments/results/riddles.jsonl \
  --output experiments/report/riddles_sweep
```

## 测试

```bash
uv run pytest              # 63 tests
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
  detectors.py            可插拔 detector（none/budget/compression/ngram/keyword/semantic）
  _monitor.py             流式 reasoning 监控
  _prefill.py             Phase 2 direct + prefill 模式
  types.py                类型定义
  _utils.py               工具函数
tests/                    63 个测试
experiments/              实验 runner、数据集、评测和分析
  datasets/               riddles、GSM8K
  evaluate/               exact-match、LLM judge
  runner.py               并发实验（断点续跑）
  analysis.py             结果分析 + 可视化
docs/                     技术方案、调研、实验说明
```

## 文档

| 文档 | 内容 |
|---|---|
| [docs/technical_proposal.md](docs/technical_proposal.md) | 完整技术方案（架构、算法、实验数据） |
| [docs/survey.md](docs/survey.md) | Reasoning model overthinking 缓解方向调研 |
| [docs/idea.md](docs/idea.md) | 研发路线和 roadmap |
| [docs/experiments.md](docs/experiments.md) | 实验运行指南 |
| [docs/testing.md](docs/testing.md) | 测试说明 |

## License

MIT
