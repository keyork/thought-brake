# thought-brake

何时停止思考：面向 reasoning model 的推理早停工具。

`thought-brake` 封装 LLM API，在流式输出中监控 `reasoning_content`，当模型出现过度推理时提前截断 thinking 阶段，再通过 direct 模式（构造新 user 消息 + 禁用推理）引导模型直接输出最终答案。目标是降低 reasoning token 成本和延迟，同时尽量保持答案质量。

## 安装

```bash
uv sync --dev
```

如果要运行实验和分析报告：

```bash
uv sync --dev --group experiments
```

## 配置

复制环境变量模板：

```bash
cp .env.example .env
```

至少填写：

```bash
THOUGHT_BRAKE_API_KEY=your-api-key
THOUGHT_BRAKE_BASE_URL=https://your-llm-api.example/v1
THOUGHT_BRAKE_MODEL=your-reasoning-model
```

也支持兼容环境变量作为 fallback：

```bash
OPENAI_API_KEY=your-api-key
OPENAI_BASE_URL=https://example.com/v1
OPENAI_MODEL=your-model
```

运行时调参也可以放在 `.env` 中：

```bash
THOUGHT_BRAKE_SOFT_BUDGET=300
THOUGHT_BRAKE_HARD_LIMIT=600
THOUGHT_BRAKE_DETECTOR=budget
THOUGHT_BRAKE_COMPRESSION_BASELINE_CHARS=200
THOUGHT_BRAKE_COMPRESSION_RECENT_CHARS=200
THOUGHT_BRAKE_COMPRESSION_THETA_CRD=0.7
THOUGHT_BRAKE_COMPRESSION_THETA_LZ=0.5
THOUGHT_BRAKE_FINALIZE_HINT="\n\n好，已经想清楚了，直接给出最终答案。"
THOUGHT_BRAKE_REASONING_START_TAG="<think>\n"
THOUGHT_BRAKE_REASONING_END_TAG="\n</think>\n\n"
THOUGHT_BRAKE_FALLBACK_EXCERPT_CHARS=300
```

## 基本用法

```python
from thought_brake import EarlyStopConfig, ThoughtBrakeClient

client = ThoughtBrakeClient()

resp = client.chat(
    messages=[{"role": "user", "content": "盲人买剪刀的问题..."}],
    config=EarlyStopConfig.for_task("chat"),
)

print(resp.content)
print(resp.metrics)
```

`resp.metrics` 中包含：

| 字段 | 含义 |
|---|---|
| `reasoning_chars` | 收集到的 reasoning 字符数 |
| `stop_reason` | `natural`、`soft`、`hard` 或 `interrupted` |
| `phase2_used` | 是否触发 Phase 2 prefill |
| `phase2_failed` | Phase 2 是否失败并走 fallback |

## Detector

当前支持三种 detector：

| detector | 说明 |
|---|---|
| `none` | 只监控，不截断；用于 baseline 测真实 reasoning 长度 |
| `budget` | soft/hard budget 截断 |
| `compression` | CRD + LZ-rate 压缩信号原型 |

## 任务预设

| preset | soft_budget | hard_limit | 适用场景 |
|---|---:|---:|---|
| `chat` | 200 | 400 | 闲聊、简单脑筋急转弯 |
| `qa` | 500 | 1000 | 一般问答 |
| `math` | 1500 | 3000 | 数学、结构化推理 |
| `complex` | 3000 | 6000 | 复杂推理任务 |

也可以直接手动配置：

```python
resp = client.chat(
    messages=[{"role": "user", "content": "问题"}],
    config=EarlyStopConfig(
        soft_budget=500,
        hard_limit=1000,
        finalize_hint="\n\n直接给出最终答案。",
    ),
)
```

禁用早停：

```python
resp = client.chat(messages, config=EarlyStopConfig(enable=False))
```

## 工作原理

完整方案见 [docs/technical_proposal.md](docs/technical_proposal.md)。

核心流程：

1. Phase 1：流式调用模型，收集 `reasoning_content`。
2. 超过 `soft_budget` 后，在句末标点处软截断。
3. 超过 `hard_limit` 时强制截断。
4. Phase 2（direct 模式，默认）：构造一条 user 消息，包含原始问题和推理摘要，通过 `enable_thinking=False` 禁用推理，直接生成最终答案。
5. Phase 2（prefill 模式，备选）：将 partial reasoning + 收束提示作为 assistant prefill 注入，适用于不支持禁用推理的模型。

如果 Phase 1 流式中断但已经收集到 partial reasoning，会标记为 `interrupted` 并尝试 Phase 2 挽救。

## 运行实验

完整流程见 [docs/experiments.md](docs/experiments.md)。

10 并发运行 riddles 实验：

```bash
uv run python experiments/runner.py \
  --dataset riddles \
  --budgets 0,100,200,500,1000 \
  --detector budget \
  --workers 10
```

运行全部数据集：

```bash
uv run python experiments/runner.py \
  --dataset all \
  --n 100 \
  --budgets 0,100,200,500,1000,2000 \
  --detector budget \
  --workers 10
```

快速 smoke test：

```bash
uv run python experiments/runner.py \
  --dataset riddles \
  --budgets 0,100,200 \
  --difficulties easy \
  --detector budget \
  --workers 10 \
  --skip-eval
```

生成分析报告：

```bash
uv run python experiments/analysis.py \
  --input experiments/results \
  --output experiments/report
```

结果文件默认写入：

```text
experiments/results/
```

报告默认写入：

```text
experiments/report/
```

## 测试与检查

完整流程见 [docs/testing.md](docs/testing.md)。

常用检查：

```bash
uv run pytest
uv run ruff check src tests experiments
uv run mypy src
uv run python experiments/runner.py --help
uv run python experiments/analysis.py --help
```

单独跑某个测试文件：

```bash
uv run pytest tests/test_client.py
```

自动修复 ruff 可处理的问题：

```bash
uv run ruff check src tests experiments --fix
```

## 项目结构

```text
src/thought_brake/        核心库
tests/                    单元测试
experiments/              实验 runner、数据集、评测和分析
docs/                     技术方案、实验说明、测试说明
examples/                 使用示例
```

## 注意事项

- `--workers` 默认是 10；如果 LLM API 限流，可以降到 5 或 3。
- `budget=0` 表示 baseline，使用 `detector="none"` 监控但不截断。
- runner 支持断点续跑：已有 `(question_id, budget, detector)` 会自动跳过。
- 不要让两个 runner 进程同时写同一个 JSONL 文件。
- 如果 GSM8K 加载失败，先确认已执行 `uv sync --dev --group experiments`。
