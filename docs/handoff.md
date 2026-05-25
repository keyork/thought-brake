# thought-brake 交接记录

日期：2026-05-24

> 历史说明：这是一份早期交接快照，记录的是 schema v2 和早期 riddle 实验状态。当前代码、实验入口和结论请以 `README.md`、`docs/experiments.md`、`docs/idea.md` 以及 `experiments/report_full_main.py` 为准。

## 当前状态

仓库当前大部分文件仍是 untracked。`git status --short --untracked-files=all` 会看到 `.gitignore`、`README.md` 已修改，以及 `src/`、`tests/`、`experiments/`、`docs/`、`.env.example`、`CLAUDE.md`、`pyproject.toml`、`uv.lock` 等新文件。

已忽略的本地/运行产物：

- `.env`
- `.ruff_cache/`、`.mypy_cache/`、`.pytest_cache/`
- `experiments/results/`
- `experiments/report/`
- `memory/`

## 已完成内容

### 核心库

- `ThoughtBrakeClient` 封装 LLM API client，对 reasoning stream 做早停。
- Phase 1 读取 `reasoning_content` 和 `content`。
- Phase 2 使用 assistant prefill，从 partial reasoning 收束到最终答案。
- `EarlyStopConfig` 集中配置，并支持 `THOUGHT_BRAKE_*` 环境变量覆盖。

### Detector 架构

新增 `src/thought_brake/detectors.py`。

已支持：

- `none`：只监控不截断，用于 baseline，能记录真实 `reasoning_chars`
- `budget`：soft/hard budget detector
- `compression`：Layer 1 CRD + LZ-rate 原型

`_monitor.py` 现在只负责流式读取，并把 reasoning chunk 喂给 detector。

### 实验 runner

- `experiments/runner.py` 支持并发 workers 和 `--detector {budget,compression}`。
- `budget=0` 使用 `detector="none"`，不再 passthrough，因此 baseline reasoning 长度可测。
- JSONL schema version 是 `2`。
- 结果字段包含：
  - `schema_version`
  - `detector`
  - `reasoning_chars`
  - `answer_chars`
  - `phase2_used`
  - `phase2_failed`
- resume key 实际为当前 schema 下的 `(question_id, budget, detector)`。

### Analysis

- `experiments/analysis.py` 会在混合结果中自动过滤最新 `schema_version`。
- summary 增加 detector 维度。

### 评测修复

- `experiments/evaluate/exact_match.py` 已支持 `70,000` 这类带逗号数字。
- 新增 `tests/test_exact_match.py`。

### Phase 2 输出收敛

- `_prefill.py` 现在会在 assistant prefill 后追加 final-only 用户提示：
  - 默认：`只输出最终答案。不要复述推理过程，不要列分析步骤，不要提到你已经思考。`
- 新增 `clean_final_answer()`，用于清理明显泄漏的推理草稿，例如 `1. 分析请求...`。
- 新增配置：
  - `final_answer_prompt`
  - `clean_phase2_answer`
  - env vars：`THOUGHT_BRAKE_FINAL_ANSWER_PROMPT`、`THOUGHT_BRAKE_CLEAN_PHASE2_ANSWER`

### 文档

- README 已改为中文。
- `docs/idea.md` 是 roadmap：
  - 可靠 baseline
  - Phase 2 final-answer-only
  - detector 对照
  - BOCPD/MDL 研究增强
- `docs/survey.md` 是谨慎版调研，不再过度 claim。
- `docs/experiments.md` 和 `docs/testing.md` 记录实验和测试流程。
- commit-target 文档里已去掉具体模型/供应商表述。

## 已观察到的实验结果

以下结果来自 Phase 2 final-only 修复前：

- `experiments/results/riddles_budget_v2.jsonl`
- `experiments/results/riddles_compression_v2.jsonl`

每个文件 80 行。

### Budget detector

- `budget=100`：quality 97.5%，savings 约 90.2%，truncation 100%，Phase 2 100%，answer length 约 5.72x baseline，latency +32.8%
- `budget=200`：quality 87.5%，savings 约 81.4%，answer length 约 5.42x
- `budget=500`：quality 85.0%，savings 约 55.7%，answer length 约 5.13x

### Compression detector

- `budget=500`：quality 90.0%，savings 约 26.1%，answer length 约 4.03x
- `budget=1000`：quality 97.5%，savings 约 9.5%，answer length 约 1.35x
- `budget=2000`：quality 97.5%，savings 约 6.7%，answer length 约 1.17x

结论：

- `budget=100` 当前最省 reasoning，但 Phase 2 答案过长是主要问题。
- `compression=1000/2000` 保守且质量稳定，但节省太少。
- Phase 2 输出收敛修复是在这些结果之后完成的，需要重跑验证。

## 下一步测试命令

先跑 20 道 riddles，baseline + `budget=100`：

```bash
uv run python experiments/runner.py \
  --dataset riddles \
  --budgets 0,100 \
  --detector budget \
  --workers 10 \
  --output experiments/results/riddles_budget100_final_only_v2.jsonl
```

分析：

```bash
uv run python experiments/analysis.py \
  --input experiments/results/riddles_budget100_final_only_v2.jsonl \
  --output experiments/report/riddles_budget100_final_only_v2
```

重点看：

- `quality_score`
- `reasoning_chars` savings
- `answer_chars`
- `phase2_used`
- `latency_ms`

## 已通过验证

```bash
uv run pytest
uv run ruff check src tests experiments
uv run mypy src
uv run python experiments/runner.py --help
uv run python experiments/analysis.py --help
```

最近一次结果：

- `pytest`：36 passed
- `ruff`：all checks passed
- `mypy src`：no issues

## 后续建议

1. 跑上面的 20 题测试。
2. 对比 `answer_chars` 是否显著低于旧的 `riddles_budget_v2.jsonl`。
3. 如果答案长度改善且质量接近 baseline，再跑全量 riddles 的 `budget=100`。
4. Phase 2 稳定后再调 compression detector。
5. 可以给 `analysis.py` 增加 `answer_ratio` 指标。
6. 提交前再次运行完整检查。

提交命令建议：

```bash
git add .
git commit -m "Add thought-brake client and experiment workflow"
git push origin main
```

注意：`git add .` 会包含 `CLAUDE.md`。该文件已改成通用 LLM API 口径。
