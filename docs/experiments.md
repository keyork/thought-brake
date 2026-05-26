# Experiment Guide

This guide describes the full workflow for running thought-brake experiments.

## 1. Prepare Environment

Install development and experiment dependencies:

```bash
uv sync --dev --group experiments
```

Create a local `.env` file:

```bash
cp .env.example .env
```

Fill in at least these values:

```bash
THOUGHT_BRAKE_API_KEY=your-api-key
THOUGHT_BRAKE_BASE_URL=https://your-llm-api.example/v1
THOUGHT_BRAKE_MODEL=your-reasoning-model
```

Runtime tuning can stay on defaults for the first run. Change the detector, budget, prompt, tag, or fallback values in `.env` only when you need to compare different configurations.

## 2. Run the Clean Main Token Experiment

This is the current recommended experiment entrypoint. New runs write schema v4
rows with token fields and `stop_detail`, then build the focused report. The
v0.1 full-run files were produced with schema v3 and remain valid for the v0.1
report.

```bash
./experiments/run_token_main.sh
```

Defaults:

```text
DATASET=all
N=100
BUDGETS=0,300,1000
WORKERS=25
PHASE2=direct
```

`N` applies to every dataset, including riddles. The local riddle set currently
contains 100 questions, so the default full run uses 100 math, 100 MMLU, and 100
riddle questions.

Outputs:

```text
experiments/results/full_budget.jsonl
experiments/results/full_compression.jsonl
experiments/results/full_keyword.jsonl
experiments/report/full_main_token/report.md
experiments/report/full_main_token/overall_tradeoff.png
experiments/report/full_main_token/dataset_decision_matrix.png
experiments/report/full_main_token/loss_vs_savings.png
```

Override values from the shell:

```bash
WORKERS=10 N=20 ./experiments/run_token_main.sh
```

Current report interpretation:

- Global default policy: `compression@1000`.
- Conservative high-quality policy: `keyword@1000`.
- Aggressive high-savings policy: `budget@300`.
- Balanced-aggressive policy: `compression@300`.
- The current report uses total-token cost as the main cost metric. It reads API
  `total_tokens` when available and falls back to `estimated_total_tokens` for
  rows where streaming usage is unavailable.
- In the latest full run, 400/1800 treated rows used API `total_tokens`, and
  1400/1800 treated rows used `estimated_total_tokens`. Treat total-token
  savings as directionally useful, not as final billing-grade measurement.

## 3. Run a Smoke Test

Run a small experiment without evaluation first. This checks credentials, streaming, early stopping, JSONL writing, and resume behavior.

```bash
uv run python experiments/runner.py \
  --dataset riddles \
  --budgets 0,100,200 \
  --difficulties easy \
  --workers 10 \
  --track-usage \
  --skip-eval
```

Expected output path:

```text
experiments/results/riddles.jsonl
```

Each row is one `(question, budget)` result. `budget=0` is the baseline with early stopping disabled.
In the current schema, `budget=0` still uses streaming monitor with `detector="none"`, so baseline `reasoning_chars` is measurable.

## 4. Run Riddle Experiments

Run the default riddle sweep with 10 parallel API workers:

```bash
uv run python experiments/runner.py \
  --dataset riddles \
  --budgets 0,100,200,500,1000 \
  --detector budget \
  --workers 10
```

Useful variants:

```bash
uv run python experiments/runner.py \
  --dataset riddles \
  --difficulties easy,medium \
  --budgets 0,100,200,500 \
  --workers 10
```

```bash
uv run python experiments/runner.py \
  --dataset riddles \
  --budgets 0,100,200,500,1000 \
  --workers 5
```

Lower `--workers` if the LLM API rate-limits or returns transient failures.

Run the Layer 1 compression detector:

```bash
uv run python experiments/runner.py \
  --dataset riddles \
  --budgets 0,500,1000,2000 \
  --detector compression \
  --workers 10 \
  --output experiments/results/riddles_compression.jsonl
```

## 5. Run GSM8K Experiments

GSM8K requires the `experiments` dependency group because it loads from HuggingFace `datasets`.

```bash
uv run python experiments/runner.py \
  --dataset gsm8k \
  --n 100 \
  --budgets 0,200,500,1000,2000 \
  --workers 10
```

For a smaller check:

```bash
uv run python experiments/runner.py \
  --dataset gsm8k \
  --n 20 \
  --budgets 0,200,500 \
  --workers 10 \
  --skip-eval
```

## 6. Run All Datasets

```bash
uv run python experiments/runner.py \
  --dataset all \
  --n 100 \
  --budgets 0,100,200,500,1000,2000 \
  --workers 10
```

By default this writes to:

```text
experiments/results/all.jsonl
```

Use `--output` to choose a separate result file:

```bash
uv run python experiments/runner.py \
  --dataset all \
  --n 100 \
  --budgets 0,100,200,500,1000,2000 \
  --workers 10 \
  --output experiments/results/all_10w.jsonl
```

## 7. Resume Behavior

The runner is append-only and resumable.

Before starting, it scans the output JSONL once and skips any existing `(question_id, budget, detector)` pair for the current schema. If a run is interrupted, run the same command again and it will continue pending pairs.

Do not run two separate runner processes against the same output file at the same time. Within one process, parallel workers are safe because only the main thread writes JSONL.

## 8. Result Fields

Important fields in each JSONL record:

| field | meaning |
|---|---|
| `schema_version` | Result schema version; analysis keeps the latest version when mixed |
| `question_id` | Dataset-local question id |
| `budget` | Soft budget; `0` means baseline |
| `detector` | `none`, `budget`, `compression`, `ngram`, `keyword`, `semantic`, or experimental `bocpd` |
| `reasoning_chars` | Collected reasoning character count |
| `truncated` | Whether early stopping was triggered |
| `stop_reason` | `natural`, `soft`, `hard`, or `interrupted` |
| `stop_detail` | Detector diagnostic detail, e.g. BOCPD `p_change`, `z`, and `r_map` |
| `answer` | Final answer returned by the client |
| `answer_chars` | Character length of the final answer |
| `phase1_*_tokens` | API usage returned for Phase 1 when available |
| `phase2_*_tokens` | API usage returned for Phase 2 when available |
| `total_tokens` | Sum of API usage tokens when available |
| `estimated_*_tokens` | Local token estimate, used when streaming usage is unavailable |
| `token_usage_source` | `api`, `estimate`, or `none` |
| `quality_score` | Evaluation score; `-1` when `--skip-eval` is used |
| `latency_ms` | End-to-end request latency |
| `phase2_used` | Whether Phase 2 recovery was used |
| `phase2_failed` | Whether Phase 2 failed and fallback was used |

## 9. Generate Reports

For the clean main experiment, use the focused report:

```bash
uv run python experiments/report_full_main.py \
  --inputs \
    experiments/results/full_budget.jsonl \
    experiments/results/full_compression.jsonl \
    experiments/results/full_keyword.jsonl \
  --output experiments/report/full_main_token
```

The older generic analysis is still useful for ad hoc files, but it should not be
used for final conclusions if `experiments/results/` contains mixed historical
JSONL files.

### Generic Analysis

After running evaluated experiments, generate tables and plots:

```bash
uv run python experiments/analysis.py \
  --input experiments/results \
  --output experiments/report
```

Generated files:

```text
experiments/report/summary.csv
experiments/report/table.md
experiments/report/pareto.png
experiments/report/reasoning_chars.png
```

Analyze one result file:

```bash
uv run python experiments/analysis.py \
  --input experiments/results/riddles.jsonl \
  --output experiments/report/riddles
```

## 10. Troubleshooting

If requests are too slow, increase `--workers` up to the LLM API's rate limit.
The clean main script defaults to `WORKERS=25`; individual `runner.py` examples
in this document use `--workers 10` as a conservative starting point.

If the LLM API rate-limits, lower `--workers` to `5` or `3`.

If `quality_score` is `-1`, the run used `--skip-eval`; rerun without it for evaluated records.

If analysis reports no evaluated records, make sure the input JSONL contains `quality_score >= 0`.

If GSM8K fails to import `datasets`, run:

```bash
uv sync --dev --group experiments
```

If Phase 2 direct mode needs a different no-thinking parameter for your LLM API,
set `THOUGHT_BRAKE_PHASE2_EXTRA_BODY` to that API's JSON object. Set it to an
empty value and use `--phase2 prefill` when the API cannot disable reasoning.
