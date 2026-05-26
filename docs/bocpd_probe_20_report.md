# BOCPD Probe 20 Report

日期：2026-05-26

输入文件：`experiments/results/bocpd_probe_20.jsonl`

## 结论

这轮 BOCPD 20 题 probe 不支持进入全量实验。关键原因不是质量或成本的单点数值，而是：

> 0 个 `soft` stop。所有截断都来自 `hard` fallback。

因此当前 BOCPD detector 还没有真正表现为 change-point monitor，只是在 `@300` / `@1000` hard limit 附近退化成了 hard-budget 策略。

## 实验规模

```text
dataset: all
n: 20
budgets: 0,300,1000
detector: bocpd
records: 180 = 60 baseline + 120 bocpd
schema_version: 3
```

注意：该结果文件仍是 schema v3，没有 `stop_detail`，所以不能看到 `p_change`、`z`、`r_map`。下一轮必须使用 schema v4 重新跑一个新文件。

## Stop Reason

| Budget | Natural | Hard | Soft |
|---:|---:|---:|---:|
| 300 | 6 | 54 | 0 |
| 1000 | 33 | 27 | 0 |

解读：

- `bocpd@300` 的 90.0% 截断全部是 hard fallback。
- `bocpd@1000` 有 45.0% 截断，也全部是 hard fallback。
- BOCPD posterior stop rule 没有在真实流上触发。

## Overall Metrics

| Config | N | Quality | Baseline Quality | ReasoningSavings | TokenSavings | Truncation | Phase2 | Phase2 API Fail | Lost | Fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bocpd@300 | 60 | 88.3% | 98.3% | 58.4% | 19.6% | 90.0% | 90.0% | 0.0% | 7 | 0 |
| bocpd@1000 | 60 | 96.7% | 98.3% | 12.4% | 2.7% | 45.0% | 45.0% | 0.0% | 3 | 1 |

这里的 Phase2 failure 指 API 调用层失败；无法解析或答错已经计入 Quality / Lost。

## Dataset Breakdown

| Dataset | Config | N | Quality | Baseline Quality | ReasoningSavings | TokenSavings | Truncation | Lost | Fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| math | bocpd@300 | 20 | 75.0% | 100.0% | 61.8% | 28.2% | 95.0% | 5 | 0 |
| math | bocpd@1000 | 20 | 95.0% | 100.0% | 1.5% | 0.5% | 55.0% | 1 | 0 |
| mmlu | bocpd@300 | 20 | 100.0% | 100.0% | 67.4% | 22.9% | 95.0% | 0 | 0 |
| mmlu | bocpd@1000 | 20 | 100.0% | 100.0% | 26.9% | 1.2% | 60.0% | 0 | 0 |
| riddle | bocpd@300 | 20 | 90.0% | 95.0% | 46.2% | 7.6% | 80.0% | 2 | 0 |
| riddle | bocpd@1000 | 20 | 95.0% | 95.0% | 9.0% | 6.2% | 20.0% | 2 | 1 |

解读：

- `bocpd@300` 最大问题在 math：质量只有 75.0%，Lost=5/20。
- `bocpd@1000` 质量基本可接受，但 total-token savings 只有 2.7%，作为节省策略意义不大。
- MMLU 在小样本上表现好，但因为没有 soft stop，不能证明 BOCPD 有效。

## Matched Comparison

以下比较只使用 BOCPD probe 覆盖到的同一批 60 个 question id。

| Config | N | Quality | Baseline Quality | ReasoningSavings | TokenSavings | Truncation | Lost | Fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| budget@300 | 60 | 87.5% | 99.2% | 74.9% | 33.5% | 100.0% | 7 | 0 |
| compression@300 | 60 | 95.0% | 95.0% | 58.8% | 17.1% | 96.7% | 2 | 2 |
| keyword@300 | 60 | 88.3% | 96.7% | 58.4% | 18.0% | 95.0% | 7 | 2 |
| bocpd@300 | 60 | 88.3% | 98.3% | 58.4% | 19.6% | 90.0% | 7 | 0 |
| budget@1000 | 60 | 94.2% | 99.2% | 34.4% | 4.9% | 78.3% | 4 | 0 |
| compression@1000 | 60 | 94.2% | 95.0% | 17.8% | 7.4% | 53.3% | 2 | 1 |
| keyword@1000 | 60 | 91.7% | 96.7% | 14.7% | 2.9% | 43.3% | 3 | 1 |
| bocpd@1000 | 60 | 96.7% | 98.3% | 12.4% | 2.7% | 45.0% | 3 | 1 |

解读：

- `bocpd@300` 没有优于 `compression@300`：质量低 6.7pp，Lost 多 5 个，TokenSavings 只高 2.5pp，且仍在 token estimate 不确定性范围内。
- `bocpd@1000` 质量高，但 TokenSavings 只有 2.7%，低于 `compression@1000` 的 7.4%。
- 当前 BOCPD 没有进入 Pareto frontier。

## 判断

当前结果说明：

1. BOCPD skeleton 能跑通 API 和实验系统。
2. Phase 2 recovery 没有出现 API 层失败。
3. 当前 stop rule 太保守，或者 `conclusion_seen` / `low_value_score` / `p_change` 的组合条件没有在真实流中对齐。
4. 在没有 `stop_detail` 前，不能判断具体卡在哪个条件。

因此下一步不是全量跑 BOCPD，而是：

1. 使用 schema v4 重新跑一个新的 probe 文件。
2. 检查 `stop_detail` 里的 `p_change`、`z`、`r_map`。
3. 再决定调哪个参数：`bocpd_stop_prob`、`bocpd_low_value_threshold`、`bocpd_min_windows`、`bocpd_window_chars` 或 `conclusion_seen` 条件。

推荐命令：

```bash
uv run python experiments/runner.py \
  --dataset all \
  --n 20 \
  --budgets 0,300,1000 \
  --detector bocpd \
  --workers 10 \
  --track-usage \
  --phase2 direct \
  --output experiments/results/bocpd_probe_20_v4.jsonl
```

