# BOCPD Probe 20 Report

日期：2026-05-26

输入文件：

- `experiments/results/bocpd_probe_20.jsonl`：schema v3 初始 probe
- `experiments/results/bocpd_probe_20_v4.jsonl`：schema v4 diagnostic probe

## 结论

两轮 BOCPD 20 题 probe 都不支持进入全量实验。关键原因不是质量或成本的单点数值，而是：

> 0 个 `soft` stop。所有截断都来自 `hard` fallback。

因此当前 BOCPD detector 还没有真正表现为 change-point monitor，只是在 `@300` / `@1000` hard limit 附近退化成了 hard-budget 策略。schema v4 的 `bocpd@1000` 数字看起来不错，但它不是 BOCPD posterior 触发出来的结果。

## 实验规模

```text
dataset: all
n: 20
budgets: 0,300,1000
detector: bocpd
records: 180 = 60 baseline + 120 bocpd
schema_version: 4
```

注意：v4 已经写入 `stop_detail`，但当前实现会被 `hard_limit` 或最后一次 `collecting` 覆盖掉最后的 posterior 细节。后续已改为保留 `last=bocpd ...` 和 blocker 信息，下一轮 probe 才能精确判断卡在哪个条件。

## Stop Reason

| Budget | Natural | Hard | Soft |
|---:|---:|---:|---:|
| 300 | 4 | 56 | 0 |
| 1000 | 33 | 27 | 0 |

解读：

- `bocpd@300` 的 93.3% 截断全部是 hard fallback。
- `bocpd@1000` 有 45.0% 截断，也全部是 hard fallback。
- BOCPD posterior stop rule 没有在真实流上触发。

## Overall Metrics

| Config | N | Quality | Baseline Quality | ReasoningSavings | TokenSavings | Truncation | Phase2 | Phase2 API Fail | Lost | Fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bocpd@300 | 60 | 89.2% | 99.2% | 60.4% | 21.5% | 93.3% | 93.3% | 0.0% | 7 | 0 |
| bocpd@1000 | 60 | 97.5% | 99.2% | 21.4% | 10.3% | 45.0% | 45.0% | 0.0% | 2 | 1 |

这里的 Phase2 failure 指 API 调用层失败；无法解析或答错已经计入 Quality / Lost。

## Dataset Breakdown

| Dataset | Config | N | Quality | Baseline Quality | ReasoningSavings | TokenSavings | Truncation | Lost | Fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| math | bocpd@300 | 20 | 90.0% | 100.0% | 68.8% | 36.6% | 95.0% | 2 | 0 |
| math | bocpd@1000 | 20 | 100.0% | 100.0% | 23.4% | 16.4% | 45.0% | 0 | 0 |
| mmlu | bocpd@300 | 20 | 95.0% | 100.0% | 66.9% | 21.5% | 100.0% | 1 | 0 |
| mmlu | bocpd@1000 | 20 | 100.0% | 100.0% | 31.3% | 6.7% | 60.0% | 0 | 0 |
| riddle | bocpd@300 | 20 | 82.5% | 97.5% | 45.6% | 6.5% | 85.0% | 4 | 0 |
| riddle | bocpd@1000 | 20 | 92.5% | 97.5% | 9.4% | 7.7% | 30.0% | 2 | 1 |

解读：

- `bocpd@300` 最大问题在 riddle：质量只有 82.5%，Lost=4/20。
- `bocpd@1000` 质量基本可接受，TokenSavings 10.3%，但这仍然来自 hard fallback / natural finish，不是 BOCPD soft stop。
- MMLU 在小样本上表现好，但因为没有 soft stop，不能证明 BOCPD 有效。

## Matched Comparison

以下比较只使用 BOCPD probe 覆盖到的同一批 60 个 question id。

| Config | N | Quality | Baseline Quality | ReasoningSavings | TokenSavings | Truncation | Lost | Fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| budget@300 | 60 | 87.5% | 99.2% | 74.9% | 33.5% | 100.0% | 7 | 0 |
| compression@300 | 60 | 95.0% | 95.0% | 58.8% | 17.1% | 96.7% | 2 | 2 |
| keyword@300 | 60 | 88.3% | 96.7% | 58.4% | 18.0% | 95.0% | 7 | 2 |
| bocpd@300 | 60 | 89.2% | 99.2% | 60.4% | 21.5% | 93.3% | 7 | 0 |
| budget@1000 | 60 | 94.2% | 99.2% | 34.4% | 4.9% | 78.3% | 4 | 0 |
| compression@1000 | 60 | 94.2% | 95.0% | 17.8% | 7.4% | 53.3% | 2 | 1 |
| keyword@1000 | 60 | 91.7% | 96.7% | 14.7% | 2.9% | 43.3% | 3 | 1 |
| bocpd@1000 | 60 | 97.5% | 99.2% | 21.4% | 10.3% | 45.0% | 2 | 1 |

解读：

- `bocpd@300` 没有优于 `compression@300`：质量低 5.8pp，Lost 多 5 个。
- `bocpd@1000` 在这批题上数值不错：Quality 97.5%，TokenSavings 10.3%，优于 `compression@1000` 的 94.2% / 7.4%。但该结果不能算 BOCPD 胜出，因为 0 个样本由 BOCPD soft stop 触发。
- 当前可记录为一个“hard fallback @1000 的偶然好点”，不能作为 v0.2 contribution。

## 判断

当前结果说明：

1. BOCPD skeleton 能跑通 API 和实验系统。
2. Phase 2 recovery 没有出现 API 层失败。
3. 当前 stop rule 太保守，或者 `conclusion_seen` / `low_value_score` / `p_change` 的组合条件没有在真实流中对齐。
4. 当前 v4 的 `stop_detail` 还不够好：hard stop 覆盖了最后一次 posterior，natural finish 又常被 collecting 状态覆盖。

因此下一步不是全量跑 BOCPD，而是：

1. 改进 `stop_detail`，保留最后一次 posterior 和 blocked condition。
2. 使用新诊断字段重跑一个 20 题 probe 文件。
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
  --output experiments/results/bocpd_probe_20_v4_detail.jsonl
```
