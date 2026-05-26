# BOCPD Probe 20 Report

日期：2026-05-26

输入文件：

- `experiments/results/bocpd_probe_20.jsonl`：schema v3 初始 probe
- `experiments/results/bocpd_probe_20_v4.jsonl`：schema v4 diagnostic probe
- `experiments/results/bocpd_probe_20_v4_detail.jsonl`：enhanced-detail probe

## 结论

三轮 BOCPD 20 题 probe 都不支持进入全量实验。关键原因不是质量或成本的单点数值，而是：

> 0 个 `soft` stop。所有截断都来自 `hard` fallback。

因此当前 BOCPD detector 还没有真正表现为 change-point monitor，只是在 `@300` / `@1000` hard limit 附近退化成了 hard-budget 策略。enhanced-detail probe 已经说明根因不是某个阈值“差一点”，而是当前信号整体没有进入 stop rule 的有效区域：

- `conclusion=0`：120/120
- `p_change`：mean 0.020，max 0.055；默认阈值 0.65
- `z`：mean 0.134，max 0.254；默认阈值 0.55

这意味着当前 BOCPD v0.2 skeleton 的设计假设没有在真实流上成立。后续如果继续 BOCPD，需要重做信号定义或 stop rule，而不是直接调低一两个阈值。

## 实验规模

```text
dataset: all
n: 20
budgets: 0,300,1000
detector: bocpd
records: 180 = 60 baseline + 120 bocpd
schema_version: 4
```

enhanced-detail 文件已经保留最后一次 posterior：

```text
hard_limit=600 last=bocpd p_change=0.016 z=0.098 r_map=1 windows=2 conclusion=0 recent=1
```

这让我们可以判断：BOCPD 不是触发失败偶然事件，而是系统性没有看到 conclusion / high change posterior / high low-value score。

## Stop Reason

| Budget | Natural | Hard | Soft |
|---:|---:|---:|---:|
| 300 | 4 | 56 | 0 |
| 1000 | 35 | 25 | 0 |

解读：

- `bocpd@300` 的 93.3% 截断全部是 hard fallback。
- `bocpd@1000` 有 41.7% 截断，也全部是 hard fallback。
- BOCPD posterior stop rule 没有在真实流上触发。

## Enhanced Detail

`bocpd_probe_20_v4_detail.jsonl` 的 120 条 BOCPD treated rows 都包含可解析 posterior detail：

| Field | Mean | Max | Stop Rule Threshold |
|---|---:|---:|---:|
| `p_change` | 0.020 | 0.055 | 0.65 |
| `z` | 0.134 | 0.254 | 0.55 |
| `conclusion` | 0.000 | 0.000 | must be 1 |
| `recent` | 0.517 | 1.000 | must be 1 |

解读：

- `conclusion` 是最硬 blocker：没有任何 treated sample 进入 `conclusion_seen=True`。
- `p_change` 也远低于阈值，即使去掉 conclusion 条件也不会触发。
- `z` 也远低于阈值，说明当前 low-value score 对真实 reasoning 流不敏感。
- `recent` 约一半为 1，但单独没有意义；其它必要条件全部失败。

## Overall Metrics

| Config | N | Quality | Baseline Quality | ReasoningSavings | TokenSavings | Truncation | Phase2 | Phase2 API Fail | Lost | Fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bocpd@300 | 60 | 93.3% | 97.5% | 56.1% | 15.5% | 93.3% | 93.3% | 0.0% | 3 | 1 |
| bocpd@1000 | 60 | 96.7% | 97.5% | 5.2% | -2.9% | 41.7% | 41.7% | 0.0% | 1 | 1 |

这里的 Phase2 failure 指 API 调用层失败；无法解析或答错已经计入 Quality / Lost。

## Dataset Breakdown

| Dataset | Config | N | Quality | Baseline Quality | ReasoningSavings | TokenSavings | Truncation | Lost | Fixed |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| math | bocpd@300 | 20 | 95.0% | 100.0% | 64.9% | 28.6% | 100.0% | 1 | 0 |
| math | bocpd@1000 | 20 | 95.0% | 100.0% | 14.6% | 8.1% | 40.0% | 1 | 0 |
| mmlu | bocpd@300 | 20 | 100.0% | 100.0% | 61.4% | 14.0% | 95.0% | 0 | 0 |
| mmlu | bocpd@1000 | 20 | 100.0% | 100.0% | 7.2% | -14.0% | 60.0% | 0 | 0 |
| riddle | bocpd@300 | 20 | 85.0% | 92.5% | 41.9% | 3.9% | 85.0% | 2 | 1 |
| riddle | bocpd@1000 | 20 | 95.0% | 92.5% | -6.1% | -2.7% | 25.0% | 0 | 1 |

解读：

- `bocpd@300` 的质量比前两轮好，但依然是 hard fallback 结果，不是 BOCPD 结果。
- `bocpd@1000` total-token savings 为负，主要因为 Phase 2 recovery 成本抵消了少量 reasoning savings。
- MMLU 在小样本上质量好，但 TokenSavings 对 `bocpd@1000` 为 -14.0%，不适合作为节省策略。

## Matched Comparison

以下比较只使用 BOCPD probe 覆盖到的同一批 60 个 question id。

| Config | N | Quality | Baseline Quality | ReasoningSavings | TokenSavings | Truncation | Lost | Fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| budget@300 | 60 | 87.5% | 99.2% | 74.9% | 33.5% | 100.0% | 7 | 0 |
| compression@300 | 60 | 95.0% | 95.0% | 58.8% | 17.1% | 96.7% | 2 | 2 |
| keyword@300 | 60 | 88.3% | 96.7% | 58.4% | 18.0% | 95.0% | 7 | 2 |
| bocpd@300 | 60 | 93.3% | 97.5% | 56.1% | 15.5% | 93.3% | 3 | 1 |
| budget@1000 | 60 | 94.2% | 99.2% | 34.4% | 4.9% | 78.3% | 4 | 0 |
| compression@1000 | 60 | 94.2% | 95.0% | 17.8% | 7.4% | 53.3% | 2 | 1 |
| keyword@1000 | 60 | 91.7% | 96.7% | 14.7% | 2.9% | 43.3% | 3 | 1 |
| bocpd@1000 | 60 | 96.7% | 97.5% | 5.2% | -2.9% | 41.7% | 1 | 1 |

解读：

- `bocpd@300` 接近 `compression@300`，但质量仍低 1.7pp，TokenSavings 也低 1.6pp，并且 0 个 soft stop。
- `bocpd@1000` 质量高，但 TokenSavings 为 -2.9%，不具备成本价值。
- 当前 BOCPD 不进入 Pareto frontier，也不能作为 v0.2 main contribution。

## 判断

当前结果说明：

1. BOCPD skeleton 能跑通 API 和实验系统。
2. Phase 2 recovery 没有出现 API 层失败。
3. 当前 stop rule 的三条核心条件都没有在真实流中对齐：`conclusion_seen` 全部失败，`p_change` 和 `z` 也远低于阈值。
4. 这不是“调低一点阈值”能解决的问题；把 `p_change_stop_prob` 从 0.65 降到 0.05 或把 `z` 阈值从 0.55 降到 0.25 会失去统计意义，基本退化成另一套 magic threshold。

因此下一步建议：

1. 不跑 BOCPD 全量。
2. 暂停 BOCPD 作为 v0.2 主线，把它标为 negative result / future work。
3. 如果未来继续 BOCPD，先离线保存 reasoning text，再重新标定 `conclusion`、feature scaling 和 change-point model。
4. 当前项目主线继续回到 v0.1 已验证的 `compression@1000` / `compression@300`，下一步优先做 cost calibration 或 cross-vendor sanity check。

当前不建议继续 rerun 同一 BOCPD probe，除非先改信号定义或记录 raw reasoning text。
