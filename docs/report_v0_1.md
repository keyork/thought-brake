# v0.1 Research Report

日期：2026-05-25

## 摘要

`thought-brake` 研究一个很具体的部署问题：黑盒 reasoning LLM API 在简单或已收敛问题上会继续生成大量低收益 reasoning，API 用户通常无法访问 logits、hidden states、采样器或模型权重，因此很难在服务端内部控制 overthinking。

v0.1 的结论是：客户端可见的 streaming reasoning 文本已经包含足够的在线信号，可以用于在过度推理继续扩大前主动中断生成；结合第二阶段 final-answer recovery，可以在保持可用回答质量的同时减少 reasoning 长度和 total token 成本。

当前 v0.1 不是一个“无参数最优停止理论”。它是一个可复现的 Layer 1 工程与实验闭环：

- client-side streaming monitor
- visible-text detector
- interrupt
- two-phase final-answer recovery
- cost-aware evaluation

## 1. 研究问题

目标场景：

- 用户只能通过 LLM API 调用 reasoning model
- 可以接收 streaming reasoning text
- 不能访问模型内部状态
- 不能修改 decoding 或训练模型
- 希望减少 overthinking 造成的 token 成本和延迟

核心问题：

> 在黑盒 LLM API 场景下，仅使用客户端可见的 streaming reasoning 文本，能否在线识别低收益推理阶段并主动截断，同时保留最终答案质量？

这和 fixed budget 的差别是：

- fixed budget 预先猜测“这道题应该推理多长”
- monitor-and-interrupt 观察“模型是否已经进入重复、犹豫、低信息密度阶段”

## 2. 方法

### 2.1 两阶段流程

```text
Phase 1: streaming reasoning + monitor
  -> detector sees visible reasoning text
  -> detector emits stop decision
  -> client interrupts generation

Phase 2: final-answer recovery
  -> direct prompt with original question and compact reasoning summary
  -> API-specific config can disable thinking for recovery
  -> final answer is evaluated
```

直接中断 Phase 1 通常拿不到完整最终答案，因此 Phase 2 是必要系统组件。当前默认采用 `direct` recovery，而不是 prefill。prefill 容易让模型沿着 partial reasoning 继续推理并泄漏草稿；direct 模式更稳定。

### 2.2 Detector

v0.1 实现并比较了 5 类 detector：

| Detector | 用途 |
|---|---|
| `budget` | 固定 soft/hard budget，作为 baseline 和 aggressive policy |
| `compression` | 用压缩率变化捕捉重复、低信息密度和推理停滞 |
| `keyword` | 捕捉犹豫、反复确认、结论后继续推理等字面信号 |
| `ngram` | 捕捉 literal n-gram overlap |
| `semantic` | 用内容词 Jaccard 捕捉粗粒度重述 |

主线 v0.1 full run 聚焦 `budget`、`compression`、`keyword` 三类策略，因为它们是当前最稳定、最容易解释、最适合发布的 Layer 1 方法。

### 2.3 `@300` / `@1000` 的含义

`compression@1000`、`keyword@300` 里的数字是 safety budget / fallback hard-limit 相关配置，单位是 reasoning chars。对 signal detector 来说，真正停止点由 detector signal 决定；fallback budget 只是避免没有信号时无限推理。

这仍然是 v0.1 的局限：当前方法有阈值和 safety budget。BOCPD / change-point detector 已经作为 v0.2 候选做过小批量验证，但当前信号没有触发 posterior soft stop，因此后续应先走 offline replay gate，再探索 value/MDL-style 本地信号。

## 3. 实验设置

主线实验输入：

```text
experiments/results/full_budget.jsonl
experiments/results/full_compression.jsonl
experiments/results/full_keyword.jsonl
```

数据规模：

- math：100 题
- mmlu：100 题
- riddle：100 题
- detector：`budget` / `compression` / `keyword`
- budgets：`0,300,1000`
- treated rows：1800
- total rows：2700
- Phase 2：direct
- token schema：v3

报告生成：

```bash
uv run python experiments/report_full_main.py
```

生成目录：

```text
experiments/report/full_main_token/
```

## 4. 指标

| 指标 | 含义 |
|---|---|
| Quality | 最终答案质量，当前由 exact match / task evaluator 得出 |
| Baseline | 同一 detector 文件中 budget=0 的 no-stop baseline 质量 |
| ReasoningSavings | Phase 1 reasoning chars 相对 no-stop baseline 的减少比例 |
| TokenSavings | total tokens 相对 no-stop baseline 的减少比例 |
| Lost | baseline 正确但 early-stop 错误的样本数 |
| Fixed | baseline 错误但 early-stop 正确的样本数 |
| Phase2Fail | Phase 2 API/recovery 调用失败率，不等同于答案错误率 |

`ReasoningSavings` 解释机制，`TokenSavings` 才是成本指标。二者不能混用。

## 5. 主结果

| Config | Quality | Baseline | ReasoningSavings | TokenSavings | Lost | Phase2Fail |
|---|---:|---:|---:|---:|---:|---:|
| budget@300 | 88.6% | 97.3% | 77.4% | 42.1% | 31 | 0.0% |
| budget@1000 | 91.0% | 97.3% | 40.1% | 20.6% | 26 | 0.0% |
| compression@300 | 92.3% | 97.7% | 62.2% | 29.1% | 20 | 0.0% |
| compression@1000 | 93.5% | 97.7% | 28.4% | 19.0% | 18 | 0.0% |
| keyword@300 | 91.0% | 96.7% | 61.2% | 28.9% | 23 | 0.0% |
| keyword@1000 | 94.3% | 96.7% | 22.1% | 14.3% | 11 | 0.0% |

当前推荐：

| 角色 | 策略 | 解释 |
|---|---|---|
| Conservative | `keyword@1000` | 质量最高，token savings 较低 |
| Default | `compression@1000` | 当前全局默认；质量接近高位，同时有更好的 token savings |
| Balanced-aggressive | `compression@300` | 更高 token savings，接受更多质量损失 |
| Aggressive | `budget@300` | 最省 token，但质量损失过大，不适合作为默认 |

## 6. 主要发现

### Finding 1：no-stop baseline 是必要参照

早停策略不是无损优化。当前 no-stop baseline 平均质量约 97.2%，所有 early-stop policy 都在质量和成本之间做 tradeoff。因此报告中必须同时展示 baseline、Quality、Lost 和 TokenSavings。

### Finding 2：reasoning savings 不能直接等价为 total-token savings

`compression@300` 的 ReasoningSavings 是 62.2%，但 TokenSavings 是 29.1%。差距来自：

- Phase 2 recovery tokens
- prompt tokens
- interrupted Phase 1 usage 缺失时的估算误差

这也是 v0.1 的 cost-aware evaluation contribution：不只报告 reasoning 长度减少，而是尽量报告 total token 成本。

### Finding 3：fixed budget 不是坏 baseline，而是 aggressive policy

`budget@300` 的 TokenSavings 最高，达到 42.1%，但 Quality 只有 88.6%，Lost=31。它在 Pareto frontier 上是一个 aggressive point，而不是应该被轻易否定的 baseline。

monitor-and-interrupt 的价值在于提供更高质量区间的折中点，例如 `compression@1000` 和 `compression@300`。

### Finding 4：默认策略应由 Quality、Lost 和 TokenSavings 共同决定

`keyword@1000` 的 Quality 是 94.3%，高于 `compression@1000` 的 93.5%，但 TokenSavings 只有 14.3%。`compression@1000` 的 TokenSavings 是 19.0%，Lost=18。考虑到当前实验规模和 token 估算误差，0.8pp Quality 差异不应过度解释；默认策略选择 `compression@1000` 是质量和成本的折中，而不是理论最优声明。

### Finding 5：数据集差异支持后续 router，但 router 还不是 v0.1 已验证产品

第一版 per-dataset 推荐：

| Dataset | Recommended | Quality | TokenSavings |
|---|---|---:|---:|
| math | `compression@300` | 93.0% | 24.9% |
| mmlu | `keyword@300` | 96.0% | 32.4% |
| riddle | `keyword@1000` | 94.0% | 12.7% |

这说明策略选择与任务类型有关。但 v0.1 还没有实现输入分类 router，因此不能 claim “自动 task routing 已完成”。当前只应把它作为 v0.2/v0.3 之后的工程方向。

## 7. 可视化证据

`experiments/report_full_main.py` 当前生成 12 张证据图：

1. `quality_gap_from_baseline.png`
2. `savings_decomposition.png`
3. `token_breakdown.png`
4. `overall_tradeoff.png`
5. `strategy_map.png`
6. `dataset_tradeoffs.png`
7. `dataset_decision_matrix.png`
8. `outcome_flow.png`
9. `loss_vs_savings.png`
10. `detector_profiles.png`
11. `token_estimate_calibration.png`
12. `latency_change.png`

这些图按证据链组织：先建立 no-stop baseline，再解释 savings 来源，再展示全局 Pareto 决策，最后说明数据集差异、质量损失来源和 token 估算不确定性。

## 8. Token 统计限制

当前 token 统计不是 billing-grade 全量 API 实测：

- 400/1800 treated rows 使用 API `total_tokens`
- 1400/1800 treated rows 使用 `estimated_total_tokens`

原因是 early-stop 会主动关闭 Phase 1 stream，而 provider 通常只在 stream 自然结束时返回最终 usage chunk。Phase 2 usage 基本可用。

在有 Phase 1 API usage 可校准的样本上，`estimated_phase1_total_tokens` mean absolute error 约 14.4%。因此：

- 可以说 total-token savings 方向成立
- 不应精确比较 0.x pp 的 TokenSavings 差异
- 后续需要 calibration run 或 provider-specific tokenizer

## 9. 与相关工作的定位

v0.1 的 niche 是：

```text
client-side + black-box LLM API + visible reasoning text
```

区别于：

- 内部信号方法：依赖 hidden states、logits、entropy 或训练
- local decoding 方法：要求本地模型或可修改 sampler
- proxy-model 方法：需要额外模型或探测过程
- fixed budget：不观察推理动态，只预设长度上限

EAT 等 black-box prior work 是重要近邻；当前项目的差异在于使用 literal text signals，不需要 proxy model，并且把 interrupt 后的 direct recovery 与 cost-aware evaluation 放在同一个实验闭环里。

## 10. 当前不应过度声明

不要 claim：

- `compression@1000` 是理论最优
- total-token savings 是完全 API billing 实测
- task router 已实现
- BOCPD 已验证
- detector 之间 0.x pp TokenSavings 差异有统计显著性
- v0.1 解决了所有 overthinking 场景

可以 claim：

- streaming reasoning text 包含可用于 client-side interrupt 的在线信号
- two-phase direct recovery 能把中断变成可用最终答案
- 当前 full run 支持在质量损失可控的情况下减少 reasoning length 和 total tokens
- cost-aware evaluation 比只报告 reasoning savings 更诚实

## 11. 下一步

v0.1 release closeout：

1. README public-facing polish
2. release note / blog outline
3. tag v0.1 或创建 GitHub release

v0.2：

1. 使用 offline replay 作为新 detector / 新信号 gate
2. 记录 BOCPD negative result，不把它作为当前 mainline contribution
3. 在不增加 LLM / embedding 调用的约束下，探索 value/MDL-style 本地文本信号
4. 只有离线 replay 通过后，才进入 20 题 API probe

v0.3：

1. token calibration
2. cross-vendor sanity check
3. latency-controlled report
