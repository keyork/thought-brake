# BOCPD / Change-point Detector Design

日期：2026-05-25

状态：v0.2 设计草案，尚未实现

## 1. 目标

v0.1 已经证明 Layer 1 literal text detectors 可以支撑 client-side monitor-and-interrupt：

- `compression@1000` 作为默认策略
- `keyword@1000` 作为保守策略
- `compression@300` 作为 balanced-aggressive 策略
- `budget@300` 作为 aggressive 策略

但 v0.1 仍有明显 magic threshold：

- `@300` / `@1000` fallback budget
- compression CRD / LZ threshold
- keyword density threshold
- consecutive windows

v0.2 的目标不是继续堆 detector，而是把“何时进入低收益推理阶段”表述成在线变化点检测问题：

> reasoning stream 的统计特征从“信息推进阶段”切换到“重复 / 犹豫 / 低信息密度阶段”时，客户端应提升 stop probability，并在 posterior 足够高时中断。

BOCPD 的价值是：把停止决策从单个手工阈值，推进为对“状态切换”的在线 posterior 估计。

## 2. 非目标

v0.2 不追求一次性解决所有问题：

- 不访问 logits / hidden states / sampler
- 不使用 embedding API
- 不训练 classifier
- 不依赖 proxy model
- 不替代 Phase 2 recovery
- 不承诺完全消除所有参数

BOCPD 仍然需要 prior、hazard、stop threshold 等配置。它的改进点是让这些参数具备统计含义，而不是散落在多个 heuristic threshold 里。

## 3. 直觉

一个正常 reasoning 过程通常有阶段性：

```text
理解题目 -> 建立约束 -> 推导/枚举 -> 得到候选答案 -> 检查/确认 -> 输出
```

overthinking 常见于后半段：

```text
得到答案 -> 继续反复确认 -> 换角度重述 -> 犹豫 -> 再次确认 -> 继续重述
```

如果把 reasoning stream 切成固定窗口，窗口特征会出现变化：

- 新信息比例下降
- 压缩率下降
- LZ factor rate 下降
- n-gram overlap 上升
- hedge / reconsider keywords 上升
- conclusion-after-thinking signal 出现

BOCPD 要检测的不是“某个特征超过阈值”，而是“当前窗口更像一个新阶段的开始，并且这个新阶段更像低收益推理”。

## 4. 输入信号

BOCPD detector 仍然使用 visible reasoning text。推荐第一版使用低维、可解释、无需外部依赖的窗口特征。

### 4.1 Windowing

建议参数：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `bocpd_window_chars` | 200 | 每个观测窗口的字符数 |
| `bocpd_min_windows` | 4 | 最少 warmup 窗口数 |
| `bocpd_max_run_length` | 64 | posterior 截断上限 |
| `hard_limit` | 1000 或 3000 | 安全兜底，仍保留 |

实现方式：

- `update(piece, total_chars)` 继续 append buffer
- 每累积一个完整 `bocpd_window_chars` 生成一个 observation
- 未满窗口时不更新 BOCPD，只返回 `should_stop=False`

### 4.2 Feature Vector

第一版建议 5 个特征：

| 特征 | 方向 | 解释 |
|---|---|---|
| `compression_ratio` | 越低越重复 | gzip compressed bytes / raw bytes |
| `lz_factor_rate` | 越低越重复 | LZ factor count / chars |
| `ngram_overlap` | 越高越重复 | recent n-grams 与历史重合 |
| `keyword_density` | 越高越犹豫 | hedge / reconsider keywords per char |
| `conclusion_seen` | bool | 是否已经出现答案/结论模式 |

为了方便建模，转换成“low-value score”方向一致的向量：

```text
x_t = [
  1 - normalized_compression_ratio,
  1 - normalized_lz_factor_rate,
  ngram_overlap,
  keyword_density,
  conclusion_seen
]
```

第一版可以进一步压缩成单变量：

```text
z_t = weighted_sum(x_t)
```

推荐先做单变量 BOCPD，原因：

- 实现简单
- 参数少
- 可解释性更好
- 样本量目前不够支持复杂多变量协方差估计

后续再考虑 diagonal Gaussian 多变量模型。

## 5. BOCPD 模型

### 5.1 状态

定义 run length：

```text
r_t = 自上一个 change point 以来经过的窗口数
```

BOCPD 维护：

```text
P(r_t | z_1:t)
```

每来一个新窗口，更新 run-length posterior。

### 5.2 Predictive Model

第一版用 Normal-Inverse-Gamma 的简化形式会偏重数学实现。为了快速落地，建议先用 Gaussian unknown mean + fixed variance 的在线版本：

```text
z_t | segment_mean ~ Normal(mu, sigma^2)
mu ~ Normal(mu0, tau0^2)
```

每个 run length 维护：

- `n`
- `mean`
- `variance` 或固定 `sigma`

预测概率：

```text
p(z_t | r_{t-1}) = NormalPDF(z_t; predictive_mean_r, predictive_var_r)
```

实现时使用 log probability，避免 underflow。

### 5.3 Hazard Function

hazard 表示每个窗口发生 change point 的 prior probability：

```text
H(r) = 1 / lambda
```

其中 `lambda` 是期望阶段长度，单位是窗口数。

推荐默认：

```text
bocpd_hazard_lambda = 12
```

如果 `window_chars=200`，则期望阶段长度约 2400 chars。这个数不是停止阈值，而是 prior：我们预期 reasoning 的统计阶段大约每 2400 chars 可能切换一次。

### 5.4 Recursive Update

对每个旧 run length `r`：

增长项：

```text
P(r_t = r + 1) += P(r_{t-1}=r) * (1 - H(r)) * p(z_t | r)
```

变化点项：

```text
P(r_t = 0) += P(r_{t-1}=r) * H(r) * p(z_t | r)
```

归一化：

```text
P(r_t | z_1:t) = P(r_t, z_1:t) / sum_r P(r_t, z_1:t)
```

实现建议：

- 全部用 log-space
- 每步只保留 top `bocpd_max_run_length`
- 若数值复杂，第一版可先用普通 float + normalize，因为 run length 很短

## 6. Stop Rule

BOCPD 只告诉我们“发生了变化点”，但不是所有变化点都应该停止。比如从理解题目切到推导阶段不是 overthinking。

因此 stop rule 应该组合两个条件：

```text
change_posterior high
AND current low-value score high
AND minimum useful reasoning reached
```

建议：

```text
change_prob = P(r_t <= bocpd_recent_run_threshold)
low_value_score = z_t

stop if:
  total_chars >= soft_budget
  and conclusion_seen
  and change_prob >= bocpd_stop_prob
  and low_value_score >= bocpd_low_value_threshold
```

默认配置草案：

| 参数 | 默认值 | 含义 |
|---|---:|---|
| `bocpd_recent_run_threshold` | 1 | `r_t <= 1` 视为刚发生变化 |
| `bocpd_stop_prob` | 0.65 | change posterior 停止阈值 |
| `bocpd_low_value_threshold` | 0.55 | 当前窗口低价值分数阈值 |
| `soft_budget` | 300 或 1000 | 最早允许 soft stop 的字符数 |
| `hard_limit` | 1000 或 3000 | 安全兜底 |

注意：这里仍有阈值，但语义比 v0.1 更清晰：

- `hazard_lambda`：先验阶段长度
- `stop_prob`：后验置信度
- `low_value_threshold`：低收益阶段强度

## 7. 与现有 detector 的关系

### 7.1 DetectorName

需要扩展：

```python
DetectorName = Literal[
    "none",
    "budget",
    "compression",
    "ngram",
    "keyword",
    "semantic",
    "bocpd",
]
```

### 7.2 Config

在 `EarlyStopConfig` 中新增：

```python
bocpd_window_chars: int = 200
bocpd_min_windows: int = 4
bocpd_max_run_length: int = 64
bocpd_hazard_lambda: float = 12.0
bocpd_stop_prob: float = 0.65
bocpd_recent_run_threshold: int = 1
bocpd_low_value_threshold: float = 0.55
bocpd_compression_weight: float = 0.30
bocpd_lz_weight: float = 0.25
bocpd_ngram_weight: float = 0.20
bocpd_keyword_weight: float = 0.15
bocpd_conclusion_weight: float = 0.10
```

并加入环境变量：

```text
THOUGHT_BRAKE_BOCPD_WINDOW_CHARS
THOUGHT_BRAKE_BOCPD_HAZARD_LAMBDA
THOUGHT_BRAKE_BOCPD_STOP_PROB
...
```

### 7.3 Class

新增：

```python
class BOCPDDetector:
    name: DetectorName = "bocpd"

    def __init__(self, config: EarlyStopConfig) -> None:
        ...

    def update(self, piece: str, total_chars: int) -> StopDecision:
        ...
```

返回 detail：

```text
bocpd p_change=0.72 z=0.61 r_map=0
```

这对实验报告很重要，后续可以画 trigger distribution。

## 8. 实现计划

### Step 1：Feature Extraction

先实现纯函数：

```python
def bocpd_features(text: str, history: str, conclusion_seen: bool) -> BOCPDFeatures:
    ...
```

测试：

- repetitive text 的 low-value score 更高
- diverse text 的 low-value score 更低
- conclusion pattern 能被记录

### Step 2：BOCPD Core

实现一个独立小类，不依赖 detector：

```python
class OnlineChangePoint:
    def update(self, value: float) -> ChangePointState:
        ...
```

测试：

- 稳定序列不会频繁触发 high change probability
- 均值突变序列会提升 `p_change`
- posterior 每步归一化
- max run length 生效

### Step 3：Detector Wrapper

实现 `BOCPDDetector`，接入：

- hard limit
- warmup
- conclusion_seen
- stop rule
- detail string

### Step 4：Runner Support

确保：

```bash
uv run python experiments/runner.py \
  --dataset all \
  --n 20 \
  --detector bocpd \
  --budgets 300,1000 \
  --workers 10 \
  --track-usage
```

可以直接运行。

### Step 5：Report Support

`report_full_main.py` 当前聚焦 full budget/compression/keyword 文件。BOCPD 初期建议新建 focused report：

```text
experiments/report_bocpd_probe.py
```

先不要污染 v0.1 main report。

## 9. 实验计划

### Probe Run

第一轮小批量：

```bash
uv run python experiments/runner.py \
  --dataset all \
  --n 20 \
  --budgets 300,1000 \
  --detector bocpd \
  --workers 10 \
  --track-usage \
  --output experiments/results/bocpd_probe.jsonl
```

目标不是赢，而是看：

- 是否稳定触发
- 是否过早触发
- 是否几乎不触发
- Phase 2 是否仍稳定
- detail 中 `p_change` 是否可解释

### Main Comparison

如果 probe 稳定，再跑：

| 策略 | 目的 |
|---|---|
| `compression@1000` | v0.1 default |
| `compression@300` | balanced-aggressive |
| `keyword@1000` | conservative |
| `budget@300` | aggressive |
| `bocpd@300` | BOCPD aggressive-ish |
| `bocpd@1000` | BOCPD default-ish |

指标：

- Quality
- ReasoningSavings
- TokenSavings
- Lost
- Fixed
- truncation rate
- trigger position distribution
- p_change distribution at stop

## 10. 判断标准

BOCPD 值得进入 v0.2 mainline 的条件：

1. Quality 不明显低于 `compression@1000`
2. TokenSavings 高于或接近 `compression@1000`
3. Lost 不明显恶化
4. trigger detail 可解释
5. 参数语义比 compression/keyword 更清晰

即使 BOCPD 不赢，也有价值：

- 如果它过早触发，说明 low-value score 或 hazard 太激进
- 如果它不触发，说明 feature shift 不足以支撑变化点建模
- 如果只在某些数据集好，说明它适合作为 router candidate

## 11. 风险

### 风险 1：BOCPD 仍然有参数

这不是失败。目标是减少“无法解释的 magic thresholds”，而不是让系统完全无参数。

### 风险 2：窗口太粗，错过短 overthinking

应通过 `bocpd_window_chars` 做 probe。不要一开始把窗口调太小，否则噪声会很大。

### 风险 3：低价值分数手工权重仍然主观

第一版用固定权重是为了可解释。后续可以从 v0.1 full run 中拟合简单权重，但不要在 v0.2 初版引入训练依赖。

### 风险 4：变化点不等于应该停止

因此必须保留 `conclusion_seen` 和 `soft_budget` 条件。没有 conclusion 的早期阶段变化不应触发 stop。

## 12. v0.2 最小交付物

代码：

- `BOCPDDetector`
- `OnlineChangePoint`
- `bocpd_features`
- config/env support
- tests

实验：

- 20-50 题 probe
- 与 v0.1 default 对比
- trigger/failure examples

文档：

- 更新 README detector table
- 更新 `docs/report_v0_1.md` 的 future work 或新增 `docs/report_v0_2.md`
- 写 BOCPD 实验结论，不论成败
