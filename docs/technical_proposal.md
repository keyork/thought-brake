# Reasoning Model 推理早停技术方案

## 1. 背景与问题

### 1.1 问题现象

调用 reasoning model 时，对于简单问题模型会产生过度推理（overthinking），表现为：

- 反复自我怀疑、横跳验证
- 沉没成本式继续推理（已得出答案但继续验证）
- 元推理过载（思考"这是不是陷阱"、"用户是不是记错了"）
- 重复模式循环

### 1.2 实测案例

输入：脑筋急转弯（盲人买剪刀 → 聋哑人买锤子）

| 指标 | 数值 |
|---|---|
| Prompt tokens | 41 |
| Reasoning tokens | 1347 |
| Completion tokens | 1540 |
| Reasoning/Prompt ratio | 32.8x |

模型在推理第 200 token 处已得出正确答案，剩余 1100+ tokens 为无效自我验证。

### 1.3 业务影响

- **成本**：reasoning tokens 按 completion 计费，浪费 70%+ token 预算
- **延迟**：P50 / P99 显著增加，影响用户体验
- **质量**：过度思考反而可能从正确答案偏移到错误答案

---

## 2. 核心洞察

**Reasoning 是 autoregressive sampling 过程，每一步 token 生成都依赖前缀上下文。这意味着：**

1. 在任意 token 边界中断 sampling 是合法操作（不破坏模型状态）
2. 可以构造任意合法的 prefix 让模型从该位置继续生成
3. "thinking 结束"本质上是一个特殊 token（`</think>`）或字段切换信号，可由客户端注入

因此**外部强制终止 thinking 阶段是数学上无损的操作**，不需要模型支持任何特殊接口。

---

## 3. 技术方案

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Wrapper                          │
│                                                              │
│  ┌──────────────┐   超出预算   ┌──────────────────────┐    │
│  │ Phase 1:     │─────────────>│ Phase 2:             │    │
│  │ Stream +     │              │ Prefill Continuation │    │
│  │ Monitor      │              │                      │    │
│  └──────┬───────┘              └──────────┬───────────┘    │
│         │ 模型自然结束                     │                 │
│         ↓                                  ↓                 │
│         └──────────────────────────────────┘                │
│                          │                                   │
│                          ↓                                   │
│                  ┌───────────────┐                          │
│                  │ Final Answer  │                          │
│                  └───────────────┘                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          ↓
                    ┌──────────┐
                    │ LLM API  │
                    └──────────┘
```

### 3.2 Phase 1：Stream 监控 + 软截断

**目标**：在不破坏推理连贯性的前提下，识别并切断过度推理。

**算法**：

```
input: messages, soft_budget, hard_limit
state: reasoning_chars = 0, buf = []

for chunk in stream(messages):
    if chunk.delta.content:           # 模型已结束 thinking
        return COMPLETE, buf
    
    if chunk.delta.reasoning_content:
        buf.append(piece)
        reasoning_chars += len(piece)
        
        if reasoning_chars >= hard_limit:
            return TRUNCATED, buf      # 硬截断
        
        if reasoning_chars >= soft_budget and ends_with_punctuation(piece):
            return TRUNCATED, buf      # 软截断:句末才断
```

**关键设计**：

1. **软截断 vs 硬截断**：
   - 软截断：超过 `soft_budget` 后等到句末标点（`。！？\n`）才断，保证语义完整
   - 硬截断：超过 `hard_limit` 强制断，防止模型陷入无限循环

2. **预算选择**（经验值）：

   | 问题类型 | soft_budget | hard_limit |
   |---|---|---|
   | 闲聊/常识 | 200 | 400 |
   | 一般问答 | 500 | 1000 |
   | 数学/代码 | 1500 | 3000 |
   | 复杂推理 | 3000 | 6000 |

3. **Token 计数近似**：用字符数代替 tokenizer，中文场景误差 <10%，省一次依赖。

### 3.3 Phase 2：引导模型输出最终答案

**目标**：从被截断的位置让模型收束并产出最终答案。

支持两种模式：`direct`（默认，推荐）和 `prefill`。

#### 3.3.1 Direct 模式（推荐）

**核心思路**：截断后保留 system/developer 控制消息，再发一条全新的 user 消息，附带原始对话、当前问题和推理摘要；如果 LLM API 支持禁用推理，则通过配置化 `phase2_extra_body` 传入对应参数，直接生成最终答案。

**Direct 构造**：

```
messages = [
    {"role": "user", "content":
        "以下是原始对话、当前问题和已完成的思考过程。请直接给出最终答案，不需要再分析。\n\n"
        "原始对话：\n<conversation>\n\n"
        "问题：<当前问题>\n\n"
        "已完成的思考：\n<reasoning_excerpt>\n\n"
        "最终答案："
    }
]
extra_body = <由 THOUGHT_BRAKE_PHASE2_EXTRA_BODY 配置>
```

**推理摘要策略**：

- 默认取 reasoning 前 150 字符（head-only）
- 仅当 reasoning 超过 400 字符时，才额外附加尾部 200 字符（head + "……" + tail）
- 短推理下 head+tail 重叠会引入 "……" 分隔符噪音，降低质量

**为什么 Direct 优于 Prefill**：

| 维度 | Prefill | Direct |
|---|---|---|
| 推理泄漏 | 严重 — 模型不尊重注入的结束标签，从 prefill 继续推理 | 低 — 取决于 LLM API 是否支持禁用推理 |
| 元评论 | 模型经常输出"我来分析一下"、"首先..."等元推理 | 极少 — 模型直接回答 |
| 延迟 | 高（模型仍在 think） | 低（跳过 thinking 阶段） |
| API 依赖 | 需要支持 assistant prefix continuation | 最好支持禁用推理参数；字段名由配置决定 |

**实验数据（riddles 数据集，20 题，budget sweep）**：

| Budget | Easy 质量 | Medium 质量 | Hard 质量 | 整体保持率 |
|---|---|---|---|---|
| 100 | 61.1% | 92.9% | 100% | 81.2% |
| 200 | 77.8% | 78.6% | 75.0% | 77.3% |
| **300** | **88.9%** | **100%** | **100%** | **95.0%** |
| 500 | 83.3% | 100% | 87.5% | 90.6% |

**Budget=300 + direct 模式** 是最佳组合，达到 95% 整体质量保持率。

**API 兼容性**：

不同 LLM API 对“禁用推理”的字段约定不同。项目不在代码中绑定具体服务商，统一通过 `THOUGHT_BRAKE_PHASE2_EXTRA_BODY` 或 `EarlyStopConfig(phase2_extra_body=...)` 配置。若 API 不支持禁用推理，可把该配置置空并使用 `prefill` 模式。

#### 3.3.2 Prefill 模式

**机制**：多数 LLM API 的 Chat Completions 形态支持"messages 列表最后一条是 assistant 消息时，模型从该消息续写"。这就是 prefill。

**Prefill 构造**：

```
messages = [
    {"role": "user", "content": <原始问题>},
    {"role": "assistant", "content": 
        "<think>\n"
        + <partial_reasoning>           # Phase 1 已生成的推理
        + "\n\n好，已经想清楚了，直接给出最终答案。\n"
        + "\n</think>\n\n"             # 显式关闭 think
    }
]
```

**已知问题**：prefill 模式在实践中表现不佳。模型可能不尊重注入的结束标签，经常从 prefill 内容继续推理而非产出最终答案。仅在 Direct 模式不可用时使用。

**Hint 文本设计**（影响很大，按效果排序）：

| Hint | 效果 | 说明 |
|---|---|---|
| `\n\n好,已经想清楚了,直接给出答案。` | ★★★★★ | 在 think 内部自然收束 |
| `\n\n综上,` | ★★★★ | 强引导结论 |
| `</think>\n\n` 直接关闭 | ★★ | 部分模型会重开 thinking |
| 不加 hint 直接关闭 | ★ | 容易续写出无关内容 |

### 3.4 数据流细节

**LLM API 的字段约定**：

```json
// Thinking 阶段
{"choices": [{"delta": {"reasoning_content": "...", "content": null}}]}

// Answer 阶段  
{"choices": [{"delta": {"reasoning_content": null, "content": "..."}}]}
```

**容错处理**：

- Phase 2 续写时，如果模型仍在 `reasoning_content` 字段输出，将其也视为答案的一部分（说明模型未识别我们注入的 `</think>`）
- 网络中断/JSON 解析失败：跳过该 chunk，不中断流程
- Phase 2 失败时 fallback 到 Phase 1 已收集的 content（若有）

---

## 4. 工程实现要点

### 4.1 异常处理矩阵

| 异常场景 | 处理策略 |
|---|---|
| Phase 1 网络中断 | 用已收集 reasoning 走 Phase 2 |
| Phase 2 API 不支持 prefill | 退化为标准调用 + 简短 system prompt |
| 模型在 Phase 2 仍输出 reasoning | 当作答案接收，记录指标告警 |
| 软截断点迟迟不出现 | hard_limit 兜底 |
| JSON 解析失败 | skip chunk，继续 |

### 4.2 Phase 2 fallback 链

不同模型/API 的 Phase 2 支持程度不同，按优先级尝试：

1. **Direct 模式**（默认）：构造 user 消息 + 配置化禁用推理参数，效果最好
2. **Prefill 模式**：assistant 前缀续写，适用于不支持禁用推理的 LLM API
3. **Fallback**：把 partial_reasoning 拼到 user message 末尾，要求模型基于此总结

### 4.3 监控指标

必须采集的指标：

```
- reasoning_chars_p50 / p99
- truncation_rate            # 触发截断的请求占比
- soft_stop_rate             # 软截断 / 总截断
- hard_stop_rate             # 硬截断 / 总截断  
- phase2_failure_rate        # 续写失败率
- answer_quality_delta       # vs baseline 的质量变化(需评测集)
- latency_p50 / p99
- cost_per_request
```

### 4.4 配置化

预算应支持运行时配置，不应硬编码：

```python
@dataclass
class EarlyStopConfig:
    detector: Literal["none", "budget", "compression", "ngram", "keyword", "semantic"] = "budget"
    soft_budget: int = 300
    hard_limit: int = 600
    phase2_mode: Literal["prefill", "direct"] = "direct"
    phase2_disable_thinking: bool = True
    phase2_extra_body: dict[str, Any] | None = {"enable_thinking": False}
    phase2_direct_template: str = "..."    # direct 模式 prompt 模板
    phase2_direct_conversation_chars: int = 1200
    phase2_direct_head_chars: int = 150    # 推理摘要头部字符数
    phase2_direct_tail_chars: int = 200    # 推理摘要尾部字符数（仅长推理时使用）
    compression_baseline_chars: int = 200
    compression_recent_chars: int = 200
    compression_theta_crd: float = 0.7
    compression_theta_lz: float = 0.5
    compression_consecutive_windows: int = 2
    finalize_hint: str = "\n\n好，已经想清楚了，直接给出最终答案。"
    sentence_end_pattern: str = r"[。！？!?\n]"
    enable: bool = True              # 总开关,便于 A/B 测试
    fallback_on_phase2_fail: bool = True
```

`detector="none"` 用于 baseline：仍然走 stream monitor 并记录真实 reasoning 长度，但不触发早停。

---

## 5. 进阶优化方向

### 5.1 动态预算（问题分类路由）

不同问题用不同预算。在 Phase 1 前用一个小模型/规则分类：

```
classify(question) → {chat, simple_qa, math, code, complex_reasoning}
                  → 查表得到 (soft_budget, hard_limit)
```

成本：分类模型 ~50ms + ~20 tokens；收益：典型场景节省 60% reasoning tokens。

### 5.2 重复模式检测（更激进）

实时检测推理流中的重复模式：

```
- 滑动窗口 n-gram 重复率 > 阈值 → 立即截断
- 关键短语复现("等等"、"让我再想想"出现第 N 次) → 截断
- Embedding 相似度: 当前段与历史段 cosine > 0.85 → 截断
```

对前文案例特别有效——模型在"标准版本 vs 用户版本"间循环 4 次，检测到第 2 次循环即可截断，节省 70%+ tokens。

### 5.3 自适应预算

基于历史数据反馈调整：

```
if 截断后答案质量 < baseline:
    soft_budget *= 1.2
elif 截断后质量 ≥ baseline 且 truncation_rate > 80%:
    soft_budget *= 0.9
```

类似 TCP 拥塞控制思路，找到 cost/quality 的甜点。

### 5.4 与非推理模型路由结合

最激进的方案：分类器判断问题是否真的需要推理：

```
needs_reasoning(question):
    yes → reasoning model + early stop
    no  → non-reasoning model 直接答
```

对前文脑筋急转弯这类问题，non-reasoning 模式往往答得更好更快。

---

## 6. 风险与权衡

### 6.1 已知风险

| 风险 | 严重度 | 缓解措施 |
|---|---|---|
| 复杂推理被错误截断导致答错 | 高 | 分类路由 + 高预算 / 关闭早停 |
| Direct 模式模型不支持禁用推理参数 | 中 | 配置为空并改用 prefill 模式 |
| Prefill 注入被模型识破，输出元评论 | 中 | 优先使用 direct 模式；prefill 优化 hint 文本 |
| 不同模型版本对 prefill 行为不同 | 中 | 模型升级时回归测试 |
| 流式中断后服务端仍计费 | 低 | 与 LLM API 服务方确认计费规则 |

### 6.2 不适用场景

- **Agent 工作流**：每步推理可能依赖前序完整推理，截断风险高
- **数学证明、定理推导**：完整推理链不可截断（注：GSM8K 常规计算题仍然适合早停）
- **多步代码调试**：reasoning 即是解题过程本身
- **LLM API 按 max_tokens 而非实际生成计费**：早停无成本收益

### 6.3 与 prompt 工程对比

| 维度 | 早停方案 | Prompt 工程 |
|---|---|---|
| 改造成本 | 中（需 wrapper） | 低（改 prompt） |
| 效果上限 | 高（强制约束） | 中（依赖模型听话） |
| 通用性 | 高（与 prompt 无关） | 低（每个场景调） |
| 可观测性 | 高（明确指标） | 低 |
| 风险 | 中 | 低 |

**建议组合使用**：prompt 工程降低 baseline，早停作为兜底。

---

## 7. 落地路线

### Phase A：可靠基线

- baseline 使用 `detector="none"`，记录真实 reasoning 长度
- 结果 schema 化，记录 `detector`、`phase2_used`、`phase2_failed`、`answer_chars`
- 修正 exact-match 数字评测边界问题
- analysis 自动过滤最新 schema，避免旧实验污染

### Phase B：Phase 2 输出收敛

- 优化 prefill/fallback prompt，减少推理草稿泄漏
- 监控 `answer_chars` 和 answer length ratio
- 对不同 LLM API 形态保留配置化 fallback

### Phase C：Detector 对照实验 ✅

- 5 种 detector：budget / compression / ngram / keyword / semantic
- 跨 3 个数据集验证：riddles / GSM8K / MMLU
- 核心结论：没有万能最优 detector，任务类型决定策略
  - Riddles（推理短且均匀）→ budget@300 (95%)
  - GSM8K（数学计算，中等长度）→ compression@1000 (90%)
  - MMLU（推理长度 615-63806，分布极宽）→ compression@300 (100%)

### Phase D：任务自适应路由（规划中）

- 基于问题特征自动选择 detector + budget
- Hybrid detector：signal guard + budget fallback
- Answer oscillation 检测（多选题）
- BOCPD / embedding 研究增强

---

## 8. 验收标准

| 指标 | Baseline | 目标 |
|---|---|---|
| 简单问题 reasoning tokens P50 | 1000+ | <400 |
| 简单问题端到端延迟 P50 | - | 降低 50%+ |
| 答案质量（人工评测） | 100% | ≥95% |
| 截断率 | 0 | 40-70% |
| Phase 2 失败率 | - | <1% |
