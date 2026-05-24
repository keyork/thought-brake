# 未来工作方向：从可靠基线到压缩信号早停

本文档记录 `thought-brake` 后续研发路线。当前判断是：**客户端黑盒 LLM API 的流式早停方向合理，但必须先把实验基线和 Phase 2 输出质量做扎实，再推进压缩信号和 BOCPD/MDL。**

## 1. 当前定位

`thought-brake` 面向只能访问 LLM API 流式文本输出的用户：

- 不能访问 hidden states / logits
- 不能修改采样器
- 不能训练或微调模型
- 需要在客户端侧低成本部署
- 希望在 reasoning model 过度推理时提前停止

因此，我们的价值不在于打败内部信号方法，而在于填补这个部署场景：

> 黑盒 LLM API + 客户端流式监控 + 可复用早停 detector。

## 2. 已有实验暴露的问题

第一轮 hard-budget 实验给了两个重要结论：

1. GSM8K 上小 budget 并不一定伤害质量，`budget=200` 的离线修正评估结果最好。
2. 现有结果还不能证明 token savings，因为旧 baseline 使用 passthrough，`reasoning_chars=0`，没有记录真实 baseline reasoning 长度。
3. Phase 2 输出经常泄漏推理草稿，answer 长度达到 baseline 的 3-7 倍，这会影响体验，也可能抵消收益。

所以后续不能直接跳到 BOCPD。正确顺序是：

1. 可靠测量 baseline reasoning
2. 修 Phase 2 final-answer-only 行为
3. 建 detector 插件体系
4. 再比较 hard budget / compression / BOCPD

## 3. 代码架构方向

当前代码已经开始按以下模块拆分：

| 模块 | 责任 |
|---|---|
| `src/thought_brake/client.py` | 对外 client、Phase 1/Phase 2 编排 |
| `src/thought_brake/_monitor.py` | 流式读取 reasoning/content |
| `src/thought_brake/detectors.py` | 可插拔早停 detector |
| `src/thought_brake/_prefill.py` | Phase 2 prefill 构造和收集 |
| `experiments/runner.py` | 并发实验、resume、schema 化 JSONL |
| `experiments/analysis.py` | 结果分析，自动过滤最新 schema |

detector 接口是后续复用的关键：

```python
class ReasoningDetector(Protocol):
    name: DetectorName

    def update(self, piece: str, total_chars: int) -> StopDecision:
        ...
```

已支持：

- `none`：baseline 监控，不截断
- `budget`：soft/hard budget
- `compression`：Layer 1 压缩信号原型

## 4. Roadmap

### Phase A：可靠实验基线

目标：让结果文件能真实回答“省了多少 reasoning”。

工作项：

- baseline 使用 `detector="none"` 走 stream monitor，不再 passthrough
- 结果写入 `schema_version`
- 结果记录 `detector`、`phase2_used`、`phase2_failed`、`answer_chars`
- analysis 只分析最新 schema，避免旧结果污染
- 修 exact-match 评测对 `70,000` 这类数字的误判

验收：

- baseline 记录真实 `reasoning_chars`
- 每个 `(question_id, budget, detector)` 可 resume
- 能计算 `avg_savings_rate`

### Phase B：Phase 2 输出收敛 ✅ 已完成

目标：早停后只输出最终答案，不泄漏 reasoning 草稿。

**已完成方案**：

实现了 `direct` 模式替代原有 `prefill` 模式：

1. 截断后发一条新 user 消息（含原始问题 + reasoning 摘要），不再用 assistant prefill
2. 通过 `enable_thinking=False` 从根本上禁用 Phase 2 推理阶段
3. 推理摘要使用 head-only 策略（短推理时避免 head+tail 重叠噪音）
4. `clean_final_answer()` 后处理去除元推理包装和尾部重复

**实验验证**（riddles 数据集，direct 模式）：

| Budget | Easy | Medium | Hard | 整体 |
|---|---|---|---|---|
| 100 | 61.1% | 92.9% | 100% | 81.2% |
| 200 | 77.8% | 78.6% | 75.0% | 77.3% |
| **300** | **88.9%** | **100%** | **100%** | **95.0%** |
| 500 | 83.3% | 100% | 87.5% | 90.6% |

**关键发现**：

- Prefill 模式在实践中失败：模型不尊重注入的 `boxed` 标签，继续推理
- `enable_thinking=False` 是关键突破 — 彻底消除 reasoning 泄漏
- Head+tail 摘要在短推理（<400 chars）时有害，"……" 分隔符引入噪音
- Budget=300 是 riddles 数据集的最优值（95% 质量保持率）

验收：

- ✅ `answer_chars` 不显著高于 baseline（direct 模式下 < 0.4x baseline）
- ✅ LLM judge 质量保持率 95%（budget=300）
- ✅ `phase2_failed` 维持低位

### Phase C：Detector 对照实验

目标：把 detector 从实现细节变成可实验变量。

对照组：

| detector | 含义 |
|---|---|
| `none` | baseline stream monitor |
| `budget` | 当前 hard/soft budget |
| `compression` | CRD + LZ-rate Layer 1 |
| `ngram` | 可选，n-gram repetition baseline |

关键指标：

- savings rate
- quality retention
- false positive rate on complex tasks
- latency
- answer length ratio

### Phase D：BOCPD / MDL 研究增强

BOCPD 是合理方向，但不是当前第一优先级。

更准确的表述：

> BOCPD 不会消除所有参数，但可以把手工阈值替换成有概率意义的建模参数，例如 hazard、prior、chunk size。

适合在 Layer 1 压缩信号确认有效后再做。

MDL / SPRT 更偏研究探索，暂时不作为工程主线。

## 5. Compression Detector Layer 1

Layer 1 用两个相对信号：

- CRD：当前窗口 gzip 压缩比相对起始窗口的衰减
- LZ-rate：当前窗口 LZ factor rate 相对起始窗口的衰减

停止规则：

```text
STOP if CRD < theta_crd OR LZ_ratio < theta_lz
持续 k 个窗口
```

这是 MVP，不应包装成“已验证的最终算法”。它的价值是：

- 无模型依赖
- 无额外 API 调用
- 可离线回放和在线使用
- 方便与 budget detector 做 A/B

## 6. 不适用边界

以下场景默认不应激进早停：

- Agent 多步工作流
- 数学证明、复杂推导
- 代码调试
- 用户要求展示完整推理过程
- LLM API 按 max tokens 而非实际生成计费

## 7. 当前优先级

按顺序推进：

1. 用新 schema 重新跑一轮小规模实验，确认 baseline reasoning 可测。
2. 修 Phase 2 输出泄漏。
3. 跑 `budget` vs `compression` 对照。
4. 根据对照结果决定是否投入 BOCPD。
5. 文献 claims 逐条核验，再决定是否写论文/报告。
