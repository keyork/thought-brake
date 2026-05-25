# Survey：Reasoning Model Overthinking 缓解方向

本文档是研发路线的背景调研，不作为最终论文综述。结论需要随文献更新继续核验。

## 1. 结论摘要

Reasoning model overthinking 的缓解方法可以粗略分为四类：

| 类别 | 信号 | 是否适合黑盒 LLM API | 说明 |
|---|---|---:|---|
| 训练阶段方法 | 训练奖励 / stop token | 否 | 需要模型权重和训练资源 |
| 内部信号方法 | hidden states / logits / entropy | 通常否 | 效果强，但 API 用户拿不到 |
| 探测调用方法 | trial answer / answer stability | 是，但贵 | 需要额外 API 调用 |
| 文本信号方法 | n-gram / compression / repetition | 是 | 最适合客户端轻量部署 |

`thought-brake` 应聚焦最后一类：**客户端流式文本信号**。

## 2. 与我们最相关的方向

### 2.0 最近的黑盒相关工作：EAT

需要在正式 report / paper 前重点核验 EAT (arXiv:2510.08146)。按当前调研记录，它是最接近本项目的 black-box reasoning early stopping 工作之一，使用 sequence-level entropy 和 proxy model 来判断停止。

当前项目和 EAT 的差异化应谨慎表述为：

1. `thought-brake` 只使用 visible reasoning text 上的字面信号，如 compression、keyword、n-gram，不依赖 proxy model。
2. `thought-brake` 的主要部署 niche 是 client-side black-box API consumer。
3. 当前实验比较了多个 detector family，而不是只验证单一停止信号。
4. 当前系统把 interrupt 后的 two-phase final-answer recovery 和 total-token cost evaluation 一起纳入评估。

这不是说 EAT 不适合黑盒场景，而是说我们的成本和部署假设更轻：不引入额外模型，不访问服务端内部状态。

### 2.1 压缩/重复信号

压缩视角的核心直觉是：

> 过度推理阶段的信息增量下降，文本更容易被压缩或出现局部重复。

这支持我们做 CRD、LZ-rate、n-gram repetition 等纯文本 detector。

需要注意：

- 压缩信号能发现“冗余/循环”
- 但不能保证“答案已经正确”
- 因此必须和质量评测、复杂任务假阳性监控配合

### 2.2 在线变化点检测

BOCPD / SPRT / MDL 这类经典统计方法可以把 detector 从固定阈值推进到更有理论解释的判据。

但当前阶段不要过度承诺“无参数”：

- BOCPD 有 hazard / prior / chunk size
- SPRT 需要定义 healthy vs overthinking 的似然
- MDL 需要设计编码方案

这些都适合作为 Layer 2 / research track，而不是第一版工程主线。

### 2.3 内部信号方法

hidden states / logits 方法通常更强，因为它们直接观察模型内部状态。但我们的目标用户拿不到这些信号，所以它们是参照物，不是直接竞争对象。

我们的定位应该是：

> 在内部信号不可用时，提供可部署的客户端替代方案。

## 3. 本项目的空白点

可主张的工程 niche：

1. 黑盒 LLM API 用户可用
2. 客户端侧实现，不修改服务端采样器
3. 不依赖额外模型、proxy model 或 embedding
4. 流式在线检测 visible reasoning text
5. detector 可插拔
6. 能和真实业务的成本/延迟/质量指标对齐

需要谨慎的学术 claim：

- “没人做过”这类表述需要逐条文献核验
- “无 magic number”应改成“参数具有统计解释”
- “已验证”应区分 toy simulation、离线回放和真实 API 实验
- “TokenSavings 精确优于某 detector”需要考虑当前 token estimate 误差，0.x pp 级差异不应过度解释

## 4. 对照基线

未来实验至少应比较：

| baseline | 作用 |
|---|---|
| `none` | 不早停，只记录真实 reasoning |
| `budget` | 当前 soft/hard budget |
| `compression` | CRD + LZ-rate |
| n-gram repetition | 字面重复 baseline |
| answer convergence | 质量参照，但成本高 |

## 5. 推荐研发顺序

当前 baseline、Phase 2 direct recovery、Layer 1 detector 对照已经完成。后续顺序应改为：

1. 收敛 v0.1 发布叙事：client-side + black-box API + visible reasoning text。
2. 补齐相关工作对比：EAT、内部信号方法、proxy-model 方法、compression-style 方法。
3. 升级最终 report：明确 TokenSavings 不确定性、Phase 2 failure 定义和 Pareto frontier。
4. 实现 Layer 2 / BOCPD，检验是否能减少 magic threshold。
5. 做 token calibration 和 cross-vendor sanity check。
6. 最后再考虑 task router、answer oscillation、embedding redundancy。

## 6. 关键风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 压缩信号把正常严谨推理误判为重复 | 复杂任务质量下降 | 按任务类型路由，复杂题关闭或提高安全 hard limit |
| Phase 2 输出推理草稿 | 成本和体验变差 | strict final prompt / output cleaning / API-specific config |
| baseline 没记录 reasoning | 无法计算 savings | 使用 `detector="none"` stream monitor |
| 文献 claims 不准 | 对外叙述风险 | 建参考文献表并逐条核验 |

## 7. 一句话总结

客户端流式文本信号是合理方向。v0.1 应先发布已经验证的 Layer 1；v0.2 再用 BOCPD / change-point detection 回应“减少 magic threshold”的研究目标。
