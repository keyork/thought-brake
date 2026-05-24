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
2. 不依赖额外模型或 embedding
3. 流式在线检测
4. detector 可插拔
5. 能和真实业务的成本/延迟/质量指标对齐

需要谨慎的学术 claim：

- “没人做过”这类表述需要逐条文献核验
- “无 magic number”应改成“参数具有统计解释”
- “已验证”应区分 toy simulation、离线回放和真实 API 实验

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

1. 先补可靠 baseline：stream but no stop。
2. 修 Phase 2 输出泄漏，保证 final answer 简洁。
3. 把 hard budget 和 compression detector 放进同一实验框架。
4. 用真实结果判断 compression 是否比 budget 更好。
5. 如果 Layer 1 有信号，再做 BOCPD。
6. 最后再考虑 MDL / SPRT 作为论文型探索。

## 6. 关键风险

| 风险 | 影响 | 缓解 |
|---|---|---|
| 压缩信号把正常严谨推理误判为重复 | 复杂任务质量下降 | 按任务类型路由，复杂题关闭或提高安全 hard limit |
| Phase 2 输出推理草稿 | 成本和体验变差 | strict final prompt / output cleaning / API-specific config |
| baseline 没记录 reasoning | 无法计算 savings | 使用 `detector="none"` stream monitor |
| 文献 claims 不准 | 对外叙述风险 | 建参考文献表并逐条核验 |

## 7. 一句话总结

客户端压缩早停是合理方向，但它不是第一步。第一步是建立可信实验基线和稳定 Phase 2；压缩 detector 是第二步；BOCPD/MDL 是第三步。
