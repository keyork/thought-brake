# 项目计划

日期：2026-05-25

这份文档记录 `thought-brake` 的短中期执行计划。它和 `docs/idea.md` 的区别是：

- `idea.md` 记录研究定位和路线判断
- `plan.md` 记录接下来实际要做什么、为什么做、做到什么算完成

## 当前判断

项目已经完成 v0.1 的核心工程与实验闭环：

- client-side streaming monitor
- early interrupt
- two-phase final-answer recovery
- 5 类 detector
- schema v3 token metrics
- focused report 和 12 张证据型可视化
- v0.1 research report draft
- 73 个测试

但项目还没有完成两个更高层目标：

- 对外发布：创建 release/tag/blog，让别人理解、安装、复现实验
- 数学美感：减少 magic threshold，推进到 BOCPD / change-point detector

因此后续不应继续无序增加 detector，而应按 milestone 推进。

## Milestone v0.1：发布 Layer 1

目标：把当前已经跑通的 compression / keyword / budget 方案整理成可以对外解释和使用的版本。

### v0.1 Scope

- Main claim：client-side + black-box API + visible reasoning text
- Method：streaming monitor + interrupt + direct recovery
- Default policy：`compression@1000`
- Metrics：quality、reasoning savings、total-token savings、Lost、Fixed、Phase 2 API/recovery failure
- Limitations：token estimate uncertainty、single-vendor evidence

### v0.1 必做

1. **重写 README 的 public-facing narrative** ✅
   - 清楚解释 contribution
   - 不把工程细节放在最前面
   - 明确 quickstart
   - 明确 limitations

2. **升级 final report** ✅
   - research question
   - setup
   - metrics
   - main findings
   - limitations
   - strategy recommendation

3. **新增 strategy map 图** ✅
   - 用一张图解释 default / aggressive / conservative 三档策略
   - 不用 TokenSavings 的 0.8pp 微差做精确排序
   - 强调 quality 和 Lost 主导 `compression@1000` / `compression@300` / `keyword@1000` 的取舍

4. **补 survey 对比** ✅
   - EAT (arXiv:2510.08146)：black-box + entropy/proxy model
   - LZ Penalty / compression-style methods：更接近 local decoding 或非 API 部署
   - ROM / NEAT / RCPD / Adaptive CoT：若依赖内部信号或训练，需要明确和本项目不同

5. **准备发布材料** ✅
   - blog 大纲
   - README quickstart
   - fixed limitations section

### v0.1 完成标准

- 一个新用户能在 5 分钟内理解项目解决什么问题
- 一个 reviewer 能看懂本项目和 EAT / internal-signal methods 的区别
- report 不夸大 total-token savings 精度
- README 不把 `compression@1000` 包装成理论最优

## Milestone v0.2：BOCPD / Layer 2

目标：回应最初“数学上更美、减少 magic number”的目标。

### 为什么需要 v0.2

当前 `compression@1000` / `compression@300` 仍然包含参数：

- `@300` / `@1000` safety budget
- compression thresholds
- LZ thresholds
- consecutive windows

这足够作为 v0.1 工程方案，但还没有真正解决 magic threshold 问题。BOCPD / change-point detection 的价值是把停止决策表述成 posterior change probability，而不是继续手调阈值。

### v0.2 Scope

- 实现一个 BOCPD-style detector
- 输入仍然是 visible reasoning text 的在线特征
- 先用 compression / LZ / repetition 等低维信号
- 对比 `compression@1000`、`compression@300`、`keyword@1000`、`budget@300`

### v0.2 必做

1. **写 Layer 2 设计文档** ✅
   - signal definition
   - hazard / prior
   - online update
   - stop criterion
   - 与 magic threshold 的关系

2. **实现 BOCPD detector**
   - 插入现有 detector 接口
   - 支持 runner 实验
   - 单测覆盖

3. **小批量验证**
   - 先跑 20-50 题
   - 看是否稳定触发
   - 看质量是否明显退化

4. **与 Layer 1 比较**
   - quality
   - reasoning savings
   - total-token savings
   - trigger distribution
   - failure examples

### v0.2 完成标准

- 能回答 BOCPD 是否真的减少参数敏感性
- 能回答 BOCPD 是否比 compression/keyword 更稳
- 即使效果不超过 Layer 1，也能明确解释失败原因

## Milestone v0.3：Cost Calibration 与 Cross-vendor

目标：让 total-token savings 更硬，并证明方法不是某一个 API 的偶然现象。

### v0.3 Scope

- token estimate calibration
- 至少一个额外 LLM API provider sanity check
- latency metrics

### v0.3 必做

1. **Token calibration**
   - 用自然结束样本对比 API usage 和 local estimate
   - 拟合简单校正函数
   - 把 MAE 从当前约 15% 尽量压低

2. **Cross-vendor sanity check**
   - 选择一个数据集
   - 跑 default policy 和 baseline
   - 不追求 full run，先验证方法可迁移

3. **Latency reporting**
   - p50 / p90 / p99 latency
   - 分别报告 baseline 和 early-stop

### v0.3 完成标准

- 能说清楚 total-token savings 的误差范围
- 至少有一个额外 provider 的 sanity result
- report 里有 latency 维度

## 暂不优先

以下方向暂时不做，除非 v0.1/v0.2/v0.3 都推进到位：

- full task router
- embedding-based redundancy
- answer oscillation
- 复杂 UI / dashboard
- 大规模 riddle full rerun

其中 task router 可以保留一个 very small version：

```text
default: compression@1000
conservative: keyword@1000
aggressive: budget@300
balanced-aggressive: compression@300
```

但不要在 v0.1 阶段把它包装成已验证的智能路由。

## 当前下一步

当前 v0.1 已经打 `v0.1.0` tag，release draft 已经落地；v0.2 BOCPD 设计文档也已启动。建议马上做：

1. 创建 GitHub Release，内容使用 `docs/release_v0_1.md`
2. 评审 `docs/bocpd_design.md`，确认 v0.2 的最小实现范围
3. 实现 BOCPD feature extraction 和 `OnlineChangePoint` core
4. 接入 `BOCPDDetector` 并跑 20-50 题 probe
