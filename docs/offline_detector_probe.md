# Offline Detector Probe

日期：2026-05-26

目标：在不调用 LLM API、不跑完整实验的前提下，快速评估 detector 信号是否有基本有效性。

## 为什么需要

BOCPD probe 暴露了一个问题：如果每次都等真实 API 实验跑完，才发现信号没有触发，迭代速度太慢，成本也高。

因此后续新 detector 或新信号必须先经过离线 replay：

1. 用 synthetic overthinking trace 做 sanity check。
2. 有 raw reasoning 历史文本时，直接读取本地文件 replay。
3. 只有离线 replay 能解释触发位置和 detail 时，才进入真实 API probe。

## 运行

默认使用内置 synthetic traces：

```bash
uv run python experiments/offline_detector_probe.py
```

只看某几个 detector：

```bash
uv run python experiments/offline_detector_probe.py \
  --detectors compression,keyword,bocpd
```

读取本地 raw reasoning 文本：

```bash
uv run python experiments/offline_detector_probe.py \
  --input tmp/reasoning.txt \
  --detectors compression,keyword,ngram,semantic,bocpd
```

## 输出解释

输出字段：

| 字段 | 含义 |
|---|---|
| `trace` | trace 名称 |
| `detector` | detector 名称 |
| `reason` | `soft` / `hard` / `natural` |
| `chars` | 停止位置或文本总长度 |
| `verdict` | 离线 sanity verdict |
| `detail` | detector detail |

`verdict`：

| verdict | 含义 |
|---|---|
| `ok_stop` | 在答案出现后触发 |
| `too_early` | 答案出现前触发 |
| `missed` | overthinking trace 没有触发 |
| `ok_no_stop` | productive trace 没触发 |

## 使用原则

离线 replay 不是最终实验证据。它只回答：

> 这个信号在一个明确构造的 overthinking 文本上，是否至少能触发，并且触发位置是否合理？

如果离线 replay 都失败，就不要进入真实 API probe。

如果离线 replay 通过，再进入小批量 API probe，最后才考虑全量实验。
