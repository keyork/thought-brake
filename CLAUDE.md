# CLAUDE.md

This file provides guidance to coding agents when working with this repository.

## Project Purpose

**thought-brake** (何时停止思考) is a client-side wrapper library for reasoning models. It detects and terminates overthinking during the streaming phase, then uses a prefill continuation to force a final answer. The core insight: a reasoning end marker is just part of the generated prefix, so the client can inject it at a controlled point to terminate the reasoning phase without breaking model state.

See `docs/technical_proposal.md` for the full design document (in Chinese).

## Commands

```bash
uv sync --dev          # install all dependencies (uses uv.lock)
uv sync --dev --group experiments  # install experiment dependencies
uv run pytest          # run tests
uv run ruff check src tests experiments   # lint
uv run mypy src        # type-check
uv run python experiments/runner.py --help
uv run python experiments/analysis.py --help
```

## Architecture

The system operates in two sequential phases per request:

**Phase 1 — Stream + Monitor**: Consume the model's `reasoning_content` stream while counting characters. A detector decides whether to stop. `none` monitors baseline without stopping, `budget` uses soft/hard budgets, and `compression` uses CRD + LZ-rate signals. If the model finishes naturally, skip Phase 2.

**Phase 2 — Prefill Continuation**: Reconstruct an `assistant` message containing the partial reasoning plus a closing hint (`\n\n好，已经想清楚了，直接给出最终答案。\n</think>\n\n`), then send a new API call to force the model to generate the final answer from that prefix.

**Target API**: LLM API with Chat Completions-style messages. Prefill works by making the last `messages` entry an `assistant` role message when the API supports assistant-prefix continuation.

## Key Design Decisions

**Budget defaults** (from `EarlyStopConfig`):
- Chat/common knowledge: `soft=200, hard=400`
- General QA: `soft=500, hard=1000`
- Math/code: `soft=1500, hard=3000`
- Complex reasoning: `soft=3000, hard=6000`

**Character count, not tokens**: Avoids a tokenizer dependency; <10% error in Chinese contexts.

**Soft vs hard truncation**: Soft waits for a sentence-ending punctuation (`。！？!?\n`) after crossing `soft_budget`. Hard fires unconditionally at `hard_limit` to prevent infinite loops.

**Detector interface**: Keep stop logic in `src/thought_brake/detectors.py`; `_monitor.py` should only stream chunks and feed the active detector.

**Hint text matters**: The in-think hint `\n\n好，已经想清楚了，直接给出最终答案。` outperforms bare `</think>` injection — models that receive the latter may re-open thinking.

**Fallback chain**:
1. Phase 2 prefill (standard: last message is `assistant`)
2. API-specific continuation options when required
3. Full fallback: append `partial_reasoning` to the user message asking for a summary

**Not suitable for**: agent workflows, multi-step code debugging, mathematical proofs — any case where the full reasoning chain is itself the output.

## Monitoring Metrics

Any implementation must emit: `reasoning_chars_p50/p99`, `truncation_rate`, `soft_stop_rate`, `hard_stop_rate`, `phase2_failure_rate`, `latency_p50/p99`, `cost_per_request`.

## Implementation Roadmap

- **Phase A**: Reliable baseline with `detector="none"`, schema-versioned results, and true reasoning length measurement
- **Phase B**: Phase 2 output convergence so final answers do not leak reasoning drafts
- **Phase C**: Detector comparisons: `budget` vs `compression`, then optional n-gram baseline
- **Phase D**: BOCPD / MDL / SPRT research track after compression signals prove useful
