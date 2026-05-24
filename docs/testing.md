# Testing Guide

This guide describes the checks to run before changing experiment code or the core client.

## 1. Install Dependencies

For core tests:

```bash
uv sync --dev
```

For experiment scripts and analysis:

```bash
uv sync --dev --group experiments
```

## 2. Run Unit Tests

```bash
uv run pytest
```

Current test areas:

| file | coverage |
|---|---|
| `tests/test_config.py` | Defaults, task presets, environment overrides |
| `tests/test_detectors.py` | Budget, no-stop, and compression detector behavior |
| `tests/test_monitor.py` | Phase 1 streaming, soft/hard stop, interrupted stream handling |
| `tests/test_prefill.py` | Phase 2 prefill message construction and streaming collection |
| `tests/test_client.py` | Client integration, passthrough, Phase 2 fallback, interruption recovery |
| `tests/test_exact_match.py` | Numeric exact-match evaluation edge cases |

Run one file:

```bash
uv run pytest tests/test_client.py
```

Run one test:

```bash
uv run pytest tests/test_client.py::test_interrupted_phase1_with_partial_reasoning_triggers_phase2
```

## 3. Run Static Checks

Ruff:

```bash
uv run ruff check src tests experiments
```

Auto-fix safe lint issues:

```bash
uv run ruff check src tests experiments --fix
```

Mypy for the package:

```bash
uv run mypy src
```

The current mypy target is `src` because the public package should stay strictly typed. Experiment scripts are still checked by ruff and covered by smoke tests.

## 4. Run Script Smoke Tests

Check CLI imports and arguments:

```bash
uv run python experiments/runner.py --help
uv run python experiments/analysis.py --help
```

Run a fast no-evaluation experiment:

```bash
uv run python experiments/runner.py \
  --dataset riddles \
  --budgets 0,100,200 \
  --difficulties easy \
  --detector budget \
  --workers 10 \
  --skip-eval \
  --output experiments/results/smoke.jsonl
```

This smoke test calls the configured API. Use it after `.env` is configured.

## 5. Full Pre-Commit Check

Run these before committing:

```bash
uv run pytest
uv run ruff check src tests experiments
uv run mypy src
uv run python experiments/runner.py --help
uv run python experiments/analysis.py --help
```

If you changed experiment execution behavior, also run a small `--skip-eval` smoke test.

## 6. What to Add Tests For

Add or update tests when changing:

- Stop reason behavior (`natural`, `soft`, `hard`, `interrupted`)
- Detector behavior (`none`, `budget`, `compression`)
- Budget configuration or `.env` parsing
- Prefill message shape or reasoning tags
- Phase 2 fallback prompt behavior
- JSONL resume semantics
- CLI arguments that affect execution

For API-dependent behavior, prefer mocked unit tests first. Use smoke tests only to verify LLM API compatibility.
