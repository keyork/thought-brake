# v0.1 Release Notes Draft

日期：2026-05-25

## Release Title

`thought-brake v0.1: client-side early stopping for black-box reasoning LLM APIs`

## Short Description

`thought-brake` v0.1 introduces a client-side streaming monitor-and-interrupt workflow for black-box reasoning LLM APIs. It watches visible reasoning text, interrupts overthinking with literal text detectors, and uses a second direct-recovery phase to produce the final answer.

## What Is Included

- `ThoughtBrakeClient` wrapper for Chat Completions-compatible LLM APIs
- Streaming reasoning monitor
- Two-phase execution:
  - Phase 1: stream + monitor + interrupt
  - Phase 2: direct final-answer recovery
- Detector interface and implementations:
  - `none`
  - `budget`
  - `compression`
  - `ngram`
  - `keyword`
  - `semantic`
- Configurable Phase 2 API body, e.g. disabling thinking where the provider supports it
- Schema v3 experiment results with:
  - Phase 1 tokens
  - Phase 2 tokens
  - total tokens
  - estimated token fallback
  - reasoning savings
  - total-token savings
  - Lost / Fixed quality accounting
- Full-run experiment runner and focused report generator
- 12 evidence-oriented report visualizations

## Main Experimental Result

Current full-run setup:

- math: 100 questions
- mmlu: 100 questions
- riddle: 100 questions
- detector families: `budget`, `compression`, `keyword`
- budgets: `0,300,1000`
- Phase 2 mode: `direct`

Current recommended policy:

| Role | Policy | Quality | Total-token savings | Notes |
|---|---|---:|---:|---|
| Conservative | `keyword@1000` | 94.3% | 14.3% | Best observed quality |
| Default | `compression@1000` | 93.5% | 19.0% | Current default balance |
| Balanced-aggressive | `compression@300` | 92.3% | 29.1% | More savings, more quality cost |
| Aggressive | `budget@300` | 88.6% | 42.1% | Highest savings, too much quality loss for default |

## Important Caveats

- `TokenSavings` is not yet billing-grade exact.
- In treated rows, 400/1800 use API `total_tokens`; 1400/1800 use `estimated_total_tokens`.
- Early-stop Phase 1 often lacks provider usage because the stream is intentionally closed before the final usage chunk.
- The local token estimate has about 14.4% mean absolute error on rows where Phase 1 API usage is available for calibration.
- `compression@1000` is a current empirical default, not a theoretical optimum.
- Task router, BOCPD, embedding redundancy, and answer oscillation are not part of v0.1.

## How To Run

Install:

```bash
uv sync --dev --group experiments
```

Configure:

```bash
cp .env.example .env
```

Run the main experiment:

```bash
./experiments/run_token_main.sh
```

Regenerate the focused report from existing results:

```bash
uv run python experiments/report_full_main.py
```

Run tests:

```bash
uv run pytest
uv run ruff check src tests experiments
uv run mypy src
```

## Blog Outline

### 1. The Problem

Reasoning models often keep thinking after the useful part of reasoning has converged. API users pay for this, but usually cannot modify the model or decoding process.

### 2. The Deployment Constraint

Most real users have only:

- streaming text
- API request cancellation
- a second API call

They do not have:

- logits
- hidden states
- sampler control
- training access

### 3. The Core Idea

Visible reasoning text is enough to detect some overthinking patterns online:

- repetition
- low information density
- hesitation
- answer reconfirmation
- conclusion followed by continued reasoning

### 4. The Two-phase System

Interrupting thinking is not enough; the user still needs a final answer. Phase 2 direct recovery turns early stopping into a usable system.

### 5. What The Experiment Shows

Use no-stop baseline as the reference. Show:

- quality drop
- reasoning savings
- total-token savings
- Lost / Fixed
- strategy map

### 6. What We Should Not Claim

Do not claim theoretical optimality or exact billing-grade savings. Present v0.1 as an empirical, client-side, black-box method.

### 7. What Comes Next

- v0.2: BOCPD / change-point detector
- v0.3: token calibration and cross-vendor sanity check
- later: task router and hybrid detector

## Suggested GitHub Release Body

```markdown
This is the first v0.1 release of thought-brake.

It provides a client-side streaming monitor-and-interrupt workflow for black-box reasoning LLM APIs. The library watches visible reasoning text, interrupts overthinking, and performs a direct final-answer recovery call.

Current full-run experiments on math=100, mmlu=100, and riddle=100 support `compression@1000` as the default policy: 93.5% answer quality and about 19.0% total-token savings under mixed API/estimated token accounting.

See:

- README.md for usage
- docs/report_v0_1.md for the research report
- docs/experiments.md for reproduction
- docs/plan.md for the roadmap
```
