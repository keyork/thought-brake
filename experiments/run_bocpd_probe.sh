#!/usr/bin/env bash
set -euo pipefail

N="${N:-20}"
WORKERS="${WORKERS:-10}"
OUTPUT="${OUTPUT:-experiments/results/bocpd_probe_${N}_v4_detail.jsonl}"

uv run python experiments/runner.py \
  --dataset all \
  --n "${N}" \
  --budgets 0,300,1000 \
  --detector bocpd \
  --workers "${WORKERS}" \
  --track-usage \
  --phase2 direct \
  --output "${OUTPUT}"
