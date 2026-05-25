#!/usr/bin/env bash
set -euo pipefail

# Clean main experiment for schema v3.
# It records API usage when the LLM API supports streaming usage and also writes
# local estimated token fields for every row.
#
# Override defaults from the shell, for example:
#   WORKERS=10 N=20 ./experiments/run_token_main.sh

DATASET="${DATASET:-all}"
N="${N:-100}"
BUDGETS="${BUDGETS:-0,300,1000}"
WORKERS="${WORKERS:-10}"
PHASE2="${PHASE2:-direct}"
RESULTS_DIR="${RESULTS_DIR:-experiments/results}"
REPORT_DIR="${REPORT_DIR:-experiments/report/full_main_token}"

mkdir -p "${RESULTS_DIR}" "${REPORT_DIR}"

run_detector() {
  local detector="$1"
  local output="${RESULTS_DIR}/full_${detector}.jsonl"

  uv run python experiments/runner.py \
    --dataset "${DATASET}" \
    --n "${N}" \
    --budgets "${BUDGETS}" \
    --detector "${detector}" \
    --phase2 "${PHASE2}" \
    --workers "${WORKERS}" \
    --track-usage \
    --output "${output}"
}

run_detector budget
run_detector compression
run_detector keyword

uv run python experiments/report_full_main.py \
  --inputs \
    "${RESULTS_DIR}/full_budget.jsonl" \
    "${RESULTS_DIR}/full_compression.jsonl" \
    "${RESULTS_DIR}/full_keyword.jsonl" \
  --output "${REPORT_DIR}"
