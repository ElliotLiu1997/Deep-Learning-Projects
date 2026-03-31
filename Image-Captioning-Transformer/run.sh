#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=1

LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="$LOG_DIR/pipeline_${TS}.log"

{
  echo "[$(date '+%F %T')] Start pipeline"
  echo "[$(date '+%F %T')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

  echo "[$(date '+%F %T')] Running train.py"
  python code/train.py

  echo "[$(date '+%F %T')] Running evaluate.py"
  python code/evaluate.py

  echo "[$(date '+%F %T')] Running figure.py"
  python code/figure.py

  echo "[$(date '+%F %T')] Pipeline finished successfully"
} >> "$PIPELINE_LOG" 2>&1

echo "Log: $PIPELINE_LOG"
