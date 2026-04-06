#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash diffusion_project/submit_nohup.sh [EPOCHS] [TRAIN_BS] [EVAL_BS] [NUM_SAMPLES] [GPU_IDS] [LOG_FILE]
#   nohup bash diffusion_project/submit_nohup.sh [EPOCHS] [TRAIN_BS] [EVAL_BS] [NUM_SAMPLES] [GPU_IDS] &
# Example:
#   bash diffusion_project/submit_nohup.sh 30 128 128 2000 0,1

EPOCHS="${1:-50}"
TRAIN_BS="${2:-128}"
EVAL_BS="${3:-128}"
NUM_SAMPLES="${4:-2000}"
GPU_IDS="${5:-0,1}"

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

OUT_DIR="diffusion_project/outputs"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${6:-$LOG_DIR/pipeline_${TS}.log}"
mkdir -p "$(dirname "$LOG_FILE")"

# Internal logging avoids external redirection path issues.
exec >> "$LOG_FILE" 2>&1

echo "Log file: $LOG_FILE"

echo "[START] $(date)"

echo "[1/4] Training..."
python diffusion_project/train.py \
  --data_dir pathmnist \
  --output_dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch_size "$TRAIN_BS" \
  --gpu_ids "$GPU_IDS"

echo "[2/4] Sampling..."
python diffusion_project/sample.py \
  --checkpoint "$OUT_DIR/latest.pt" \
  --data_dir pathmnist \
  --output_dir "$OUT_DIR" \
  --num_samples 16 \
  --ddim_steps 100 50 \
  --gpu_ids "$GPU_IDS"

echo "[3/4] Saving real test grid..."
python diffusion_project/save_real_grid.py \
  --data_dir pathmnist \
  --output_dir "$OUT_DIR" \
  --num_samples 16

echo "[4/4] Evaluating..."
python diffusion_project/evaluate.py \
  --checkpoint "$OUT_DIR/latest.pt" \
  --data_dir pathmnist \
  --output_dir "$OUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --batch_size "$EVAL_BS" \
  --ddim_steps 100 50 \
  --gpu_ids "$GPU_IDS"

echo "[DONE] $(date)"
