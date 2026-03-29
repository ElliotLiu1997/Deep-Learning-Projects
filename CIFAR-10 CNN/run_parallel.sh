#!/usr/bin/env bash
set -euo pipefail

# Usage:
# ./run_parallel.sh [data_dir] [output_dir] [epochs] [batch_size] [num_workers] [lr] [seed]
#
# Example:
# ./run_parallel.sh .. ../outputs_parallel 25 128 4 0.001 123

DATA_DIR="${1:-..}"
OUTPUT_DIR="${2:-../outputs_parallel}"
EPOCHS="${3:-25}"
BATCH_SIZE="${4:-128}"
NUM_WORKERS="${5:-4}"
LR="${6:-0.001}"
SEED="${7:-42}"
LOG_DIR="${OUTPUT_DIR}/logs"

MODELS=(
  "baseline"
  "deeper"
  "residual"
  "baseline_bn"
  "deeper_bn"
  "residual_bn"
)

GPU_IDS=(0 1)
MAX_JOBS=${#GPU_IDS[@]}
running_jobs=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting parallel training with ${MAX_JOBS} GPUs"
echo "DATA_DIR=${DATA_DIR}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

for idx in "${!MODELS[@]}"; do
  model="${MODELS[$idx]}"
  gpu="${GPU_IDS[$((idx % MAX_JOBS))]}"
  log_file="${LOG_DIR}/${model}.log"

  echo "Launch model=${model} on GPU=${gpu} (log: ${log_file})"
  nohup env CUDA_VISIBLE_DEVICES="${gpu}" python main.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    --lr "${LR}" \
    --seed "${SEED}" \
    --model "${model}" > "${log_file}" 2>&1 &

  running_jobs=$((running_jobs + 1))
  if [[ "${running_jobs}" -ge "${MAX_JOBS}" ]]; then
    wait -n
    running_jobs=$((running_jobs - 1))
  fi
done

wait
echo "All models finished."
