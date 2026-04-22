#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_CSV="${PROJECT_ROOT}/info.csv"
IMAGE_DIR="${PROJECT_ROOT}/images/images_normalized"
ENCODER_CKPT="${PROJECT_ROOT}/classification_only/outputs/resnet/best_model.pt"
OUT_DIR="${PROJECT_ROOT}/share_encoder/outputs/lstm_attn"
GPU_IDS="0,1"

EPOCHS=30
BATCH_SIZE=128
LR=1e-4
ENCODER_LR=3e-5
TEACHER_FORCING_RATIO=0.8
NUM_WORKERS=8

mkdir -p "${OUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

CLS_LOSS_WEIGHTS=("1.1" "1.2" "1.4" "1.6" "1.8" "2.0" "2.5" "3.0" "3.5" "4.0")

for CLS_LOSS_WEIGHT in "${CLS_LOSS_WEIGHTS[@]}"; do
  RUN_DIR="${OUT_DIR}/clsw_${CLS_LOSS_WEIGHT}"
  mkdir -p "${RUN_DIR}"

  echo "============================================================"
  echo "[share_encoder] train cls_loss_weight=${CLS_LOSS_WEIGHT} -> ${RUN_DIR}"
  echo "============================================================"
  python -m share_encoder.train \
    --data_csv "${DATA_CSV}" \
    --image_dir "${IMAGE_DIR}" \
    --encoder_checkpoint "${ENCODER_CKPT}" \
    --output_dir "${RUN_DIR}" \
    --gpu_ids "${GPU_IDS}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --encoder_lr "${ENCODER_LR}" \
    --teacher_forcing_ratio "${TEACHER_FORCING_RATIO}" \
    --cls_loss_weight "${CLS_LOSS_WEIGHT}" \
    --num_workers "${NUM_WORKERS}" \
    2>&1 | tee "${RUN_DIR}/train.log"

  echo "============================================================"
  echo "[share_encoder] eval cls_loss_weight=${CLS_LOSS_WEIGHT} -> ${RUN_DIR}"
  echo "============================================================"
  python -m share_encoder.evaluate \
    --data_csv "${DATA_CSV}" \
    --image_dir "${IMAGE_DIR}" \
    --encoder_checkpoint "${ENCODER_CKPT}" \
    --model_path "${RUN_DIR}/best_model.pt" \
    --output_dir "${RUN_DIR}" \
    --gpu_ids "${GPU_IDS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}" \
    2>&1 | tee "${RUN_DIR}/eval.log"
done

echo "Done all cls_loss_weight runs under: ${OUT_DIR}"
