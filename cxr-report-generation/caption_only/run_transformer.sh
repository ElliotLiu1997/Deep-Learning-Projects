#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATA_CSV="${PROJECT_ROOT}/info.csv"
IMAGE_DIR="${PROJECT_ROOT}/images/images_normalized"
ENCODER_CKPT="${PROJECT_ROOT}/classification_only/outputs/resnet/best_model.pt"
OUT_DIR="${PROJECT_ROOT}/caption_only/outputs/transformer"
GPU_IDS="0,1"

EPOCHS=30
BATCH_SIZE=64
LR=1e-4
ENCODER_LR=3e-5
TEACHER_FORCING_RATIO=0.8
LABEL_SMOOTHING=0.0
CAPTION_BALANCE_ALPHA=0.0
TOKEN_BALANCE_ALPHA=0.0
NUM_WORKERS=8

mkdir -p "${OUT_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

echo "[transformer] train -> ${OUT_DIR}"
python -m caption_only.train \
  --decoder_type transformer \
  --data_csv "${DATA_CSV}" \
  --image_dir "${IMAGE_DIR}" \
  --encoder_checkpoint "${ENCODER_CKPT}" \
  --output_dir "${OUT_DIR}" \
  --gpu_ids "${GPU_IDS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --encoder_lr "${ENCODER_LR}" \
  --teacher_forcing_ratio "${TEACHER_FORCING_RATIO}" \
  --label_smoothing "${LABEL_SMOOTHING}" \
  --caption_balance_alpha "${CAPTION_BALANCE_ALPHA}" \
  --token_balance_alpha "${TOKEN_BALANCE_ALPHA}" \
  --early_stopping_patience 6 \
  --num_workers "${NUM_WORKERS}" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo "[transformer] eval (greedy)"
python -m caption_only.evaluate \
  --decoder_type transformer \
  --data_csv "${DATA_CSV}" \
  --image_dir "${IMAGE_DIR}" \
  --encoder_checkpoint "${ENCODER_CKPT}" \
  --model_path "${OUT_DIR}/best_model.pt" \
  --output_dir "${OUT_DIR}" \
  --gpu_ids "${GPU_IDS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --decode_method greedy \
  2>&1 | tee "${OUT_DIR}/eval_greedy.log"

echo "[transformer] eval (beam)"
python -m caption_only.evaluate \
  --decoder_type transformer \
  --data_csv "${DATA_CSV}" \
  --image_dir "${IMAGE_DIR}" \
  --encoder_checkpoint "${ENCODER_CKPT}" \
  --model_path "${OUT_DIR}/best_model.pt" \
  --output_dir "${OUT_DIR}/beam" \
  --gpu_ids "${GPU_IDS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --decode_method beam \
  --beam_size 3 \
  --length_penalty 0.7 \
  2>&1 | tee "${OUT_DIR}/eval_beam.log"

echo "Done: ${OUT_DIR}"
