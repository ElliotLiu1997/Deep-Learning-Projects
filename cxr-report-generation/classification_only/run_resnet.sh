#!/usr/bin/env bash
set -euo pipefail

# Train + evaluate RESNET only.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# =========================
# Edit parameters here only
# =========================
DATA_CSV="${PROJECT_ROOT}/info.csv"
IMAGE_DIR="${PROJECT_ROOT}/images/images_normalized"
OUT_DIR="${PROJECT_ROOT}/classification_only/outputs/resnet"
GPU_IDS="0,1"
EPOCHS=25
BATCH_SIZE=128
HEAD_LR=1e-4
BACKBONE_LR=1e-5
UNFREEZE_EPOCH=3
EARLY_STOP_PATIENCE=5
EARLY_STOP_MIN_DELTA=1e-4

mkdir -p "${OUT_DIR}"

echo "============================================================"
echo "[resnet] Training"
echo "out_dir=${OUT_DIR}"
echo "============================================================"

python "${PROJECT_ROOT}/classification_only/train.py" \
  --encoder resnet \
  --data_csv "${DATA_CSV}" \
  --image_dir "${IMAGE_DIR}" \
  --gpu_ids "${GPU_IDS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${HEAD_LR}" \
  --backbone_lr "${BACKBONE_LR}" \
  --unfreeze_epoch "${UNFREEZE_EPOCH}" \
  --scheduler plateau \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
  --compute_auc_val \
  --save_path "${OUT_DIR}/best_model.pt" \
  --history_csv "${OUT_DIR}/train_history.csv" \
  --loss_plot "${OUT_DIR}/train_loss.png" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo "============================================================"
echo "[resnet] Evaluation (test only)"
echo "============================================================"

python "${PROJECT_ROOT}/classification_only/eval.py" \
  --checkpoint "${OUT_DIR}/best_model.pt" \
  --encoder resnet \
  --data_csv "${DATA_CSV}" \
  --image_dir "${IMAGE_DIR}" \
  --gpu_ids "${GPU_IDS}" \
  --search_threshold_on_val \
  --threshold_min 0.05 \
  --threshold_max 0.95 \
  --threshold_step 0.05 \
  --threshold_search_csv "${OUT_DIR}/threshold_search.csv" \
  --compute_auc \
  --metrics_csv "${OUT_DIR}/eval_metrics.csv" \
  --per_class_csv "${OUT_DIR}/eval_per_class_metrics.csv" \
  2>&1 | tee "${OUT_DIR}/eval.log"

echo "Done. Results saved under: ${OUT_DIR}"
