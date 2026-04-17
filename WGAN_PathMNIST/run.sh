#!/usr/bin/env bash
set -euo pipefail

mkdir -p outputs

nohup python gan_project/train.py \
  --epochs 150 \
  --batch_size 256 \
  --eval_batch_size 256 \
  --num_samples 5000 \
  --seed 123 \
  --device cuda \
  --gpu_ids 1 \
  > outputs/train_gan_nohup.log 2>&1 &
