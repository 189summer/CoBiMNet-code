#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-./dataset}"
DEVICE="${DEVICE:-cuda:0}"

for pred_len in 96 192 336 720; do
  python main.py \
    --dataset ETTh2 \
    --data_dir "${DATA_DIR}" \
    --seq_len 720 \
    --pred_len "${pred_len}" \
    --cycle_len 24 \
    --period_len 24 \
    --batch_size 256 \
    --lr 1e-4 \
    --cycle_lr 5e-4 \
    --cycle_weight_decay 1e-4 \
    --lr_schedule exp \
    --epochs 300 \
    --patience 50 \
    --select_from_epoch 5 \
    --seed 2025 \
    --device "${DEVICE}"
done
