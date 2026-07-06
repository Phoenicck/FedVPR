#!/bin/bash
# FedOSS RetinalOCT easy-protocol training script
# Usage:
#   bash scripts/train_OCT_easy.sh
#   DEVICE_ID=1 bash scripts/train_OCT_easy.sh

set -e

DEVICE_ID="${DEVICE_ID:-1}"
DATA_ROOT="${DATA_ROOT:-./datasets/RetinalOCT_Dataset}"
LOG_DIR="${LOG_DIR:-./logs}"

echo "====== FedOSS RetinalOCT Training (easy_k3) ======"
echo "DEVICE_ID=${DEVICE_ID}"
echo ""

# Pretrain (seed 0, known=3, unknown=5)
python main.py \
    --data_root="${DATA_ROOT}" \
    --lr=5e-4 \
    --backbone='Resnet18' \
    --dataset='RetinalOCT' \
    --protocol_mode='easy_k3' \
    --known_class=3 \
    --unknown_class=5 \
    --seed=0 \
    --batchsize=8 \
    --epoches=50 \
    --client_num=8 \
    --worker_steps=1 \
    --mode='Pretrain' \
    --dirichlet=0.5 \
    --save_interval=5 \
    --log_dir="${LOG_DIR}" \
    --device_id="${DEVICE_ID}"

# Finetune
python main.py \
    --data_root="${DATA_ROOT}" \
    --lr=1e-4 \
    --backbone='Resnet18' \
    --dataset='RetinalOCT' \
    --protocol_mode='easy_k3' \
    --known_class=3 \
    --unknown_class=5 \
    --seed=0 \
    --batchsize=8 \
    --epoches=30 \
    --client_num=8 \
    --worker_steps=1 \
    --mode='Finetune' \
    --eps=0.1 \
    --num_steps=1 \
    --unknown_weight=1. \
    --dirichlet=0.5 \
    --start_epoch='[5,10,15,20,25]' \
    --sample_from=8 \
    --log_dir="${LOG_DIR}" \
    --device_id="${DEVICE_ID}"

echo ""
echo "====== Training Complete (easy_k3) ======"
