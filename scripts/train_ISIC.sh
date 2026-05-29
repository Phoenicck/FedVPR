#!/bin/bash
# FedOSS ISIC 2019 training script
# Usage: bash scripts/train_ISIC.sh

echo "====== FedOSS ISIC 2019 Training ======"
echo ""

# Small /dev/shm friendly defaults. Override with NUM_WORKERS=4 for faster large-server runs.
NUM_WORKERS=${NUM_WORKERS:-0}
PREFETCH_FACTOR=${PREFETCH_FACTOR:-2}
echo "DataLoader NUM_WORKERS=${NUM_WORKERS}, PREFETCH_FACTOR=${PREFETCH_FACTOR}"

# Pretrain (seed 0, known=5, unknown=3)
python main.py \
    --data_root='./datasets' \
    --lr=5e-4 \
    --backbone='Resnet18' \
    --dataset='ISIC' \
    --known_class=5 \
    --unknown_class=3 \
    --seed=0 \
    --batchsize=8 \
    --epoches=100 \
    --client_num=8 \
    --worker_steps=1 \
    --mode='Pretrain' \
    --dirichlet=0.5 \
    --save_interval=5 \
    --log_dir='./logs' \
    --num_workers=${NUM_WORKERS} \
    --prefetch_factor=${PREFETCH_FACTOR}

# Finetune
python main.py \
    --data_root='./datasets' \
    --lr=1e-4 \
    --backbone='Resnet18' \
    --dataset='ISIC' \
    --known_class=5 \
    --unknown_class=3 \
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
    --log_dir='./logs' \
    --num_workers=${NUM_WORKERS} \
    --prefetch_factor=${PREFETCH_FACTOR}

echo ""
echo "====== Training Complete ======"
