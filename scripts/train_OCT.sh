#!/bin/bash
# FedOSS RetinalOCT training script
# Usage: bash scripts/train_OCT.sh

echo "====== FedOSS RetinalOCT Training ======"
echo ""

# Pretrain (seed 0, known=5, unknown=3)
# python main.py \
#     --data_root='./datasets/RetinalOCT_Dataset' \
#     --lr=5e-4 \
#     --backbone='Resnet18' \
#     --dataset='RetinalOCT' \
#     --known_class=5 \
#     --unknown_class=3 \
#     --seed=0 \
#     --batchsize=8 \
#     --epoches=50 \
#     --client_num=8 \
#     --worker_steps=1 \
#     --mode='Pretrain' \
#     --dirichlet=0.5 \
#     --save_interval=5 \
#     --log_dir='./logs'

# Finetune
python main.py \
    --data_root='./datasets/RetinalOCT_Dataset' \
    --lr=1e-4 \
    --backbone='Resnet18' \
    --dataset='RetinalOCT' \
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
    --device_id=0

echo ""
echo "====== Training Complete ======"
