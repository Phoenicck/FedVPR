#!/bin/bash
# FedVPR ISIC 2019 training script
# Usage: bash scripts/train_ISIC.sh

echo "====== FedVPR ISIC 2019 Training ======"
echo ""

# Pretrain (seed 0, known=5, unknown=3, virtue_num=3)
python main.py \
    --data_root='./datasets' \
    --lr=5e-4 \
    --backbone='Resnet18' \
    --dataset='ISIC' \
    --protocol_mode='easy' \
    --known_class=5 \
    --unknown_class=3 \
    --virtue_num=3 \
    --seed=0 \
    --batchsize=8 \
    --epoches=100 \
    --client_num=8 \
    --worker_steps=1 \
    --mode='Pretrain' \
    --dirichlet=0.5 \
    --save_interval=5 \
    --log_dir='./logs'

# Finetune
python main.py \
    --data_root='./datasets' \
    --lr=1e-4 \
    --backbone='Resnet18' \
    --dataset='ISIC' \
    --protocol_mode='easy' \
    --known_class=5 \
    --unknown_class=3 \
    --virtue_num=3 \
    --seed=0 \
    --batchsize=8 \
    --epoches=30 \
    --client_num=8 \
    --worker_steps=1 \
    --mode='Finetune' \
    --eps=0.1 \
    --num_steps=1 \
    --dirichlet=0.5 \
    --start_epoch='[5,10,15,20,25]' \
    --sample_from=8 \
    --lups_mode='diag' \
    --lups_space='pooled' \
    --lups_pool_size=2 \
    --lups_local_weight=0.1 \
    --lups_global_weight=0.01 \
    --rank_weight=0.05 \
    --rank_margin=0.2 \
    --lups_sample_strategy='low_density' \
    --lups_candidates=100 \
    --log_dir='./logs'

echo ""
echo "====== Training Complete ======"
