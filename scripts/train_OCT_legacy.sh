#!/bin/bash
# FedVPR RetinalOCT legacy pretrain recipe for clean hard-protocol comparison
# Usage: bash scripts/train_OCT_legacy.sh

echo "====== FedVPR RetinalOCT Legacy Training ======"
echo ""

python main.py \
    --data_root='./datasets/RetinalOCT_Dataset' \
    --lr=5e-4 \
    --backbone='Resnet18' \
    --dataset='RetinalOCT' \
    --protocol_mode='easy' \
    --known_class=5 \
    --unknown_class=3 \
    --virtue_num=3 \
    --seed=0 \
    --batchsize=8 \
    --epoches=50 \
    --client_num=8 \
    --worker_steps=1 \
    --mode='Pretrain' \
    --dirichlet=0.5 \
    --vir_weight_warmup=0.5 \
    --vir_weight_main=0.01 \
    --vir_warmup_epochs=4 \
    --vir_anneal_epochs=0 \
    --vir_margin=1.0 \
    --vir_margin_weight=0.0 \
    --save_interval=5 \
    --log_dir='./logs'

echo ""
echo "====== Legacy Training Complete ======"
