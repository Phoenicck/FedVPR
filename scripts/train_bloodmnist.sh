#!/bin/bash
set -e

# FedVPR BloodMNIST training script with multi-virtual heads + LUPS.
# Usage:
#   bash scripts/train_bloodmnist.sh              # full run
#   SMOKE=1 bash scripts/train_bloodmnist.sh      # fast smoke test
#   STAGE=finetune bash scripts/train_bloodmnist.sh

PYTHON="/workspace/wanghengzhuo/miniconda3/bin/conda run -n pfllib python"

DATA_ROOT="${DATA_ROOT:-./datasets/MedMNIST/bloodmnist.npz}"
SEED="${SEED:-1}"
KNOWN="${KNOWN:-5}"
UNKNOWN="${UNKNOWN:-3}"
VIRTUAL="${VIRTUAL:-3}"
DIRICHLET="${DIRICHLET:-0.5}"
STAGE="${STAGE:-all}"  # all, pretrain, finetune

if [[ "${SMOKE:-0}" == "1" ]]; then
    CLIENTS="${CLIENTS:-2}"
    BATCHSIZE="${BATCHSIZE:-16}"
    PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-1}"
    FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-2}"
    MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-20}"
    MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-20}"
    LUPS_MIN_COUNT="${LUPS_MIN_COUNT:-0}"
    LUPS_CANDIDATES="${LUPS_CANDIDATES:-10}"
    START_EPOCH="${START_EPOCH:-[0]}"
else
    CLIENTS="${CLIENTS:-8}"
    BATCHSIZE="${BATCHSIZE:-8}"
    PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-100}"
    FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-30}"
    MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
    MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"
    LUPS_MIN_COUNT="${LUPS_MIN_COUNT:-10}"
    LUPS_CANDIDATES="${LUPS_CANDIDATES:-100}"
    START_EPOCH="${START_EPOCH:-[5,10,15,20,25]}"
fi
SAMPLE_FROM="${SAMPLE_FROM:-${CLIENTS}}"

mkdir -p logs

echo "====== FedVPR BloodMNIST Training ======"
echo "Stage=${STAGE}, Seed=${SEED}, K=${KNOWN}, U=${UNKNOWN}, V=${VIRTUAL}, Clients=${CLIENTS}"
echo ""

# Stage 1: Pretrain with virtual prototype reservation.
if [[ "${STAGE}" == "all" || "${STAGE}" == "pretrain" ]]; then
    $PYTHON main.py \
        --data_root="${DATA_ROOT}" \
        --lr=5e-4 \
        --backbone='Resnet18' \
        --dataset='Bloodmnist' \
        --known_class=${KNOWN} \
        --unknown_class=${UNKNOWN} \
        --virtue_num=${VIRTUAL} \
        --seed=${SEED} \
        --batchsize=${BATCHSIZE} \
        --epoches=${PRETRAIN_EPOCHS} \
        --client_num=${CLIENTS} \
        --worker_steps=1 \
        --mode='Pretrain' \
        --dirichlet=${DIRICHLET} \
        --save_interval=5 \
        --max_train_batches=${MAX_TRAIN_BATCHES} \
        --max_eval_batches=${MAX_EVAL_BATCHES} \
        --log_dir='./logs'
fi

# Stage 2: Finetune with LUPS diagonal sampling and ranking regularization.
if [[ "${STAGE}" == "all" || "${STAGE}" == "finetune" ]]; then
    $PYTHON main.py \
        --data_root="${DATA_ROOT}" \
        --lr=1e-4 \
        --backbone='Resnet18' \
        --dataset='Bloodmnist' \
        --known_class=${KNOWN} \
        --unknown_class=${UNKNOWN} \
        --virtue_num=${VIRTUAL} \
        --seed=${SEED} \
        --batchsize=${BATCHSIZE} \
        --epoches=${FINETUNE_EPOCHS} \
        --client_num=${CLIENTS} \
        --worker_steps=1 \
        --mode='Finetune' \
        --eps=0.1 \
        --num_steps=1 \
        --dirichlet=${DIRICHLET} \
        --start_epoch="${START_EPOCH}" \
        --sample_from=${SAMPLE_FROM} \
        --lups_mode='diag' \
        --lups_space='pooled' \
        --lups_pool_size=2 \
        --lups_min_count=${LUPS_MIN_COUNT} \
        --lups_min_var=1e-4 \
        --lups_var_scale=1.0 \
        --lups_candidates=${LUPS_CANDIDATES} \
        --lups_sample_strategy='low_density' \
        --lups_local_weight=0.1 \
        --lups_global_weight=0.01 \
        --rank_weight=0.05 \
        --rank_margin=0.2 \
        --max_train_batches=${MAX_TRAIN_BATCHES} \
        --max_eval_batches=${MAX_EVAL_BATCHES} \
        --log_dir='./logs'
fi

echo ""
echo "====== FedVPR BloodMNIST Training Complete ======"
