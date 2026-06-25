#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_DIR="results_stage1_grid/20260624_003127/2_hard_lambda0p05"
CHECKPOINT="${RUN_DIR}/result_dir_snapshot/best_ckpt_Pretrain_known_class_5_unknown_class_3_seed_0.pth"
OUT_DIR="${RUN_DIR}/result_dir_snapshot/analysis/stage2b_minimal_pilot"

python train_stage2b_minimal.py \
  --checkpoint "${CHECKPOINT}" \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUT_DIR}" \
  --rounds 20 \
  --stage2b-lr 0.0001 \
  --lambda-reserve 0.05 \
  --lambda-pseudo 0.1 \
  --pseudo-ratio 0.25 \
  --pseudo-per-anchor-cap 64 \
  --boundary-fraction 0.10 \
  --generator-steps 5 \
  --generator-step-size 0.005 \
  --generator-max-feature-distance 0.5 \
  --analysis-batch-size 128 \
  --analysis-num-workers 4 \
  --eval-every 1
