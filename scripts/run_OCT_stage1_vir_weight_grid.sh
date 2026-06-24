#!/bin/bash
# FedVPR RetinalOCT Stage-1 reserve ablation:
# protocol_mode in {hard, random} x lambda_reserve in {0.05, 0.20} by default.
#
# Usage:
#   bash scripts/run_OCT_stage1_vir_weight_grid.sh
#   bash scripts/run_OCT_stage1_vir_weight_grid.sh 0.05 0.10 0.20
#   nohup bash scripts/run_OCT_stage1_vir_weight_grid.sh > overnight_stage1_grid.out 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.." || exit 1

CONFIG="./configs/stage1_retinaoct_hard.yaml"
DATA_ROOT="/workspace/Phoenic/claude0527/FedOSS/datasets/RetinalOCT_Dataset"
PROTOCOLS=(hard random)
WEIGHTS=("$@")

if [ ${#WEIGHTS[@]} -eq 0 ]; then
  WEIGHTS=(0.01 0.05 0.20)
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_ROOT="./logs_stage1_grid/${TIMESTAMP}"
ARCHIVE_ROOT="./results_stage1_grid/${TIMESTAMP}"
BASE_SAVE_DIR="./results/MPretrain-DRetinalOCT-Msoftmax-BResnet18/LR0.0005-K5-U3-Seed0-RsvW15-V3"

mkdir -p "$LOG_ROOT" "$ARCHIVE_ROOT"

echo "===== FedVPR Stage-1 reserve grid ====="
echo "timestamp: ${TIMESTAMP}"
echo "config: ${CONFIG}"
echo "data_root: ${DATA_ROOT}"
echo "protocols: ${PROTOCOLS[*]}"
echo "lambda_reserve values: ${WEIGHTS[*]}"
echo "log root: ${LOG_ROOT}"
echo "archive root: ${ARCHIVE_ROOT}"
echo

run_idx=0
for protocol in "${PROTOCOLS[@]}"; do
  for weight in "${WEIGHTS[@]}"; do
    run_idx=$((run_idx + 1))
    weight_tag="${weight//./p}"
    run_tag="${run_idx}_${protocol}_lambda${weight_tag}"
    run_log_dir="${LOG_ROOT}/${run_tag}"
    run_archive_dir="${ARCHIVE_ROOT}/${run_tag}"

    mkdir -p "$run_log_dir" "$run_archive_dir"

    echo "----- [${run_idx}] protocol=${protocol} lambda_reserve=${weight} -----"
    echo "logs -> ${run_log_dir}"
    echo "archive -> ${run_archive_dir}"

    python main.py \
      --config "$CONFIG" \
      --data_root "$DATA_ROOT" \
      --protocol_mode "$protocol" \
      --lambda_reserve "$weight" \
      --log_dir "$run_log_dir"

    if [ -d "$BASE_SAVE_DIR" ]; then
      cp -a "$BASE_SAVE_DIR" "${run_archive_dir}/result_dir_snapshot"
      printf 'protocol_mode: %s\nlambda_reserve: %s\nsource_config: %s\n' "$protocol" "$weight" "$CONFIG" > "${run_archive_dir}/run_meta.txt"
      echo "snapshot saved -> ${run_archive_dir}/result_dir_snapshot"
    else
      echo "warning: expected result dir not found: ${BASE_SAVE_DIR}"
    fi
    echo
  done
done

echo "===== All runs finished ====="
echo "logs: ${LOG_ROOT}"
echo "snapshots: ${ARCHIVE_ROOT}"
