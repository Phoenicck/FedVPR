#!/bin/bash
# FedVPR RetinalOCT Stage-1 reserve training script
set -e
cd "$(dirname "$0")/.." || exit 1
python main.py --config ./configs/stage1_retinaoct_hard.yaml
