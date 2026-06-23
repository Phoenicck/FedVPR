#!/bin/bash
# FedVPR Stage-1 reserve visualization example (hard protocol)
set -e
cd "$(dirname "$0")/.." || exit 1

python visualize/tsne_virtual_anchors.py \
  --data_root /workspace/Phoenic/claude0527/FedOSS/datasets/RetinalOCT_Dataset \
  --checkpoint /workspace/Phoenic/claude0527/FedVPR/results/MPretrain-DRetinalOCT-Msoftmax-BResnet18/LR0.0005-K5-U3-Seed0-RsvW15-V3/best_ckpt_Pretrain_known_class_5_unknown_class_3_seed_0.pth \
  --output_dir /workspace/Phoenic/claude0527/FedVPR/visualize/hard_stage1_example \
  --protocol_mode hard \
  --method pca \
  --anchor_source fixed \
  --seed 0 \
  --client_num 8
