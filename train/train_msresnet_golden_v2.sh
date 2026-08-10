#!/bin/bash
# ============================================================
# MSResNet 训练启动脚本 (Golden 数据集, level0) - 修复版
# 强制关闭 AMP，修复 inplace ReLU 导致的 nan
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 关闭 OpenMP 警告
export OMP_NUM_THREADS=1

echo "========================================"
echo "  MSResNet Training on Golden (level0) - FIXED"
echo "========================================"

python train/train_msresnet_golden_v2.py \
  --data_train "/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/train" \
  --data_val "/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val" \
  --task_name "MSResNet_Golden_v2" \
  --num_epochs 50 \
  --batch_size 8 \
  --val_batch_size 8 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --num_workers 8 \
  --grad_clip 1.0 \
  --work_dir "/root/autodl-tmp/SPPrompt-Water-master/work_dir"
