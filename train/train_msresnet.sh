#!/bin/bash
# ============================================================
# MSResNet 训练启动脚本 (GID 数据集, 单尺度 level0)
# 修复: 关闭 AMP, 降低 batch_size 和 lr 避免 NaN
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "  MSResNet Training on Gaofen (level0)"
echo "========================================"

python train/train_msresnet_gaofen.py \
  --data_train "/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/train" \
  --data_val "/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val" \
  --task_name "MSResNet_GID" \
  --num_epochs 50 \
  --batch_size 8 \
  --val_batch_size 8 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --num_workers 8 \
  --work_dir "/root/autodl-tmp/SPPrompt-Water-master/work_dir"
