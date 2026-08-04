#!/bin/bash
# ============================================================
# QTNUnet 训练启动脚本 (GID 数据集, 单尺度 level0)
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "========================================"
echo "  QTNUnet Training on GID (level0)"
echo "========================================"

python train/train_qtnunet.py \
  --data_train "/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/train" \
  --data_val "/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val" \
  --task_name "QTNUnet_GID" \
  --num_epochs 50 \
  --batch_size 2 \
  --val_batch_size 4 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --num_workers 8 \
  --work_dir "/root/autodl-tmp/SPPrompt-Water-master/work_dir"
