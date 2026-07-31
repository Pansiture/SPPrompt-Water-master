#!/bin/bash
# ============================================================
# UNet 训练启动脚本 (Gaofen 数据集, 单尺度 level0)
# 用法: bash train/train_unet_gaofen.sh
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# 训练
# ============================================================
echo "========================================"
echo "  UNet Training on Gaofen (level0)"
echo "========================================"

python train/train_unet_gaofen.py \
  --model_type "unet" \
  --data_train "/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/train" \
  --data_val "/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val" \
  --task_name "UNet_Gaofen" \
  --num_epochs 50 \
  --batch_size 16 \
  --val_batch_size 16 \
  --lr 5e-4 \
  --input_size 1024 \
  --num_workers 8 \
  --work_dir "/root/autodl-tmp/SPPrompt-Water-master/work_dir"
