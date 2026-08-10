#!/bin/bash
# ============================================================
# SwinTransformer 训练启动脚本 (Gaofen 数据集, level0)
# 用法: bash train/train_swin_gaofen.sh
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# 训练
# ============================================================
echo "========================================"
echo "  SwinTransformer Training on Gaofen (level0)"
echo "========================================"

python train/train_swin_gaofen.py \
  --data_train "/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/train" \
  --data_val "/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val" \
  --task_name "Swin_Gaofen" \
  --swin_pretrained "/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth" \
  --num_epochs 50 \
  --batch_size 8 \
  --val_batch_size 8 \
  --lr 1e-3 \
  --num_workers 8 \
  --work_dir "/root/autodl-tmp/SPPrompt-Water-master/work_dir"
