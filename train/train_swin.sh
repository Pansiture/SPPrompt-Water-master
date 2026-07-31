#!/bin/bash
# ============================================================
# SwinTransformer 训练启动脚本
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_baseline.py \
  --model_type "swin" \
  --data_train "/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/train" \
  --data_val "/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val" \
  --task_name "Swin_GID" \
  --swin_pretrained "/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth" \
  --num_epochs 50 \
  --batch_size 8 \
  --val_batch_size 8 \
  --grad_accum_steps 1 \
  --lr 1e-3 \
  --num_workers 8 \
  --work_dir "/root/autodl-tmp/SPPrompt-Water-master/work_dir"
