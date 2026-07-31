#!/bin/bash
# ============================================================
# UNet 训练启动脚本 (梯度累积: batch=8, accum=2, 等效 batch=16)
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train_baseline.py \
  --model_type "unet" \
  --data_train "/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/train" \
  --data_val "/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val" \
  --task_name "UNet_GID" \
  --num_epochs 50 \
  --batch_size 8 \
  --val_batch_size 8 \
  --grad_accum_steps 2 \
  --lr 5e-4 \
  --input_size 1024 \
  --num_workers 8 \
  --work_dir "/root/autodl-tmp/SPPrompt-Water-master/work_dir"
