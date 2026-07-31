#!/bin/bash
# SPPrompt-Water Training Script for GID Dataset
# Usage: bash train.sh

cd /root/autodl-tmp/SPPrompt-Water-master

# ---- Configuration ----
# Data paths (preprocessed GID dataset)
DATA_TRAIN="/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/train"
DATA_VAL=""  # auto-derived from DATA_TRAIN by replacing 'train' with 'val'

# Pretrained weights
SAM_CKPT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth"
SWIN_CKPT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth"

# Training hyperparameters
TASK_NAME="SPP_GID"
NUM_EPOCHS=50
BATCH_SIZE=6
VAL_BATCH_SIZE=6
LR=0.0001
WEIGHT_DECAY=0.01
NUM_WORKERS=8
DEVICE="cuda:0"

# Output
WORK_DIR="/root/autodl-tmp/SPPrompt-Water-master/work_dir"

# Optional: resume from checkpoint
RESUME=""

# Optional: freeze SAM prompt module (default: false, recommended to keep false for fine-tuning)
# Set to "true" only if you want to freeze the SAM encoder/decoder
FREEZE_PROMPT="false"

# Build optional arguments
FREEZE_ARG=""
if [ "$FREEZE_PROMPT" = "true" ]; then
    FREEZE_ARG="--freeze_prompt true"
fi

RESUME_ARG=""
if [ -n "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
fi

# ---- Launch Training ----
python train_promptwaternet.py \
  --data_train "${DATA_TRAIN}" \
  --promptcp "${SAM_CKPT}" \
  --SwintransformerPretrain "${SWIN_CKPT}" \
  --work_dir "${WORK_DIR}" \
  --task_name "${TASK_NAME}" \
  -num_epochs ${NUM_EPOCHS} \
  -batch_size ${BATCH_SIZE} \
  -val_batch_size ${VAL_BATCH_SIZE} \
  -lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_workers ${NUM_WORKERS} \
  --device ${DEVICE} \
  --use_amp \
  ${FREEZE_ARG} \
  ${RESUME_ARG}
