#!/bin/bash
# SPPrompt-WaterNet Training Script for Gaofen_processed_v2 Dataset
# All important paths and hyperparameters are explicitly specified

cd /root/autodl-tmp/SPPrompt-Water-master

# Environment
export PYTHONPATH=/root/autodl-tmp/SPPrompt-Water-master:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Dataset paths
DATA_TRAIN="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/train"
DATA_VAL="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val"

# Pretrained weights
SAM_CHECKPOINT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth"
SWIN_CHECKPOINT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth"

# Training hyperparameters
TASK_NAME="SPP_Gaofen_v2"
NUM_EPOCHS=50
BATCH_SIZE=4
VAL_BATCH_SIZE=4
LR=0.0001
WEIGHT_DECAY=0.01
NUM_WORKERS=8

# Output
WORK_DIR="/root/autodl-tmp/SPPrompt-Water-master/work_dir"

# Optional: resume from checkpoint (leave empty to train from scratch)
RESUME=""
# RESUME="/root/autodl-tmp/SPPrompt-Water-master/work_dir/SOME_CHECKPOINT.pth"

# Run training
python train/train_promptwaternet_gaofen.py \
    --data_train ${DATA_TRAIN} \
    --data_val ${DATA_VAL} \
    --promptcp ${SAM_CHECKPOINT} \
    --SwintransformerPretrain ${SWIN_CHECKPOINT} \
    --work_dir ${WORK_DIR} \
    --task_name ${TASK_NAME} \
    --num_epochs ${NUM_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --val_batch_size ${VAL_BATCH_SIZE} \
    --lr ${LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --num_workers ${NUM_WORKERS} \
    --device cuda:0 \
    --use_amp \
    --freeze_prompt False \
    ${RESUME:+--resume ${RESUME}}
