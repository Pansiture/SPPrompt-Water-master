#!/bin/bash
export PYTHONPATH=/root/autodl-tmp/SPPrompt-Water-master:$PYTHONPATH

# Kill any existing training process to free GPU memory
pkill -f "train_promptwaternet_gaofen.py" || true
sleep 2

cd /root/autodl-tmp/SPPrompt-Water-master

# Set memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python train/train_promptwaternet_gaofen.py \
    --data_train data/Gaofen_processed_v2/level0/train \
    --promptcp pretrain/sam_vit_b_01ec64.pth \
    --SwintransformerPretrain pretrain/swin_tiny_patch4_window7_224.pth \
    --task_name SPP_Gaofen_v2_multiscale \
    --batch_size 8 \
    --num_epochs 50 \
    --lr 0.0001 \
    --num_workers 4 \
    --use_amp
