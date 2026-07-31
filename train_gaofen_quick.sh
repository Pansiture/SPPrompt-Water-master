#!/bin/bash
export PYTHONPATH=/root/autodl-tmp/SPPrompt-Water-master:$PYTHONPATH

cd /root/autodl-tmp/SPPrompt-Water-master

python train/train_promptwaternet_gaofen.py \
    --data_train data/Gaofen_processed_v2/level0/train \
    --promptcp pretrain/sam_vit_b_01ec64.pth \
    --SwintransformerPretrain pretrain/swin_tiny_patch4_window7_224.pth \
    --task_name SPP_Gaofen_v2 \
    --batch_size 6 \
    --num_epochs 50 \
    --lr 0.0001 \
    --num_workers 8 \
    --use_amp
