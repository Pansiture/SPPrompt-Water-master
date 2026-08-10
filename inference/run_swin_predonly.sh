#!/bin/bash
# Swin 推理：只输出黑白预测 mask（使用 infer_swin_gaofen_predonly.py）
cd /root/autodl-tmp/SPPrompt-Water-master

python inference/infer_swin_gaofen_predonly.py \
  --image_list "101_aug0_v0.png,13_aug0_v0.png,140_aug0_v0.png,175_aug0_v0.png,184_aug0_v0.png" \
  --output_dir /root/autodl-tmp/SPPrompt-Water-master/output/swin_pred_masks
