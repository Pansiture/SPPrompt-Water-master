#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Project Directory Structure for SPPrompt-Water Baseline Experiments

This file documents the purpose of each directory and key script.
"""

# ============================================================
# 1. DATA
# ============================================================
# data/GLH-water/                    <- 原始 GLH-Water 数据集
#   ├── test/
#   │   ├── img/                     <- 25 张原始高分辨率测试图像 (.jpg)
#   │   └── label/                   <- 25 张对应标签 (.png)
#   ├── train/                       <- 原始训练图像/标签 (200 对)
#   ├── val/                         <- 原始验证图像/标签 (25 对)
#   ├── level0/                      <- 预处理后的 1x 分辨率 (1024x1024 patches)
#   │   ├── train/imgs, gts, prompt_mask_256
#   │   ├── val/imgs, gts, prompt_mask_256
#   │   └── test/imgs, gts, prompt_mask_256
#   ├── level1/                      <- 预处理后的 2x 下采样
#   └── level2/                      <- 预处理后的 4x 下采样
#
# 预处理脚本: pretrain/preprocess_glh_water.py
# 用法: python pretrain/preprocess_glh_water.py --input_dir data/GLH-water --output_dir /autodl-pub/glh_data

# ============================================================
# 2. MODEL DEFINITIONS (核心网络)
# ============================================================
# SPP_model/modeling/
#   ├── prompt_water_net.py          <- SPPrompt-Water (论文主模型)
#   ├── swin_net.py                  <- SwinTransformer + UPerHead (baseline)
#   ├── swin_mmcv.py                 <- Swin Transformer backbone (mmcv 实现)
#   ├── uper_head.py                 <- UPerHead 分割头
#   ├── Unet.py                      <- U-Net baseline (已存在)
#   ├── other_model/                 <- 其他对比模型 (MECNet, MSResNet, etc.)
#   └── ...

# ============================================================
# 3. TRAINING SCRIPTS
# ============================================================
# 根目录下:
#   train_promptwaternet.py          <- 训练 SPPrompt-Water (原论文代码)
#   train_baseline.py                <- 训练 U-Net / SwinTransformer baseline
#                                     用法:
#                                     python train_baseline.py --model_type unet --num_epochs 50
#                                     python train_baseline.py --model_type swin --num_epochs 50
#
# SPP_model/modeling/20241230-2152_train_swintransformer_linux.py
#                                     <- 旧的 SwinTransformer 训练脚本 (参考用)

# ============================================================
# 4. TEST / INFERENCE SCRIPTS
# ============================================================
# 根目录下:
#   test_baseline_patch.py           <- Patch 级别测试 (输出 OA/mIoU/F1 + 分割图)
#                                     用法:
#                                     python test_baseline_patch.py --model_type unet --resume <ckpt>
#                                     python test_baseline_patch.py --model_type swin --resume <ckpt>
#
#   test_baseline_full_image.py      <- 全图滑动窗口测试 (生成论文对比图)
#                                     用法:
#                                     python test_baseline_full_image.py --model_type unet --resume <ckpt>
#                                     python test_baseline_full_image.py --model_type swin --resume <ckpt>
#
# inference/                         <- 原论文的 SPPrompt-Water 推理脚本
#   ├── inference_waterprompt.py     <- 标准滑动窗口推理 (SPPrompt-Water)
#   ├── PyramidPromptSeg_mutilayer.py<- 金字塔渐进式推理
#   └── ...

# ============================================================
# 5. PRETRAINED WEIGHTS
# ============================================================
# pretrain/
#   ├── sam_vit_b_01ec64.pth         <- SAM ViT-B 预训练权重 (用于 prompt module)
#   ├── swin_tiny_patch4_window7_224.pth <- Swin Transformer 预训练权重 (ImageNet)
#   └── preprocess_glh_water.py      <- 数据预处理脚本

# ============================================================
# 6. WORK DIRECTORY (训练输出)
# ============================================================
# work_dir/
#   ├── UCFSP_GLH-20260714-1728/     <- SPPrompt-Water 训练输出
#   ├── level0_40epoch_e35_0.8514_Valscore2.7147_withSwin/  <- 旧实验
#   ├── levelfusion_14epoch_e11_0.8219_valscore2.6524/       <- 旧实验
#   └── ... (新 baseline 训练输出会在这里生成)

# ============================================================
# 7. OUTPUT (测试结果)
# ============================================================
# output/
#   ├── test_patches/                <- patch 级别分割结果
#   │   ├── UNET/                    <- U-Net 分割图
#   │   └── SWIN/                    <- SwinTransformer 分割图
#   └── test_full_images/            <- 全图分割结果 (论文对比图)
#       ├── UNET_xxx_pred.png
#       └── SWIN_xxx_pred.png

# ============================================================
# 8. DATA LOADER
# ============================================================
# RSDataloader.py                    <- PromptDataset_GID5 等数据加载类

# ============================================================
# 9. QUICK START (推荐执行顺序)
# ============================================================
# Step 1: 训练 U-Net baseline
#   python train_baseline.py --model_type unet --num_epochs 50 --batch_size 4
#
# Step 2: 训练 SwinTransformer baseline
#   python train_baseline.py --model_type swin --num_epochs 50 --batch_size 4
#
# Step 3: Patch 级别测试 (快速指标)
#   python test_baseline_patch.py --model_type unet --resume work_dir/UNET_GLH-xxx/best_model_xxx.pth
#   python test_baseline_patch.py --model_type swin --resume work_dir/SWIN_GLH-xxx/best_model_xxx.pth
#
# Step 4: 全图测试 (生成论文对比图)
#   python test_baseline_full_image.py --model_type unet --resume <ckpt>
#   python test_baseline_full_image.py --model_type swin --resume <ckpt>
