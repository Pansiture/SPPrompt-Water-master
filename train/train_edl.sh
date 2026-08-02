#!/bin/bash
# ============================================================
# SPPrompt-Water + EDL + SAM Referee  Training Script
# Usage: bash train_edl.sh
# ============================================================

set -euo pipefail

# Fix CUDA memory fragmentation (prevents OOM with large attention blocks)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/autodl-tmp/SPPrompt-Water-master

# ============================================================
# 1. 关键路径配置 (请根据实际环境修改)
# ============================================================
# 数据路径 (预处理后的 GID 数据集)
DATA_TRAIN="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/train"
DATA_VAL=""                           # 留空则自动将 train 替换为 val

# 预训练权重路径
SAM_CKPT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth"
SWIN_CKPT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth"

# 输出与工作目录
WORK_DIR="/root/autodl-tmp/SPPrompt-Water-master/work_dir"
TASK_NAME="SPP_GID_EDL"

# 日志目录
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="${LOG_DIR}/train_edl_${TIMESTAMP}.log"

# ============================================================
# 2. 训练超参数配置
# ============================================================
NUM_EPOCHS=40
BATCH_SIZE=12                    # 单步 12，显存估算 ~26GB，安全余量 6GB
VAL_BATCH_SIZE=16                # 验证只 forward，可以更大
LR=0.00002
WEIGHT_DECAY=0.01
NUM_WORKERS=8
DEVICE="cuda:0"

# Gradient accumulation: 1 step => 单步优化，无需累积（显存足够）
GRAD_ACCUM=1

# ============================================================
# 3. EDL 特有超参数
# ============================================================
REFEREE_WEIGHT=0.1
KL_ANNEAL_RATIO=0.3          # 前 50% epoch (20 epochs) 完成 KL 升温，推迟 KL 惩罚让模型先恢复旧性能
REFEREE_INTERVAL=1
WARMUP_EPOCHS=10               # 10 epoch warmup 让 EDL 新层充分学习
KL_SCALE=0.1                   # 恢复 epoch 9 配置：低 KL 惩罚
POS_WEIGHT=20.0                # GLH-water 前景 ~3%，加大前景权重对抗类别不平衡

# 0.82 基线 SPP 模型 (levelfusion 14epoch)，用于热启动 EDL 头
PRETRAIN_CKPT="/root/autodl-tmp/SPPrompt-Water-master/work_dir/levelfusion_14epoch_e11_0.8219_valscore2.6524/best_model_e11_Valscore2.6524.pth"

# ============================================================
# 4. 可选开关
# ============================================================
FREEZE_PROMPT="true"          # 冻结 SAM prompt 模块 (基线 0.82 已优化，backbone 全部冻结)
USE_AMP="True"                # 混合精度训练 (关闭 AMP，避免 cuDNN 错误)
USE_WANDB="false"             # 是否使用 wandb 记录
USE_REFEREE="false"           # 是否启用 SAM 裁判分支 (false=省显存，只保留 EDL 不确定性)
RESUME="/root/autodl-tmp/SPPrompt-Water-master/work_dir/GID_EDL_15epoch_e9_0.8043_Valscore2.6115/GID_EDL_15epoch_e9_0.8043_Valscore2.6115.pth"  # 断点续训路径，从 best epoch 9 (mIoU 0.8043) 恢复

# ============================================================
# 5. 打印完整配置并保存到日志
# ============================================================
{
  echo "========================================"
  echo "  SPPrompt-Water + EDL Training Config"
  echo "  Start Time: ${TIMESTAMP}"
  echo "  Host: $(hostname)"
  echo "  GPU: $(nvidia-smi -L 2>/dev/null | head -n 1 || echo 'N/A')"
  echo "========================================"
  echo ""
  echo "[Paths]"
  echo "  DATA_TRAIN:        ${DATA_TRAIN}"
  echo "  DATA_VAL:          ${DATA_VAL:-<auto-derived>}"
  echo "  SAM_CKPT:          ${SAM_CKPT}"
  echo "  SWIN_CKPT:         ${SWIN_CKPT}"
  echo "  WORK_DIR:          ${WORK_DIR}"
  echo "  TASK_NAME:         ${TASK_NAME}"
  echo "  LOG_FILE:          ${LOG_FILE}"
  echo ""
  echo "[Training Hyperparameters]"
  echo "  NUM_EPOCHS:        ${NUM_EPOCHS}"
  echo "  BATCH_SIZE:        ${BATCH_SIZE}"
  echo "  VAL_BATCH_SIZE:    ${VAL_BATCH_SIZE}"
  echo "  LR:                ${LR}"
  echo "  WEIGHT_DECAY:      ${WEIGHT_DECAY}"
  echo "  NUM_WORKERS:       ${NUM_WORKERS}"
  echo "  DEVICE:            ${DEVICE}"
  echo ""
  echo "[EDL-specific Parameters]"
  echo "  REFEREE_WEIGHT:    ${REFEREE_WEIGHT}"
  echo "  KL_ANNEAL_RATIO:   ${KL_ANNEAL_RATIO}"
  echo "  REFEREE_INTERVAL:  ${REFEREE_INTERVAL}"
  echo "  WARMUP_EPOCHS:     ${WARMUP_EPOCHS}"
  echo "  KL_SCALE:          ${KL_SCALE}"
  echo "  POS_WEIGHT:        ${POS_WEIGHT}"
  echo "  PRETRAIN_CKPT:     ${PRETRAIN_CKPT}"
  echo ""
  echo "[Optional Flags]"
  echo "  FREEZE_PROMPT:     ${FREEZE_PROMPT}"
  echo "  USE_AMP:           ${USE_AMP}"
  echo "  USE_WANDB:         ${USE_WANDB}"
  echo "  USE_REFEREE:       ${USE_REFEREE}"
  echo "  RESUME:            ${RESUME:-<none>}"
  echo ""
  echo "========================================"
  echo "  Launching training ..."
  echo "========================================"
  echo ""
} | tee -a "${LOG_FILE}"

# ============================================================
# 6. 构建可选参数
# ============================================================
FREEZE_ARG=""
if [ "${FREEZE_PROMPT}" = "true" ]; then
    FREEZE_ARG="--freeze_prompt true"
fi

AMP_ARG=""
if [ "${USE_AMP}" = "true" ]; then
    AMP_ARG="--use_amp"
fi

WANDB_ARG=""
if [ "${USE_WANDB}" = "true" ]; then
    WANDB_ARG="--use_wandb true"
fi

REFEREE_ARG=""
if [ "${USE_REFEREE}" = "true" ]; then
    REFEREE_ARG="--use_referee true"
fi

RESUME_ARG=""
if [ -n "${RESUME}" ]; then
    RESUME_ARG="--resume ${RESUME}"
fi

# 如果 DATA_VAL 为空，不传 --data_val，Python 内部会自动推导
VAL_ARG=""
if [ -n "${DATA_VAL}" ]; then
    VAL_ARG="--data_val ${DATA_VAL}"
fi

# ============================================================
# 7. 启动训练 (Python 输出 + bash 配置统一写入 LOG_FILE)
# ============================================================
python train/train_edl.py \
  --data_train "${DATA_TRAIN}" \
  ${VAL_ARG} \
  --promptcp "${SAM_CKPT}" \
  --SwintransformerPretrain "${SWIN_CKPT}" \
  --work_dir "${WORK_DIR}" \
  --task_name "${TASK_NAME}" \
  --num_epochs ${NUM_EPOCHS} \
  --batch_size ${BATCH_SIZE} \
  --val_batch_size ${VAL_BATCH_SIZE} \
  --lr ${LR} \
  --weight_decay ${WEIGHT_DECAY} \
  --num_workers ${NUM_WORKERS} \
  --device ${DEVICE} \
  --referee_weight ${REFEREE_WEIGHT} \
  --kl_anneal_ratio ${KL_ANNEAL_RATIO} \
  --referee_interval ${REFEREE_INTERVAL} \
  --warmup_epochs ${WARMUP_EPOCHS} \
  --kl_scale ${KL_SCALE} \
  --pos_weight ${POS_WEIGHT} \
  --seg_anchor_weight 1.0 \
  --grad_accum ${GRAD_ACCUM} \
  --pretrain_ckpt "${PRETRAIN_CKPT}" \
  ${REFEREE_ARG} \
  ${FREEZE_ARG} \
  ${AMP_ARG} \
  ${WANDB_ARG} \
  ${RESUME_ARG} \
  2>&1 | tee -a "${LOG_FILE}"

# ============================================================
# 8. 结束信息
# ============================================================
{
  echo ""
  echo "========================================"
  echo "  Training finished at $(date +"%Y%m%d-%H%M%S")"
  echo "  Full log saved to: ${LOG_FILE}"
  