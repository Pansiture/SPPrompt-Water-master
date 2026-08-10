#!/bin/bash
# ============================================================
# SPPrompt-Water + EDL 多尺度训练启动脚本 (Gaofen_processed_v2)
# 三尺度 level0/1/2 同时训练 (ConcatDataset)
# 用法: bash train/train_edl_gaofen_multiscale.sh
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# 1. 关键路径配置
# ============================================================
DATA_TRAIN="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/train"
DATA_VAL=""                           # 留空则自动将 train 替换为 val

SAM_CKPT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth"
SWIN_CKPT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth"

WORK_DIR="/root/autodl-tmp/SPPrompt-Water-master/work_dir"
TASK_NAME="SPP_Gaofen_EDL_MultiScale"

LOG_DIR="${WORK_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="${LOG_DIR}/train_edl_gaofen_multiscale_${TIMESTAMP}.log"

# ============================================================
# 2. 训练超参数配置
# ============================================================
NUM_EPOCHS=50
BATCH_SIZE=6
VAL_BATCH_SIZE=6
LR=0.0001
WEIGHT_DECAY=0.01
NUM_WORKERS=8
DEVICE="cuda:0"

# ============================================================
# 3. EDL 特有超参数
# ============================================================
REFEREE_WEIGHT=0.1
KL_ANNEAL_RATIO=0.5
REFEREE_INTERVAL=1
WARMUP_EPOCHS=5
KL_SCALE=0.5

# 可选：加载之前 SPPromptWaterNet 的最佳权重作为热启动
# PRETRAIN_CKPT="/path/to/Golden_SPPrompt_13epoch_e9_0.8295_2.7022.pth"
PRETRAIN_CKPT=""

# ============================================================
# 4. 可选开关
# ============================================================
FREEZE_PROMPT="false"
USE_AMP="false"               # 多尺度 + EDL 建议关闭 AMP，避免 nan
USE_WANDB="false"
USE_REFEREE="false"
RESUME=""

# ============================================================
# 5. 打印配置并启动
# ============================================================
{
  echo "========================================"
  echo "  SPPrompt-Water + EDL (Gaofen MultiScale)"
  echo "  Start Time: ${TIMESTAMP}"
  echo "========================================"
  echo "  DATA_TRAIN: ${DATA_TRAIN}"
  echo "  DATA_VAL:   ${DATA_VAL:-<auto>}"
  echo "  BATCH_SIZE: ${BATCH_SIZE}"
  echo "  LR:         ${LR}"
  echo "  USE_AMP:    ${USE_AMP}"
  echo "========================================"
} | tee -a "${LOG_FILE}"

# 构建参数
AMP_ARG=""
if [ "${USE_AMP}" = "true" ]; then AMP_ARG="--use_amp"; fi
RESUME_ARG=""
if [ -n "${RESUME}" ]; then RESUME_ARG="--resume ${RESUME}"; fi
PRETRAIN_ARG=""
if [ -n "${PRETRAIN_CKPT}" ]; then PRETRAIN_ARG="--pretrain_ckpt ${PRETRAIN_CKPT}"; fi

python train/train_edl_gaofen_multiscale.py \
  --data_train "${DATA_TRAIN}" \
  ${VAL_ARG:-} \
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
  --warmup_epochs ${WARMUP_EPOCHS} \
  --kl_scale ${KL_SCALE} \
  ${PRETRAIN_ARG} \
  ${AMP_ARG} \
  ${RESUME_ARG} \
  2>&1 | tee -a "${LOG_FILE}"
