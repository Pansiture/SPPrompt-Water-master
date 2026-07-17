#!/bin/bash
# train.sh - Unified Training Launcher for UCFSP (Uncertainty-Calibrated Foundation-Scale Progressive) Water Extraction
# Usage: bash train.sh [glh|gid] [epochs] [batch_size]
# Example: bash train.sh glh 50 8

set -e  # 遇到错误立即退出

# ======================== 用户配置区 ========================
# 数据集根目录（根据你的实际路径修改）
DATA_ROOT_GLH="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water"
DATA_ROOT_GID="/root/autodl-tmp/SPPrompt-Water-master/data/GID"

# 预训练权重路径（已根据实际环境配置）
SAM_CHECKPOINT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth"
SWIN_PRETRAIN="/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth"

# 工作目录
WORK_DIR="./work_dir"
EXP_LOG_DIR="./exp_logs"

# 默认超参数
DEFAULT_EPOCHS=50
DEFAULT_BATCH_SIZE=6
DEFAULT_LR=0.0001
DEFAULT_NUM_WORKERS=8
# ============================================================

# 解析参数
DATASET=${1:-glh}
NUM_EPOCHS=${2:-$DEFAULT_EPOCHS}
BATCH_SIZE=${3:-$DEFAULT_BATCH_SIZE}

# 根据数据集选择路径
if [ "$DATASET" == "glh" ] || [ "$DATASET" == "GLH" ]; then
    DATA_TRAIN="$DATA_ROOT_GLH/level0/train"
    TASK_NAME="UCFSP_GLH"
    DATASET_FULL="GLH-Water"
elif [ "$DATASET" == "gid" ] || [ "$DATASET" == "GID" ]; then
    DATA_TRAIN="$DATA_ROOT_GID/level0/train"
    TASK_NAME="UCFSP_GID"
    DATASET_FULL="GID"
else
    echo "[ERROR] Unknown dataset: $DATASET. Supported: glh, gid"
    exit 1
fi

# 创建日志目录
mkdir -p "$EXP_LOG_DIR"
mkdir -p "$WORK_DIR"

# 生成日志文件名（带时间戳）
RUN_ID=$(date +"%Y%m%d-%H%M%S")
LOG_FILE="$EXP_LOG_DIR/${TASK_NAME}_${RUN_ID}.log"

# 设置 PYTHONPATH
export PYTHONPATH=/root/autodl-tmp/SPPrompt-Water-master:$PYTHONPATH

echo "========================================" | tee -a "$LOG_FILE"
echo "  UCFSP Training Launcher" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 打印硬件信息
echo "[Hardware Info]" | tee -a "$LOG_FILE"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)" | tee -a "$LOG_FILE"
echo "  CUDA: $(nvcc --version 2>/dev/null | grep 'release' | awk '{print $5}' || echo 'N/A')" | tee -a "$LOG_FILE"
echo "  PyTorch: $(python -c 'import torch; print(torch.__version__)')" | tee -a "$LOG_FILE"
echo "  Available GPU Memory: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1) MiB" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 打印路径配置
echo "[Path Configuration]" | tee -a "$LOG_FILE"
echo "  Dataset:              $DATASET_FULL" | tee -a "$LOG_FILE"
echo "  Training Data:        $DATA_TRAIN" | tee -a "$LOG_FILE"
echo "  SAM Checkpoint:       $SAM_CHECKPOINT" | tee -a "$LOG_FILE"
echo "  Swin Pretrain:        $SWIN_PRETRAIN" | tee -a "$LOG_FILE"
echo "  Work Directory:       $WORK_DIR" | tee -a "$LOG_FILE"
echo "  Log File:             $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 检查关键文件是否存在
echo "[File Existence Check]" | tee -a "$LOG_FILE"
if [ -f "$SAM_CHECKPOINT" ]; then
    echo "  [OK] SAM checkpoint found: $SAM_CHECKPOINT" | tee -a "$LOG_FILE"
else
    echo "  [WARN] SAM checkpoint NOT found: $SAM_CHECKPOINT" | tee -a "$LOG_FILE"
fi

if [ -f "$SWIN_PRETRAIN" ]; then
    echo "  [OK] Swin pretrain found: $SWIN_PRETRAIN" | tee -a "$LOG_FILE"
else
    echo "  [WARN] Swin pretrain NOT found: $SWIN_PRETRAIN" | tee -a "$LOG_FILE"
fi

if [ -d "$DATA_TRAIN" ]; then
    echo "  [OK] Training data dir exists: $DATA_TRAIN" | tee -a "$LOG_FILE"
    echo "  [INFO] Training samples: $(find $DATA_TRAIN/imgs -type f 2>/dev/null | wc -l)" | tee -a "$LOG_FILE"
else
    echo "  [ERROR] Training data dir NOT found: $DATA_TRAIN" | tee -a "$LOG_FILE"
    exit 1
fi
echo "" | tee -a "$LOG_FILE"

# 打印训练超参数
echo "[Training Hyperparameters]" | tee -a "$LOG_FILE"
echo "  Epochs:               $NUM_EPOCHS" | tee -a "$LOG_FILE"
echo "  Batch Size:           $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Val Batch Size:       $BATCH_SIZE" | tee -a "$LOG_FILE"
echo "  Learning Rate:        $DEFAULT_LR" | tee -a "$LOG_FILE"
echo "  Weight Decay:         0.01" | tee -a "$LOG_FILE"
echo "  Num Workers:          $DEFAULT_NUM_WORKERS" | tee -a "$LOG_FILE"
echo "  Loss:                 0.6*DiceFocal + 0.4*Tversky + 0.5*EDL + 0.3*FM_Consistency" | tee -a "$LOG_FILE"
echo "  AMP:                  Enabled" | tee -a "$LOG_FILE"
echo "  Optimizer:            AdamW + CosineAnnealingLR" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 打印模型配置
echo "[Model Configuration]" | tee -a "$LOG_FILE"
echo "  Backbone:             Swin-Tiny + UPerHead" | tee -a "$LOG_FILE"
echo "  Prompt Module:        SAM ViT-B" | tee -a "$LOG_FILE"
echo "  Uncertainty Head:     Evidential Head (Enabled)" | tee -a "$LOG_FILE"
echo "  FM Consistency:       SAM ViT-B Frozen Encoder (Enabled)" | tee -a "$LOG_FILE"
echo "  UGPF:                 Enabled (Inference)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "========================================" | tee -a "$LOG_FILE"
echo "  Training Started at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 启动训练（使用 tee 同时输出到控制台和日志文件）
python train_promptwaternet.py \
    --data_train="$DATA_TRAIN" \
    --promptcp="$SAM_CHECKPOINT" \
    --SwintransformerPretrain="$SWIN_PRETRAIN" \
    -num_epochs=$NUM_EPOCHS \
    -batch_size=$BATCH_SIZE \
    -val_batch_size=$BATCH_SIZE \
    --num_workers=$DEFAULT_NUM_WORKERS \
    -lr=$DEFAULT_LR \
    -work_dir="$WORK_DIR" \
    -task_name="$TASK_NAME" \
    -use_amp \
    2>&1 | tee -a "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "  Training Finished at $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "  Exit Code:            $EXIT_CODE" | tee -a "$LOG_FILE"
echo "  Best Model Saved to:  $WORK_DIR/$TASK_NAME-*/" | tee -a "$LOG_FILE"
echo "  Full Log:             $LOG_FILE" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

exit $EXIT_CODE
