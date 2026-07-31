#!/bin/bash
# ============================================================
# SwinTransformer 推理 + 指标评估启动脚本
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"

# ============================================================
# 1. 配置（请根据实际情况填写权重路径）
# ============================================================
SWIN_CKPT="/root/autodl-tmp/SPPrompt-Water-master/work_dir/.../swin_best.pth"

IMAGE_FOLDER="/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/imgs"
GT_FOLDER="/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/gts"

OUTPUT_DIR="./output/swin_eval"
SAVE_VISUALS="true"

# ============================================================
# 2. 检查
# ============================================================
for p in "$SWIN_CKPT" "$IMAGE_FOLDER" "$GT_FOLDER"; do
    if [ ! -e "$p" ]; then
        echo "[ERROR] Not found: $p"
        exit 1
    fi
done

echo "========================================"
echo "  SwinTransformer Evaluation"
echo "========================================"
echo "  Checkpoint:   $SWIN_CKPT"
echo "  Images:       $IMAGE_FOLDER"
echo "  GT:           $GT_FOLDER"
echo "  Output:       $OUTPUT_DIR"
echo "========================================"

# ============================================================
# 3. 启动评估
# ============================================================
python eval_baseline.py \
  --model_type "swin" \
  --resume "$SWIN_CKPT" \
  --image_folder "$IMAGE_FOLDER" \
  --gt_folder "$GT_FOLDER" \
  --output_dir "$OUTPUT_DIR" \
  --save_visuals "$SAVE_VISUALS"

echo ""
echo "========================================"
echo "  Done. Check:"
echo "  - $OUTPUT_DIR/eval_log.txt"
echo "  - $OUTPUT_DIR/metrics.csv"
echo "  - $OUTPUT_DIR/per_image_metrics.csv"
echo "========================================"
