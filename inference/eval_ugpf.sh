#!/bin/bash
# ============================================================
# UGPF 推理 + 指标评估脚本
# 输出：mIoU, Acc, F1 + 日志 + 可视化结果
# ============================================================

set -euo pipefail

cd /root/autodl-tmp/SPPrompt-Water-master
export PYTHONPATH="/root/autodl-tmp/SPPrompt-Water-master:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
# 1. 配置
# ============================================================
EDL_CKPT="/root/autodl-tmp/SPPrompt-Water-master/work_dir/GID_EDL_15epoch_e9_0.8043_Valscore2.6115/GID_EDL_15epoch_e9_0.8043_Valscore2.6115.pth"
SAM_CKPT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth"
SWIN_CKPT="/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth"

# 数据路径（GLH-water level0 val 集）
IMAGE_FOLDER="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/val/imgs"
GT_FOLDER="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/val/gts"
PROMPT_FOLDER="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/val/prompt_mask_256"

OUTPUT_DIR="./output/ugpf_eval"
UNCERTAINTY_THRESHOLD=0.5
FUSE_SCALES="true"
SAVE_VISUALS="true"

# ============================================================
# 2. 检查
# ============================================================
for p in "$EDL_CKPT" "$IMAGE_FOLDER" "$GT_FOLDER" "$PROMPT_FOLDER"; do
    if [ ! -e "$p" ]; then
        echo "[ERROR] Not found: $p"
        exit 1
    fi
done

echo "========================================"
echo "  UGPF Evaluation"
echo "========================================"
echo "  Checkpoint:   $EDL_CKPT"
echo "  Images:       $IMAGE_FOLDER"
echo "  GT:           $GT_FOLDER"
echo "  Prompt:       $PROMPT_FOLDER"
echo "  Output:       $OUTPUT_DIR"
echo "  UGPF:         $FUSE_SCALES"
echo "  Save visuals: $SAVE_VISUALS"
echo "========================================"

# ============================================================
# 3. 启动评估
# ============================================================
python inference/eval_ugpf.py \
  --resume "$EDL_CKPT" \
  --SwintransformerPretrain "$SWIN_CKPT" \
  --promptcp "$SAM_CKPT" \
  --image_folder "$IMAGE_FOLDER" \
  --gt_folder "$GT_FOLDER" \
  --prompt_folder "$PROMPT_FOLDER" \
  --output_dir "$OUTPUT_DIR" \
  --uncertainty_threshold "$UNCERTAINTY_THRESHOLD" \
  --fuse_scales "$FUSE_SCALES" \
  --save_visuals "$SAVE_VISUALS"

echo ""
echo "========================================"
echo "  Done. Check:"
echo "  - $OUTPUT_DIR/eval_log.txt"
echo "  - $OUTPUT_DIR/metrics.csv"
echo "========================================"
