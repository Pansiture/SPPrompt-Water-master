#!/usr/bin/env python
"""
SPPromptWaterNetEDL 推理脚本：在 Gaofen val 上逐张推理，
计算每张图的 mIoU / F1 / Acc，输出黑白预测图，汇总成 CSV 和 log。
"""
import os
import sys
import argparse
import csv
import time
import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPP_model.modeling.prompt_water_net_edl import SPPromptWaterNetEDL
from SPP_model.modeling.edl_utils import evidence_to_prob_uncertainty


def calculate_single_metrics(pred, gt, num_classes=2):
    flat_pred = pred.flatten().astype(int)
    flat_label = gt.flatten().astype(int)
    cm = confusion_matrix(flat_label, flat_pred, labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = intersection / (union + 1e-6)
    miou = np.nanmean(iou)
    precision = intersection / (cm.sum(axis=0) + 1e-6)
    recall = intersection / (cm.sum(axis=1) + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy = intersection.sum() / (cm.sum() + 1e-6)
    return float(miou), float(accuracy), float(np.mean(f1))


def save_png(pred_binary, path):
    arr = np.array(pred_binary).squeeze()
    img = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/imgs")
    parser.add_argument("--gt_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/gts")
    parser.add_argument("--prompt_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/prompt_mask_256")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--promptcp", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth")
    parser.add_argument("--swin_pretrained", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_list", type=str, default=None,
                        help="Comma-separated image names to process")
    parser.add_argument("--save_visuals", type=int, default=1)
    args = parser.parse_args()

    save_visuals = bool(args.save_visuals)
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置 log
    log_file = os.path.join(args.output_dir, "inference.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # 加载模型
    model = SPPromptWaterNetEDL(
        promptcheckpoint=args.promptcp,
        swin_pretrained=args.swin_pretrained,
        freeze_prompt=False,
        use_referee=False
    )
    logger.info(f"Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()
    logger.info("Model loaded.")

    img_dir = Path(args.image_folder)
    gt_dir = Path(args.gt_folder)
    prompt_dir = Path(args.prompt_folder)

    # 解析 image_list
    if args.image_list:
        names = [n.strip() for n in args.image_list.split(",") if n.strip()]
        img_list = []
        for name in names:
            stem = Path(name).stem
            p = img_dir / (stem + ".png")
            if not p.exists():
                p = img_dir / (stem + ".tif")
            if p.exists():
                img_list.append(p)
        logger.info(f"Image list: {len(names)} names, found {len(img_list)} images.")
    else:
        img_list = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff", ".png")])
    logger.info(f"Found {len(img_list)} images to evaluate")

    per_image_results = []
    start_time = time.time()

    for img_path in tqdm(img_list, desc="Evaluating"):
        prefix = img_path.stem

        # 加载图像
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32)  # 不除255，与训练一致
        image = torch.from_numpy(img).to(device)

        # 加载 prompt mask
        prompt_path = prompt_dir / (prefix + ".png")
        if not prompt_path.exists():
            prompt_path = prompt_dir / (prefix + ".tif")
        if not prompt_path.exists():
            logger.warning(f"Prompt mask not found for {prefix}, skip")
            continue

        prompt_mask = io.imread(prompt_path)
        prompt_mask = np.expand_dims(prompt_mask, axis=0)  # (1, H, W)
        prompt_mask = np.expand_dims(prompt_mask, axis=0).astype(np.float32) / 255.0  # (1, 1, H, W)
        prompt_mask = torch.from_numpy(prompt_mask).to(device)

        # 推理
        with torch.no_grad():
            evidence = model(image, prompt_mask, return_evidence=True)
            prob, uncertainty = evidence_to_prob_uncertainty(evidence)
            prob_fg = prob[:, 1:2, :, :]  # (1, 1, 1024, 1024)

        prob_np = prob_fg.squeeze().cpu().numpy()

        # 释放显存
        del image, prompt_mask, evidence, prob, uncertainty, prob_fg
        torch.cuda.empty_cache()

        # 加载 GT
        gt_path = gt_dir / (prefix + ".png")
        if not gt_path.exists():
            gt_path = gt_dir / (prefix + ".tif")
        if not gt_path.exists():
            logger.warning(f"GT not found for {prefix}, skip")
            continue

        gt = io.imread(gt_path)
        gt = (gt > 0).astype(np.uint8)

        # Resize prob 到 GT 尺寸
        if prob_np.shape != gt.shape:
            prob_t = torch.from_numpy(prob_np).unsqueeze(0).unsqueeze(0).float()
            prob_t = F.interpolate(prob_t, size=gt.shape, mode='bilinear', align_corners=False)
            prob_np = prob_t.squeeze().cpu().numpy()

        pred_binary = (prob_np > 0.5).astype(np.uint8)

        miou, acc, f1 = calculate_single_metrics(pred_binary, gt)

        per_image_results.append({
            "image": prefix,
            "mIoU": round(miou, 6),
            "Acc": round(acc, 6),
            "F1": round(f1, 6)
        })

        logger.info(f"{prefix}: mIoU={miou:.4f}, Acc={acc:.4f}, F1={f1:.4f}")

        if save_visuals:
            save_png(pred_binary, Path(args.output_dir) / f"{prefix}_pred.png")

    elapsed = time.time() - start_time

    # 汇总
    avg_miou = float(np.mean([r["mIoU"] for r in per_image_results]))
    avg_acc = float(np.mean([r["Acc"] for r in per_image_results]))
    avg_f1 = float(np.mean([r["F1"] for r in per_image_results]))

    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info(f"Images: {len(per_image_results)}")
    logger.info(f"Time: {elapsed:.2f} s ({elapsed / len(per_image_results):.2f} s/img)")
    logger.info(f"Avg mIoU: {avg_miou:.4f}")
    logger.info(f"Avg Acc:  {avg_acc:.4f}")
    logger.info(f"Avg F1:   {avg_f1:.4f}")
    logger.info("=" * 60)

    # 保存单图 CSV
    per_image_csv = os.path.join(args.output_dir, "per_image_metrics.csv")
    with open(per_image_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mIoU", "Acc", "F1"])
        for r in per_image_results:
            w.writerow([r["image"], r["mIoU"], r["Acc"], r["F1"]])
    logger.info(f"Per-image metrics saved to: {per_image_csv}")

    # 保存汇总 CSV
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mIoU", avg_miou])
        w.writerow(["Acc", avg_acc])
        w.writerow(["F1", avg_f1])
        w.writerow(["num_images", len(per_image_results)])
        w.writerow(["time_seconds", round(elapsed, 2)])
    logger.info(f"Summary metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
