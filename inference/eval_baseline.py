#!/usr/bin/env python
"""
Baseline (UNet / SwinTransformer) 推理 + 指标评估脚本
支持：批量推理、mIoU/Acc/F1 计算、日志记录、单图指标 CSV
"""
import os
import sys
import argparse
import logging
import time
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPP_model.modeling.other_model.Unet import UNet
from SPP_model.modeling.swin_net import SwinTransformerNet


def calculate_metrics(preds, labels, num_classes=2):
    preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
    miou_list, f1_list, acc_list = [], [], []
    for pred, label in zip(preds, labels):
        pred = pred.squeeze()
        pred = (pred > 0.5).astype(int)
        label = label.squeeze()
        flat_pred = pred.flatten()
        flat_label = label.flatten()
        cm = confusion_matrix(flat_label, flat_pred, labels=list(range(num_classes)))
        intersection = np.diag(cm)
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
        iou = intersection / (union + 1e-6)
        miou = np.nanmean(iou)
        precision = intersection / (cm.sum(axis=0) + 1e-6)
        recall = intersection / (cm.sum(axis=1) + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        accuracy = intersection.sum() / (cm.sum() + 1e-6)
        miou_list.append(miou)
        f1_list.append(np.mean(f1))
        acc_list.append(accuracy)
    return np.mean(miou_list), np.mean(acc_list), np.mean(f1_list)


def save_png(tensor, path):
    tensor = tensor.squeeze().detach().cpu()
    if tensor.ndim == 2:
        img = (tensor * 255).clamp(0, 255).numpy().astype(np.uint8)
        Image.fromarray(img, mode='L').save(path)
    elif tensor.ndim == 3:
        img = (tensor * 255).clamp(0, 255).numpy().astype(np.uint8)
        if img.shape[0] == 3:
            img = img.transpose(1, 2, 0)
        Image.fromarray(img).save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--gt_folder", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True, choices=["unet", "swin"])
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/baseline_eval")
    parser.add_argument("--input_size", type=int, default=512, help="Resize to this size for UNet (0 to disable)")
    parser.add_argument("--out_channels", type=int, default=1, help="UNet out channels (1 for sigmoid, 2 for softmax)")
    parser.add_argument("--save_visuals", type=str, default="true", help="Save prediction masks")
    args = parser.parse_args()

    save_visuals = args.save_visuals.lower() in ("true", "1", "yes", "t")

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "eval_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Baseline Evaluation (%s)", args.model_type.upper())
    logger.info("=" * 60)
    logger.info("Resume: %s", args.resume)
    logger.info("Image: %s", args.image_folder)
    logger.info("GT: %s", args.gt_folder)
    logger.info("Output: %s", args.output_dir)
    logger.info("Input size: %s", args.input_size if args.model_type == "unet" else "original")
    logger.info("Save visuals: %s", save_visuals)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_type == "unet":
        model = UNet(in_channels=3, out_channels=args.out_channels)
    else:
        model = SwinTransformerNet(pretrained=None)

    logger.info("Loading checkpoint: %s", args.resume)
    ckpt = torch.load(args.resume, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()

    img_dir = Path(args.image_folder)
    gt_dir = Path(args.gt_folder)
    img_list = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")])
    logger.info("Found %d images", len(img_list))

    all_preds = []
    all_gts = []
    per_image_results = []
    start_time = time.time()

    for img_path in tqdm(img_list, desc="Evaluating"):
        prefix = img_path.stem
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0
        image = torch.from_numpy(img).to(device)

        if args.model_type == "unet" and args.input_size > 0:
            image = F.interpolate(image, size=(args.input_size, args.input_size), mode='bilinear', align_corners=False)

        with torch.no_grad():
            logit = model(image)
            if args.model_type == "unet":
                if args.out_channels == 1:
                    prob = torch.sigmoid(logit)
                else:
                    prob = torch.softmax(logit, dim=1)[:, 1:2, :, :]
            else:
                # Swin output is already 1 channel logit
                prob = torch.sigmoid(logit)

        # GT
        gt_path = gt_dir / (prefix + ".png")
        if not gt_path.exists():
            gt_path = gt_dir / (prefix + ".tif")
        if gt_path.exists():
            gt = io.imread(gt_path)
            gt = np.expand_dims(gt, axis=0)
            gt = gt / 255.0
        else:
            logger.warning("GT not found for %s, skip", prefix)
            continue

        pred_resized = F.interpolate(prob, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        all_preds.append(pred_resized)
        all_gts.append(torch.from_numpy(gt).float())

        img_miou, img_acc, img_f1 = calculate_metrics(pred_resized, torch.from_numpy(gt).float().unsqueeze(0))
        per_image_results.append({
            "image": prefix,
            "mIoU": img_miou,
            "Acc": img_acc,
            "F1": img_f1
        })

        if save_visuals:
            pred_mask = (pred_resized > 0.5).float()
            save_png(pred_mask, Path(args.output_dir) / f"{prefix}_pred.png")

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    miou, acc, f1 = calculate_metrics(all_preds, all_gts)
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("Images: %d", len(img_list))
    logger.info("Time: %.2f s (%.2f s/img)", elapsed, elapsed / len(img_list))
    logger.info("mIoU: %.4f", miou)
    logger.info("Acc:  %.4f", acc)
    logger.info("F1:   %.4f", f1)
    logger.info("=" * 60)

    per_image_csv = os.path.join(args.output_dir, "per_image_metrics.csv")
    with open(per_image_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mIoU", "Acc", "F1"])
        for r in per_image_results:
            w.writerow([r["image"], r["mIoU"], r["Acc"], r["F1"]])
    logger.info("Per-image metrics saved to: %s", per_image_csv)

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mIoU", miou])
        w.writerow(["Acc", acc])
        w.writerow(["F1", f1])
        w.writerow(["num_images", len(img_list)])
        w.writerow(["time_seconds", elapsed])
    logger.info("Metrics saved to: %s", csv_path)


if __name__ == "__main__":
    main()
