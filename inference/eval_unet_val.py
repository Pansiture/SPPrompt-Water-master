#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UNet validation evaluation script.
Runs patch-level inference on the val set, saves per-image predictions,
and outputs per-image metrics (mIoU, Acc, F1) plus summary CSV.
"""
import argparse
import csv
import os
import sys
import logging
from pathlib import Path

# OOM protection: allow expandable segments and frequent cache clear
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from skimage import io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPP_model.modeling.other_model.Unet import UNet
from RSDataloader import PromptDataset_GID5


def cm_from_flats(pred_flat, label_flat, num_classes=2):
    """Compute confusion matrix from flattened arrays (int labels)."""
    mask = (label_flat >= 0) & (label_flat < num_classes)
    return confusion_matrix(label_flat[mask], pred_flat[mask], labels=list(range(num_classes)))


def metrics_from_cm(cm):
    """mIoU, OA, F1 from confusion matrix."""
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = intersection / np.maximum(union, 1e-6)
    miou = np.nanmean(iou)
    precision = intersection / np.maximum(cm.sum(axis=0), 1e-6)
    recall = intersection / np.maximum(cm.sum(axis=1), 1e-6)
    f1 = 2 * (precision * recall) / np.maximum(precision + recall, 1e-6)
    accuracy = intersection.sum() / np.maximum(cm.sum(), 1e-6)
    return miou, accuracy, np.nanmean(f1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_val", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/val",
                        help="Path to validation patch data")
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to UNet checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./output/unet_val_eval",
                        help="Directory to save predictions and CSVs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "eval_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("UNet Val Evaluation")
    logger.info("=" * 60)
    logger.info("Checkpoint: %s", args.resume)
    logger.info("Val Data:   %s", args.data_val)
    logger.info("Output:     %s", args.output_dir)
    logger.info("=" * 60)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    model = UNet(in_channels=3, out_channels=1).to(device)
    checkpoint = torch.load(args.resume, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    logger.info("Loaded checkpoint: %s", args.resume)
    model.eval()

    dataset = PromptDataset_GID5(args.data_val)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    logger.info("Number of val samples: %d", len(dataset))

    pred_dir = os.path.join(args.output_dir, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    per_image_results = []
    all_preds = []
    all_gts = []

    with torch.no_grad():
        for images, labels, _, img_names in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device).float()

            preds = model(images)
            pred_probs = torch.sigmoid(preds)
            pred_binary = (pred_probs > 0.5).float()

            # Per-image metrics and save
            for i in range(pred_binary.shape[0]):
                pred_img = pred_binary[i].squeeze().cpu().numpy()
                gt_img = labels[i].squeeze().cpu().numpy()

                miou, acc, f1 = calculate_single_metrics(pred_img, gt_img)
                name = img_names[i] if isinstance(img_names[i], str) else str(img_names[i])
                per_image_results.append({
                    "image": name,
                    "mIoU": miou,
                    "Acc": acc,
                    "F1": f1
                })

                # Save prediction image
                pred_uint8 = (pred_img * 255).astype(np.uint8)
                out_path = os.path.join(pred_dir, name)
                io.imsave(out_path, pred_uint8, check_contrast=False)

                # Accumulate for overall metrics (CPU to avoid OOM)
                all_preds.append(pred_img)
                all_gts.append(gt_img)

            # Clear cache after each batch to prevent fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ---- overall metrics ----
    all_preds = np.stack(all_preds, axis=0)
    all_gts = np.stack(all_gts, axis=0)
    overall_miou, overall_acc, overall_f1 = calculate_single_metrics(all_preds, all_gts)

    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("Images: %d", len(per_image_results))
    logger.info("mIoU: %.4f", overall_miou)
    logger.info("Acc:  %.4f", overall_acc)
    logger.info("F1:   %.4f", overall_f1)
    logger.info("=" * 60)

    # ---- save per-image CSV ----
    per_image_csv = os.path.join(args.output_dir, "per_image_metrics.csv")
    with open(per_image_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mIoU", "Acc", "F1"])
        for r in per_image_results:
            w.writerow([r["image"], r["mIoU"], r["Acc"], r["F1"]])
    logger.info("Per-image metrics saved to: %s", per_image_csv)

    # ---- save summary CSV ----
    summary_csv = os.path.join(args.output_dir, "metrics.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mIoU", overall_miou])
        w.writerow(["Acc", overall_acc])
        w.writerow(["F1", overall_f1])
        w.writerow(["num_images", len(per_image_results)])
    logger.info("Summary metrics saved to: %s", summary_csv)


if __name__ == "__main__":
    main()
