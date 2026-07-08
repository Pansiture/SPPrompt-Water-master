#!/usr/bin/env python
"""Evaluate pyramid segmentation results against whole-scene labels."""
import os
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.metrics import confusion_matrix
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pred_dir", type=str, required=True, help="Directory with prediction PNGs")
parser.add_argument("--label_dir", type=str, required=True, help="Directory with label PNGs")
parser.add_argument("--suffix", type=str, default="_final_result", help="Prediction filename suffix before .png")
args = parser.parse_args()


def calculate_metrics(pred, label, num_classes=2):
    """Calculate mIoU, accuracy, and F1-score."""
    flat_pred = pred.flatten()
    flat_label = label.flatten()
    cm = confusion_matrix(flat_label, flat_pred, labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = intersection / (union + 1e-6)
    miou = np.mean(iou)
    precision = intersection / (cm.sum(axis=0) + 1e-6)
    recall = intersection / (cm.sum(axis=1) + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy = intersection.sum() / (cm.sum() + 1e-6)
    return miou, accuracy, np.mean(f1), np.mean(f1[1])  # last is F1-water


def main():
    pred_dir = Path(args.pred_dir)
    label_dir = Path(args.label_dir)

    pred_files = sorted([p for p in pred_dir.glob(f"*{args.suffix}.png")])
    if not pred_files:
        print(f"No prediction files found in {pred_dir} with suffix '{args.suffix}'")
        return

    miou_list, acc_list, f1_list, f1w_list = [], [], [], []

    for pred_path in pred_files:
        stem = pred_path.stem.replace(args.suffix, "")
        label_path = label_dir / f"{stem}.png"
        if not label_path.exists():
            print(f"Warning: label not found for {stem}, skipping")
            continue

        pred = np.array(Image.open(pred_path).convert('L'))
        label = np.array(Image.open(label_path).convert('L'))

        # Binarize: 255 -> 1, 0 -> 0
        pred = (pred > 127).astype(np.uint8)
        label = (label > 127).astype(np.uint8)

        miou, acc, f1, f1w = calculate_metrics(pred, label)

        # Free memory immediately for large images
        del pred, label

        miou_list.append(miou)
        acc_list.append(acc)
        f1_list.append(f1)
        f1w_list.append(f1w)

        print(f"{stem}: mIoU={miou:.4f}, Acc={acc:.4f}, F1={f1:.4f}, F1-water={f1w:.4f}")

    print(f"\n{'='*50}")
    print(f"Average over {len(miou_list)} images:")
    print(f"mIoU: {np.mean(miou_list):.4f}")
    print(f"Acc:  {np.mean(acc_list):.4f}")
    print(f"F1:   {np.mean(f1_list):.4f}")
    print(f"F1-water: {np.mean(f1w_list):.4f}")
    print(f"Valscore: {np.mean(miou_list)+np.mean(acc_list)+np.mean(f1_list):.4f}")


if __name__ == "__main__":
    main()
