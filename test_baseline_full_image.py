#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full-image test script for baseline models (U-Net and SwinTransformer)
using sliding-window inference on high-resolution test images.
Saves full segmentation maps suitable for paper figures (TGRS).
"""
import argparse
import os
import numpy as np
import torch
from skimage import io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import glob

from SPP_model.modeling.other_model.Unet import UNet
from SPP_model.modeling.swin_net import SwinTransformerNet


def calculate_metrics(pred, label, num_classes=2):
    """Calculate mIoU, OA (Accuracy), and F1-score for a single image."""
    pred = pred.squeeze().flatten()
    label = label.squeeze().flatten()
    cm = confusion_matrix(label, pred, labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = intersection / (union + 1e-6)
    miou = np.nanmean(iou)
    precision = intersection / (cm.sum(axis=0) + 1e-6)
    recall = intersection / (cm.sum(axis=1) + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy = intersection.sum() / (cm.sum() + 1e-6)
    return miou, accuracy, np.mean(f1)


def sliding_window_predict(model, image, window_size=1024, stride=1024, device='cuda'):
    """
    Sliding-window inference for a large image.
    image: numpy array (H, W, 3), uint8
    Returns: (H, W) binary mask, uint8 (0 or 255)
    """
    h, w = image.shape[:2]
    # Pad to multiples of window_size
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    image_padded = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    h_pad, w_pad = image_padded.shape[:2]

    pred_sum = np.zeros((h_pad, w_pad), dtype=np.float32)
    count = np.zeros((h_pad, w_pad), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        # Calculate number of windows for tqdm
        ny = (h_pad - window_size) // stride + 1
        nx = (w_pad - window_size) // stride + 1
        total = ny * nx
        pbar = tqdm(total=total, desc="Sliding window")
        for y in range(0, h_pad - window_size + 1, stride):
            for x in range(0, w_pad - window_size + 1, stride):
                patch = image_padded[y:y+window_size, x:x+window_size]
                patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
                pred = torch.sigmoid(model(patch_tensor)).squeeze().cpu().numpy()
                pred_sum[y:y+window_size, x:x+window_size] += pred
                count[y:y+window_size, x:x+window_size] += 1
                pbar.update(1)
        pbar.close()

    pred_avg = pred_sum / (count + 1e-6)
    pred_mask = (pred_avg > 0.5).astype(np.uint8) * 255
    # Crop back to original size
    return pred_mask[:h, :w]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["unet", "swin"], required=True)
    parser.add_argument("--img_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/test/img")
    parser.add_argument("--label_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/test/label")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./output/test_full_images")
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--swin_pretrain", type=str,
                        default="./pretrain/swin_tiny_patch4_window7_224.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- load model ----
    if args.model_type == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model_type == "swin":
        model = SwinTransformerNet(pretrained=args.swin_pretrain).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    checkpoint = torch.load(args.resume, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.resume}")

    # ---- discover image/label pairs ----
    img_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff")
    img_files = []
    for f in sorted(os.listdir(args.img_dir)):
        if f.lower().endswith(img_exts):
            img_files.append(f)

    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_type.upper()
    results = []

    for img_name in tqdm(img_files, desc="Images"):
        img_path = os.path.join(args.img_dir, img_name)
        # Match label by basename (remove extension)
        base_name = os.path.splitext(img_name)[0]
        label_path = None
        for ext in (".png", ".jpg", ".tif"):
            cand = os.path.join(args.label_dir, base_name + ext)
            if os.path.isfile(cand):
                label_path = cand
                break
        if label_path is None:
            print(f"Warning: label not found for {img_name}, skipping.")
            continue

        image = io.imread(img_path)
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] > 3:
            image = image[:, :, :3]

        label = io.imread(label_path)
        if label.ndim == 3:
            label = label[:, :, 0]
        label = (label > 127).astype(np.uint8)  # binary 0/1

        # Predict
        pred_mask = sliding_window_predict(model, image, args.window_size, args.stride, device)
        pred_binary = (pred_mask > 127).astype(np.uint8)

        # Metrics
        miou, acc, f1 = calculate_metrics(pred_binary, label)
        results.append({
            "name": img_name,
            "mIoU": miou,
            "OA": acc,
            "F1": f1
        })
        print(f"{img_name}: OA={acc:.4f}, mIoU={miou:.4f}, F1={f1:.4f}")

        # Save result image
        out_path = os.path.join(args.output_dir, f"{model_name}_{base_name}_pred.png")
        io.imsave(out_path, pred_mask, check_contrast=False)

    # Overall metrics
    if len(results) > 0:
        avg_miou = np.mean([r["mIoU"] for r in results])
        avg_oa = np.mean([r["OA"] for r in results])
        avg_f1 = np.mean([r["F1"] for r in results])
        print(f"\nOverall {model_name}: OA={avg_oa:.4f}, mIoU={avg_miou:.4f}, F1={avg_f1:.4f}")

        # Save summary
        summary_path = os.path.join(args.output_dir, f"{model_name}_fullimage_summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Checkpoint: {args.resume}\n")
            f.write(f"Image dir: {args.img_dir}\n")
            f.write(f"Label dir: {args.label_dir}\n")
            f.write(f"Window size: {args.window_size}, Stride: {args.stride}\n\n")
            f.write(f"Overall OA: {avg_oa:.4f}  ({avg_oa*100:.2f}%)\n")
            f.write(f"Overall mIoU: {avg_miou:.4f}  ({avg_miou*100:.2f}%)\n")
            f.write(f"Overall F1: {avg_f1:.4f}  ({avg_f1*100:.2f}%)\n\n")
            f.write("Per-image results:\n")
            for r in results:
                f.write(f"  {r['name']}: OA={r['OA']:.4f}, mIoU={r['mIoU']:.4f}, F1={r['F1']:.4f}\n")
        print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
