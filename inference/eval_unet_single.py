#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single-image UNet inference for a specific val image."""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from skimage import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SPP_model.modeling.other_model.Unet import UNet
from sklearn.metrics import confusion_matrix


def calculate_single_metrics(pred, label, num_classes=2):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", type=str, required=True, help="Path to input image (e.g. imgs/106_p59_v1.png)")
    parser.add_argument("--mask", type=str, default=None, help="Path to ground-truth mask (gts/xxx.png). If None, uses val/gts/<img_basename>")
    parser.add_argument("--prompt", type=str, default=None, help="Path to prompt mask (prompt_mask_256/xxx.png). If None, uses val/prompt_mask_256/<img_basename>")
    parser.add_argument("--resume", type=str, required=True, help="Path to UNet checkpoint")
    parser.add_argument("--output", type=str, default="./output/single_unet.png", help="Where to save prediction")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=3, out_channels=1).to(device)
    ckpt = torch.load(args.resume, map_location=device)

    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
        print("Loaded 'model' key from checkpoint")
    elif "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
        print("Loaded 'state_dict' key from checkpoint")
    else:
        model.load_state_dict(ckpt)
        print("Loaded raw checkpoint (no wrapper key)")

    print("Checkpoint keys:", list(ckpt.keys())[:10])
    print("Model param sample:", model.final_conv.weight[0, 0, 0, 0].item())

    model.eval()

    img = io.imread(args.img)
    print(f"Image shape: {img.shape}, dtype: {img.dtype}, min/max: {img.min()}/{img.max()}")

    # Infer mask/prompt paths if not provided
    img_basename = os.path.basename(args.img)
    img_dir = os.path.dirname(args.img)
    if args.mask is None:
        # val/imgs/xxx.png -> val/gts/xxx.png
        args.mask = os.path.join(os.path.dirname(img_dir), "gts", img_basename)
    if args.prompt is None:
        args.prompt = os.path.join(os.path.dirname(img_dir), "prompt_mask_256", img_basename)

    mask = io.imread(args.mask)
    prompt = io.imread(args.prompt)
    print(f"Mask shape: {mask.shape}, unique: {np.unique(mask)}")

    # Match training: PromptDataset_GID5 does NOT divide by 255
    img = img.astype(np.float32)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float()
    img_tensor = img_tensor.to(device)

    # UNet only takes image, no prompt needed
    with torch.no_grad():
        pred = model(img_tensor)
        print(f"Raw pred range: {pred.min().item():.4f} ~ {pred.max().item():.4f}")
        pred_prob = torch.sigmoid(pred)
        print(f"Sigmoid prob range: {pred_prob.min().item():.4f} ~ {pred_prob.max().item():.4f}")
        pred_binary = (pred_prob > 0.5).float()
        print(f"Binary positive ratio: {pred_binary.mean().item():.4f}")
        
        # Also try threshold at 0.9 to see if model is just overconfident
        pred_binary_09 = (pred_prob > 0.9).float()
        print(f"Binary (th=0.9) positive ratio: {pred_binary_09.mean().item():.4f}")

    pred_np = pred_binary.squeeze().cpu().numpy()
    pred_np_09 = pred_binary_09.squeeze().cpu().numpy()
    mask_np = (mask > 0).astype(np.uint8)

    miou, acc, f1 = calculate_single_metrics(pred_np, mask_np)
    miou_09, acc_09, f1_09 = calculate_single_metrics(pred_np_09, mask_np)
    print(f"Image:    {img_basename}")
    print(f"Mask:     {args.mask}")
    print(f"Prompt:   {args.prompt}")
    print(f"mIoU (th=0.5): {miou:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
    print(f"mIoU (th=0.9): {miou_09:.4f} | Acc: {acc_09:.4f} | F1: {f1_09:.4f}")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    io.imsave(args.output, (pred_np * 255).astype(np.uint8), check_contrast=False)
    io.imsave(args.output.replace(".png", "_th09.png"), (pred_np_09 * 255).astype(np.uint8), check_contrast=False)
    print(f"Saved prediction (th=0.5) to: {args.output}")
    print(f"Saved prediction (th=0.9) to: {args.output.replace('.png', '_th09.png')}")


if __name__ == "__main__":
    main()
