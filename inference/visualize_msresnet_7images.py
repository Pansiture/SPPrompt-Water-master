#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Visualize MSResNet inference on 7 selected images from GLH-Water level0 val.
"""
import os
import sys
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from SPP_model.modeling.other_model.MSResNet import MSResNet

join = os.path.join


def load_model(checkpoint_path, device):
    model = MSResNet(in_channels=3, num_classes=1, pretrained=False).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess(image_np):
    """HWC -> CHW, normalize to [0,1], to tensor."""
    if image_np.max() > 1.0:
        image_np = image_np.astype(np.float32) / 255.0
    image_tensor = torch.from_numpy(image_np.transpose(2, 0, 1)).float().unsqueeze(0)
    return image_tensor


def inference(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        pred = model(image_tensor)
        pred = torch.sigmoid(pred)
    pred = pred.squeeze().cpu().numpy()
    return pred


def visualize(image_np, gt_np, pred_np, save_path, threshold=0.5):
    """Plot original image, GT mask, prediction mask, and overlay."""
    pred_mask = (pred_np > threshold).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    # Original image
    axes[0].imshow(image_np)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Ground Truth
    axes[1].imshow(gt_np, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Ground Truth")
    axes[1].axis("off")

    # Prediction
    axes[2].imshow(pred_mask, cmap="gray", vmin=0, vmax=1)
    axes[2].set_title(f"Prediction (thr={threshold})")
    axes[2].axis("off")

    # Overlay: original + red mask for prediction
    overlay = image_np.copy()
    if overlay.max() <= 1.0:
        overlay = (overlay * 255).astype(np.uint8)
    pred_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    pred_rgb[..., 0] = (pred_mask * 255).astype(np.uint8)  # red channel
    overlay = (overlay * 0.6 + pred_rgb * 0.4).astype(np.uint8)
    axes[3].imshow(overlay)
    axes[3].set_title("Overlay (Prediction)")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/work_dir/GLH_MSResNet_9epoch_e3_0.7595_2.5194/GLH_MSResNet_9epoch_e3_0.7595_2.5194.pth")
    parser.add_argument("--val_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/val")
    parser.add_argument("--out_dir", type=str, default="./visualize_results_7images")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # 7 images from screenshot
    target_images = [
        "04_p134_v0.png",
        "106_p59_v1.png",
        "106_p93_v2.png",
        "116_p125_v2.png",
        "118_p58_v0.png",
        "137_p55_v0.png",
        "143_p33_v2.png",
    ]

    img_dir = join(args.val_dir, "imgs")
    gt_dir = join(args.val_dir, "gts")

    model = load_model(args.checkpoint, device)

    for img_name in target_images:
        img_path = join(img_dir, img_name)
        gt_path = join(gt_dir, img_name)

        if not os.path.exists(img_path):
            print(f"[SKIP] Image not found: {img_path}")
            continue
        if not os.path.exists(gt_path):
            print(f"[SKIP] GT not found: {gt_path}")
            continue

        image = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("L"))
        gt = (gt > 127).astype(np.float32)

        image_tensor = preprocess(image)
        pred = inference(model, image_tensor, device)

        # resize pred to original image size if needed
        if pred.shape != image.shape[:2]:
            pred_pil = Image.fromarray((pred * 255).astype(np.uint8))
            pred_pil = pred_pil.resize((image.shape[1], image.shape[0]), Image.BILINEAR)
            pred = np.array(pred_pil).astype(np.float32) / 255.0

        save_path = join(args.out_dir, f"vis_{img_name}")
        visualize(image, gt, pred, save_path, threshold=args.threshold)

    print(f"\nAll visualizations saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
