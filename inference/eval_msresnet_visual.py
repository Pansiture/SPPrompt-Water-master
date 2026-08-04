#!/usr/bin/env python
"""
MSResNet 单图推理可视化脚本
输出: 原图 | GT | 预测结果 拼接图
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SPP_model.modeling.other_model.MSResNet import MSResNet


def inference_single(model, img_path, device):
    """对单张图推理，返回预测概率图 (H, W)"""
    img = io.imread(img_path)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, 0).astype(np.float32) / 255.0
    image = torch.from_numpy(img).to(device)

    with torch.no_grad():
        pred = model(image)
        prob = torch.sigmoid(pred)

    prob_np = prob.squeeze().cpu().numpy()
    return prob_np


def save_visualization(img_path, gt_path, pred_prob, save_path, threshold=0.5):
    """
    拼接可视化: [原图, GT, 预测结果]
    """
    img = io.imread(img_path)
    gt = io.imread(gt_path)
    pred_mask = (pred_prob > threshold).astype(np.uint8) * 255

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(img)
    axes[0].set_title("Image")
    axes[0].axis('off')

    axes[1].imshow(gt, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(pred_mask, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title(f"Prediction (thr={threshold})")
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, required=True, help="MSResNet checkpoint path")
    parser.add_argument("--image_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/imgs")
    parser.add_argument("--gt_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/gts")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/SPPrompt-Water-master/output/msresnet_visual")
    parser.add_argument("--image_names", type=str, required=True,
                        help="Semicolon-separated image names (without extension)")
    parser.add_argument("--pretrained_path", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/pretrain/resnet34-b627a593.pth")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    model = MSResNet(in_channels=3, num_classes=1, pretrained_path=args.pretrained_path).to(device)
    print(f"Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("Model loaded.")

    # Parse image names
    names = [n.strip() for n in args.image_names.split(';') if n.strip()]
    print(f"Processing {len(names)} images...")

    for name in names:
        img_path = os.path.join(args.image_folder, name + ".png")
        if not os.path.isfile(img_path):
            print(f"[WARN] Image not found: {img_path}")
            continue

        gt_path = os.path.join(args.gt_folder, name + ".png")
        if not os.path.isfile(gt_path):
            print(f"[WARN] GT not found: {gt_path}")
            continue

        print(f"Processing: {name}")
        pred_prob = inference_single(model, img_path, device)
        save_path = os.path.join(args.output_dir, f"{name}_visual.png")
        save_visualization(img_path, gt_path, pred_prob, save_path)

    print(f"\nAll done. Visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
