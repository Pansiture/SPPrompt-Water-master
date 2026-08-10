#!/usr/bin/env python
"""
DeepLabV3+ 单图推理脚本 - 只输出预测结果图
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from skimage import io
from PIL import Image
from torchvision import models

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_deeplabv3plus_res101(num_classes=1):
    """Build DeepLabV3+ with ResNet-101 backbone."""
    model = models.segmentation.deeplabv3_resnet101(weights=None)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


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
        pred = model(image)['out']
        prob = torch.sigmoid(pred)

    prob_np = prob.squeeze().cpu().numpy()
    return prob_np


def save_pred_mask(pred_prob, save_path, threshold=0.5):
    """保存预测mask为PNG"""
    pred_mask = (pred_prob > threshold).astype(np.uint8) * 255
    Image.fromarray(pred_mask, mode='L').save(save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, required=True, help="DeepLabV3+ checkpoint path")
    parser.add_argument("--image_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/imgs")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-tmp/SPPrompt-Water-master/output/deeplabv3plus_pred")
    parser.add_argument("--image_names", type=str, required=True,
                        help="Semicolon-separated image names (without extension)")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Load model
    model = build_deeplabv3plus_res101(num_classes=1).to(device)

    print(f"Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # Filter out aux_classifier keys if model doesn't have aux_classifier
    filtered_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('aux_classifier.')}
    model.load_state_dict(filtered_state_dict, strict=False)
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

        print(f"Processing: {name}")
        pred_prob = inference_single(model, img_path, device)
        save_path = os.path.join(args.output_dir, f"{name}_pred.png")
        save_pred_mask(pred_prob, save_path)
        print(f"  Saved: {save_path}")

    print(f"\nAll done. Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
