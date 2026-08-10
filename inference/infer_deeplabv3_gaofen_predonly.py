#!/usr/bin/env python
"""
DeepLabV3+ 单图推理：只输出黑白预测 mask（0/255 PNG）。
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from skimage import io
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from torchvision.models.segmentation import deeplabv3_resnet101


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/imgs")
    parser.add_argument("--weights", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/work_dir/Golden_DeepLabV3Plus_12epoch_e11_0.8372_2.713/Golden_DeepLabV3Plus_12epoch_e11_0.8372_2.713.pth")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/output/deeplabv3_pred_masks")
    parser.add_argument("--image_list", type=str, required=True, help="逗号或换行分隔的图片文件名（不含路径），例如 img1.png,img2.tif")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = deeplabv3_resnet101(pretrained=False)
    model.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    # pretrained=False 时 aux_classifier 可能为 None，推理用不到，跳过即可
    if model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

    ckpt = torch.load(args.weights, map_location=device)
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    # strict=False 忽略 aux_classifier 等不匹配的参数
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"Loaded weights from {args.weights}")

    img_dir = Path(args.image_folder)
    names = [n.strip() for n in args.image_list.replace(',', '\n').split('\n') if n.strip()]

    for name in names:
        stem = Path(name).stem
        img_path = img_dir / (stem + ".png")
        if not img_path.exists():
            img_path = img_dir / (stem + ".tif")
        if not img_path.exists():
            img_path = img_dir / (stem + ".tiff")
        if not img_path.exists():
            print(f"Warning: image not found for {name}, skip")
            continue

        img = io.imread(img_path)
        orig_h, orig_w = img.shape[:2]
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        # 动态归一化：若最大值大于 1，则认为是 0-255 范围
        if img.max() > 1.0:
            img = img / 255.0
        img = np.expand_dims(img, 0)
        image = torch.from_numpy(img).to(device)

        with torch.no_grad():
            pred = model(image)
            if isinstance(pred, dict):
                pred = pred['out']
            prob = torch.sigmoid(pred)
            # 插值回原始尺寸
            prob = F.interpolate(prob, size=(orig_h, orig_w), mode='bilinear', align_corners=False)

        prob_np = prob.squeeze().cpu().numpy()
        pred_mask = (prob_np > 0.5).astype(np.uint8) * 255

        out_path = Path(args.output_dir) / f"{stem}_pred.png"
        Image.fromarray(pred_mask, mode='L').save(out_path)
        print(f"Saved: {out_path}")

    print(f"\nAll predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
