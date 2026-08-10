#!/usr/bin/env python
"""
SwinTransformer 单图推理：只输出黑白预测 mask（0/255 PNG）。
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from skimage import io
from PIL import Image
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SPP_model.modeling.swin_net import SwinTransformerNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/imgs")
    parser.add_argument("--weights", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/work_dir/Golden_Swin_30epoch_e26_0.8189_2.68/Golden_Swin_30epoch_e26_0.8189_2.68.pth")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/output/swin_pred_masks")
    parser.add_argument("--image_list", type=str, required=True, help="逗号或换行分隔的图片文件名")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = SwinTransformerNet(pretrained=None)
    ckpt = torch.load(args.weights, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
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
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)
        image = torch.from_numpy(img).to(device)

        with torch.no_grad():
            logit = model(image)
            prob = torch.sigmoid(logit)

        prob_np = prob.squeeze().cpu().numpy()
        pred = (prob_np > 0.5).astype(np.uint8) * 255

        out_path = Path(args.output_dir) / f"{stem}_pred.png"
        Image.fromarray(pred, mode='L').save(out_path)
        print(f"Saved: {out_path}")

    print(f"\nAll predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
