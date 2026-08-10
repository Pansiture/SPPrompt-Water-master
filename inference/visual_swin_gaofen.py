#!/usr/bin/env python
"""
SwinTransformer 可视化推理脚本：对指定几张图片推理并生成 side-by-side 对比图。
"""
import os
import sys
import argparse
from pathlib import Path
import numpy as np
from skimage import io
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SPP_model.modeling.swin_net import SwinTransformerNet


def overlay_mask(image, mask, color=(1, 0, 0), alpha=0.4):
    """将二值 mask 以指定颜色叠加到 RGB 图像上。"""
    overlay = image.copy().astype(np.float32) / 255.0
    for c in range(3):
        overlay[..., c] = np.where(mask > 0, overlay[..., c] * (1 - alpha) + color[c] * alpha, overlay[..., c])
    return (overlay * 255).clip(0, 255).astype(np.uint8)


def infer_and_visualize(model, img_path, gt_path, device, out_path):
    prefix = img_path.stem
    img = io.imread(img_path)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
    
    img_np = img.copy()
    img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)
    image = torch.from_numpy(img).to(device)
    
    with torch.no_grad():
        logit = model(image)
        prob = torch.sigmoid(logit)
    
    prob_np = prob.squeeze().cpu().numpy()
    pred = (prob_np > 0.5).astype(np.uint8)
    
    gt = io.imread(gt_path)
    gt = (gt > 0).astype(np.uint8)
    
    if pred.shape != gt.shape:
        prob_t = torch.from_numpy(prob_np).unsqueeze(0).unsqueeze(0).float()
        prob_t = F.interpolate(prob_t, size=gt.shape, mode='bilinear', align_corners=False)
        prob_np = prob_t.squeeze().cpu().numpy()
        pred = (prob_np > 0.5).astype(np.uint8)
    
    # 创建三合一图
    h, w = gt.shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图
    axes[0].imshow(img_np)
    axes[0].set_title(f"Image: {prefix}")
    axes[0].axis('off')
    
    # GT overlay
    gt_overlay = overlay_mask(img_np, gt, color=(0, 1, 0), alpha=0.4)
    axes[1].imshow(gt_overlay)
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')
    
    # Pred overlay
    pred_overlay = overlay_mask(img_np, pred, color=(1, 0, 0), alpha=0.4)
    axes[2].imshow(pred_overlay)
    axes[2].set_title("Prediction")
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/imgs")
    parser.add_argument("--gt_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/gts")
    parser.add_argument("--weights", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/work_dir/Golden_Swin_30epoch_e26_0.8189_2.68/Golden_Swin_30epoch_e26_0.8189_2.68.pth")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/output/visual_swin_gaofen")
    parser.add_argument("--image_list", type=str, required=True, help="逗号分隔或换行分隔的图片文件名（不含路径）")
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
    gt_dir = Path(args.gt_folder)
    
    # Parse image list
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
        
        gt_path = gt_dir / (stem + ".png")
        if not gt_path.exists():
            gt_path = gt_dir / (stem + ".tif")
        if not gt_path.exists():
            print(f"Warning: GT not found for {stem}, skip")
            continue
        
        out_path = Path(args.output_dir) / f"{stem}_vis.png"
        infer_and_visualize(model, img_path, gt_path, device, out_path)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
