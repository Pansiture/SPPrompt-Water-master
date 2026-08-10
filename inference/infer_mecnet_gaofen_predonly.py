#!/usr/bin/env python
"""
MECNet 单图推理：只输出黑白预测 mask（0/255 PNG）。
支持滑窗分块推理，避免大图 OOM。
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
from SPP_model.modeling.other_model.MECNet import MECNet


def sliding_window_inference(model, image_tensor, patch_size=512, overlap=256, device='cuda'):
    """
    对单张图片 (1, C, H, W) 进行滑窗推理，避免显存溢出。
    overlap: 滑窗重叠像素数。
    """
    _, C, H, W = image_tensor.shape
    if H <= patch_size and W <= patch_size:
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                pred = model(image_tensor)
                prob = torch.sigmoid(pred)
        return prob.float()

    # 计算需要 padding 的尺寸，使其能被 (patch_size - overlap) 整除
    stride = patch_size - overlap
    pad_h = (stride - H % stride) % stride
    pad_w = (stride - W % stride) % stride
    if pad_h > 0 or pad_w > 0:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, H_pad, W_pad = image_tensor.shape
    else:
        H_pad, W_pad = H, W

    prob_map = torch.zeros((1, 1, H_pad, W_pad), device=device)
    count_map = torch.zeros((1, 1, H_pad, W_pad), device=device)

    model.eval()
    with torch.no_grad():
        for y in range(0, H_pad - patch_size + 1, stride):
            for x in range(0, W_pad - patch_size + 1, stride):
                patch = image_tensor[:, :, y:y+patch_size, x:x+patch_size]
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    pred = model(patch)
                    prob = torch.sigmoid(pred)
                prob_map[:, :, y:y+patch_size, x:x+patch_size] += prob.float()
                count_map[:, :, y:y+patch_size, x:x+patch_size] += 1
                # 及时清理缓存
                if device == 'cuda':
                    torch.cuda.empty_cache()

    prob_map /= count_map
    # 去除 padding
    prob_map = prob_map[:, :, :H, :W]
    return prob_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/imgs")
    parser.add_argument("--weights", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/work_dir/Golden_MECNet_10epoch_e6_0.8296_2.6978/Golden_MECNet_10epoch_e6_0.8296_2.6978.pth")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/output/mecnet_pred_masks")
    parser.add_argument("--image_list", type=str, required=True,
                        help="逗号或换行分隔的图片文件名（不含路径），例如 img1.png,img2.tif")
    parser.add_argument("--patch_size", type=int, default=512, help="滑窗块大小")
    parser.add_argument("--overlap", type=int, default=256, help="滑窗重叠像素")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    model = MECNet(visualization=False)
    ckpt = torch.load(args.weights, map_location='cpu')  # 先加载到 CPU 避免瞬间显存占用
    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    model.to(device).eval()
    print(f"Loaded weights from {args.weights}")
    if device.type == 'cuda':
        torch.cuda.empty_cache()

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
        if img.max() > 1.0:
            img = img / 255.0
        img = np.expand_dims(img, 0)
        image = torch.from_numpy(img).to(device)

        prob = sliding_window_inference(model, image, patch_size=args.patch_size, overlap=args.overlap, device=args.device)
        prob_np = prob.squeeze().cpu().numpy()
        pred_mask = (prob_np > 0.5).astype(np.uint8) * 255

        out_path = Path(args.output_dir) / f"{stem}_pred.png"
        Image.fromarray(pred_mask, mode='L').save(out_path)
        print(f"Saved: {out_path}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print(f"\nAll predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
