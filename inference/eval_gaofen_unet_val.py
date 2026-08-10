#!/usr/bin/env python
"""
推理脚本：使用 UNet 权重在 Gaofen_processed_v2 level0 val 集上逐张推理，
计算每张图的 mIoU / F1 / Acc，汇总成 CSV，并避免 OOM（结果即时落 CPU / 磁盘）。
"""
import os
import sys
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPP_model.modeling.other_model.Unet import UNet


def calculate_single_metrics(pred, gt, num_classes=2):
    """
    pred: numpy array, 2D, binary 0/1
    gt:   numpy array, 2D, binary 0/1
    """
    flat_pred = pred.flatten().astype(int)
    flat_label = gt.flatten().astype(int)
    cm = confusion_matrix(flat_label, flat_pred, labels=list(range(num_classes)))
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = intersection / (union + 1e-6)
    miou = np.nanmean(iou)
    precision = intersection / (cm.sum(axis=0) + 1e-6)
    recall = intersection / (cm.sum(axis=1) + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    accuracy = intersection.sum() / (cm.sum() + 1e-6)
    return float(miou), float(accuracy), float(np.mean(f1))


def save_png(tensor_or_np, path):
    if torch.is_tensor(tensor_or_np):
        arr = tensor_or_np.squeeze().detach().cpu().numpy()
    else:
        arr = np.array(tensor_or_np).squeeze()
    img = (arr * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img, mode='L').save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/imgs")
    parser.add_argument("--gt_folder", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/val/gts")
    parser.add_argument("--resume", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/work_dir/Golden_UNet_10epoch_5epoch_0.8142_2.67/Golden_UNet_10epoch_5epoch_0.8142_2.67.pth")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/output/eval_gaofen_unet_val")
    parser.add_argument("--save_visuals", type=int, default=1, help="1=save pred masks, 0=don't save")
    parser.add_argument("--image_list", type=str, default=None, help="Optional: path to a text file with image names (one per line) to process only those images")
    parser.add_argument("--out_channels", type=int, default=1)
    args = parser.parse_args()

    save_visuals = bool(args.save_visuals)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load model ---
    model = UNet(in_channels=3, out_channels=args.out_channels)
    print(f"Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.to(device).eval()

    img_dir = Path(args.image_folder)
    gt_dir = Path(args.gt_folder)

    # --- 如果指定了 image_list，只处理这些图片 ---
    if args.image_list is not None and os.path.isfile(args.image_list):
        with open(args.image_list, "r") as f:
            target_names = [line.strip() for line in f if line.strip()]
        img_list = []
        for name in target_names:
            found = list(img_dir.glob(name))
            if not found:
                found = list(img_dir.glob(name + ".*"))
            img_list.extend(found)
        img_list = sorted(list(set(img_list)))
        print(f"Image list provided: {len(target_names)} names, found {len(img_list)} images.")
    else:
        img_list = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")])
    print(f"Found {len(img_list)} images to evaluate")

    per_image_results = []
    start_time = time.time()

    # 为了避免 OOM，逐张推理，pred 立即回 CPU / numpy，不累积在 GPU
    for img_path in tqdm(img_list, desc="Evaluating"):
        prefix = img_path.stem

        # --- Load image ---
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0).astype(np.float32) / 255.0  # match training: divided by 255
        image = torch.from_numpy(img).to(device)

        with torch.no_grad():
            logit = model(image)
            prob = torch.sigmoid(logit)

        # 立即回 CPU 并转为 numpy，释放 GPU 显存
        prob_np = prob.squeeze().cpu().numpy()

        # 清空当前图的显存缓存
        del image, logit, prob
        torch.cuda.empty_cache()

        # --- Load GT ---
        gt_path = gt_dir / (prefix + ".png")
        if not gt_path.exists():
            gt_path = gt_dir / (prefix + ".tif")
        if not gt_path.exists():
            print(f"Warning: GT not found for {prefix}, skip")
            continue

        gt = io.imread(gt_path)
        gt = (gt > 0).astype(np.uint8)

        # Resize prob to GT size if needed
        if prob_np.shape != gt.shape:
            prob_t = torch.from_numpy(prob_np).unsqueeze(0).unsqueeze(0).float()
            prob_t = F.interpolate(prob_t, size=gt.shape, mode='bilinear', align_corners=False)
            prob_np = prob_t.squeeze().cpu().numpy()

        pred_binary = (prob_np > 0.5).astype(np.uint8)

        miou, acc, f1 = calculate_single_metrics(pred_binary, gt)

        per_image_results.append({
            "image": prefix,
            "mIoU": round(miou, 6),
            "Acc": round(acc, 6),
            "F1": round(f1, 6)
        })

        if save_visuals:
            save_png(pred_binary, Path(args.output_dir) / f"{prefix}_pred.png")

    elapsed = time.time() - start_time

    # --- 汇总总体指标 ---
    avg_miou = float(np.mean([r["mIoU"] for r in per_image_results]))
    avg_acc = float(np.mean([r["Acc"] for r in per_image_results]))
    avg_f1 = float(np.mean([r["F1"] for r in per_image_results]))

    print("=" * 60)
    print("Evaluation Complete")
    print(f"Images: {len(per_image_results)}")
    print(f"Time: {elapsed:.2f} s ({elapsed / len(per_image_results):.2f} s/img)")
    print(f"Avg mIoU: {avg_miou:.4f}")
    print(f"Avg Acc:  {avg_acc:.4f}")
    print(f"Avg F1:   {avg_f1:.4f}")
    print("=" * 60)

    # --- 保存单图 CSV ---
    per_image_csv = os.path.join(args.output_dir, "per_image_metrics.csv")
    with open(per_image_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mIoU", "Acc", "F1"])
        for r in per_image_results:
            w.writerow([r["image"], r["mIoU"], r["Acc"], r["F1"]])
    print(f"Per-image metrics saved to: {per_image_csv}")

    # --- 保存汇总 CSV ---
    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mIoU", avg_miou])
        w.writerow(["Acc", avg_acc])
        w.writerow(["F1", avg_f1])
        w.writerow(["num_images", len(per_image_results)])
        w.writerow(["time_seconds", round(elapsed, 2)])
    print(f"Summary metrics saved to: {csv_path}")


if __name__ == "__main__":
    main()
