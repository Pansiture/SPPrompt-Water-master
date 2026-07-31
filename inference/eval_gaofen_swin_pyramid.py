#!/usr/bin/env python
"""
使用 SwinTransformer + 多尺度金字塔 在 Gaofen_processed_v2 上推理评估。
支持 level0/level1/level2 金字塔输入，计算每张图的 mIoU/Acc/F1，输出 CSV。
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
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPP_model.modeling.swin_net import SwinTransformerNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_test", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/test",
                        help="Path to test data (with imgs/ and gts/ subfolders)")
    parser.add_argument("--level1_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level1/test",
                        help="Path to level1 downsampled data")
    parser.add_argument("--level2_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level2/test",
                        help="Path to level2 downsampled data")
    parser.add_argument("--resume", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/work_dir/GID_Swin_28epoch_20epoch_0.6885_Valscore2.3706/GID_Swin_28epoch_20epoch_0.6885_Valscore2.3706.pth")
    parser.add_argument("--output_dir", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/output/eval_gaofen_swin_pyramid")
    parser.add_argument("--use_pyramid", type=int, default=1,
                        help="1=use multi-scale pyramid (level0+1+2), 0=level0 only")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


def calculate_single_metrics(pred, gt, num_classes=2):
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


class GaofenPyramidDataset(Dataset):
    """
    Dataset for Gaofen with optional multi-scale pyramid.
    Returns: (img_l0, img_l1, img_l2, gt, name) if use_pyramid else (img_l0, gt, name)
    """
    def __init__(self, level0_dir, level1_dir=None, level2_dir=None, use_pyramid=True):
        self.use_pyramid = use_pyramid
        self.img_dir_l0 = Path(level0_dir) / "imgs"
        self.gt_dir = Path(level0_dir) / "gts"
        self.img_list = sorted([p.name for p in self.img_dir_l0.glob("*.png")])

        if use_pyramid:
            self.img_dir_l1 = Path(level1_dir) / "imgs"
            self.img_dir_l2 = Path(level2_dir) / "imgs"

    def __len__(self):
        return len(self.img_list)

    def _load_img(self, path):
        img = io.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        elif img.shape[2] > 3:
            img = img[:, :, :3]
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))
        return img

    def _load_gt(self, path):
        gt = io.imread(path)
        if gt.ndim == 3:
            gt = gt[:, :, 0]
        gt = (gt > 0).astype(np.float32)
        return gt

    def __getitem__(self, idx):
        name = self.img_list[idx]
        img_l0 = self._load_img(self.img_dir_l0 / name)
        gt = self._load_gt(self.gt_dir / name)

        if self.use_pyramid:
            img_l1 = self._load_img(self.img_dir_l1 / name)
            img_l2 = self._load_img(self.img_dir_l2 / name)
            return (
                torch.from_numpy(img_l0),
                torch.from_numpy(img_l1),
                torch.from_numpy(img_l2),
                torch.from_numpy(gt),
                name
            )
        else:
            return torch.from_numpy(img_l0), torch.from_numpy(gt), name


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Use pyramid: {bool(args.use_pyramid)}")

    # Load model
    model = SwinTransformerNet(pretrained=None).to(device)
    print(f"Loading checkpoint: {args.resume}")
    ckpt = torch.load(args.resume, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model.eval()

    # Dataset
    dataset = GaofenPyramidDataset(
        args.data_test,
        args.level1_dir if args.use_pyramid else None,
        args.level2_dir if args.use_pyramid else None,
        use_pyramid=bool(args.use_pyramid)
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Dataset: {len(dataset)} images")

    per_image_results = []
    start_time = time.time()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            if args.use_pyramid:
                img_l0, img_l1, img_l2, gt, names = batch
                img_l0 = img_l0.to(device)
                img_l1 = img_l1.to(device)
                img_l2 = img_l2.to(device)
                # SwinTransformerNet only takes single input, so we ensemble predictions
                pred_l0 = torch.sigmoid(model(img_l0))
                pred_l1 = torch.sigmoid(model(img_l1))
                pred_l2 = torch.sigmoid(model(img_l2))
                # Upsample l1, l2 to l0 size
                pred_l1 = F.interpolate(pred_l1, size=pred_l0.shape[2:], mode='bilinear', align_corners=False)
                pred_l2 = F.interpolate(pred_l2, size=pred_l0.shape[2:], mode='bilinear', align_corners=False)
                # Average ensemble
                pred = (pred_l0 + pred_l1 + pred_l2) / 3.0
            else:
                img_l0, gt, names = batch
                img_l0 = img_l0.to(device)
                pred = torch.sigmoid(model(img_l0))

            # Move to CPU immediately
            pred_np = pred.squeeze().cpu().numpy()
            gt_np = gt.squeeze().cpu().numpy()

            # Clear GPU
            if args.use_pyramid:
                del img_l0, img_l1, img_l2, pred_l0, pred_l1, pred_l2, pred
            else:
                del img_l0, pred
            torch.cuda.empty_cache()

            # Binarize
            pred_binary = (pred_np > 0.5).astype(np.uint8)
            gt_binary = (gt_np > 0).astype(np.uint8)

            # Calculate metrics per image in batch
            for i in range(pred_binary.shape[0] if pred_binary.ndim == 3 else 1):
                if pred_binary.ndim == 3:
                    p = pred_binary[i]
                    g = gt_binary[i]
                    name = names[i]
                else:
                    p = pred_binary
                    g = gt_binary
                    name = names[0]

                miou, acc, f1 = calculate_single_metrics(p, g)
                per_image_results.append({
                    "image": name,
                    "mIoU": round(miou, 6),
                    "Acc": round(acc, 6),
                    "F1": round(f1, 6)
                })

    elapsed = time.time() - start_time

    # Summary
    avg_miou = float(np.mean([r["mIoU"] for r in per_image_results]))
    avg_acc = float(np.mean([r["Acc"] for r in per_image_results]))
    avg_f1 = float(np.mean([r["F1"] for r in per_image_results]))

    print("=" * 60)
    print(f"Images: {len(per_image_results)}")
    print(f"Time: {elapsed:.2f} s ({elapsed / len(per_image_results):.2f} s/img)")
    print(f"Avg mIoU: {avg_miou:.4f}")
    print(f"Avg Acc:  {avg_acc:.4f}")
    print(f"Avg F1:   {avg_f1:.4f}")
    print("=" * 60)

    # Save CSVs
    per_image_csv = os.path.join(args.output_dir, "per_image_metrics.csv")
    with open(per_image_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mIoU", "Acc", "F1"])
        for r in per_image_results:
            w.writerow([r["image"], r["mIoU"], r["Acc"], r["F1"]])

    summary_csv = os.path.join(args.output_dir, "metrics.csv")
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mIoU", avg_miou])
        w.writerow(["Acc", avg_acc])
        w.writerow(["F1", avg_f1])
        w.writerow(["num_images", len(per_image_results)])
        w.writerow(["time_seconds", round(elapsed, 2)])
        w.writerow(["use_pyramid", int(args.use_pyramid)])

    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
