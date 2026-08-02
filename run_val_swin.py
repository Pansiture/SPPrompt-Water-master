#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validation inference for SwinTransformer baseline on GLH-Water val set.
Outputs: per-image result visualization + per-image miou/acc/f1 + CSV summary.
OOM-safe: small batch size, periodic cache clear, no_grad.
"""
import argparse
import os
import sys
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from SPP_model.modeling.swin_net import SwinTransformerNet
from RSDataloader import PromptDataset_GID5

join = os.path.join


def calculate_metrics(preds, labels, num_classes=2):
    """Calculate per-sample mIoU, OA, F1."""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()
    miou_list, f1_list, acc_list = [], [], []
    for pred, label in zip(preds, labels):
        pred = pred.squeeze()
        pred = (pred > 0.5).astype(int)
        label = label.squeeze()
        flat_pred = pred.flatten()
        flat_label = label.flatten()
        cm = confusion_matrix(flat_label, flat_pred, labels=list(range(num_classes)))
        intersection = np.diag(cm)
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
        iou = intersection / (union + 1e-6)
        miou = np.nanmean(iou)
        precision = intersection / (cm.sum(axis=0) + 1e-6)
        recall = intersection / (cm.sum(axis=1) + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        accuracy = intersection.sum() / (cm.sum() + 1e-6)
        miou_list.append(miou)
        f1_list.append(np.mean(f1))
        acc_list.append(accuracy)
    return np.array(miou_list), np.array(acc_list), np.array(f1_list)


def overlay_image(original_image, predicted_mask, ground_truth_mask):
    """Create RGB overlay visualization."""
    original_image = np.array(original_image)
    if original_image.max() <= 1.0:
        original_image = (original_image * 255).astype(np.uint8)
    else:
        original_image = original_image.astype(np.uint8)
    predicted_mask = (predicted_mask * 255).astype(np.uint8)
    ground_truth_mask = (ground_truth_mask * 255).astype(np.uint8)
    overlay = np.zeros((*predicted_mask.shape, 3), dtype=np.uint8)
    overlay[:, :, 0] = np.where(predicted_mask == 255, 255, original_image[:, :, 0])
    overlay[:, :, 1] = np.where(predicted_mask == 255, 0,   original_image[:, :, 1])
    overlay[:, :, 2] = np.where(predicted_mask == 255, 0,   original_image[:, :, 2])
    combined_image = np.zeros((predicted_mask.shape[0], predicted_mask.shape[1] * 3, 3), dtype=np.uint8)
    combined_image[:, :predicted_mask.shape[1], :] = original_image
    combined_image[:, predicted_mask.shape[1]:2*predicted_mask.shape[1], :] = np.stack([ground_truth_mask]*3, axis=-1)
    combined_image[:, 2*predicted_mask.shape[1]:, :] = overlay
    return combined_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/work_dir/GLH_swin_7epoch_e6_0.7324_2.4671/GLH_swin_7epoch_e6_0.7324_2.4671.pth")
    parser.add_argument("--data_val", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/val")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save results. Defaults to <checkpoint_dir>/val_results")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for inference (reduce if OOM)")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_overlay", action="store_true", default=True,
                        help="Save visualization images")
    parser.add_argument("--skip_existing", action="store_true", default=False,
                        help="Skip images that already have result files")
    args = parser.parse_args()

    if args.output_dir is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.output_dir = join(checkpoint_dir, "val_results")

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # ---- Load model ----
    print("Loading SwinTransformerNet...")
    model = SwinTransformerNet(pretrained=None).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # ---- Dataset ----
    val_dataset = PromptDataset_GID5(args.data_val)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    print(f"Val samples: {len(val_dataset)}, batches: {len(val_dataloader)}")

    # ---- CSV setup ----
    csv_path = join(args.output_dir, "val_metrics.csv")
    csv_exists = os.path.exists(csv_path)
    csv_file = open(csv_path, "a", newline="")
    csv_writer = csv.writer(csv_file)
    if not csv_exists:
        csv_writer.writerow(["filename", "mIoU", "Acc", "F1"])
        csv_file.flush()

    # ---- Inference ----
    all_miou, all_acc, all_f1 = [], [], []

    with torch.no_grad():
        for batch_idx, (image, labels, _, filenames) in enumerate(tqdm(val_dataloader, desc="Val Inference")):
            image = image.to(device)
            labels = labels.to(device).float()

            preds = model(image)
            preds = torch.sigmoid(preds)

            # Compute per-sample metrics
            miou_b, acc_b, f1_b = calculate_metrics(preds, labels)
            all_miou.extend(miou_b.tolist())
            all_acc.extend(acc_b.tolist())
            all_f1.extend(f1_b.tolist())

            # Save results per image
            for i, fname in enumerate(filenames):
                base_name = os.path.splitext(fname)[0]

                # Skip if already done
                if args.skip_existing:
                    overlay_path = join(args.output_dir, f"{base_name}_result.png")
                    if os.path.exists(overlay_path):
                        continue

                # Write CSV row
                csv_writer.writerow([fname, f"{miou_b[i]:.6f}", f"{acc_b[i]:.6f}", f"{f1_b[i]:.6f}"])

                if args.save_overlay:
                    pred_mask = (preds[i, 0].cpu().numpy() > 0.5).astype(np.float32)
                    gt_mask = labels[i, 0].cpu().numpy()
                    # original image for overlay: denormalize
                    orig = image[i].cpu().numpy()
                    mean = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
                    std = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
                    orig = orig * std + mean
                    orig = np.transpose(orig, (1, 2, 0))
                    orig = np.clip(orig, 0, 1)

                    combined = overlay_image(orig, pred_mask, gt_mask)
                    Image.fromarray(combined).save(join(args.output_dir, f"{base_name}_result.png"))

            # Periodic flush and cache clear to prevent OOM / hang
            if batch_idx % 10 == 0:
                csv_file.flush()
                torch.cuda.empty_cache()

    csv_file.close()

    # ---- Summary ----
    print("\n========== Validation Summary ==========")
    print(f"Total images: {len(all_miou)}")
    print(f"Mean mIoU: {np.mean(all_miou):.6f}")
    print(f"Mean Acc:  {np.mean(all_acc):.6f}")
    print(f"Mean F1:   {np.mean(all_f1):.6f}")
    print(f"CSV saved to: {csv_path}")
    print(f"Result images saved to: {args.output_dir}")
    print("========================================")


if __name__ == "__main__":
    main()
