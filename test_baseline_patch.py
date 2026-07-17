#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for baseline models (U-Net and SwinTransformer) on patch-level test set.
Outputs: OA, mIoU, F1-score, and segmentation result images for each patch.
"""
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from skimage import io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from SPP_model.modeling.other_model.Unet import UNet
from SPP_model.modeling.swin_net import SwinTransformerNet
from RSDataloader import PromptDataset_GID5


def calculate_metrics(preds, labels, num_classes=2):
    """Calculate mIoU, OA (Accuracy), and F1-score."""
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
    return np.mean(miou_list), np.mean(acc_list), np.mean(f1_list)


def infer_and_save(model, dataloader, device, output_dir):
    """Run inference, compute metrics, and save segmentation maps."""
    model.eval()
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(dataloader)
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for image, labels, _, img_names in tqdm(dataloader, desc="Testing"):
            image, labels = image.to(device), labels.to(device).float()
            pred = model(image)
            pred_prob = torch.sigmoid(pred)
            pred_binary = (pred_prob > 0.5).float()

            # metrics per batch
            miou, acc, f1 = calculate_metrics(pred_binary.detach(), labels)
            miou_sum += miou
            acc_sum += acc
            f1_sum += f1

            # save each prediction image
            if output_dir:
                for i in range(pred_binary.shape[0]):
                    pred_img = pred_binary[i].squeeze().cpu().numpy() * 255
                    pred_img = pred_img.astype(np.uint8)
                    out_name = img_names[i] if isinstance(img_names[i], str) else str(img_names[i])
                    out_path = os.path.join(output_dir, out_name)
                    io.imsave(out_path, pred_img, check_contrast=False)

    miou_avg = miou_sum / num_batches
    acc_avg = acc_sum / num_batches
    f1_avg = f1_sum / num_batches
    print(f"Test Results: mIoU: {miou_avg:.4f}, OA: {acc_avg:.4f}, F1: {f1_avg:.4f}")
    return miou_avg, acc_avg, f1_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["unet", "swin"], required=True,
                        help="Baseline model to test: unet or swin")
    parser.add_argument("--data_test", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/test",
                        help="Path to patch-level test data")
    parser.add_argument("--resume", type=str, required=True,
                        help="Path to trained checkpoint (.pth)")
    parser.add_argument("--output_dir", type=str, default="./output/test_patches",
                        help="Directory to save segmentation result images")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--swin_pretrain", type=str,
                        default="./pretrain/swin_tiny_patch4_window7_224.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- load model ----
    if args.model_type == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model_type == "swin":
        model = SwinTransformerNet(pretrained=args.swin_pretrain).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    if not os.path.isfile(args.resume):
        raise FileNotFoundError(f"Checkpoint not found: {args.resume}")

    checkpoint = torch.load(args.resume, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.resume}")

    # ---- dataset ----
    dataset = PromptDataset_GID5(args.data_test)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    print(f"Number of test samples: {len(dataset)}")

    model_name = args.model_type.upper()
    output_dir = os.path.join(args.output_dir, model_name)

    # ---- infer ----
    miou, acc, f1 = infer_and_save(model, dataloader, device, output_dir)

    # ---- save summary ----
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, f"{model_name}_test_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Checkpoint: {args.resume}\n")
        f.write(f"Test Data: {args.data_test}\n")
        f.write(f"OA: {acc:.4f}  ({acc*100:.2f}%)\n")
        f.write(f"mIoU: {miou:.4f}  ({miou*100:.2f}%)\n")
        f.write(f"F1: {f1:.4f}  ({f1*100:.2f}%)\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
