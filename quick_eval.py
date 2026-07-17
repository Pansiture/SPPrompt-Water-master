#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick evaluation script for baseline models on a small subset of test patches.
Useful for quickly checking model performance without running full inference.
"""
import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from skimage import io
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

from SPP_model.modeling.other_model.Unet import UNet
from SPP_model.modeling.swin_net import SwinTransformerNet
from RSDataloader import PromptDataset_GID5


def calculate_metrics(preds, labels, num_classes=2):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["unet", "swin"], required=True)
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--data_test", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/test")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of test samples to evaluate (default: 100)")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--swin_pretrain", type=str,
                        default="./pretrain/swin_tiny_patch4_window7_224.pth")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    if args.model_type == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model_type == "swin":
        model = SwinTransformerNet(pretrained=args.swin_pretrain).to(device)

    checkpoint = torch.load(args.resume, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    print(f"Loaded checkpoint from {args.resume}")

    # Dataset (subset for quick eval)
    dataset = PromptDataset_GID5(args.data_test)
    if args.num_samples < len(dataset):
        indices = np.random.choice(len(dataset), args.num_samples, replace=False)
        dataset = Subset(dataset, indices)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)

    model.eval()
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(dataloader)

    with torch.no_grad():
        for image, labels, _, _ in tqdm(dataloader, desc="Quick Eval"):
            image, labels = image.to(device), labels.to(device).float()
            pred = model(image)
            pred_binary = (torch.sigmoid(pred) > 0.5).float()
            miou, acc, f1 = calculate_metrics(pred_binary.detach(), labels)
            miou_sum += miou
            acc_sum += acc
            f1_sum += f1

    print(f"\nQuick Eval ({args.num_samples} samples):")
    print(f"  OA:   {acc_sum/num_batches:.4f}  ({acc_sum/num_batches*100:.2f}%)")
    print(f"  mIoU: {miou_sum/num_batches:.4f}  ({miou_sum/num_batches*100:.2f}%)")
    print(f"  F1:   {f1_sum/num_batches:.4f}  ({f1_sum/num_batches*100:.2f}%)")


if __name__ == "__main__":
    main()
