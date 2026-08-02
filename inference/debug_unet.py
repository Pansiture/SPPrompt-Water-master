#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug UNet single image using same logic as training validate()."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from skimage import io
from sklearn.metrics import confusion_matrix

from SPP_model.modeling.other_model.Unet import UNet
from RSDataloader import PromptDataset_GID5
from torch.utils.data import DataLoader


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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet(in_channels=3, out_channels=1).to(device)

ckpt = torch.load(
    "/root/autodl-tmp/SPPrompt-Water-master/work_dir/GLH_UNet_20epoch_e7_0.7304_Valscore2.4664/UNet_GLH_20epoch_e7_0.7304_Valscore2.4664.pth",
    map_location=device
)
model.load_state_dict(ckpt["model"])
model.eval()

# Method 1: DataLoader (same as training validate)
dataset = PromptDataset_GID5("/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/val")
loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

for i, (img, label, _, name) in enumerate(loader):
    if "106_p59_v1" in str(name[0]):
        img, label = img.to(device), label.to(device).float()
        with torch.no_grad():
            pred = model(img)
            prob = torch.sigmoid(pred)
            print(f"[DataLoader] pred range: {pred.min().item():.4f} ~ {pred.max().item():.4f}")
            print(f"[DataLoader] prob range: {prob.min().item():.4f} ~ {prob.max().item():.4f}")
            print(f"[DataLoader] prob mean: {prob.mean().item():.4f}")
            miou, acc, f1 = calculate_metrics(prob.detach(), label)
            print(f"[DataLoader] mIoU={miou:.4f} Acc={acc:.4f} F1={f1:.4f}")
        break

# Method 2: Manual load (same as my script)
img_np = io.imread("/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/val/imgs/106_p59_v1.png")
img_np = img_np.astype(np.float32) / 255.0
img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
with torch.no_grad():
    pred = model(img_tensor)
    prob = torch.sigmoid(pred)
    print(f"[Manual] pred range: {pred.min().item():.4f} ~ {pred.max().item():.4f}")
    print(f"[Manual] prob range: {prob.min().item():.4f} ~ {prob.max().item():.4f}")
    print(f"[Manual] prob mean: {prob.mean().item():.4f}")

# Compare raw pixel values
print(f"\n[DataLoader] img min/max: {img.min().item():.4f} / {img.max().item():.4f}")
print(f"[Manual] img min/max: {img_tensor.min().item():.4f} / {img_tensor.max().item():.4f}")
print(f"[DataLoader] img mean: {img.mean().item():.4f}")
print(f"[Manual] img mean: {img_tensor.mean().item():.4f}")
