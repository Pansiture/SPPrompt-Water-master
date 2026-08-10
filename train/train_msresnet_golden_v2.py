#!/usr/bin/env python
"""
MSResNet 训练脚本 (Golden 数据集, level0) - 修复版
修复: 强制关闭 AMP, 修复 inplace ReLU 导致的 nan, 添加梯度裁剪
"""
import os
import sys
import argparse
import logging
import shutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import glob
from skimage import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import monai

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SPP_model.modeling.other_model.MSResNet import MSResNet


class SimpleDataset(Dataset):
    """兼容 png/tif 的简单数据集，无需 prompt。"""
    def __init__(self, data_root, inference=False):
        self.inference = inference
        self.img_path = os.path.join(data_root, "imgs")
        self.gt_path = os.path.join(data_root, "gts")
        
        files = []
        for ext in ["*.png", "*.tif", "*.tiff"]:
            files.extend(glob.glob(os.path.join(self.gt_path, "**", ext), recursive=True))
        self.gt_files = sorted(list(set(files)))
        
        self.gt_files = [f for f in self.gt_files
                         if os.path.isfile(os.path.join(self.img_path, Path(f).stem + ".png"))
                         or os.path.isfile(os.path.join(self.img_path, Path(f).stem + ".tif"))
                         or os.path.isfile(os.path.join(self.img_path, Path(f).stem + ".tiff"))]
        
        print(f"number of images: {len(self.gt_files)}")

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        stem = Path(self.gt_files[index]).stem
        
        img_path = os.path.join(self.img_path, stem + ".png")
        if not os.path.isfile(img_path):
            img_path = os.path.join(self.img_path, stem + ".tif")
        if not os.path.isfile(img_path):
            img_path = os.path.join(self.img_path, stem + ".tiff")
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        # 动态归一化：若最大值大于 1，则认为是 0-255 范围
        if img.max() > 1.0:
            img = img / 255.0
        
        gt = io.imread(self.gt_files[index])
        gt = np.expand_dims(gt, axis=0).astype(np.float32)
        if gt.max() > 1.0:
            gt = gt / 255.0
        
        if self.inference:
            return torch.tensor(img).float(), torch.tensor(gt).float(), self.gt_files[index]
        else:
            return torch.tensor(img).float(), torch.tensor(gt).long(), stem


def calculate_metrics(preds, labels, num_classes=2):
    preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
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


def validate(model, val_dataloader, device, loss_fn):
    model.eval()
    val_loss = 0
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(val_dataloader)
    
    with torch.no_grad():
        for image, labels, _ in tqdm(val_dataloader, desc="Validation"):
            image = image.to(device)
            labels = labels.to(device).float()
            
            pred = model(image)
            if pred.shape[2:] != labels.shape[2:]:
                pred = F.interpolate(pred, size=labels.shape[2:], mode='bilinear', align_corners=False)
            
            loss = loss_fn(pred, labels)
            prob = torch.sigmoid(pred)
            
            miou, acc, f1 = calculate_metrics(prob.detach(), labels)
            val_loss += loss.item()
            miou_sum += miou
            acc_sum += acc
            f1_sum += f1
    
    model.train()
    return val_loss / num_batches, miou_sum / num_batches, acc_sum / num_batches, f1_sum / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_train", type=str, required=True)
    parser.add_argument("--data_val", type=str, required=True)
    parser.add_argument("--work_dir", type=str, default="/root/autodl-tmp/SPPrompt-Water-master/work_dir")
    parser.add_argument("--task_name", type=str, default="MSResNet_Golden_v2")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    # 关键修复：强制关闭 AMP，MSResNet 的 inplace ReLU 与 FP16 不兼容
    parser.add_argument("--use_amp", action="store_true", default=False,
                        help="Enable AMP mixed precision (default: False, forced off for MSResNet)")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping max norm")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = os.path.join(args.work_dir, f"{args.task_name}-{run_id}")
    os.makedirs(model_save_path, exist_ok=True)
    device = torch.device(args.device)

    log_file = os.path.join(model_save_path, f"{run_id}_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("Training MSResNet on Golden (level0) - FIXED VERSION")
    logger.info("=" * 60)
    logger.info("Data train: %s", args.data_train)
    logger.info("Data val:   %s", args.data_val)
    logger.info("Epochs:     %d", args.num_epochs)
    logger.info("Batch:      %d", args.batch_size)
    logger.info("Grad accum: %d (effective batch=%d)", args.grad_accum_steps, args.batch_size * args.grad_accum_steps)
    logger.info("LR:         %.6f", args.lr)
    logger.info("AMP:        %s (FORCED OFF for MSResNet)", args.use_amp)
    logger.info("Grad clip:  %.2f", args.grad_clip)
    logger.info("=" * 60)

    model = MSResNet(in_channels=3, num_classes=1)
    model.to(device).train()
    logger.info("Total params: %d", sum(p.numel() for p in model.parameters()))
    logger.info("Trainable params: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 关键修复：过滤 requires_grad，避免不可训练参数干扰
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    dice_focal_loss = monai.losses.DiceFocalLoss(sigmoid=True, reduction="mean", squared_pred=True, alpha=0.75)
    tversky_loss = monai.losses.TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
    def combined_loss(pred, target):
        return 0.6 * dice_focal_loss(pred, target) + 0.4 * tversky_loss(pred, target)

    train_dataset = SimpleDataset(args.data_train)
    val_dataset = SimpleDataset(args.data_val)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    logger.info("Train samples: %d", len(train_dataset))
    logger.info("Val samples:   %d", len(val_dataset))

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # 强制禁用 AMP
    if args.use_amp:
        logger.warning("AMP requested but disabled for MSResNet due to inplace ReLU incompatibility")
    use_amp = False
    logger.info("Using full FP32 precision")

    train_loss = []
    val_loss_list = []
    best_Valscore = 0.0
    previous_best_model = None

    for epoch in range(start_epoch, args.num_epochs):
        epoch_loss = 0
        optimizer.zero_grad()
        for step, (image, labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            image = image.to(device)
            labels = labels.to(device).float()
            
            is_accum_end = ((step + 1) % args.grad_accum_steps == 0) or (step + 1 == len(train_loader))

            # 纯 FP32 训练，无 AMP
            pred = model(image)
            if pred.shape[2:] != labels.shape[2:]:
                pred = F.interpolate(pred, size=labels.shape[2:], mode='bilinear', align_corners=False)
            loss = combined_loss(pred, labels) / args.grad_accum_steps
            loss.backward()
            
            # 梯度裁剪，防止爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            
            if is_accum_end:
                optimizer.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * args.grad_accum_steps
        
        epoch_loss /= (step + 1)
        train_loss.append(epoch_loss)
        
        # 检查 loss 是否为 nan
        if np.isnan(epoch_loss):
            logger.error("Epoch loss is NaN! Stopping training.")
            break
        
        logger.info("Epoch: %d, Loss: %.4f, LR: %.6f", epoch, epoch_loss, optimizer.param_groups[0]['lr'])

        val_loss, val_miou, val_acc, val_f1 = validate(model, val_loader, device, combined_loss)
        val_loss_list.append(val_loss)
        Valscore = val_miou + val_acc + val_f1
        logger.info("Validation - Loss: %.4f, mIoU: %.4f, Acc: %.4f, F1: %.4f, Valscore: %.4f", 
                    val_loss, val_miou, val_acc, val_f1, Valscore)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }

        if Valscore > best_Valscore:
            best_Valscore = Valscore
            new_best_model_filename = f"best_model_e{epoch}_Valscore{Valscore:.4f}.pth"
            new_best_model_path = os.path.join(model_save_path, new_best_model_filename)
            if previous_best_model and os.path.exists(previous_best_model):
                os.remove(previous_best_model)
            torch.save(checkpoint, new_best_model_path)
            previous_best_model = new_best_model_path
            logger.info("New best model saved: %s", new_best_model_filename)

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(model_save_path, f"checkpoint_e{epoch}.pth"))
        
        scheduler.step()

        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss_list, label="Val Loss")
        plt.legend()
        plt.savefig(os.path.join(model_save_path, f"{args.task_name}_loss_curve.png"))
        plt.close()

    logger.info("Training finished. Best Valscore: %.4f", best_Valscore)


if __name__ == "__main__":
    main()
