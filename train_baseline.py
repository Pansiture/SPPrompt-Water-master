#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for baseline models (U-Net and SwinTransformer)
on GLH-Water dataset (level0, 1024x1024 patches).
"""
import argparse
import logging
import sys
import os
import shutil
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import monai
from sklearn.metrics import confusion_matrix

from SPP_model.modeling.other_model.Unet import UNet
from SPP_model.modeling.swin_net import SwinTransformerNet
from RSDataloader import PromptDataset_GID5

join = os.path.join


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


def validate(model, val_dataloader, device, loss_fn):
    """Validation loop."""
    model.eval()
    val_loss = 0
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(val_dataloader)
    with torch.no_grad():
        for image, labels, _, _ in tqdm(val_dataloader, desc="Validation"):
            image, labels = image.to(device), labels.to(device).float()
            pred = model(image)
            loss = loss_fn(pred, labels)
            val_loss += loss.item()
            miou, acc, f1 = calculate_metrics(torch.sigmoid(pred).detach(), labels)
            miou_sum += miou
            acc_sum += acc
            f1_sum += f1
    return val_loss / num_batches, miou_sum / num_batches, acc_sum / num_batches, f1_sum / num_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, choices=["unet", "swin"], required=True,
                        help="Baseline model: unet or swin")
    parser.add_argument("--data_train", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GLH-water/level0/train",
                        help="Path to training data")
    parser.add_argument("--data_val", type=str, default=None,
                        help="Path to validation data (default: replace 'train' with 'val')")
    parser.add_argument("--work_dir", type=str, default="./work_dir")
    parser.add_argument("--task_name", type=str, default=None,
                        help="Task name for save directory")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--val_batch_size", type=int, default=6)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--swin_pretrain", type=str,
                        default="./pretrain/swin_tiny_patch4_window7_224.pth")
    args = parser.parse_args()
    if args.model_type == "unet":
        args.swin_pretrain = "N/A (U-Net does not use Swin pretrain)"

    if args.data_val is None:
        args.data_val = args.data_train.replace("train", "val")
    if args.task_name is None:
        args.task_name = f"{args.model_type.upper()}_GLH"

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    device = torch.device(args.device)
    os.makedirs(model_save_path, exist_ok=True)

    # ---- logging ----
    log_file = join(model_save_path, f"{run_id}_training.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info("Logging to: %s", log_file)
    logger.info("=" * 60)
    logger.info("Training Configuration:")
    for k, v in vars(args).items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)

    # ---- model ----
    if args.model_type == "unet":
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model_type == "swin":
        model = SwinTransformerNet(pretrained=args.swin_pretrain).to(device)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model.train()
    logger.info("Total parameters: %d", sum(p.numel() for p in model.parameters()))
    logger.info("Trainable parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ---- optimizer / scheduler ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    # ---- loss ----
    dice_focal_loss = monai.losses.DiceFocalLoss(
        sigmoid=True, reduction="mean", squared_pred=True, alpha=0.75
    )
    tversky_loss = monai.losses.TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)

    def combined_loss(pred, target):
        return 0.6 * dice_focal_loss(pred, target) + 0.4 * tversky_loss(pred, target)

    # ---- datasets ----
    train_dataset = PromptDataset_GID5(args.data_train)
    val_dataset = PromptDataset_GID5(args.data_val)
    logger.info("Train samples: %d", len(train_dataset))
    logger.info("Val samples: %d", len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )

    start_epoch = 0
    best_Valscore = 0.0
    previous_best_model = None
    train_loss_list = []
    val_loss_list = []

    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Resumed from %s at epoch %d", args.resume, start_epoch)

    if args.use_amp:
        scaler = torch.amp.GradScaler('cuda')

    # ---- training loop ----
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0
        for step, (image, labels, _, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            image, labels = image.to(device), labels.to(device).float()
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(image)
                    loss = combined_loss(pred, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(image)
                loss = combined_loss(pred, labels)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= (step + 1)
        train_loss_list.append(epoch_loss)

        val_loss, val_miou, val_acc, val_f1 = validate(model, val_dataloader, device, combined_loss)
        val_loss_list.append(val_loss)
        Valscore = val_miou + val_acc + val_f1
        logger.info(
            'Epoch %d - Loss: %.4f, Val Loss: %.4f, mIoU: %.4f, OA: %.4f, F1: %.4f, Valscore: %.4f',
            epoch, epoch_loss, val_loss, val_miou, val_acc, val_f1, Valscore
        )

        # ---- save checkpoint ----
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if Valscore > best_Valscore:
            best_Valscore = Valscore
            new_best_filename = f"best_model_e{epoch}_Valscore{Valscore:.4f}.pth"
            new_best_path = join(model_save_path, new_best_filename)
            if previous_best_model and os.path.exists(previous_best_model):
                os.remove(previous_best_model)
            torch.save(checkpoint, new_best_path)
            previous_best_model = new_best_path
            logger.info("New best model saved: %s", new_best_filename)

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, join(model_save_path, f"checkpoint_e{epoch}.pth"))

        scheduler.step()

        # ---- plot loss curve ----
        plt.plot(train_loss_list, label="Train Loss")
        plt.plot(val_loss_list, label="Val Loss")
        plt.title("Train and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(join(model_save_path, args.task_name + "_loss_curve.png"))
        plt.close()

    logger.info("Training complete. Best Valscore: %.4f", best_Valscore)


if __name__ == "__main__":
    main()
