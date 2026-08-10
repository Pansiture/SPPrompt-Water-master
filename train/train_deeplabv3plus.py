#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for DeepLabV3+ (ResNet-101 backbone) on GID dataset (level0, single-scale).
"""
import argparse
import logging
import sys
import os
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import monai
from sklearn.metrics import confusion_matrix
from torchvision import models

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_baseline import SimpleDataset

join = os.path.join


def calculate_metrics(preds, labels, num_classes=2):
    """Calculate mIoU, OA (Accuracy), and F1-score."""
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
    """Validation loop."""
    model.eval()
    val_loss = 0
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(val_dataloader)
    with torch.no_grad():
        for image, labels, _ in tqdm(val_dataloader, desc="Validation"):
            image, labels = image.to(device), labels.to(device).float()
            pred = model(image)['out']
            if pred.shape[2:] != labels.shape[2:]:
                pred = F.interpolate(pred, size=labels.shape[2:], mode='bilinear', align_corners=False)
            loss = loss_fn(pred, labels)
            val_loss += loss.item()
            miou, acc, f1 = calculate_metrics(torch.sigmoid(pred).detach(), labels)
            miou_sum += miou
            acc_sum += acc
            f1_sum += f1
    model.train()
    return val_loss / num_batches, miou_sum / num_batches, acc_sum / num_batches, f1_sum / num_batches


def build_deeplabv3plus_res101(num_classes=1, pretrained=True):
    """
    Build DeepLabV3+ with ResNet-101 backbone.
    torchvision.models.segmentation.deeplabv3_resnet101 output channels = num_classes.
    """
    weights = models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT if pretrained else None
    model = models.segmentation.deeplabv3_resnet101(weights=weights)
    # Replace classifier head for binary segmentation
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    # Also replace auxiliary classifier if present
    if hasattr(model, 'aux_classifier') and model.aux_classifier is not None:
        model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_train", type=str,
                        default="/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/train")
    parser.add_argument("--data_val", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default="/root/autodl-tmp/SPPrompt-Water-master/work_dir")
    parser.add_argument("--task_name", type=str, default="DeepLabV3Plus_GID")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--no_pretrained", action="store_true", default=False,
                        help="Train from scratch without ImageNet pretraining")
    args = parser.parse_args()

    if args.data_val is None:
        args.data_val = args.data_train.replace("train", "val")

    run_id = datetime.now().strftime("%Y%m%d-%H%M")
    model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
    device = torch.device(args.device)
    os.makedirs(model_save_path, exist_ok=True)

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
    model = build_deeplabv3plus_res101(num_classes=1, pretrained=not args.no_pretrained).to(device)
    model.train()
    logger.info("Total parameters: %d", sum(p.numel() for p in model.parameters()))
    logger.info("Trainable parameters: %d", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ---- optimizer / scheduler ----
    # Use different lr for backbone and classifier (common practice)
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.lr * 0.1},
        {'params': head_params, 'lr': args.lr}
    ], weight_decay=args.weight_decay)

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
    train_dataset = SimpleDataset(args.data_train)
    val_dataset = SimpleDataset(args.data_val)
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
        logger.info("AMP enabled (GradScaler).")
    else:
        logger.info("AMP disabled (full fp32).")

    # ---- training loop ----
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        epoch_loss = 0
        for step, (image, labels, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            image, labels = image.to(device), labels.to(device).float()
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    pred = model(image)['out']
                    if pred.shape[2:] != labels.shape[2:]:
                        pred = F.interpolate(pred, size=labels.shape[2:], mode='bilinear', align_corners=False)
                    loss = combined_loss(pred, labels)
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(image)['out']
                if pred.shape[2:] != labels.shape[2:]:
                    pred = F.interpolate(pred, size=labels.shape[2:], mode='bilinear', align_corners=False)
                loss = combined_loss(pred, labels)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
