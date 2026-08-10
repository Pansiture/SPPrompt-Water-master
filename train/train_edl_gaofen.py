#!/usr/bin/env python
"""
SPPrompt-Water + EDL 训练脚本 (Gaofen_processed_v2 数据集)
基于 train_edl.py 修改，适配 Gaofen 数据格式 (png, 无 tif)。
"""
import logging
import sys
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from skimage import io
from tqdm import tqdm
from datetime import datetime
import shutil
import os
from pathlib import Path
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SPP_model.modeling.prompt_water_net_edl import SPPromptWaterNetEDL
from SPP_model.modeling.edl_utils import edl_loss, evidence_to_prob_uncertainty
from sklearn.metrics import confusion_matrix

join = os.path.join


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class PromptDataset_Gaofen(Dataset):
    """适配 Gaofen_processed_v2 (png 格式, 无 tif) 的 PromptDataset。"""
    def __init__(self, data_root, inference=False, bbox_shift=20):
        self.inference = inference
        self.data_root = data_root
        self.gt_path = join(data_root, "gts")
        self.img_path = join(data_root, "imgs")
        self.prompt_path = join(data_root, "prompt_mask_256")

        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.png"), recursive=True)
        )
        self.gt_path_files = [
            file for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, Path(file).stem + ".png"))
        ]
        self.bbox_shift = bbox_shift
        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def _augment(self, img, gt, prompt):
        import random
        if random.random() > 0.5:
            img = np.flip(img, axis=2).copy()
            gt = np.flip(gt, axis=2).copy()
            prompt = np.flip(prompt, axis=2).copy()
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()
            prompt = np.flip(prompt, axis=1).copy()
        if random.random() > 0.5:
            k = random.randint(1, 3)
            img = np.rot90(img, k, axes=(1, 2)).copy()
            gt = np.rot90(gt, k, axes=(1, 2)).copy()
            prompt = np.rot90(prompt, k, axes=(1, 2)).copy()
        return img, gt, prompt

    def _resize_to_1024(self, img, is_mask=False):
        if img.shape[-2:] == (1024, 1024):
            return img
        t = torch.from_numpy(img).float().unsqueeze(0)
        mode = 'nearest' if is_mask else 'bilinear'
        t = F.interpolate(t, size=(1024, 1024), mode=mode, align_corners=None if mode == 'nearest' else False)
        return t.squeeze(0).numpy()

    def _resize_prompt_to_256(self, img):
        if img.shape[-2:] == (256, 256):
            return img
        t = torch.from_numpy(img).float().unsqueeze(0)
        t = F.interpolate(t, size=(256, 256), mode='nearest')
        return t.squeeze(0).numpy()

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        base_name = Path(img_name).stem

        img_path = join(self.img_path, base_name + ".png")
        img_1024 = io.imread(img_path)
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        img_1024 = self._resize_to_1024(img_1024, is_mask=False)

        prompt_img = io.imread(join(self.prompt_path, img_name))
        prompt_img = np.expand_dims(prompt_img, axis=0)
        prompt_img = prompt_img / 255.0
        prompt_img = self._resize_prompt_to_256(prompt_img)

        gt = io.imread(join(self.gt_path, img_name))
        gt = np.expand_dims(gt, axis=0)
        gt = gt / 255.0
        gt = self._resize_to_1024(gt, is_mask=True)

        if not self.inference:
            img_1024, gt, prompt_img = self._augment(img_1024, gt, prompt_img)
            import random
            if random.random() < 0.15:
                prompt_img = np.zeros_like(prompt_img)

        if self.inference:
            return (torch.tensor(img_1024).float(), torch.tensor(prompt_img).float(), join(self.gt_path, img_name))
        else:
            return (
                torch.tensor(img_1024).float(),
                torch.tensor(gt).long(),
                torch.tensor(prompt_img).float(),
                img_name,
            )


parser = argparse.ArgumentParser()
parser.add_argument("--data_train", type=str,
                    default="/root/autodl-tmp/SPPrompt-Water-master/data/Gaofen_processed_v2/level0/train")
parser.add_argument("--data_val", type=str, default=None)
parser.add_argument("--promptcp", type=str,
                    default="/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth")
parser.add_argument("--freeze_prompt", type=str2bool, default=False)
parser.add_argument("--SwintransformerPretrain", type=str,
                    default="/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth")
parser.add_argument("--work_dir", type=str, default="/root/autodl-tmp/SPPrompt-Water-master/work_dir")
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--task_name", type=str, default="SPP_Gaofen_EDL")
parser.add_argument("--num_epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--val_batch_size", type=int, default=4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--use_wandb", type=str2bool, default=False)
parser.add_argument("--use_amp", action="store_true", default=False)
parser.add_argument("--resume", type=str, default="")
parser.add_argument("--device", type=str, default="cuda:0")

# EDL-specific
parser.add_argument("--referee_weight", type=float, default=0.1)
parser.add_argument("--kl_anneal_ratio", type=float, default=0.5)
parser.add_argument("--referee_interval", type=int, default=1)
parser.add_argument("--warmup_epochs", type=int, default=5)
parser.add_argument("--pretrain_ckpt", type=str, default="")
parser.add_argument("--kl_scale", type=float, default=0.5)
parser.add_argument("--use_referee", type=str2bool, default=False)

args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.makedirs(model_save_path, exist_ok=True)

log_file = join(model_save_path, f"{run_id}_training.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(log_file, encoding="utf-8"), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")

logger.info("=" * 60)
logger.info("Training Configuration (EDL Version - Gaofen):")
logger.info("  data_train:          %s", args.data_train)
logger.info("  data_val:            %s", args.data_val if args.data_val else args.data_train.replace("train", "val"))
logger.info("  promptcp (SAM ckpt): %s", args.promptcp)
logger.info("  Swin Pretrain:       %s", args.SwintransformerPretrain)
logger.info("  freeze_prompt:       %s", args.freeze_prompt)
logger.info("  work_dir:            %s", args.work_dir)
logger.info("  task_name:           %s", args.task_name)
logger.info("  num_epochs:          %d", args.num_epochs)
logger.info("  batch_size:          %d", args.batch_size)
logger.info("  val_batch_size:      %d", args.val_batch_size)
logger.info("  lr:                  %.6f", args.lr)
logger.info("  weight_decay:        %.6f", args.weight_decay)
logger.info("  device:              %s", args.device)
logger.info("  use_amp:             %s", args.use_amp)
logger.info("  referee_weight:      %.3f", args.referee_weight)
logger.info("  kl_anneal_ratio:     %.2f", args.kl_anneal_ratio)
logger.info("  warmup_epochs:       %d", args.warmup_epochs)
logger.info("=" * 60)


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


def validate(prompt_water_net, val_dataloader, device, epoch, num_epochs,
             use_referee=False, kl_scale=0.5, kl_anneal_ratio=0.5, referee_weight=0.1):
    prompt_water_net.eval()
    val_loss = 0
    val_edl_loss = 0
    val_ref_loss = 0
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for image, labels, prompts, _ in tqdm(val_dataloader, desc="Validation"):
            image, prompts, labels = image.to(device), prompts.to(device), labels.to(device).float()

            evidence = prompt_water_net(image, prompts, return_evidence=True)
            if labels.shape[-2:] != evidence.shape[-2:]:
                labels = F.interpolate(labels, size=evidence.shape[-2:], mode='nearest')

            loss, mse, kl, annealing = edl_loss(
                evidence, labels, epoch, num_epochs,
                kl_anneal_epochs_ratio=kl_anneal_ratio, kl_scale=kl_scale
            )

            if use_referee:
                sam_prob = prompt_water_net.forward_referee(image)
                prob, _ = evidence_to_prob_uncertainty(evidence)
                ref_loss = F.mse_loss(prob[:, 1:2], sam_prob)
            else:
                ref_loss = torch.tensor(0.0, device=device)

            total_loss = loss + referee_weight * ref_loss
            val_loss += total_loss.item()
            val_edl_loss += loss.item()
            val_ref_loss += ref_loss.item()

            prob, _ = evidence_to_prob_uncertainty(evidence)
            prob_fg = prob[:, 1:2, :, :]
            miou, acc, f1 = calculate_metrics(prob_fg.detach(), labels)
            miou_sum += miou
            acc_sum += acc
            f1_sum += f1

    return (val_loss / num_batches, val_edl_loss / num_batches, val_ref_loss / num_batches,
            miou_sum / num_batches, acc_sum / num_batches, f1_sum / num_batches)


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(__file__, join(model_save_path, run_id + "_" + os.path.basename(__file__)))

    promptcheckpoint = args.promptcp if args.promptcp else None
    prompt_water_net = SPPromptWaterNetEDL(
        promptcheckpoint,
        args.SwintransformerPretrain,
        freeze_prompt=args.freeze_prompt,
        use_referee=args.use_referee
    ).to(device)
    prompt_water_net.train()

    logger.info("Total parameters: %s", sum(p.numel() for p in prompt_water_net.parameters()))
    logger.info("Trainable parameters: %s", sum(p.numel() for p in prompt_water_net.parameters() if p.requires_grad))

    base_params = []
    edl_new_params = []
    for name, p in prompt_water_net.named_parameters():
        if not p.requires_grad:
            continue
        if "uperhead" in name or "conv_fusion" in name:
            edl_new_params.append(p)
        else:
            base_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": base_params, "lr": args.lr, "weight_decay": args.weight_decay},
            {"params": edl_new_params, "lr": args.lr * 10, "weight_decay": args.weight_decay},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-5)

    warmup_scheduler = None
    if args.warmup_epochs > 0:
        def warmup_fn(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            return 1.0
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_fn)

    logger.info("=" * 60)
    logger.info("Weight Loading Paths:")
    logger.info("  SAM prompt checkpoint:  %s", args.promptcp if args.promptcp else "None")
    logger.info("  Swin Transformer:       %s", args.SwintransformerPretrain)
    logger.info("  Resume checkpoint:      %s", args.resume if args.resume else "None")
    logger.info("=" * 60)

    num_epochs = args.num_epochs
    train_loss = []
    val_loss_list = []
    val_edl_loss_list = []
    val_ref_loss_list = []
    best_Valscore = 0.001
    previous_best_model = None

    # 单尺度 level0 数据集
    train_dataset = PromptDataset_Gaofen(args.data_train)
    val_data_path = args.data_val if args.data_val else args.data_train.replace("train", "val")
    val_dataset = PromptDataset_Gaofen(val_data_path, inference=True)

    logger.info("Number of training samples: %s", len(train_dataset))
    logger.info("Number of val samples:      %s", len(val_dataset))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=True
    )

    start_epoch = 0
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        start_epoch = checkpoint.get("epoch", 0) + 1
        prompt_water_net.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        logger.info("Resumed from epoch %d", start_epoch)

    if args.pretrain_ckpt and os.path.isfile(args.pretrain_ckpt):
        logger.info("Loading compatible weights from pretrain checkpoint: %s", args.pretrain_ckpt)
        pretrain_ckpt = torch.load(args.pretrain_ckpt, map_location=device)
        pretrain_state = pretrain_ckpt.get("model", pretrain_ckpt)
        model_state = prompt_water_net.state_dict()
        compatible_state = {}
        skipped_keys = []
        for k, v in model_state.items():
            if k in pretrain_state and pretrain_state[k].shape == v.shape:
                compatible_state[k] = pretrain_state[k]
            else:
                skipped_keys.append(k)
        model_state.update(compatible_state)
        prompt_water_net.load_state_dict(model_state, strict=False)
        logger.info("Loaded %d / %d parameters from pretrain_ckpt", len(compatible_state), len(model_state))
        if skipped_keys:
            logger.info("Skipped %d keys: %s", len(skipped_keys), ", ".join(skipped_keys[:10]) + ("..." if len(skipped_keys) > 10 else ""))

    if args.use_amp:
        scaler = torch.amp.GradScaler('cuda')
        logger.info("AMP enabled.")
    else:
        logger.info("AMP disabled (full fp32).")

    # Smoke test
    logger.info("=" * 60)
    logger.info("Running smoke test on one batch...")
    try:
        prompt_water_net.train()
        batch = next(iter(train_dataloader))
        img_smoke, lbl_smoke, prm_smoke, _ = batch
        img_smoke = img_smoke[:1].to(device)
        lbl_smoke = lbl_smoke[:1].to(device).float()
        prm_smoke = prm_smoke[:1].to(device)

        optimizer.zero_grad()
        ev = prompt_water_net(img_smoke, prm_smoke, return_evidence=True)
        if lbl_smoke.shape[-2:] != ev.shape[-2:]:
            lbl_smoke = F.interpolate(lbl_smoke, size=ev.shape[-2:], mode='nearest')
        loss_smoke, mse_smoke, kl_smoke, anneal_smoke = edl_loss(
            ev, lbl_smoke, 0, 1, kl_anneal_epochs_ratio=args.kl_anneal_ratio, kl_scale=args.kl_scale
        )
        if args.use_referee:
            sp = prompt_water_net.forward_referee(img_smoke)
            prob_smoke, _ = evidence_to_prob_uncertainty(ev)
            ref_smoke = F.mse_loss(prob_smoke[:, 1:2], sp)
            loss_smoke = loss_smoke + args.referee_weight * ref_smoke
        loss_smoke.backward()
        optimizer.step()

        prompt_water_net.eval()
        with torch.no_grad():
            ev_val = prompt_water_net(img_smoke, prm_smoke, return_evidence=True)
            prob_val, _ = evidence_to_prob_uncertainty(ev_val)
            prob_fg_val = prob_val[:, 1:2, :, :]
            miou_smoke, acc_smoke, f1_smoke = calculate_metrics(prob_fg_val.detach(), lbl_smoke)
        logger.info("Smoke test passed: mIoU=%.4f, Acc=%.4f, F1=%.4f", miou_smoke, acc_smoke, f1_smoke)
        logger.info("=" * 60)
    except Exception as e:
        logger.error("Smoke test FAILED: %s", str(e))
        raise

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        epoch_edl_loss = 0
        epoch_ref_loss = 0
        prompt_water_net.train()

        for step, (image, labels, prompts, _) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
            optimizer.zero_grad()
            image, prompts, labels = image.to(device), prompts.to(device), labels.to(device).float()

            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    evidence = prompt_water_net(image, prompts, return_evidence=True)
                    if labels.shape[-2:] != evidence.shape[-2:]:
                        labels = F.interpolate(labels, size=evidence.shape[-2:], mode='nearest')
                    loss, mse, kl, annealing = edl_loss(
                        evidence, labels, epoch, num_epochs,
                        kl_anneal_epochs_ratio=args.kl_anneal_ratio, kl_scale=args.kl_scale
                    )
                    if args.use_referee and step % args.referee_interval == 0:
                        sam_prob = prompt_water_net.forward_referee(image)
                        prob, _ = evidence_to_prob_uncertainty(evidence)
                        ref_loss = F.mse_loss(prob[:, 1:2], sam_prob)
                    else:
                        ref_loss = torch.tensor(0.0, device=device)
                    total_loss = loss + args.referee_weight * ref_loss
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                evidence = prompt_water_net(image, prompts, return_evidence=True)
                if labels.shape[-2:] != evidence.shape[-2:]:
                    labels = F.interpolate(labels, size=evidence.shape[-2:], mode='nearest')
                loss, mse, kl, annealing = edl_loss(
                    evidence, labels, epoch, num_epochs,
                    kl_anneal_epochs_ratio=args.kl_anneal_ratio, kl_scale=args.kl_scale
                )
                if args.use_referee and step % args.referee_interval == 0:
                    sam_prob = prompt_water_net.forward_referee(image)
                    prob, _ = evidence_to_prob_uncertainty(evidence)
                    ref_loss = F.mse_loss(prob[:, 1:2], sam_prob)
                else:
                    ref_loss = torch.tensor(0.0, device=device)
                total_loss = loss + args.referee_weight * ref_loss
                total_loss.backward()
                optimizer.step()

            epoch_loss += total_loss.item()
            epoch_edl_loss += loss.item()
            epoch_ref_loss += ref_loss.item()

        epoch_loss /= (step + 1)
        epoch_edl_loss /= (step + 1)
        epoch_ref_loss /= (step + 1)
        train_loss.append(epoch_loss)

        logger.info(
            'Epoch: %d, Loss: %.4f (EDL: %.4f, Ref: %.4f), LR: base=%.6f head=%.6f, KL_coef: %.4f',
            epoch, epoch_loss, epoch_edl_loss, epoch_ref_loss,
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'], annealing
        )

        val_total, val_edl, val_ref, val_miou, val_acc, val_f1 = validate(
            prompt_water_net, val_dataloader, device, epoch, num_epochs,
            use_referee=args.use_referee, kl_scale=args.kl_scale,
            kl_anneal_ratio=args.kl_anneal_ratio, referee_weight=args.referee_weight
        )
        val_loss_list.append(val_total)
        val_edl_loss_list.append(val_edl)
        val_ref_loss_list.append(val_ref)

        Valscore = val_miou + val_acc + val_f1
        logger.info(
            'Validation - Loss: %.4f (EDL: %.4f, Ref: %.4f), mIoU: %.4f, Acc: %.4f, F1: %.4f, Valscore: %.4f',
            val_total, val_edl, val_ref, val_miou, val_acc, val_f1, Valscore
        )

        checkpoint = {
            "model": prompt_water_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }

        if Valscore > best_Valscore:
            best_Valscore = Valscore
            new_best_model_filename = f"best_model_e{epoch}_Valscore{Valscore:.4f}.pth"
            new_best_model_path = join(model_save_path, new_best_model_filename)
            if previous_best_model is not None and os.path.exists(previous_best_model):
                os.remove(previous_best_model)
            torch.save(checkpoint, new_best_model_path)
            previous_best_model = new_best_model_path
            logger.info("New best model saved: %s", new_best_model_filename)

        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, join(model_save_path, f'checkpoint_e{epoch}.pth'))

        if warmup_scheduler is not None and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            scheduler.step()

        plt.figure(figsize=(10, 6))
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss_list, label="Val Total Loss")
        plt.plot(val_edl_loss_list, label="Val EDL Loss")
        plt.plot(val_ref_loss_list, label="Val Ref Loss")
        plt.title("Train and Validation Loss (EDL)")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(join(model_save_path, args.task_name + "_loss_curve.png"))
        plt.close()


if __name__ == "__main__":
    main()
