# author MengFanlin
# time 2024/12/6
# filename train_promptwaternet
# description:train
import logging
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset
from skimage import io
from tqdm import tqdm
from datetime import datetime
import shutil
import os
from SPP_model.modeling.prompt_water_net import SPPromptWaterNet
from RSDataloader import PromptDataset_GID5
from sklearn.metrics import confusion_matrix


join = os.path.join

parser = argparse.ArgumentParser()

parser.add_argument("--data_train", type=str, default=r"/autodl-pub/glh_data/level0/train",
                    help="path to training data; 3 subfolders: gts , imgs and prompt_mask_256")
parser.add_argument("--data_val", type=str, default=None,
                    help="Optional: explicit val data path. If None, will replace 'train' with 'val' in data_train")
parser.add_argument("--promptcp", type=str, default=r"",
                    help="The checkpoint of Prompt model (SAM vit-b pretrain weights, optional)")
parser.add_argument(
    "-freeze_prompt", type=bool, default=False, help="Freeze the prompt module (default False to allow fine-tuning SAM)"
)
parser.add_argument("--SwintransformerPretrain", type=str, default="./pretrain/swin_tiny_patch4_window7_224.pth",
                    help="Path to Swin Transformer pretrain weights. Download from https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth")
parser.add_argument("-work_dir", type=str, default=r"./work_dir")

parser.add_argument("--num_workers", type=int, default=8)
# ---------------------------------------------------------------------------------
parser.add_argument("-task_name", type=str, default="SPP_GLH")

# train
parser.add_argument("-num_epochs", type=int, default=50)
parser.add_argument("-batch_size", type=int, default=4)
parser.add_argument("-val_batch_size", type=int, default=4)
# Optimizer parameters
parser.add_argument(
    "-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)"
)
parser.add_argument(
    "-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)"
)
parser.add_argument(
    "-use_wandb", type=bool, default=False, help="use wandb to monitor training"
)
parser.add_argument("-use_amp", action="store_true", default=True, help="use amp") #
parser.add_argument(
    "--resume", type=str, default="",
    help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")

args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)

os.makedirs(model_save_path, exist_ok=True)

# ---- logging setup ----
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
logger.info(f"Logging to: {log_file}")

# ---- print all training hyperparameters ----
logger.info("=" * 60)
logger.info("Training Configuration:")
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
logger.info("  num_workers:         %d", args.num_workers)
logger.info("  device:              %s", args.device)
logger.info("  use_amp:             %s", args.use_amp)
logger.info("  use_wandb:           %s", args.use_wandb)
logger.info("  resume:              %s", args.resume if args.resume else "None")
logger.info("=" * 60)



def calculate_metrics(preds, labels, num_classes=2):
    """Calculate mIoU, accuracy, and F1-score."""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    miou_list = []
    f1_list = []
    acc_list = []

    for pred, label in zip(preds, labels):
        pred = pred.squeeze()
        #pred = pred.argmax(axis=0)  # 获取预测类别
        pred = (pred > 0.5).astype(int)  # 二值化预测
        label = label.squeeze()  # 标签类别
        flat_pred = pred.flatten()
        flat_label = label.flatten()

        # 计算混淆矩阵
        cm = confusion_matrix(flat_label, flat_pred, labels=list(range(num_classes)))

        intersection = np.diag(cm)  # 交集
        union = cm.sum(axis=0) + cm.sum(axis=1) - intersection  # 并集
        iou = intersection / (union + 1e-6)
        miou = np.nanmean(iou)  # mIoU

        precision = intersection / (cm.sum(axis=0) + 1e-6)  # 精确率
        recall = intersection / (cm.sum(axis=1) + 1e-6)  # 召回率
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)  # F1分数

        accuracy = intersection.sum() / (cm.sum() + 1e-6)  # 准确率

        # 过滤 NaN
        # miou_list.append(miou)
        # f1_list.append(np.nanmean(f1))
        # acc_list.append(accuracy)
        miou_list.append(miou)
        f1_list.append(np.mean(f1))
        acc_list.append(accuracy)
    # return np.nanmean(miou_list), np.nanmean(acc_list), np.nanmean(f1_list)
    return np.mean(miou_list), np.mean(acc_list), np.mean(f1_list)

def validate(prompt_water_net, val_dataloader, device, loss_fn):
    """Validation loop to compute metrics."""
    prompt_water_net.eval()
    val_loss = 0
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for image, labels, prompts, _ in tqdm(val_dataloader, desc="Validation"):
            image, prompts, labels = image.to(device), prompts.to(device), labels.to(device).float()
            labels_256 = F.interpolate(labels, size=(256, 256), mode='nearest')
            seg_logits, evidence, prob_256, uncertainty_256, consistency = prompt_water_net(image, prompts)
            loss = loss_fn(seg_logits, labels)
            val_loss += loss.item()

            # 计算指标 (仍然基于 1024x1024 分割输出)
            miou, acc, f1 = calculate_metrics(torch.sigmoid(seg_logits).detach(), labels)
            miou_sum += miou
            acc_sum += acc
            f1_sum += f1

    val_loss /= num_batches
    miou_avg = miou_sum / num_batches
    acc_avg = acc_sum / num_batches
    f1_avg = f1_sum / num_batches

    return val_loss, miou_avg, acc_avg, f1_avg


def main():
    os.makedirs(model_save_path, exist_ok=True)

    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    promptcheckpoint = args.promptcp if args.promptcp else None
    prompt_water_net = SPPromptWaterNet(promptcheckpoint,args.SwintransformerPretrain,freeze_prompt=args.freeze_prompt).to(device)
    prompt_water_net.train()

    logger.info(
        "Number of total parameters: %s",
        sum(p.numel() for p in prompt_water_net.parameters()),
    )
    logger.info(
        "Number of trainable parameters: %s",
        sum(p.numel() for p in prompt_water_net.parameters() if p.requires_grad),
    )
    prompt_module_encdec_params = list(prompt_water_net.prompt_module.image_encoder.parameters()) + list(
        prompt_water_net.prompt_module.mask_decoder.parameters()
    )
    # optimizer = torch.optim.AdamW(
    #     prompt_module_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    # )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, prompt_water_net.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay)

    # Add cosine annealing LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )

    logger.info(
        "Number of image encoder and mask decoder parameters: %s",
        sum(p.numel() for p in prompt_module_encdec_params if p.requires_grad),
    )
    logger.info("=" * 60)
    logger.info("Weight Loading Paths:")
    logger.info("  SAM prompt checkpoint (promptcp):  %s", args.promptcp if args.promptcp else "None (loading default)")
    logger.info("  Swin Transformer pretrained:       %s", args.SwintransformerPretrain)
    logger.info("  Resume checkpoint:                 %s", args.resume if args.resume else "None (training from scratch)")
    logger.info("=" * 60)

    logger.info("Model Architecture Details:")
    logger.info("  Total parameters:         %s", sum(p.numel() for p in prompt_water_net.parameters()))
    logger.info("  Trainable parameters:     %s", sum(p.numel() for p in prompt_water_net.parameters() if p.requires_grad))
    logger.info("  SAM enc+dec trainable:    %s", sum(p.numel() for p in prompt_module_encdec_params if p.requires_grad))
    logger.info("  freeze_prompt:            %s", args.freeze_prompt)
    logger.info("  Optimizer:                AdamW")
    logger.info("  LR Scheduler:             CosineAnnealingLR (T_max=%d, eta_min=1e-6)", args.num_epochs)
    logger.info("  Loss:                     0.6*DiceFocalLoss(alpha=0.75) + 0.4*TverskyLoss(alpha=0.3, beta=0.7)")
    logger.info("  AMP (mixed precision):    %s", args.use_amp)
    logger.info("=" * 60)

    logger.info("Training Hyperparameters:")
    logger.info("  num_epochs:      %d", args.num_epochs)
    logger.info("  batch_size:      %d", args.batch_size)
    logger.info("  val_batch_size:  %d", args.val_batch_size)
    logger.info("  lr:              %.6f", args.lr)
    logger.info("  weight_decay:    %.6f", args.weight_decay)
    logger.info("  num_workers:     %d", args.num_workers)
    logger.info("  device:          %s", args.device)
    logger.info("=" * 60)

    dice_focal_loss = monai.losses.DiceFocalLoss(sigmoid=True, reduction="mean", squared_pred=True, alpha=0.75)
    tversky_loss = monai.losses.TverskyLoss(sigmoid=True, alpha=0.3, beta=0.7)
    
    def combined_loss(pred, target):
        return 0.6 * dice_focal_loss(pred, target) + 0.4 * tversky_loss(pred, target)

    class EvidentialLoss(nn.Module):
        def __init__(self, lambda_u=1.0):
            super().__init__()
            self.lambda_u = lambda_u
        
        def forward(self, evidence, target):
            target = target.float()
            alpha = evidence + 1.0
            S = alpha[:, 0:1] + alpha[:, 1:2]
            p = alpha[:, 0:1] / S
            u = 2.0 / S
            loss = target * (1 - p)**2 + (1 - target) * p**2 + self.lambda_u * u
            return loss.mean()

    class FMConsistencyLoss(nn.Module):
        def forward(self, consistency, target):
            return F.mse_loss(consistency, target.float(), reduction='mean')

    edl_loss_fn = EvidentialLoss(lambda_u=1.0)
    fm_loss_fn = FMConsistencyLoss()

    # 设置训练和验证数据集
    num_epochs = args.num_epochs
    # iter_num = 0
    train_loss = []
    val_loss_list = []
    best_loss = 1e10
    best_Valscore = 0.001
    previous_best_model = None

    # ---- Multi-level unified dataset (no file copy needed) ----
    levels = ["level0", "level1", "level2"]
    train_datasets = []
    for lv in levels:
        ds = PromptDataset_GID5(args.data_train.replace("level0", lv))
        train_datasets.append(ds)
    train_dataset = ConcatDataset(train_datasets)

    logger.info("Number of training samples: %s", len(train_dataset))
    logger.info("  (level0: %d, level1: %d, level2: %d)", len(train_datasets[0]), len(train_datasets[1]), len(train_datasets[2]))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_datasets = []
    for lv in levels:
        ds = PromptDataset_GID5(args.data_train.replace("train", "val").replace("level0", lv))
        val_datasets.append(ds)
    val_dataset = ConcatDataset(val_datasets)
    logger.info("Number of validation samples: %s", len(val_dataset))
    logger.info("  (level0: %d, level1: %d, level2: %d)", len(val_datasets[0]), len(val_datasets[1]), len(val_datasets[2]))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    logger.info("Dataset Configuration (UNIFIED Multi-Level):")
    logger.info("  Mode:            ConcatDataset (level0 + level1 + level2)")
    for i, lv in enumerate(levels):
        logger.info("  %s:", lv.upper())
        logger.info("    Train path:    %s", args.data_train.replace("level0", lv))
        logger.info("    Train samples: %d", len(train_datasets[i]))
        logger.info("    Val path:      %s", args.data_train.replace("train", "val").replace("level0", lv))
        logger.info("    Val samples:   %d", len(val_datasets[i]))
    logger.info("  TOTAL Train:     %d", len(train_dataset))
    logger.info("  TOTAL Val:       %d", len(val_dataset))
    logger.info("  Zero-prompt prob: 15%% (paper-style default prompt token)")
    logger.info("=" * 60)


    start_epoch = 0
    best_val_loss = 0.01

    if args.resume is not None and args.resume != "":
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            prompt_water_net.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    if args.use_amp:
        scaler = torch.amp.GradScaler('cuda')

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, labels, prompts, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            image, prompts, labels = image.to(device), prompts.to(device), labels.to(device).float()
            labels_256 = F.interpolate(labels, size=(256, 256), mode='nearest')
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    seg_logits, evidence, prob_256, uncertainty_256, consistency = prompt_water_net(image, prompts)
                    loss_seg = combined_loss(seg_logits, labels)
                    loss_edl = edl_loss_fn(evidence, labels_256)
                    loss_fm = fm_loss_fn(consistency, labels_256)
                    loss = loss_seg + 0.5 * loss_edl + 0.3 * loss_fm
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                seg_logits, evidence, prob_256, uncertainty_256, consistency = prompt_water_net(image, prompts)
                loss_seg = combined_loss(seg_logits, labels)
                loss_edl = edl_loss_fn(evidence, labels_256)
                loss_fm = fm_loss_fn(consistency, labels_256)
                loss = loss_seg + 0.5 * loss_edl + 0.3 * loss_fm
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= step + 1
        train_loss.append(epoch_loss)

        logger.info('Time: %s, Epoch: %d, Loss: %.4f, LR: %.6f', datetime.now().strftime("%Y%m%d-%H%M"), epoch, epoch_loss, optimizer.param_groups[0]['lr'])
        #torch.cuda.empty_cache()
        # 验证过程
        val_loss, val_miou, val_acc, val_f1 = validate(prompt_water_net, val_dataloader, device, combined_loss)
        val_loss_list.append(val_loss)
        Valscore = val_miou+val_acc+val_f1
        logger.info('Validation - Loss: %.4f, mIoU: %.4f, Acc: %.4f, F1: %.4f, Valscore: %.4f', val_loss, val_miou, val_acc, val_f1, Valscore)

        # 保存模型（只保存最佳模型，不保存每个epoch以节省磁盘）
        checkpoint = {
            "model": prompt_water_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }

        if Valscore > best_Valscore:
            best_Valscore = Valscore

            # 构造新模型的文件名
            new_best_model_filename = f"best_model_e{epoch}_Valscore{Valscore:.4f}.pth"
            new_best_model_path = join(model_save_path, new_best_model_filename)

            # 删除上一个最佳模型文件（如果存在）
            if previous_best_model is not None and os.path.exists(previous_best_model):
                os.remove(previous_best_model)

            # 保存新的最佳模型
            torch.save(checkpoint, new_best_model_path)

            # 更新 previous_best_model 为当前的最佳模型文件路径
            previous_best_model = new_best_model_path
            logger.info("New best model saved: %s", new_best_model_filename)

        # 可选：每10个epoch保存一次常规检查点
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, join(model_save_path, f'checkpoint_e{epoch}.pth'))

        # Step the scheduler every epoch
        scheduler.step()

        # 绘制损失曲线
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss_list, label="Val Loss")
        plt.title("Train and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(join(model_save_path, args.task_name + "_loss_curve.png"))
        plt.close()





if __name__ == "__main__":


    main()