# author MengFanlin
# time 2024/12/6
# filename train_promptwaternet
# description:train
import logging
import sys
import argparse
import torch
import monai
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
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
    "-freeze_prompt", type=bool, default=True, help="Freeze the prompt module (default True to preserve SAM feature extraction)"
)
parser.add_argument("--SwintransformerPretrain", type=str, default="./pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth",
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

def validate(prompt_water_net, val_dataloader, device, dicefocal_loss):
    """Validation loop to compute metrics."""
    prompt_water_net.eval()
    val_loss = 0
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for image, labels, prompts, _ in tqdm(val_dataloader, desc="Validation"):
            image, prompts, labels = image.to(device), prompts.to(device), labels.to(device).float()
            medsam_pred = prompt_water_net(image, prompts)
            loss = dicefocal_loss(medsam_pred, labels)
            val_loss += loss.item()

            # 计算指标
            miou, acc, f1 = calculate_metrics(torch.sigmoid(medsam_pred).detach(), labels)
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
    promptcheckpoint = args.promptcp
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

    logger.info(
        "Number of image encoder and mask decoder parameters: %s",
        sum(p.numel() for p in prompt_module_encdec_params if p.requires_grad),
    )
    dicefocal_loss = monai.losses.dice_focal(sigmoid=True, reduction="mean", squared_pred=True)

    # 设置训练和验证数据集
    num_epochs = args.num_epochs
    # iter_num = 0
    train_loss = []
    best_loss = 1e10
    best_Valscore = 0.001
    previous_best_model = None

    train_dataset = PromptDataset_GID5(args.data_train)



    logger.info("Number of training samples: %s", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    val_dataset = PromptDataset_GID5(args.data_train.replace("train", "val"))
    logger.info("Number of validation samples: %s", len(val_dataset))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )


    start_epoch = 0
    best_val_loss = 0.01

    if args.resume is not None:
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
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = prompt_water_net(image, prompts)
                    loss = dicefocal_loss(medsam_pred, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                medsam_pred = prompt_water_net(image, prompts)
                loss = dicefocal_loss(medsam_pred, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= step
        train_loss.append(epoch_loss)

        logger.info('Time: %s, Epoch: %d, Loss: %.4f', datetime.now().strftime("%Y%m%d-%H%M"), epoch, epoch_loss)
        #torch.cuda.empty_cache()
        # 验证过程
        val_loss, val_miou, val_acc, val_f1 = validate(prompt_water_net, val_dataloader, device, dicefocal_loss)
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


        # # 有点问题，没输出出来
        # if Valscore > best_Valscore:
        #     best_Valscore = Valscore
        #     torch.save(checkpoint, join(model_save_path, f"best_model_e{epoch}_Valscore{Valscore:.4f}.pth"))


        # if val_loss < best_loss:
        #     best_loss = val_loss
        #     torch.save(checkpoint, join(model_save_path, "best_model.pth"))

        # 绘制损失曲线
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.title("Train and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(join(model_save_path, args.task_name + "_loss_curve.png"))
        plt.close()





if __name__ == "__main__":


    main()