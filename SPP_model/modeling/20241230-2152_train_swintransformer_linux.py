# author MengFanlin
# time 2024/12/26
# filename train_transformer_linux
# description:xxx
# author MengFanlin
# time 2024/12/25
# filename train_swintransformer
# description:只进行swintransformer

# author MengFanlin
# time 2024/12/6
# filename train_promptwaternet
# description:xxx
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
from SPP_model.modeling.prompt_water_net import PromptWaterNet
from SPP_model.modeling.swin_net import SwinTransformerNet
from RSDataloader import PromptDataset
from sklearn.metrics import confusion_matrix


join = os.path.join

parser = argparse.ArgumentParser()

# Add relevant arguments
parser.add_argument("--data_train", type=str, default="/scratch/cuihao/water_1024_NET/train",
                    help="path to training data; 3 subfolders: gts , imgs and prompt_mask_256")
# parser.add_argument("--model_type", type=str, default="vit_b")
#
# parser.add_argument("--promptcp", type=str, default="/project/cuihao/PromptWater/sam_vit_b_01ec64.pth",
#                     help="The checkpoint of Prompt model")
# parser.add_argument("--promptcp", type=str, default=r"D:\deeplearning\code\MedSAM-main\work_dir\MedSAM-ViT-B-20241205-2139\medsam_model_best_93e_Loss0.08.pth",
#                     help="The checkpoint of Prompt model")
parser.add_argument("--num_workers", type=int, default=3)
# ---------------------------------------------------------------------------------
parser.add_argument("-task_name", type=str, default="swin-_halfdata_no_prompt")
  #
# parser.add_argument(
#     "-checkpoint", type=str, default=r"D:\deeplearning\sam_vit_b_01ec64.pth",
# #"-checkpoint", type=str, default=r"D:\deeplearning\code\MedSAM-main\work_dir\MedSAM-ViT-B-20240718-1108\medsam_model_best.pth"
# help='SAM模型'
# )

parser.add_argument(
    "--load_pretrain", type=bool, default=True, help="load pretrain model"
)

parser.add_argument("-pretrain_model_path", type=str, default="")
parser.add_argument("-work_dir", type=str, default="/project/cuihao/PromptWater/work_dir")
# train
parser.add_argument("-num_epochs", type=int, default=50)
parser.add_argument("-batch_size", type=int, default=3)
parser.add_argument("-val_batch_size", type=int, default=3)
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
parser.add_argument("-use_amp", action="store_true", default=True, help="use amp") # 孟改
parser.add_argument(
    "--resume", type=str, default="",
    help="Resuming training from checkpoint"
)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--SwintransformerPretrain", type=str, default="/project/cuihao/PromptWater/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth")
args = parser.parse_args()

run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)


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

def validate(net, val_dataloader, device, dicefocal_loss):
    """Validation loop to compute metrics."""
    net.eval()
    val_loss = 0
    miou_sum, acc_sum, f1_sum = 0, 0, 0
    num_batches = len(val_dataloader)

    with torch.no_grad():
        for image, labels, _, _ in tqdm(val_dataloader, desc="Validation"):
            image, labels = image.to(device), labels.to(device)
            medsam_pred = net(image)
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
    # promptcheckpoint = args.promptcp
    swin = SwinTransformerNet(args.SwintransformerPretrain).to(device)

    swin.train()

    print(
        "Number of total parameters: ",
        sum(p.numel() for p in swin.parameters()),
    )
    print(
        "Number of trainable parameters: ",
        sum(p.numel() for p in swin.parameters() if p.requires_grad),
    )
    # prompt_module_encdec_params = list(swin.prompt_module.image_encoder.parameters()) + list(
    #     swin.prompt_module.mask_decoder.parameters()
    # )
    # optimizer = torch.optim.AdamW(
    #     prompt_module_encdec_params, lr=args.lr, weight_decay=args.weight_decay
    # )
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, swin.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay)

    # print(
    #     "Number of image encoder and mask decoder parameters: ",
    #     sum(p.numel() for p in prompt_module_encdec_params if p.requires_grad),
    # )
    dicefocal_loss = monai.losses.dice_focal(sigmoid=True, reduction="mean", squared_pred=True)

    # 设置训练和验证数据集
    num_epochs = args.num_epochs
    # iter_num = 0
    train_loss = []
    best_loss = 1e10
    best_Valscore = 0.001
    previous_best_model = None

    train_dataset = PromptDataset(args.data_train)



    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_dataset = PromptDataset(args.data_train.replace("train", "val"))
    print("Number of validation samples: ", len(val_dataset))
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


    start_epoch = 0
    best_val_loss = 0.01

    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            swin.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])

    if args.use_amp:
        scaler = torch.cuda.amp.GradScaler()

    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for step, (image, labels, _, _) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            image, labels = image.to(device), labels.to(device)
            if args.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    medsam_pred = swin(image)
                    loss = dicefocal_loss(medsam_pred, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                medsam_pred = swin(image)
                loss = dicefocal_loss(medsam_pred, labels)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= step
        train_loss.append(epoch_loss)

        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Loss: {epoch_loss}')
        #torch.cuda.empty_cache()
        # 验证过程
        val_loss, val_miou, val_acc, val_f1 = validate(swin, val_dataloader, device, dicefocal_loss)
        Valscore = val_miou+val_acc+val_f1
        print(f'Validation - Loss: {val_loss:.4f}, mIoU: {val_miou:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f},Valscore:{Valscore:.4f}')

        # 保存模型
        checkpoint = {
            "model": swin.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, f'{epoch}_Loss{epoch_loss:.2f}_ValLoss{val_loss:.2f}_Valscore{Valscore:.4f}.pth'))

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

