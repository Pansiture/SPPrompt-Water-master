# author MengFanlin
# time 2024/12/15
# filename inference_waterprompt
# description:sliding window prediction


import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from skimage import io
from tqdm import tqdm
# from SPP_model import sam_model_registry
# from SPP_model.modeling.SAM_mask import SamMask
# from RSDataloader import GIDDataset  # Assuming the same dataset class is used
import argparse
from SPP_model.modeling.prompt_water_net import SPPromptWaterNet
from RSDataloader import PromptDataset_GID5
from sklearn.metrics import confusion_matrix

parser = argparse.ArgumentParser()

# Add relevant arguments
parser.add_argument("--data_test", type=str, default=r"/autodl-pub/glh_data/level0/val",
                    help="path for test data; 3 subfolders: gts , imgs and prompt_mask_256")
parser.add_argument("--resume", type=str,
                    default=r"./work_dir/SPP_GLH-XXXXXXXX/best_model_eXX_ValscoreX.XXXX.pth",
                    help=r"Path to trained checkpoint")
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--output_dir", type=str,
                    default=r"./output/val", help="output directory for predictions")

parser.add_argument("--SwintransformerPretrain", type=str, default="./pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth")
parser.add_argument("--promptcp", type=str, default=r"",
                    help="The checkpoint of Prompt model (SAM vit-b, optional)")

args = parser.parse_args()

def calculate_metrics(preds, labels, num_classes=2):
    """Calculate mIoU, accuracy, and F1-score."""
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    miou_list = []
    f1_list = []
    acc_list = []
    f1_water_list = []

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
        f1_water_list.append(np.mean(f1[1]))
    # return np.nanmean(miou_list), np.nanmean(acc_list), np.nanmean(f1_list)
    return np.mean(miou_list), np.mean(acc_list), np.mean(f1_list),np.mean(f1_water_list)


# Define the inference function
def infer_and_save(model, dataloader, device, output_dir):
    model.eval()

    miou_sum, acc_sum, f1_sum, f1water_sum = 0, 0, 0, 0
    num_batches = len(dataloader)

    with torch.no_grad():
        # for step, (image, prompts, image_path) in enumerate(tqdm(dataloader)):
        for image, labels, prompts, image_path in tqdm(dataloader):
            image, prompts = image.to(device), prompts.to(device)
            pred = model(image, prompts)
            pred = torch.sigmoid(pred)  # Apply sigmoid to get probabilities
            pred = (pred > 0.5).float()  # Binarize the predictions at threshold 0.5
            pred = (pred * 255).byte()  # Convert to 0 and 255

            # Save the prediction as an image
            #save_prediction_as_image(pred.cpu().numpy(), image_path[0], output_dir)

            miou, acc, f1 ,f1water= calculate_metrics(pred.detach(), labels)
            miou_sum += miou
            acc_sum += acc
            f1_sum += f1
            f1water_sum += f1water

    miou_avg = miou_sum / num_batches
    acc_avg = acc_sum / num_batches
    f1_avg = f1_sum / num_batches
    f1water_avg =f1water_sum / num_batches
    Valscore = miou_avg + acc_avg + f1_avg
    print(f'mIoU: {miou_avg:.4f}, Acc: {acc_avg:.4f}, F1: {f1_avg:.4f},Valscore:{Valscore:.4f},f1water_avg:{f1water_avg:.4f}')


def save_prediction_as_image(prediction, image_path, output_dir):
    # Assume the prediction is a single-channel mask
    pred_image = prediction.squeeze()  # Remove batch and channel dimensions if necessary

    # Construct output file path
    filename = os.path.basename(image_path)
    name_without_extension = os.path.splitext(filename)[0]
    output_path = os.path.join(output_dir, f"{name_without_extension}.png")

    # Save the image
    io.imsave(output_path, pred_image)


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    promptcheckpoint = args.promptcp
    swin_pretrained = args.SwintransformerPretrain
    prompt_water_net = SPPromptWaterNet(promptcheckpoint, swin_pretrained, freeze_prompt=True).to(device)

    #
    if args.resume is not None and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        prompt_water_net.load_state_dict(checkpoint["model"])

    # Set up the dataset and dataloader
    dataset = PromptDataset_GID5(args.data_test
                         )  # Make sure to modify the dataset class to handle inference mode

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Run inference and save predictions
    infer_and_save(prompt_water_net, dataloader, device, args.output_dir)

    print("Inference complete. Predictions saved as images.")


if __name__ == "__main__":


    main()
