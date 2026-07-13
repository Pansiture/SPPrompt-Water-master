import os
import glob
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from pathlib import Path
from skimage import io
import torchvision.transforms.functional as TF
from SPP_model.modeling.prompt_water_net import SPPromptWaterNet

# 路径配置
img_dir = './data/level0/val/imgs'
prompt_dir = './data/level0/val/prompt_mask_256'
gt_dir = './data/level0/val/gts'
sam_cp = './pretrain/sam_vit_b_01ec64.pth'
swin_cp = './pretrain/swin_tiny_patch4_window7_224.pth'
resume = './work_dir/levelfusion_14epoch_e11_0.8219_valscore2.6524/best_model_e11_Valscore2.6524.pth'

device = torch.device('cuda')
model = SPPromptWaterNet(sam_cp, swin_cp, freeze_prompt=True).to(device).eval()
ckpt = torch.load(resume, map_location=device)
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

def run_with_prompt(image, prompt):
    """简单推理：image, prompt 都是 [1, C, H, W] 的 torch tensor"""
    image = image.to(device)
    prompt = prompt.to(device)
    # resize 到 1024x1024 / 256x256
    Xi = F.interpolate(image, size=(1024, 1024), mode='bilinear', align_corners=False)
    Pi = F.interpolate(prompt, size=(256, 256), mode='bilinear', align_corners=False)
    with torch.no_grad():
        pred = torch.sigmoid(model(Xi, Pi))
    pred = F.interpolate(pred, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
    return (pred > 0.5).float()

img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
miou_list = []

for i, img_path in enumerate(img_files[:100]):  # 先跑 100 张快速验证
    basename = os.path.basename(img_path)
    stem = os.path.splitext(basename)[0]
    
    # load image
    img = io.imread(img_path)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0).float()
    
    # load prompt from prompt_mask_256
    prompt_path = os.path.join(prompt_dir, basename)
    if os.path.exists(prompt_path):
        prompt = io.imread(prompt_path)
        prompt = np.expand_dims(prompt, axis=0)
        prompt = prompt / 255.0
    else:
        prompt = np.zeros((1, img.shape[2], img.shape[3]))
    prompt = torch.tensor(prompt).unsqueeze(0).float()
    
    # inference
    pred = run_with_prompt(img, prompt)
    
    # load gt and compute miou
    gt_path = os.path.join(gt_dir, basename)
    if not os.path.exists(gt_path):
        continue
    gt = np.array(Image.open(gt_path).convert('L'))
    pred_arr = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
    if pred_arr.shape != gt.shape:
        pred_arr = np.array(Image.fromarray((pred_arr * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
        pred_arr = (pred_arr > 127).astype(np.uint8)
    
    pred_flat = pred_arr.flatten()
    gt_flat = (gt > 127).astype(np.uint8).flatten()
    cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
    iou = np.diag(cm) / (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm) + 1e-6)
    miou = np.mean(iou)
    miou_list.append(miou)
    
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(img_files[:100])} done, current mean mIoU: {np.mean(miou_list):.4f}')

print(f'Evaluated: {len(miou_list)} images')
print(f'Mean mIoU with REAL prompt: {np.mean(miou_list):.4f}')
print(f'Median mIoU: {np.median(miou_list):.4f}')
print(f'Min: {np.min(miou_list):.4f}, Max: {np.max(miou_list):.4f}')
