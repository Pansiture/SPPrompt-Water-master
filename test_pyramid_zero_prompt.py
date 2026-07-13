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

# ============================
# 关键配置信息
# ============================
img_dir = './data/level0/val/imgs'
gt_dir = './data/level0/val/gts'
sam_cp = './pretrain/sam_vit_b_01ec64.pth'
swin_cp = './pretrain/swin_tiny_patch4_window7_224.pth'
resume = './work_dir/levelfusion_14epoch_e11_0.8219_valscore2.6524/best_model_e11_Valscore2.6524.pth'

print("=" * 60)
print("[Config] Pyramid Zero-Prompt Inference")
print("=" * 60)
print(f"[Config] Image dir:        {img_dir}")
print(f"[Config] GT dir:           {gt_dir}")
print(f"[Config] SAM checkpoint:    {sam_cp}")
print(f"[Config] Swin pretrain:     {swin_cp}")
print(f"[Config] Trained weight:    {resume}")
print(f"[Config] Prompt mode:      ZERO PROMPT (all zeros)")
print("=" * 60)

device = torch.device('cuda')
model = SPPromptWaterNet(sam_cp, swin_cp, freeze_prompt=True).to(device).eval()

print(f"[Init] Loading trained weight: {resume}")
ckpt = torch.load(resume, map_location=device)
if 'model' in ckpt:
    model.load_state_dict(ckpt['model'])
    print("[Init] Loaded 'model' key from checkpoint")
else:
    model.load_state_dict(ckpt)
    print("[Init] Loaded full checkpoint")
print("[Init] Model ready for inference")

def run_pyramid_with_zero_prompt(image):
    """3层金字塔推理，顶层用 ZERO prompt，下层用上层输出"""
    image = image.to(device)
    
    B, C, H, W = image.shape
    n_level = 2
    
    # 构建金字塔
    X = image
    memory = {}
    for i in range(n_level + 1):
        if i > 0:
            X = F.interpolate(X, scale_factor=0.5, mode='area')
        memory[f'layer{i}_X'] = X
    
    # 自顶向下推理
    for level in range(n_level, -1, -1):
        X_level = memory[f'layer{level}_X']
        H_l, W_l = X_level.shape[2], X_level.shape[3]
        
        if level == n_level:
            # 顶层：使用 ZERO prompt
            P_level = torch.zeros(B, 1, H_l, W_l, device=device)
        else:
            # 下层：用上层输出
            parent = memory[f'layer{level+1}_Y']
            P_level = F.interpolate(parent, size=(H_l, W_l), mode='bilinear', align_corners=False)
        
        # resize 到模型输入
        Xi = F.interpolate(X_level, size=(1024, 1024), mode='bilinear', align_corners=False)
        Pi = F.interpolate(P_level, size=(256, 256), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            Yi = torch.sigmoid(model(Xi, Pi))
        Yi = F.interpolate(Yi, size=(H_l, W_l), mode='bilinear', align_corners=False)
        Yi = (Yi > 0.5).float()
        
        memory[f'layer{level}_Y'] = Yi
    
    return memory['layer0_Y'][:, :, :H, :W]

img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
miou_list = []

print(f"[Run] Found {len(img_files)} images, starting zero-prompt pyramid inference...")

for i, img_path in enumerate(img_files):
    basename = os.path.basename(img_path)
    
    img = io.imread(img_path)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img).unsqueeze(0).float()
    
    # ZERO prompt inference
    pred = run_pyramid_with_zero_prompt(img)
    
    gt_path = os.path.join(gt_dir, basename)
    if not os.path.exists(gt_path):
        continue
    gt = np.array(Image.open(gt_path).convert('L'))
    pred_arr = pred.squeeze().cpu().numpy()
    if pred_arr.shape != gt.shape:
        pred_arr = np.array(Image.fromarray((pred_arr * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
        pred_arr = (pred_arr > 127).astype(np.uint8)
    else:
        pred_arr = (pred_arr > 0.5).astype(np.uint8)
    
    pred_flat = pred_arr.flatten()
    gt_flat = (gt > 127).astype(np.uint8).flatten()
    cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
    iou = np.diag(cm) / (cm.sum(axis=0) + cm.sum(axis=1) - np.diag(cm) + 1e-6)
    miou = np.mean(iou)
    miou_list.append(miou)
    
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(img_files)} done, current mean mIoU: {np.mean(miou_list):.4f}')

print(f'Evaluated: {len(miou_list)} images')
print(f'Mean mIoU (Pyramid + ZERO Prompt): {np.mean(miou_list):.4f}')
print(f'Median: {np.median(miou_list):.4f}, Min: {np.min(miou_list):.4f}, Max: {np.max(miou_list):.4f}')
