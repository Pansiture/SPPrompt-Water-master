import os
import sys
import glob
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from skimage import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SPP_model.modeling.prompt_water_net import SPPromptWaterNet

# ============================
# 关键配置信息 (GID Best Checkpoint)
# ============================
img_dir = '/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/imgs'
prompt_dir = '/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/prompt_mask_256'
gt_dir = '/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/gts'
sam_cp = '/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth'
swin_cp = '/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth'
resume = '/root/autodl-tmp/SPPrompt-Water-master/work_dir/GID_levelfusion_25epoch_e23_0.8194_Valscore2.6695/GID_levelfusion_25epoch_e23_0.8194_Valscore2.6695.pth'

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


def run_pyramid(image, prompt_mode='real'):
    """3层金字塔推理"""
    image = image.to(device)
    if prompt_mode == 'real' and 'real_prompt' in run_pyramid.__globals__:
        real_prompt = run_pyramid.__globals__.get('real_prompt', None)
    
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
            if prompt_mode == 'zero':
                P_level = torch.zeros(B, 1, H_l, W_l, device=device)
            else:
                P_level = real_prompt.to(device)
                if P_level.shape[2:] != (H_l, W_l):
                    P_level = F.interpolate(P_level, size=(H_l, W_l), mode='bilinear', align_corners=False)
        else:
            parent = memory[f'layer{level+1}_Y']
            P_level = F.interpolate(parent, size=(H_l, W_l), mode='bilinear', align_corners=False)
        
        Xi = F.interpolate(X_level, size=(1024, 1024), mode='bilinear', align_corners=False)
        Pi = F.interpolate(P_level, size=(256, 256), mode='bilinear', align_corners=False)
        
        with torch.no_grad():
            Yi = torch.sigmoid(model(Xi, Pi))
        Yi = F.interpolate(Yi, size=(H_l, W_l), mode='bilinear', align_corners=False)
        Yi = (Yi > 0.5).float()
        
        memory[f'layer{level}_Y'] = Yi
    
    return memory['layer0_Y'][:, :, :H, :W]


def evaluate_all_metrics(pred_arr, gt):
    """计算完整指标: mIoU, Acc, Precision, Recall, F1, Water IoU"""
    pred_flat = pred_arr.flatten()
    gt_flat = gt.flatten()
    
    cm = confusion_matrix(gt_flat, pred_flat, labels=[0, 1])
    
    intersection = np.diag(cm)
    union = cm.sum(axis=0) + cm.sum(axis=1) - intersection
    iou = intersection / (union + 1e-6)
    miou = np.mean(iou)
    water_iou = iou[1]
    
    accuracy = cm.diagonal().sum() / (cm.sum() + 1e-6)
    
    precision = intersection / (cm.sum(axis=0) + 1e-6)
    recall = intersection / (cm.sum(axis=1) + 1e-6)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
    
    return {
        'mIoU': miou,
        'Water_IoU': water_iou,
        'Acc': accuracy,
        'Precision': precision[1],
        'Recall': recall[1],
        'F1': f1[1],
    }


def run_inference(prompt_mode='real'):
    img_files = sorted(glob.glob(os.path.join(img_dir, '*.png')))
    
    metrics_list = []
    
    print(f"\n[Run] Prompt mode: {prompt_mode.upper()}, images: {len(img_files)}")
    
    for i, img_path in enumerate(img_files):
        basename = os.path.basename(img_path)
        
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).float()
        
        if prompt_mode == 'real':
            prompt_path = os.path.join(prompt_dir, basename)
            if os.path.exists(prompt_path):
                prompt = io.imread(prompt_path)
                prompt = np.expand_dims(prompt, axis=0) / 255.0
            else:
                prompt = np.zeros((1, img.shape[2], img.shape[3]))
            global real_prompt
            real_prompt = torch.tensor(prompt).unsqueeze(0).float()
            pred = run_pyramid(img, prompt_mode='real')
        else:
            pred = run_pyramid(img, prompt_mode='zero')
        
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
        
        gt = (gt > 127).astype(np.uint8)
        
        metrics = evaluate_all_metrics(pred_arr, gt)
        metrics_list.append(metrics)
        
        if (i + 1) % 100 == 0:
            mean_miou = np.mean([m['mIoU'] for m in metrics_list])
            print(f'  {i+1}/{len(img_files)} done, current mean mIoU: {mean_miou:.4f}')
    
    # 汇总
    print(f"\n{'='*60}")
    print(f"  Prompt Mode: {prompt_mode.upper()}")
    print(f"{'='*60}")
    print(f"  Evaluated images: {len(metrics_list)}")
    for key in ['mIoU', 'Water_IoU', 'Acc', 'Precision', 'Recall', 'F1']:
        vals = [m[key] for m in metrics_list]
        print(f"  {key:12s}: {np.mean(vals):.4f}  (min: {np.min(vals):.4f}, max: {np.max(vals):.4f})")
    print(f"{'='*60}")
    
    return metrics_list


if __name__ == '__main__':
    # 1. Real Prompt 推理
    real_metrics = run_inference(prompt_mode='real')
    
    # 2. Zero Prompt 推理
    zero_metrics = run_inference(prompt_mode='zero')
    
    # 3. 对比汇总
    print(f"\n{'='*60}")
    print("  FINAL COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':12s} {'Real Prompt':>12s} {'Zero Prompt':>12s} {'Diff':>10s}")
    print(f"  {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
    for key in ['mIoU', 'Water_IoU', 'Acc', 'Precision', 'Recall', 'F1']:
        real_mean = np.mean([m[key] for m in real_metrics])
        zero_mean = np.mean([m[key] for m in zero_metrics])
        diff = real_mean - zero_mean
        print(f"  {key:12s} {real_mean:12.4f} {zero_mean:12.4f} {diff:+10.4f}")
    print(f"{'='*60}")
