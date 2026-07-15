import os
import sys
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from skimage import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from SPP_model.modeling.prompt_water_net import SPPromptWaterNet

img_dir = '/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/imgs'
prompt_dir = '/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/prompt_mask_256'
gt_dir = '/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/gts'
sam_cp = '/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth'
swin_cp = '/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth'
resume = '/root/autodl-tmp/SPPrompt-Water-master/work_dir/GID_levelfusion_25epoch_e23_0.8194_Valscore2.6695/GID_levelfusion_25epoch_e23_0.8194_Valscore2.6695.pth'

MAX_IMAGES = 200
BATCH_SIZE = 2

device = torch.device('cuda')
model = SPPromptWaterNet(sam_cp, swin_cp, freeze_prompt=True).to(device).eval()
ckpt = torch.load(resume, map_location=device)
model.load_state_dict(ckpt['model'] if 'model' in ckpt else ckpt)

class GIDValDataset(Dataset):
    def __init__(self, img_dir, prompt_dir, gt_dir):
        self.img_dir = img_dir
        self.prompt_dir = prompt_dir
        self.gt_dir = gt_dir
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])[:MAX_IMAGES]
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        basename = self.img_files[idx]
        img = io.imread(os.path.join(self.img_dir, basename))
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        img = np.transpose(img, (2, 0, 1)).astype(np.float32)
        prompt_path = os.path.join(self.prompt_dir, basename)
        if os.path.exists(prompt_path):
            prompt = io.imread(prompt_path).astype(np.float32) / 255.0
        else:
            prompt = np.zeros((img.shape[1], img.shape[2]), dtype=np.float32)
        prompt = np.expand_dims(prompt, axis=0)
        gt_path = os.path.join(self.gt_dir, basename)
        gt = np.array(Image.open(gt_path).convert('L'))
        gt = (gt > 127).astype(np.uint8)
        return torch.from_numpy(img), torch.from_numpy(prompt), torch.from_numpy(gt), basename

def run_pyramid_batched(images, prompts, mode='real'):
    images = images.to(device)
    B, C, H, W = images.shape
    n_level = 2
    X, P = images, prompts.to(device) if prompts is not None else None
    memory = {}
    for i in range(n_level + 1):
        if i > 0:
            X = F.interpolate(X, scale_factor=0.5, mode='area')
            if P is not None:
                P = F.interpolate(P, scale_factor=0.5, mode='area')
        memory[f'layer{i}_X'] = X
        memory[f'layer{i}_P'] = P
    for level in range(n_level, -1, -1):
        X_level = memory[f'layer{level}_X']
        H_l, W_l = X_level.shape[2], X_level.shape[3]
        if level == n_level:
            if mode == 'zero':
                P_level = torch.zeros(B, 1, H_l, W_l, device=device)
            else:
                P_level = memory[f'layer{level}_P']
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
    return {'mIoU': miou, 'Water_IoU': water_iou, 'Acc': accuracy, 'Precision': precision[1], 'Recall': recall[1], 'F1': f1[1]}

def run_batched_inference(prompt_mode='real'):
    dataset = GIDValDataset(img_dir, prompt_dir, gt_dir)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, drop_last=False)
    metrics_list = []
    print(f"[Run] {prompt_mode.upper()} prompt, subset: {len(dataset)} images, batch={BATCH_SIZE}")
    for images, prompts, gts, basenames in dataloader:
        preds = run_pyramid_batched(images, prompts, mode=prompt_mode)
        preds = preds.squeeze(1).cpu().numpy()
        gts = gts.numpy()
        for j in range(preds.shape[0]):
            pred_arr = preds[j]
            gt = gts[j]
            if pred_arr.shape != gt.shape:
                pred_arr = np.array(Image.fromarray((pred_arr * 255).astype(np.uint8)).resize((gt.shape[1], gt.shape[0]), Image.NEAREST))
                pred_arr = (pred_arr > 127).astype(np.uint8)
            else:
                pred_arr = (pred_arr > 0.5).astype(np.uint8)
            metrics = evaluate_all_metrics(pred_arr, gt)
            metrics_list.append(metrics)
    print(f"  {prompt_mode.upper()} Done. mIoU={np.mean([m['mIoU'] for m in metrics_list]):.4f}, Water_IoU={np.mean([m['Water_IoU'] for m in metrics_list]):.4f}, F1={np.mean([m['F1'] for m in metrics_list]):.4f}")
    return metrics_list

real_metrics = run_batched_inference(prompt_mode='real')
zero_metrics = run_batched_inference(prompt_mode='zero')

print(f"\n{'='*60}")
print("  FINAL COMPARISON (200-image subset)")
print(f"{'='*60}")
for key in ['mIoU', 'Water_IoU', 'Acc', 'Precision', 'Recall', 'F1']:
    real_mean = np.mean([m[key] for m in real_metrics])
    zero_mean = np.mean([m[key] for m in zero_metrics])
    print(f"  {key:12s}: Real={real_mean:.4f}, Zero={zero_mean:.4f}, Diff={real_mean-zero_mean:+.4f}")
print(f"{'='*60}")
