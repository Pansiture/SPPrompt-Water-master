import os
import sys
import glob
import logging
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
IMG_DIR = '/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/imgs'
PROMPT_DIR = '/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/prompt_mask_256'
GT_DIR = '/root/autodl-tmp/SPPrompt-Water-master/data/GID_processed/level0/val/gts'
SAM_CKPT = '/root/autodl-tmp/SPPrompt-Water-master/pretrain/sam_vit_b_01ec64.pth'
SWIN_CKPT = '/root/autodl-tmp/SPPrompt-Water-master/pretrain/swin_tiny_patch4_window7_224.pth'
RESUME = '/root/autodl-tmp/SPPrompt-Water-master/work_dir/GID_levelfusion_25epoch_e23_0.8194_Valscore2.6695/GID_levelfusion_25epoch_e23_0.8194_Valscore2.6695.pth'
OUTPUT_DIR = '/root/autodl-tmp/SPPrompt-Water-master/output/inference_gid_best'
LOG_FILE = os.path.join(OUTPUT_DIR, 'inference.log')
MAX_IMAGES = 50  # 快速验证只跑50张

os.makedirs(os.path.join(OUTPUT_DIR, 'real_prompt'), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, 'zero_prompt'), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

device = torch.device('cuda')
model = SPPromptWaterNet(SAM_CKPT, SWIN_CKPT, freeze_prompt=True).to(device).eval()

logger.info("=" * 60)
logger.info("Inference Configuration")
logger.info("=" * 60)
logger.info("  IMG_DIR:        %s", IMG_DIR)
logger.info("  PROMPT_DIR:     %s", PROMPT_DIR)
logger.info("  GT_DIR:         %s", GT_DIR)
logger.info("  SAM_CKPT:       %s", SAM_CKPT)
logger.info("  SWIN_CKPT:      %s", SWIN_CKPT)
logger.info("  RESUME:         %s", RESUME)
logger.info("  OUTPUT_DIR:     %s", OUTPUT_DIR)
logger.info("  MAX_IMAGES:     %s", MAX_IMAGES)
logger.info("  Device:         %s", device)
logger.info("=" * 60)

logger.info("[Init] Loading trained weight: %s", RESUME)
ckpt = torch.load(RESUME, map_location=device)
if 'model' in ckpt:
    model.load_state_dict(ckpt['model'])
    logger.info("[Init] Loaded 'model' key from checkpoint")
else:
    model.load_state_dict(ckpt)
    logger.info("[Init] Loaded full checkpoint")
logger.info("[Init] Model ready for inference")


def run_pyramid(image, real_prompt=None, mode='real'):
    image = image.to(device)
    B, C, H, W = image.shape
    n_level = 2
    
    X = image
    memory = {}
    for i in range(n_level + 1):
        if i > 0:
            X = F.interpolate(X, scale_factor=0.5, mode='area')
        memory[f'layer{i}_X'] = X
    
    for level in range(n_level, -1, -1):
        X_level = memory[f'layer{level}_X']
        H_l, W_l = X_level.shape[2], X_level.shape[3]
        
        if level == n_level:
            if mode == 'zero':
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
    img_files = sorted(glob.glob(os.path.join(IMG_DIR, '*.png')))[:MAX_IMAGES]
    metrics_list = []
    
    logger.info("[Run] Prompt mode: %s, images: %d", prompt_mode.upper(), len(img_files))
    
    for i, img_path in enumerate(img_files):
        basename = os.path.basename(img_path)
        
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img).unsqueeze(0).float()
        
        if prompt_mode == 'real':
            prompt_path = os.path.join(PROMPT_DIR, basename)
            if os.path.exists(prompt_path):
                prompt = io.imread(prompt_path)
                prompt = np.expand_dims(prompt, axis=0) / 255.0
            else:
                prompt = np.zeros((1, img.shape[2], img.shape[3]))
            real_prompt = torch.tensor(prompt).unsqueeze(0).float()
            pred = run_pyramid(img, real_prompt=real_prompt, mode='real')
        else:
            pred = run_pyramid(img, mode='zero')
        
        gt_path = os.path.join(GT_DIR, basename)
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
        
        # Save prediction image
        out_dir = os.path.join(OUTPUT_DIR, prompt_mode + '_prompt')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, basename)
        Image.fromarray((pred_arr * 255).astype(np.uint8)).save(out_path)
        
        metrics = evaluate_all_metrics(pred_arr, gt)
        metrics_list.append(metrics)
        
        if (i + 1) % 10 == 0:
            mean_miou = np.mean([m['mIoU'] for m in metrics_list])
            logger.info('  %d/%d done, current mean mIoU: %.4f', i+1, len(img_files), mean_miou)
    
    logger.info("=" * 60)
    logger.info("Prompt Mode: %s", prompt_mode.upper())
    logger.info("Evaluated images: %d", len(metrics_list))
    for key in ['mIoU', 'Water_IoU', 'Acc', 'Precision', 'Recall', 'F1']:
        vals = [m[key] for m in metrics_list]
        logger.info("  %s: %.4f (min: %.4f, max: %.4f)", key, np.mean(vals), np.min(vals), np.max(vals))
    logger.info("=" * 60)
    
    return metrics_list


if __name__ == '__main__':
    real_metrics = run_inference(prompt_mode='real')
    zero_metrics = run_inference(prompt_mode='zero')
    
    logger.info("\n" + "=" * 60)
    logger.info("FINAL COMPARISON (%d images)", MAX_IMAGES)
    logger.info("=" * 60)
    logger.info("  %-12s %-12s %-12s %-10s", "Metric", "Real", "Zero", "Diff")
    logger.info("  " + "-" * 46)
    for key in ['mIoU', 'Water_IoU', 'Acc', 'Precision', 'Recall', 'F1']:
        real_mean = np.mean([m[key] for m in real_metrics])
        zero_mean = np.mean([m[key] for m in zero_metrics])
        diff = real_mean - zero_mean
        logger.info("  %-12s %-12.4f %-12.4f %-+10.4f", key, real_mean, zero_mean, diff)
    logger.info("=" * 60)
