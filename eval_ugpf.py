#!/usr/bin/env python
"""
UGPF 推理 + 指标评估脚本
支持：批量推理、多尺度 UGPF 融合、mIoU/Acc/F1 计算、日志记录
"""
import os
import sys
import argparse
import logging
import time
import csv
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from skimage import io
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from SPP_model.modeling.prompt_water_net_edl import SPPromptWaterNetEDL
from SPP_model.modeling.edl_utils import evidence_to_prob_uncertainty


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def calculate_metrics(preds, labels, num_classes=2):
    preds = preds.cpu().numpy() if torch.is_tensor(preds) else preds
    labels = labels.cpu().numpy() if torch.is_tensor(labels) else labels
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


class PyramidSegmenterEDL:
    def __init__(self, prompt_net, output_dir='./output_ugpf', uncertainty_threshold=0.5, fuse_scales=True, save_visuals=True):
        self.prompt_net = prompt_net.to('cuda').eval()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.uncertainty_threshold = uncertainty_threshold
        self.fuse_scales = fuse_scales
        self.save_visuals = save_visuals
        self.memory_cache = {}

    def _pad_to_pow2_square(self, image):
        _, _, H, W = image.shape
        max_hw = max(H, W)
        target_size = max(1024, 2 ** ((max_hw - 1).bit_length()))
        pad_h, pad_w = target_size - H, target_size - W
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return image, (H, W)

    def _save_png(self, tensor, path):
        if not self.save_visuals:
            return
        tensor = tensor.squeeze().detach().cpu()
        if tensor.ndim == 2:
            img = (tensor * 255).clamp(0, 255).numpy().astype(np.uint8)
            Image.fromarray(img, mode='L').save(path)
        elif tensor.ndim == 3:
            img = (tensor * 255).clamp(0, 255).numpy().astype(np.uint8)
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            Image.fromarray(img).save(path)

    def _save_heatmap(self, tensor, path):
        if not self.save_visuals:
            return
        tensor = tensor.squeeze().detach().cpu().numpy()
        if tensor.ndim == 3:
            tensor = tensor[0]
        vmin, vmax = tensor.min(), tensor.max()
        if vmax - vmin > 1e-6:
            norm = (tensor - vmin) / (vmax - vmin)
        else:
            norm = np.zeros_like(tensor)
        img = (norm * 255).astype(np.uint8)
        Image.fromarray(img, mode='L').save(path)

    def _crop_tensor(self, tensor, num_splits):
        B, C, H, W = tensor.shape
        patch_size = H // num_splits
        patches = {}
        for row in range(num_splits):
            for col in range(num_splits):
                crop = tensor[..., row * patch_size:(row + 1) * patch_size,
                              col * patch_size:(col + 1) * patch_size]
                patches[(row, col)] = crop
        return patches

    def _merge_patches(self, patches):
        if not patches:
            return None
        rows = max(row for row, col in patches.keys()) + 1
        cols = max(col for row, col in patches.keys()) + 1
        merged_rows = []
        for row in range(rows):
            row_patches = [patches[(row, col)] for col in range(cols)]
            merged_row = torch.cat(row_patches, dim=3)
            merged_rows.append(merged_row)
        return torch.cat(merged_rows, dim=2)

    def run(self, image, prompt, prefix=""):
        self.memory_cache.clear()
        image = image.to('cuda')
        prompt = prompt.to('cuda')
        image, orig_size = self._pad_to_pow2_square(image)
        prompt, _ = self._pad_to_pow2_square(prompt)

        B, C, H, W = image.shape
        n_level = 2

        # Build pyramid
        X, P = image, prompt
        for i in range(n_level + 1):
            if i > 0:
                X = F.interpolate(X, scale_factor=0.5, mode='area')
                P = F.interpolate(P, scale_factor=0.5, mode='area')
            self.memory_cache[f'layer{i}_X'] = X
            self.memory_cache[f'layer{i}_P'] = P

        # Top-down inference
        for level in range(n_level, -1, -1):
            X_level = self.memory_cache[f'layer{level}_X']
            patch_size = 1024
            H_level, W_level = X_level.shape[2], X_level.shape[3]
            num_splits = max(1, H_level // patch_size)
            X_patches = self._crop_tensor(X_level, num_splits)

            if level == n_level:
                P_level = self.memory_cache[f'layer{level}_P']
                P_patches = self._crop_tensor(P_level, num_splits)
            else:
                parent_prob = self.memory_cache[f'layer{level+1}_prob']
                parent_unc = self.memory_cache[f'layer{level+1}_unc']
                P_level = self.memory_cache[f'layer{level}_P']
                parent_prob = F.interpolate(parent_prob, size=P_level.shape[2:], mode='bilinear', align_corners=False)
                parent_unc = F.interpolate(parent_unc, size=P_level.shape[2:], mode='bilinear', align_corners=False)
                P_patches_base = self._crop_tensor(P_level, num_splits)
                P_patches_parent = self._crop_tensor(parent_prob, num_splits)
                U_patches_parent = self._crop_tensor(parent_unc, num_splits)
                P_patches = {}
                for key in P_patches_base.keys():
                    u = U_patches_parent[key]
                    weight = (u < self.uncertainty_threshold).float()
                    P_patches[key] = weight * P_patches_parent[key] + (1 - weight) * P_patches_base[key]

            Y_prob_patches = {}
            Y_unc_patches = {}
            Y_mask_patches = {}
            for (row, col), Xi in X_patches.items():
                Pi = P_patches[(row, col)]
                Xi_model = F.interpolate(Xi, size=(1024, 1024), mode='bilinear', align_corners=False)
                Pi_model = F.interpolate(Pi, size=(256, 256), mode='bilinear', align_corners=False)
                with torch.no_grad():
                    evidence = self.prompt_net(Xi_model, Pi_model, return_evidence=True)
                    prob, uncertainty = evidence_to_prob_uncertainty(evidence)
                    Yi_prob = prob[:, 1:2, :, :]
                    Yi_unc = uncertainty
                Yi_prob = F.interpolate(Yi_prob, size=(Xi.shape[2], Xi.shape[3]), mode='bilinear', align_corners=False)
                Yi_unc = F.interpolate(Yi_unc, size=(Xi.shape[2], Xi.shape[3]), mode='bilinear', align_corners=False)
                Yi_mask = (Yi_prob > 0.5).float()
                Y_prob_patches[(row, col)] = Yi_prob
                Y_unc_patches[(row, col)] = Yi_unc
                Y_mask_patches[(row, col)] = Yi_mask

            Y_prob_level = self._merge_patches(Y_prob_patches)
            Y_unc_level = self._merge_patches(Y_unc_patches)
            Y_mask_level = self._merge_patches(Y_mask_patches)
            self.memory_cache[f'layer{level}_prob'] = Y_prob_level
            self.memory_cache[f'layer{level}_unc'] = Y_unc_level
            self.memory_cache[f'layer{level}_Y'] = Y_mask_level

        # UGPF fusion
        if self.fuse_scales:
            level_probs = []
            level_uncerts = []
            for level in range(n_level + 1):
                prob = self.memory_cache[f'layer{level}_prob'][:, :, :orig_size[0], :orig_size[1]]
                unc = self.memory_cache[f'layer{level}_unc'][:, :, :orig_size[0], :orig_size[1]]
                if level > 0:
                    prob = F.interpolate(prob, size=orig_size, mode='bilinear', align_corners=False)
                    unc = F.interpolate(unc, size=orig_size, mode='bilinear', align_corners=False)
                level_probs.append(prob)
                level_uncerts.append(unc)

            weights = [1.0 - u for u in level_uncerts]
            sum_weights = sum(weights)
            prob_fused = sum(w * p for w, p in zip(weights, level_probs)) / (sum_weights + 1e-6)
            unc_fused = sum(w * u for w, u in zip(weights, level_uncerts)) / (sum_weights + 1e-6)
            final_result = (prob_fused > 0.5).float()
            final_prob = prob_fused
            final_unc = unc_fused

            self._save_heatmap(final_prob, self.output_dir / f'{prefix}_final_prob.png')
            self._save_heatmap(final_unc, self.output_dir / f'{prefix}_final_unc.png')
            self._save_png(final_result, self.output_dir / f'{prefix}_final_result.png')
        else:
            final_result = self.memory_cache['layer0_Y'][:, :, :orig_size[0], :orig_size[1]]
            self._save_png(final_result, self.output_dir / f'{prefix}_final_result.png')

        self.memory_cache.clear()
        del image, prompt, X, P
        return final_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--gt_folder", type=str, required=True, help="Ground truth folder (png masks)")
    parser.add_argument("--prompt_folder", type=str, required=True, help="Prompt mask folder (png/tif)")
    parser.add_argument("--resume", type=str, required=True)
    parser.add_argument("--SwintransformerPretrain", type=str, required=True)
    parser.add_argument("--promptcp", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./output/ugpf_eval")
    parser.add_argument("--uncertainty_threshold", type=float, default=0.5)
    parser.add_argument("--fuse_scales", type=str2bool, default=True)
    parser.add_argument("--save_visuals", type=str2bool, default=True, help="Save prob/unc/result images")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "eval_log.txt")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("UGPF Evaluation")
    logger.info("=" * 60)
    logger.info("Resume: %s", args.resume)
    logger.info("Image: %s", args.image_folder)
    logger.info("GT: %s", args.gt_folder)
    logger.info("Prompt: %s", args.prompt_folder)
    logger.info("Output: %s", args.output_dir)
    logger.info("Fuse scales: %s", args.fuse_scales)
    logger.info("Uncertainty threshold: %.2f", args.uncertainty_threshold)
    logger.info("Save visuals: %s", args.save_visuals)
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_net = SPPromptWaterNetEDL(
        args.promptcp if args.promptcp else None,
        args.SwintransformerPretrain,
        freeze_prompt=True
    ).to(device)

    logger.info("Loading checkpoint: %s", args.resume)
    ckpt = torch.load(args.resume, map_location=device)
    if "model" in ckpt:
        prompt_net.load_state_dict(ckpt["model"])
    else:
        prompt_net.load_state_dict(ckpt)

    segmenter = PyramidSegmenterEDL(
        prompt_net,
        output_dir=args.output_dir,
        uncertainty_threshold=args.uncertainty_threshold,
        fuse_scales=args.fuse_scales,
        save_visuals=args.save_visuals
    )

    img_dir = Path(args.image_folder)
    gt_dir = Path(args.gt_folder)
    prompt_dir = Path(args.prompt_folder)

    img_list = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")])
    logger.info("Found %d images", len(img_list))

    all_preds = []
    all_gts = []
    per_image_results = []
    start_time = time.time()

    for img_path in tqdm(img_list, desc="Evaluating"):
        prefix = img_path.stem
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        image = torch.tensor(img).float()

        # Prompt
        prompt_path = prompt_dir / (prefix + ".png")
        if not prompt_path.exists():
            prompt_path = prompt_dir / (prefix + ".tif")
        if prompt_path.exists():
            pimg = io.imread(prompt_path)
            if pimg.ndim == 2:
                pimg = np.expand_dims(pimg, axis=2)
            pimg = np.transpose(pimg, (2, 0, 1))
            pimg = np.expand_dims(pimg, axis=0)
            pimg = pimg / 255.0
        else:
            pimg = np.zeros((1, 1, image.shape[2], image.shape[3]))
        prompt = torch.tensor(pimg).float()

        # GT
        gt_path = gt_dir / (prefix + ".png")
        if not gt_path.exists():
            gt_path = gt_dir / (prefix + ".tif")
        if gt_path.exists():
            gt = io.imread(gt_path)
            gt = np.expand_dims(gt, axis=0)
            gt = gt / 255.0
        else:
            logger.warning("GT not found for %s, skip", prefix)
            continue

        with torch.no_grad():
            pred = segmenter.run(image, prompt, prefix=prefix if args.save_visuals else "")

        pred_resized = F.interpolate(pred, size=gt.shape[-2:], mode='bilinear', align_corners=False)
        all_preds.append(pred_resized)
        all_gts.append(torch.tensor(gt).float())

        # Calculate per-image metrics
        img_miou, img_acc, img_f1 = calculate_metrics(pred_resized, torch.tensor(gt).float().unsqueeze(0))
        per_image_results.append({
            "image": prefix,
            "mIoU": img_miou,
            "Acc": img_acc,
            "F1": img_f1
        })

    all_preds = torch.cat(all_preds, dim=0)
    all_gts = torch.cat(all_gts, dim=0)
    miou, acc, f1 = calculate_metrics(all_preds, all_gts)
    elapsed = time.time() - start_time

    logger.info("=" * 60)
    logger.info("Evaluation Complete")
    logger.info("Images: %d", len(img_list))
    logger.info("Time: %.2f s (%.2f s/img)", elapsed, elapsed / len(img_list))
    logger.info("mIoU: %.4f", miou)
    logger.info("Acc:  %.4f", acc)
    logger.info("F1:   %.4f", f1)
    logger.info("=" * 60)

    # Save per-image metrics
    per_image_csv = os.path.join(args.output_dir, "per_image_metrics.csv")
    with open(per_image_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "mIoU", "Acc", "F1"])
        for r in per_image_results:
            w.writerow([r["image"], r["mIoU"], r["Acc"], r["F1"]])
    logger.info("Per-image metrics saved to: %s", per_image_csv)

    csv_path = os.path.join(args.output_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["mIoU", miou])
        w.writerow(["Acc", acc])
        w.writerow(["F1", f1])
        w.writerow(["num_images", len(img_list)])
        w.writerow(["time_seconds", elapsed])
    logger.info("Metrics saved to: %s", csv_path)


if __name__ == "__main__":
    main()
