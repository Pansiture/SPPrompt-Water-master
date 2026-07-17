# author MengFanlin + EDL modification
# time 2025/3/31
# filename PyramidPromptSeg_edl
# description: Pyramid segmentation with Uncertainty-Guided Progressive Fusion (UGPF) and EDL

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from skimage import io
import time
from SPP_model.modeling.prompt_water_net_edl import SPPromptWaterNetEDL
from SPP_model.modeling.edl_utils import evidence_to_prob_uncertainty, uncertainty_guided_fusion
import argparse
import os
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default=r"/autodl-pub/glh_data/level0/val/imgs")
parser.add_argument("--prompt_path", type=str, default=None)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--resume_dir", type=str, default="./work_dir", help="dir with layer{X}.pth (multi-scale training weights)")
parser.add_argument("--output_dir", type=str, default=r"./output/pyramid_edl")
parser.add_argument("--SwintransformerPretrain", type=str, default="./pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth")
parser.add_argument("--promptcp", type=str, default=r"", help="SAM vit-b checkpoint path")
parser.add_argument("--uncertainty_threshold", type=float, default=0.5, help="Uncertainty threshold for prompt gating")
parser.add_argument("--fuse_scales", type=str2bool, default=True, help="Use UGPF to fuse all scales at the end")
args = parser.parse_args()


class PyramidSegmenterEDL:
    """
    Pyramid segmentation with EDL evidence and Uncertainty-Guided Progressive Fusion (UGPF).
    Key features:
    1. Each level outputs evidence, probability, and uncertainty maps.
    2. Prompt propagation uses uncertainty as gate (low uncertainty -> trust parent mask).
    3. Final UGPF fuses all scales weighted by confidence (1 - uncertainty).
    """

    def __init__(self, prompt_net, output_dir='./output_edl', save_intermediate=True,
                 resume_dir=None, save_intermediate2=True, uncertainty_threshold=0.5, fuse_scales=True):
        self.prompt_net = prompt_net.to('cuda').eval()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.save_intermediate = save_intermediate
        self.save_intermediate2 = save_intermediate2
        self.memory_cache = {}
        self.resume_dir = Path(resume_dir) if resume_dir else None
        self.current_prefix = ""
        self.uncertainty_threshold = uncertainty_threshold
        self.fuse_scales = fuse_scales

    def _load_layer_weights(self, level):
        """Load layer-specific weights if available."""
        if self.resume_dir is None:
            return
        pth = self.resume_dir / f"layer{level}.pth"
        if pth.exists():
            print(f"[Load weight] Using weights: {pth}")
            ckpt = torch.load(pth, map_location='cuda')
            if "model" in ckpt:
                self.prompt_net.load_state_dict(ckpt["model"])
            else:
                self.prompt_net.load_state_dict(ckpt)

    def _pad_to_pow2_square(self, image):
        _, _, H, W = image.shape
        max_hw = max(H, W)
        target_size = max(1024, 2 ** ((max_hw - 1).bit_length()))
        pad_h, pad_w = target_size - H, target_size - W
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='constant', value=0)
        return image, (H, W)

    def _save_png(self, tensor, path):
        tensor = tensor.squeeze().detach().cpu()
        if tensor.ndim == 3:
            img = tensor.byte().numpy().transpose(1, 2, 0)
            Image.fromarray(img).save(path)
        elif tensor.ndim == 2:
            img = tensor.byte().numpy()
            Image.fromarray(img).save(path)
        elif tensor.ndim == 1:
            # Scalar or small vector, skip
            pass
        else:
            img = (tensor * 255).clamp(0, 255).numpy().astype(np.uint8)
            Image.fromarray(img).save(path)

    def _save_heatmap(self, tensor, path, cmap='jet'):
        """Save uncertainty/probability heatmap."""
        tensor = tensor.squeeze().detach().cpu().numpy()
        if tensor.ndim == 3:
            tensor = tensor[0]  # take first channel
        # Normalize to 0-255
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
        self.current_prefix = prefix
        image = image.to('cuda')
        prompt = prompt.to('cuda')
        image, orig_size = self._pad_to_pow2_square(image)
        prompt, _ = self._pad_to_pow2_square(prompt)

        B, C, H, W = image.shape
        n_level = 2
        print(f"[Pyramid Config] Padded image size: {H}x{W}, levels: {n_level} (0~{n_level})")

        layer_patch_sizes = [1024, 1024, 1024]

        # Build pyramid
        X, P = image, prompt
        for i in range(n_level + 1):
            if i > 0:
                X = F.interpolate(X, scale_factor=0.5, mode='area')
                P = F.interpolate(P, scale_factor=0.5, mode='area')
            self.memory_cache[f'layer{i}_X'] = X
            self.memory_cache[f'layer{i}_P'] = P
            print(f"[Pyramid Build] Layer {i}: size {X.shape[2]}x{X.shape[3]}")
            if self.save_intermediate:
                self._save_png(X, self.output_dir / f'{prefix}_layer{i}_X.png')

        # Top-down inference with uncertainty
        for level in range(n_level, -1, -1):
            print(f"\n[Inference] ==== Processing level {level} ====")
            self._load_layer_weights(level)

            X_level = self.memory_cache[f'layer{level}_X']
            patch_size = layer_patch_sizes[level]
            H_level, W_level = X_level.shape[2], X_level.shape[3]
            num_splits = max(1, H_level // patch_size)

            print(f"[Inference] Level {level}: image={H_level}x{W_level}, patch_size={patch_size}, splits={num_splits}x{num_splits}")

            X_patches = self._crop_tensor(X_level, num_splits)

            # Prepare prompt: use parent probability if confident, else use original prompt
            if level == n_level:
                P_level = self.memory_cache[f'layer{level}_P']
                P_patches = self._crop_tensor(P_level, num_splits)
            else:
                parent_prob = self.memory_cache[f'layer{level+1}_prob']
                parent_unc = self.memory_cache[f'layer{level+1}_unc']

                # Uncertainty-gated prompt: if parent is uncertain, fallback to original prompt
                P_level = self.memory_cache[f'layer{level}_P']
                # Resize parent prob/unc to match current level size before cropping
                parent_prob = F.interpolate(parent_prob, size=P_level.shape[2:], mode='bilinear', align_corners=False)
                parent_unc = F.interpolate(parent_unc, size=P_level.shape[2:], mode='bilinear', align_corners=False)
                P_patches_base = self._crop_tensor(P_level, num_splits)
                P_patches_parent = self._crop_tensor(parent_prob, num_splits)
                U_patches_parent = self._crop_tensor(parent_unc, num_splits)

                P_patches = {}
                for key in P_patches_base.keys():
                    u = U_patches_parent[key]
                    # Gate: if uncertainty > threshold, use original prompt (less trust)
                    # else use parent probability (more trust)
                    weight = (u < self.uncertainty_threshold).float()
                    P_patches[key] = weight * P_patches_parent[key] + (1 - weight) * P_patches_base[key]

            # Patch-wise inference with EDL
            Y_prob_patches = {}
            Y_unc_patches = {}
            Y_mask_patches = {}
            for (row, col), Xi in X_patches.items():
                Pi = P_patches[(row, col)]

                if self.save_intermediate2:
                    self._save_png(Xi, self.output_dir / f'{prefix}_layer{level}_X_{row}_{col}.png')
                    self._save_png(Pi * 255, self.output_dir / f'{prefix}_layer{level}_P_{row}_{col}.png')

                Xi_model = F.interpolate(Xi, size=(1024, 1024), mode='bilinear', align_corners=False)
                Pi_model = F.interpolate(Pi, size=(256, 256), mode='bilinear', align_corners=False)

                with torch.no_grad():
                    evidence = self.prompt_net(Xi_model, Pi_model, return_evidence=True)
                    prob, uncertainty = evidence_to_prob_uncertainty(evidence)
                    Yi_prob = prob[:, 1:2, :, :]  # foreground probability
                    Yi_unc = uncertainty

                # Resize back to patch size
                Yi_prob = F.interpolate(Yi_prob, size=(Xi.shape[2], Xi.shape[3]), mode='bilinear', align_corners=False)
                Yi_unc = F.interpolate(Yi_unc, size=(Xi.shape[2], Xi.shape[3]), mode='bilinear', align_corners=False)
                Yi_mask = (Yi_prob > 0.5).float()

                Y_prob_patches[(row, col)] = Yi_prob
                Y_unc_patches[(row, col)] = Yi_unc
                Y_mask_patches[(row, col)] = Yi_mask

                if self.save_intermediate2:
                    self._save_heatmap(Yi_prob, self.output_dir / f'{prefix}_layer{level}_prob_{row}_{col}.png')
                    self._save_heatmap(Yi_unc, self.output_dir / f'{prefix}_layer{level}_unc_{row}_{col}.png')
                    self._save_png(Yi_mask * 255, self.output_dir / f'{prefix}_layer{level}_Y_{row}_{col}.png')

                print(f"[Inference] Patch ({row},{col}): prob range=[{Yi_prob.min():.3f},{Yi_prob.max():.3f}], unc mean={Yi_unc.mean():.3f}")

            # Merge patches
            Y_prob_level = self._merge_patches(Y_prob_patches)
            Y_unc_level = self._merge_patches(Y_unc_patches)
            Y_mask_level = self._merge_patches(Y_mask_patches)

            self.memory_cache[f'layer{level}_prob'] = Y_prob_level
            self.memory_cache[f'layer{level}_unc'] = Y_unc_level
            self.memory_cache[f'layer{level}_Y'] = Y_mask_level

            print(f"[Inference] Level {level} merged: prob shape={Y_prob_level.shape}, unc mean={Y_unc_level.mean():.3f}")

            if self.save_intermediate:
                self._save_heatmap(Y_prob_level, self.output_dir / f'{prefix}_layer{level}_prob.png')
                self._save_heatmap(Y_unc_level, self.output_dir / f'{prefix}_layer{level}_unc.png')
                self._save_png(Y_mask_level * 255, self.output_dir / f'{prefix}_layer{level}_Y.png')

        # Uncertainty-Guided Progressive Fusion (UGPF) across all scales
        if self.fuse_scales:
            print("\n[UGPF] Fusing multi-scale predictions with uncertainty-guided weights...")
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

            # Weighted fusion: higher confidence (1 - uncertainty) gets higher weight
            weights = [1.0 - u for u in level_uncerts]
            sum_weights = sum(weights)
            prob_fused = sum(w * p for w, p in zip(weights, level_probs)) / (sum_weights + 1e-6)
            unc_fused = sum(w * u for w, u in zip(weights, level_uncerts)) / (sum_weights + 1e-6)

            final_result = (prob_fused > 0.5).float()
            final_prob = prob_fused
            final_unc = unc_fused

            # Save fused outputs
            self._save_heatmap(final_prob, self.output_dir / f'{prefix}_final_prob.png')
            self._save_heatmap(final_unc, self.output_dir / f'{prefix}_final_unc.png')
            self._save_png(final_result * 255, self.output_dir / f'{prefix}_final_result.png')
            print(f"[UGPF] Final uncertainty mean: {final_unc.mean():.4f}, max: {final_unc.max():.4f}")
        else:
            final_result = self.memory_cache['layer0_Y'][:, :, :orig_size[0], :orig_size[1]]
            final_path = self.output_dir / f'{prefix}_final_result.png'
            self._save_png(final_result * 255, final_path)
            print(f"[Output] Final result saved: {final_path}")

        self.memory_cache.clear()
        del image, prompt, X, P
        return final_result


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("[Config] Pyramid EDL Segmentation - UGPF Enabled")
    print("=" * 60)
    print(f"[Config] Device:              {device}")
    print(f"[Config] Image folder:        {args.image_folder}")
    print(f"[Config] Prompt path:         {args.prompt_path if args.prompt_path else 'None (ZERO PROMPT)'}")
    print(f"[Config] Resume weight:       {args.resume if args.resume else 'None'}")
    print(f"[Config] Resume dir:          {args.resume_dir}")
    print(f"[Config] Output dir:          {args.output_dir}")
    print(f"[Config] Uncertainty thresh:  {args.uncertainty_threshold}")
    print(f"[Config] Fuse scales (UGPF):  {args.fuse_scales}")
    print("=" * 60)

    prompt_net = SPPromptWaterNetEDL(
        args.promptcp if args.promptcp else None,
        args.SwintransformerPretrain,
        freeze_prompt=True
    ).to(device)

    if args.resume and os.path.isfile(args.resume):
        print(f"[Init] Loading TRAINED weight: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if "model" in ckpt:
            prompt_net.load_state_dict(ckpt["model"])
        else:
            prompt_net.load_state_dict(ckpt)
    else:
        print("[WARNING] No trained weight loaded! Using random init for non-backbone parts.")

    segmenter = PyramidSegmenterEDL(
        prompt_net,
        output_dir=args.output_dir,
        save_intermediate=True,
        resume_dir=args.resume_dir,
        save_intermediate2=True,
        uncertainty_threshold=args.uncertainty_threshold,
        fuse_scales=args.fuse_scales
    )

    input_dir = Path(args.image_folder)
    image_list = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")])
    print(f"Found {len(image_list)} images in {input_dir}")

    for img_path in image_list:
        prefix = img_path.stem
        print(f"\n==== Processing {prefix} ====")
        img = io.imread(img_path)
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        image = torch.tensor(img).float()

        if args.prompt_path and os.path.isfile(args.prompt_path):
            prompt_path = Path(args.prompt_path)
            if prompt_path.suffix.lower() in ('.tif', '.tiff'):
                prompt = io.imread(args.prompt_path, plugin='tifffile')
            else:
                prompt = io.imread(args.prompt_path)
            prompt = TF.to_tensor(prompt).unsqueeze(0)
        else:
            prompt = torch.zeros(1, 1, image.shape[2], image.shape[3])
        prompt = prompt / 255

        result = segmenter.run(image, prompt, prefix=prefix)

    print(f"\nAll done. Total time: {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
