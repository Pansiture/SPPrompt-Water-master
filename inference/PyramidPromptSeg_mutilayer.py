# author MengFanlin
# time 2025/3/31
# filename PyramidPromptSeg
# description: Pyramid segmentation with per-layer weights (Batch version)

import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF
from skimage import io
import time
from SPP_model.modeling.prompt_water_net import SPPromptWaterNet
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--image_folder", type=str, default=r"/autodl-pub/glh_data/level0/val/imgs")
parser.add_argument("--prompt_path", type=str, default=None)
parser.add_argument("--resume", type=str, default=None)
parser.add_argument("--resume_dir", type=str, default="./work_dir", help="dir with layer{X}.pth (multi-scale training weights)")
parser.add_argument("--output_dir", type=str, default=r"./output/pyramid")
parser.add_argument("--SwintransformerPretrain", type=str, default="./pretrain/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth")
parser.add_argument("--promptcp", type=str, default=r"", help="SAM vit-b checkpoint path")
args = parser.parse_args()


class PyramidSegmenterPNG:
    def __init__(self, prompt_net, output_dir='./output_png', save_intermediate=False, resume_dir=None, save_intermediate2=False):
        self.prompt_net = prompt_net.to('cuda').eval()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.save_intermediate = save_intermediate
        self.save_intermediate2 = save_intermediate2
        self.memory_cache = {}
        self.resume_dir = Path(resume_dir) if resume_dir else None
        self.current_prefix = ""  # 新增：记录当前图片前缀

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
        else:
            # Silently use previous weights for missing layers
            pass

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
        """增加 prefix 参数，使输出文件带上图片名前缀"""
        self.current_prefix = prefix
        image = image.to('cuda')
        prompt = prompt.to('cuda')
        image, orig_size = self._pad_to_pow2_square(image)
        prompt, _ = self._pad_to_pow2_square(prompt)

        B, C, H, W = image.shape
        # 固定为3层金字塔（level0, level1, level2），对应论文设定
        n_level = 2
        print(f"[Pyramid Config] Padded image size: {H}x{W}, fixed pyramid levels: {n_level} (layers 0~{n_level})")

        # 各层模型输入 patch 尺寸（在金字塔图像坐标系中，模型输入固定为 1024×1024）
        # level2 (1/4 图): 1024×1024 整图输入
        # level1 (1/2 图): 1024×1024 patch，对应原图 2048×2048
        # level0 (原图):   1024×1024 patch，对应原图 1024×1024
        layer_patch_sizes = [1024, 1024, 1024]  # layer 0, 1, 2

        # 构建金字塔
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

        # 自顶向下推理
        for level in range(n_level, -1, -1):
            print(f"\n[Inference] ==== Processing level {level} ====")
            self._load_layer_weights(level)

            X_level = self.memory_cache[f'layer{level}_X']
            patch_size = layer_patch_sizes[level]
            H_level, W_level = X_level.shape[2], X_level.shape[3]
            num_splits = max(1, H_level // patch_size)

            print(f"[Inference] Level {level}: image={H_level}x{W_level}, patch_size={patch_size}, splits={num_splits}x{num_splits}")

            # 切分图像
            X_patches = self._crop_tensor(X_level, num_splits)

            # 准备 prompt
            if level == n_level:
                # 顶层：使用原始下采样后的 prompt
                P_level = self.memory_cache[f'layer{level}_P']
                P_patches = self._crop_tensor(P_level, num_splits)
            else:
                # 下层：使用上一层输出的 mask 作为 prompt
                parent_Y = self.memory_cache[f'layer{level+1}_Y']
                # parent 尺寸是当前层的一半，切分份数相同
                P_patches = self._crop_tensor(parent_Y, num_splits)

            # 逐 patch 推理
            Y_patches = {}
            U_patches = {}
            for (row, col), Xi in X_patches.items():
                Pi = P_patches[(row, col)]

                if self.save_intermediate2:
                    self._save_png(Xi, self.output_dir / f'{prefix}_layer{level}_X_{row}_{col}.png')
                    self._save_png(Pi * 255, self.output_dir / f'{prefix}_layer{level}_P_{row}_{col}.png')

                # Resize 到模型输入尺寸
                Xi_model = F.interpolate(Xi, size=(1024, 1024), mode='bilinear', align_corners=False)
                Pi_model = F.interpolate(Pi, size=(256, 256), mode='bilinear', align_corners=False)

                with torch.no_grad():
                    seg_logits, evidence, prob_256, uncertainty_256, consistency = self.prompt_net(Xi_model, Pi_model)
                    Yi_model = torch.sigmoid(seg_logits)
                
                # Resize 回原始 patch 尺寸
                Yi = F.interpolate(Yi_model, size=(Xi.shape[2], Xi.shape[3]), mode='bilinear', align_corners=False)
                
                # UGPF: 不确定性引导的 prompt 融合
                u_resized = F.interpolate(uncertainty_256, size=(Xi.shape[2], Xi.shape[3]), mode='bilinear', align_corners=False)
                adaptive_prompt = Yi * (1.0 - u_resized)
                
                Y_patches[(row, col)] = adaptive_prompt
                U_patches[(row, col)] = u_resized
                if self.save_intermediate2:
                    self._save_png(adaptive_prompt * 255, self.output_dir / f'{prefix}_layer{level}_Y_{row}_{col}.png')
                print(f"[Inference] Patch ({row},{col}): Xi_in={Xi.shape}, Pi_in={Pi.shape}, model_in=1024x1024/256x256, Yi_out={Yi.shape}, positive={(Yi > 0).sum().item()}")

            # 合并 patches
            Y_level = self._merge_patches(Y_patches)
            U_level = self._merge_patches(U_patches) if U_patches else None
            self.memory_cache[f'layer{level}_Y'] = Y_level
            if U_level is not None:
                self.memory_cache[f'layer{level}_U'] = U_level
            print(f"[Inference] Level {level} merged output: {Y_level.shape}, positive pixels={(Y_level > 0).sum().item()}")

            if self.save_intermediate:
                self._save_png(Y_level * 255, self.output_dir / f'{prefix}_layer{level}_Y.png')
                if U_level is not None:
                    u_vis = (U_level[0, 0].detach().cpu().numpy() * 255).astype(np.uint8)
                    Image.fromarray(u_vis).save(self.output_dir / f'{prefix}_layer{level}_U.png')

        # 最终结果
        final_result = self.memory_cache['layer0_Y'][:, :, :orig_size[0], :orig_size[1]]
        final_path = self.output_dir / f'{prefix}_final_result.png'
        self._save_png(final_result * 255, final_path)
        print(f"[Output] Final result saved: {final_path}, shape={final_result.shape}, positive pixels={(final_result > 0).sum().item()}")

        self.memory_cache.clear()
        del image, prompt, X, P
        return final_result


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("[Config] Pyramid Prompt Segmentation - Startup Config")
    print("=" * 60)
    print(f"[Config] Device:              {device}")
    print(f"[Config] Image folder:        {args.image_folder}")
    print(f"[Config] Prompt path:         {args.prompt_path if args.prompt_path else 'None (ZERO PROMPT)'}")
    print(f"[Config] Resume weight:       {args.resume if args.resume else 'None'}")
    print(f"[Config] Resume dir:          {args.resume_dir}")
    print(f"[Config] Output dir:          {args.output_dir}")
    print(f"[Config] Swin pretrain:       {args.SwintransformerPretrain}")
    print(f"[Config] SAM checkpoint:      {args.promptcp if args.promptcp else 'None'}")
    print("=" * 60)

    prompt_net = SPPromptWaterNet(args.promptcp if args.promptcp else None, args.SwintransformerPretrain, freeze_prompt=True).to(device)

    # fallback: load global resume if provided
    if args.resume and os.path.isfile(args.resume):
        print(f"[Init] Loading TRAINED weight: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if "model" in ckpt:
            prompt_net.load_state_dict(ckpt["model"])
            print(f"[Init] Loaded 'model' key from checkpoint")
        else:
            prompt_net.load_state_dict(ckpt)
            print(f"[Init] Loaded full checkpoint")
    else:
        print(f"[WARNING] No trained weight loaded! Using random init for non-backbone parts.")

    segmenter = PyramidSegmenterPNG(
        prompt_net,
        output_dir=args.output_dir,
        save_intermediate=True,
        resume_dir=args.resume_dir,
        save_intermediate2=True
    )

    input_dir = Path(args.image_folder)
    image_list = sorted([p for p in input_dir.glob("*") if p.suffix.lower() in (".tif", ".tiff", ".png", ".jpg", ".jpeg")])


    print(f"Found {len(image_list)} images in {input_dir}")

    for img_path in image_list:
        prefix = img_path.stem
        print(f"\n==== Processing {prefix} ====")
        img = io.imread(img_path)
        if img.ndim == 2:  # 灰度图兼容
            img = np.expand_dims(img, axis=2)
        img = np.transpose(img, (2, 0, 1))  # 3,H,W
        img = np.expand_dims(img, axis=0)  # 1,3,H,W
        image = torch.tensor(img).float()

        # prompt 若无路径，则自动生成空白
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
