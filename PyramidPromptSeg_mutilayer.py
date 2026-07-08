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

        original_image = image.clone()
        B, C, H, W = image.shape
        n_level = (H // 1024).bit_length() - 1
        print(f"Number of pyramid levels: {n_level}")

        X, P = image, prompt
        for i in range(n_level + 1):
            if i > 0:
                Hx, Wx = X.shape[2:]
                X = F.interpolate(X, size=(Hx // 2, Wx // 2), mode='area')
                P = F.interpolate(P, size=(Hx // 2, Wx // 2), mode='area')
            self.memory_cache[f'layer{i}_X'] = X
            if self.save_intermediate:
                self._save_png(X, self.output_dir / f'{prefix}_layer{i}_X.png')

        P = F.avg_pool2d(P, kernel_size=2)
        P = F.avg_pool2d(P, kernel_size=2)
        self.memory_cache[f'layer{n_level}_P'] = P
        if self.save_intermediate:
            self._save_png(P*255, self.output_dir / f'{prefix}_layer{n_level}_P.png')

        # ---- top layer inference ----
        self._load_layer_weights(n_level)
        X = self.memory_cache[f'layer{n_level}_X']
        P = self.memory_cache[f'layer{n_level}_P']
        with torch.no_grad():
            Yn = torch.sigmoid(self.prompt_net(X, P))
            Yn = (Yn > 0.5).float()
        self.memory_cache[f'layer{n_level}_Y'] = Yn

        if self.save_intermediate:
            self._save_png(Yn*255, self.output_dir / f'{prefix}_layer{n_level}_Y.png')

        # ---- pyramid refine ----
        for level in reversed(range(n_level + 1)):
            print(f"\nProcessing level {level}")

            self._load_layer_weights(level)
            Xn = self.memory_cache[f'layer{level}_X']
            Yn = self.memory_cache.get(f'layer{level}_Y', None)

            num_splits = 2 ** (n_level - level)
            if num_splits > 1:
                Xn_crops = self._crop_tensor(Xn, num_splits)
                Ys = {}
                for (row, col), Xi in Xn_crops.items():
                    parent_row = row // 2
                    parent_col = col // 2
                    parent_key = f'layer{level+1}_Y_{parent_row}_{parent_col}'

                    if parent_key not in self.memory_cache:
                        parent_Y = self.memory_cache[f'layer{level+1}_Y']
                        parent_crops = self._crop_tensor(parent_Y, 2 ** (n_level - (level + 1)))
                        for (r, c), crop in parent_crops.items():
                            self.memory_cache[f'layer{level+1}_Y_{r}_{c}'] = crop

                    Pi_parent = self.memory_cache[f'layer{level+1}_Y_{parent_row}_{parent_col}']
                    _, _, Hp, Wp = Pi_parent.shape
                    sub_h = max(1, Hp // 2)
                    sub_w = max(1, Wp // 2)
                    child_r = row % 2
                    child_c = col % 2
                    Pi = Pi_parent[..., child_r * sub_h:(child_r + 1) * sub_h,
                                  child_c * sub_w:(child_c + 1) * sub_w]

                    Pi = F.interpolate(Pi, scale_factor=0.5, mode='nearest')

                    with torch.no_grad():
                        if self.save_intermediate2:
                            self._save_png(Xi, self.output_dir / f'{prefix}_layer{level}_X_{row}_{col}.png')
                            self._save_png(Pi*255, self.output_dir / f'{prefix}_layer{level}_P_{row}_{col}.png')
                        Yi = torch.sigmoid(self.prompt_net(Xi, Pi))
                        Yi = (Yi > 0.5).float()
                    self.memory_cache[f'layer{level}_Y_{row}_{col}'] = Yi
                    if self.save_intermediate2:
                        self._save_png(Yi*255, self.output_dir / f'{prefix}_layer{level}_Y_{row}_{col}.png')
                    Ys[(row, col)] = Yi

                Yn = self._merge_patches(Ys)

            self.memory_cache[f'layer{level}_Y'] = Yn
            if self.save_intermediate:
                self._save_png(Yn*255, self.output_dir / f'{prefix}_layer{level}_Y.png')

        final_result = Yn[:, :, :orig_size[0], :orig_size[1]]
        self._save_png(final_result*255, self.output_dir / f'{prefix}_final_result.png')

        self.memory_cache.clear()
        del image, prompt, X, P, Yn

        return final_result


def main():
    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    prompt_net = SPPromptWaterNet(args.promptcp, args.SwintransformerPretrain, freeze_prompt=True).to(device)

    # fallback: load global resume if provided
    if args.resume and os.path.isfile(args.resume):
        print(f"[Init] load base weight: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if "model" in ckpt:
            prompt_net.load_state_dict(ckpt["model"])
        else:
            prompt_net.load_state_dict(ckpt)

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
