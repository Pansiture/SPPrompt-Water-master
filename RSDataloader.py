# author MengFanlin
# time 2024/7/2
# filename RSDataloader
# description:xxx
import os
import numpy as np
import os
join = os.path.join
import torch
from torch.utils.data import Dataset, DataLoader
import random
import glob
from skimage import io
from typing import Any, Dict, List, Tuple

# 

class PromptDataset_GID5(Dataset):
    def __init__(self, data_root,inference=False, bbox_shift=20):
        self.inference=inference
        self.data_root = data_root
        self.gt_path = join(data_root, "gts") #label
        self.img_path = join(data_root, "imgs")
        self.prompt_path = join(data_root, "prompt_mask_256")


        #这个里改数据路径
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.png"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if (os.path.isfile(join(self.img_path, os.path.splitext(os.path.basename(file))[0] + ".tif"))
                or os.path.isfile(join(self.img_path, os.path.splitext(os.path.basename(file))[0] + ".png")))
        ]
        self.bbox_shift = bbox_shift

        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def _augment(self, img, gt, prompt):
        """Apply spatial and color augmentations."""
        # Random horizontal flip
        if random.random() > 0.5:
            img = np.flip(img, axis=2).copy()
            gt = np.flip(gt, axis=2).copy()
            prompt = np.flip(prompt, axis=2).copy()

        # Random vertical flip
        if random.random() > 0.5:
            img = np.flip(img, axis=1).copy()
            gt = np.flip(gt, axis=1).copy()
            prompt = np.flip(prompt, axis=1).copy()

        # Random 90-degree rotations (0, 90, 180, 270)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            img = np.rot90(img, k, axes=(1, 2)).copy()
            gt = np.rot90(gt, k, axes=(1, 2)).copy()
            prompt = np.rot90(prompt, k, axes=(1, 2)).copy()

        # Color jitter removed: brightness & contrast changes hurt water segmentation
        # by confusing water with shadows/buildings in remote sensing imagery

        return img, gt, prompt

    def _resize_to_1024(self, img, is_mask=False):
        """Resize array to 1024x1024 if needed."""
        if img.shape[-2:] == (1024, 1024):
            return img
        import torch.nn.functional as F
        t = torch.from_numpy(img).float().unsqueeze(0)
        mode = 'nearest' if is_mask else 'bilinear'
        t = F.interpolate(t, size=(1024, 1024), mode=mode, align_corners=None if mode == 'nearest' else False)
        return t.squeeze(0).numpy()

    def _resize_prompt_to_256(self, img):
        """Resize prompt mask to 256x256 if needed."""
        if img.shape[-2:] == (256, 256):
            return img
        import torch.nn.functional as F
        t = torch.from_numpy(img).float().unsqueeze(0)
        t = F.interpolate(t, size=(256, 256), mode='nearest')
        return t.squeeze(0).numpy()

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_path_files[index])
        base_name = os.path.splitext(img_name)[0]

        tif_path = os.path.join(self.img_path, base_name + ".tif")
        png_path = os.path.join(self.img_path, base_name + ".png")

        if os.path.isfile(tif_path):
            img_1024 = io.imread(tif_path, plugin='tifffile')
        elif os.path.isfile(png_path):
            img_1024 = io.imread(png_path)
        else:
            raise FileNotFoundError(f"Image not found for {base_name} (.tif or .png)")

        img_1024 = np.transpose(img_1024, (2, 0, 1))  # (3, H, W)
        # Resize image and masks to 1024x1024 for uniform batching
        img_1024 = self._resize_to_1024(img_1024, is_mask=False)

        prompt_img = io.imread(join(self.prompt_path, img_name))  # (H, W)
        prompt_img = np.expand_dims(prompt_img, axis=0)  # (1, H, W)
        prompt_img = prompt_img / 255
        prompt_img = self._resize_prompt_to_256(prompt_img)  # ensure (1, 256, 256)

        gt = io.imread(join(self.gt_path, img_name))  # (H, W)
        gt = np.expand_dims(gt, axis=0)  # (1, H, W)
        gt = gt / 255
        gt = self._resize_to_1024(gt, is_mask=True)

        # Apply online augmentation during training
        if not self.inference:
            img_1024, gt, prompt_img = self._augment(img_1024, gt, prompt_img)
            # Paper-style default prompt token: 15% zero-prompt for robust top-layer inference
            if random.random() < 0.15:
                prompt_img = np.zeros_like(prompt_img)

        if self.inference:
            return (torch.tensor(img_1024).float(),torch.tensor(prompt_img).float(),join(self.gt_path, img_name))
        else:

            return (
                torch.tensor(img_1024).float(),
                torch.tensor(gt).long(),
                torch.tensor(prompt_img).float(),

                img_name,
            )


class PromptDataset_old(Dataset):
    def __init__(self, data_root,inference=False, bbox_shift=20):
        self.inference=inference
        self.data_root = data_root
        self.gt_path = join(data_root, "gts") #label
        self.img_path = join(data_root, "imgs")
        self.prompt_path = join(data_root, "prompt_mask_256")
        # self.prompt_gts_path = join(data_root, "prompt_gts")
        # self.swin_gts_path = join(data_root, "swin_gts")

        #这个里改数据路径
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.tif"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift

        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        # img_1024 = np.load(
        #     join(self.img_path, img_name), "r", allow_pickle=True
        # )  # (1024, 1024, 3)
        img_1024 = io.imread(os.path.join(self.img_path, img_name), plugin='tifffile')
        #img_1024 = io.imread(os.path.join(self.img_path, img_name))
        #img_1024=io.imread(join(self.img_path, img_name))

        # 移除第一个通道，只保留后三个通道
        # img_1024 = img_1024[:, :, 1:] #如果有GID近红外的话就取消注释
        # convert the shape to (3, H, W)

        img_1024 = np.transpose(img_1024, (2, 0, 1))
        prompt_img= io.imread(join(self.prompt_path, img_name))  # (1024, 1024, 1)
        prompt_img = np.expand_dims(prompt_img, axis=0)
        prompt_img = prompt_img / 255

        gt = io.imread(join(self.gt_path, img_name)) # multiple labels [0, 1,4,5...], (256,256)
        gt = np.expand_dims(gt, axis=0)
        gt = gt / 255

        if not self.inference:
            # Paper-style default prompt token: 15% zero-prompt for robust top-layer inference
            if random.random() < 0.15:
                prompt_img = np.zeros_like(prompt_img)

        if self.inference:
            return (torch.tensor(img_1024).float(),torch.tensor(prompt_img).float(),join(self.gt_path, img_name))
        else:

            return (
                torch.tensor(img_1024).float(),
                torch.tensor(gt).long(),
                torch.tensor(prompt_img).float(),
                # torch.tensor(prompt_gts).float(),
                # torch.tensor(swin_gts).float(),
                img_name,
            )

class PromptDataset_our_noprompt(Dataset):
    def __init__(self, data_root,inference=False, bbox_shift=20):
        self.inference=inference
        self.data_root = data_root
        self.gt_path = join(data_root, "gts") #label
        self.img_path = join(data_root, "imgs")
        # self.prompt_path = join(data_root, "prompt_mask_256")
        # self.prompt_gts_path = join(data_root, "prompt_gts")
        # self.swin_gts_path = join(data_root, "swin_gts")

        #这个里改数据路径
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.tif"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift

        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        # img_1024 = np.load(
        #     join(self.img_path, img_name), "r", allow_pickle=True
        # )  # (1024, 1024, 3)
        img_1024 = io.imread(os.path.join(self.img_path, img_name), plugin='tifffile')
        #img_1024 = io.imread(os.path.join(self.img_path, img_name))
        #img_1024=io.imread(join(self.img_path, img_name))

        # 移除第一个通道，只保留后三个通道
        # img_1024 = img_1024[:, :, 1:] #如果有GID近红外的话就取消注释
        # convert the shape to (3, H, W)

        img_1024 = np.transpose(img_1024, (2, 0, 1))
        # prompt_img= io.imread(join(self.prompt_path, img_name))  # (1024, 1024, 1)我猜的
        # prompt_img = np.expand_dims(prompt_img, axis=0)
        # prompt_img = prompt_img / 255

        gt = io.imread(join(self.gt_path, img_name)) # multiple labels [0, 1,4,5...], (256,256)
        gt = np.expand_dims(gt, axis=0)
        gt = gt / 255

        # prompt_gts = io.imread(join(self.prompt_gts_path, img_name)) # multiple labels [0, 1,4,5...], (256,256)
        # prompt_gts = np.expand_dims(prompt_gts, axis=0)
        # prompt_gts = prompt_gts / 255
        #
        # swin_gts = io.imread(join(self.swin_gts_path, img_name)) # multiple labels [0, 1,4,5...], (256,256)
        # swin_gts = np.expand_dims(swin_gts, axis=0)
        # swin_gts = swin_gts / 255

        if self.inference:
            return (torch.tensor(img_1024).float(),join(self.gt_path, img_name))
        else:

            return (
                torch.tensor(img_1024).float(),
                torch.tensor(gt).long(),

                # torch.tensor(prompt_gts).float(),
                # torch.tensor(swin_gts).float(),
                img_name,
            )



class GIDDataset(Dataset):
    def __init__(self, data_root,inference=False, bbox_shift=20):
        self.inference=inference
        self.data_root = data_root
        self.gt_path = join(data_root, "gts") #label
        self.img_path = join(data_root, "imgs")
        self.prompt_path = join(data_root, "prompt_mask_256")
        #这个里改数据路径
        self.gt_path_files = sorted(
            glob.glob(join(self.gt_path, "**/*.tif"), recursive=True)
        )
        self.gt_path_files = [
            file
            for file in self.gt_path_files
            if os.path.isfile(join(self.img_path, os.path.basename(file)))
        ]
        self.bbox_shift = bbox_shift

        print(f"number of images: {len(self.gt_path_files)}")

    def __len__(self):
        return len(self.gt_path_files)

    def __getitem__(self, index):
        # load npy image (1024, 1024, 3), [0,1]
        img_name = os.path.basename(self.gt_path_files[index])
        # img_1024 = np.load(
        #     join(self.img_path, img_name), "r", allow_pickle=True
        # )  # (1024, 1024, 3)
        img_1024 = io.imread(os.path.join(self.img_path, img_name), plugin='tifffile')
        #img_1024 = io.imread(os.path.join(self.img_path, img_name))
        #img_1024=io.imread(join(self.img_path, img_name))

        # 移除第一个通道，只保留后三个通道
        # img_1024 = img_1024[:, :, 1:] #如果有近红外的话就取消注释
        # convert the shape to (3, H, W)
        img_1024 = np.transpose(img_1024, (2, 0, 1))

        # assert (
        #     np.max(img_1024) <= 1.0 and np.min(img_1024) >= 0.0
        # ), "image should be normalized to [0, 1]"


        prompt_1024= io.imread(join(self.prompt_path, img_name))  # (1024, 1024, 1)
        prompt_1024 = np.expand_dims(prompt_1024, axis=0)
        prompt_1024 = prompt_1024 / 255

        gt = io.imread(join(self.gt_path, img_name)) # multiple labels [0, 1,4,5...], (256,256)
        gt = np.expand_dims(gt, axis=0)
        gt = gt / 255

        if not self.inference:
            # Paper-style default prompt token: 15% zero-prompt for robust top-layer inference
            if random.random() < 0.15:
                prompt_1024 = np.zeros_like(prompt_1024)

        if self.inference:
            return (torch.tensor(img_1024).float(),torch.tensor(prompt_1024).float(),join(self.gt_path, img_name))
        else:

            return (
                torch.tensor(img_1024).float(),
                torch.tensor(gt).long(),
                torch.tensor(prompt_1024).float(),
                img_name,
            )


class MultiScaleGaofenDataset(Dataset):
    """
    Multi-scale dataset for Gaofen that returns level0/1/2 simultaneously.
    All levels are resized to 1024x1024 to match model input.
    Only uses files that exist in ALL three levels (intersection).
    """
    def __init__(self, level0_root, level1_root=None, level2_root=None, inference=False):
        self.inference = inference
        self.level0 = PromptDataset_GID5(level0_root, inference=inference)
        self.use_multiscale = (level1_root is not None and level2_root is not None)
        
        if self.use_multiscale:
            self.level1 = PromptDataset_GID5(level1_root, inference=inference)
            self.level2 = PromptDataset_GID5(level2_root, inference=inference)
            
            # Find intersection of basenames across all levels
            l0_names = {os.path.basename(f): f for f in self.level0.gt_path_files}
            l1_names = {os.path.basename(f): f for f in self.level1.gt_path_files}
            l2_names = {os.path.basename(f): f for f in self.level2.gt_path_files}
            
            common_basenames = sorted(list(set(l0_names.keys()) & set(l1_names.keys()) & set(l2_names.keys())))
            self.common_files = common_basenames  # store basenames
            print(f"MultiScaleGaofenDataset: level0={len(l0_names)}, level1={len(l1_names)}, level2={len(l2_names)}")
            print(f"  Common files (intersection): {len(self.common_files)}")
            
            # Build index mapping: basename -> index in each dataset
            self.l0_index_map = {os.path.basename(f): i for i, f in enumerate(self.level0.gt_path_files)}
            self.l1_index_map = {os.path.basename(f): i for i, f in enumerate(self.level1.gt_path_files)}
            self.l2_index_map = {os.path.basename(f): i for i, f in enumerate(self.level2.gt_path_files)}
    
    def __len__(self):
        if self.use_multiscale:
            return len(self.common_files)
        return len(self.level0)
    
    def _resize_to_1024(self, tensor, is_mask=False):
        """Resize tensor to 1024x1024 if smaller."""
        if tensor.shape[-2:] == (1024, 1024):
            return tensor
        orig_dtype = tensor.dtype
        t = tensor.float().unsqueeze(0)
        mode = 'nearest' if is_mask else 'bilinear'
        t = torch.nn.functional.interpolate(t, size=(1024, 1024), mode=mode, align_corners=None if mode == 'nearest' else False)
        return t.squeeze(0).to(orig_dtype)
    
    def _resize_prompt_to_256(self, tensor):
        """Resize prompt tensor to 256x256."""
        if tensor.shape[-2:] == (256, 256):
            return tensor
        orig_dtype = tensor.dtype
        t = tensor.float().unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(256, 256), mode='nearest')
        return t.squeeze(0).to(orig_dtype)
    
    def __getitem__(self, index):
        if self.use_multiscale:
            filename = self.common_files[index]
            
            img0, gt0, prompt0, name0 = self.level0[self.l0_index_map[filename]]
            img1, gt1, prompt1, name1 = self.level1[self.l1_index_map[filename]]
            img2, gt2, prompt2, name2 = self.level2[self.l2_index_map[filename]]
            
            # Resize level1 and level2 images/gts to 1024x1024
            # Prompt resize to 256x256 (model expects 256x256 prompt)
            img1 = self._resize_to_1024(img1, is_mask=False)
            gt1 = self._resize_to_1024(gt1, is_mask=True)
            prompt1 = self._resize_prompt_to_256(prompt1)
            
            img2 = self._resize_to_1024(img2, is_mask=False)
            gt2 = self._resize_to_1024(gt2, is_mask=True)
            prompt2 = self._resize_prompt_to_256(prompt2)
            
            # Stack multi-scale inputs: (3, C, H, W) where 3 = num_scales
            img_ms = torch.stack([img0, img1, img2], dim=0)
            gt_ms = torch.stack([gt0, gt1, gt2], dim=0)
            prompt_ms = torch.stack([prompt0, prompt1, prompt2], dim=0)
            
            return img_ms, gt_ms, prompt_ms, name0
        else:
            return self.level0[index]
