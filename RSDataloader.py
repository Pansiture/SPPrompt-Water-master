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


        img_1024 = np.transpose(img_1024, (2, 0, 1)) #变为3*1024*1024
        prompt_img= io.imread(join(self.prompt_path, img_name))  # (256, 256)
        prompt_img = np.expand_dims(prompt_img, axis=0)# (1,256, 256)
        prompt_img = prompt_img / 255

        gt = io.imread(join(self.gt_path, img_name)) # multiple labels [0, 1,4,5...], (256,256)
        gt = np.expand_dims(gt, axis=0)
        gt = gt / 255



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
        prompt_img= io.imread(join(self.prompt_path, img_name))  # (1024, 1024, 1)我猜的
        prompt_img = np.expand_dims(prompt_img, axis=0)
        prompt_img = prompt_img / 255

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


        prompt_1024= io.imread(join(self.prompt_path, img_name))  # (1024, 1024, 1)我猜的
        prompt_1024 = np.expand_dims(prompt_1024, axis=0)
        prompt_1024 = prompt_1024 / 255




        gt = io.imread(join(self.gt_path, img_name)) # multiple labels [0, 1,4,5...], (256,256)
        gt = np.expand_dims(gt, axis=0)
        gt = gt / 255

        # assert img_name == os.path.basename(self.gt_path_files[index]), (
        #     "img gt name error" + self.gt_path_files[index] + self.npy_files[index]
        # )

        # label_ids = np.unique(gt)[1:]
        # gt2D = np.uint8(
        #     gt == random.choice(label_ids.tolist())
        # )  # only one label, (256, 256)
        # assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "ground truth should be 0, 1"
        # y_indices, x_indices = np.where(gt2D > 0)
        # x_min, x_max = np.min(x_indices), np.max(x_indices)
        # y_min, y_max = np.min(y_indices), np.max(y_indices)
        # # add perturbation to bounding box coordinates
        # H, W = gt2D.shape
        # x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        # x_max = min(W, x_max + random.randint(0, self.bbox_shift))
        # y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        # y_max = min(H, y_max + random.randint(0, self.bbox_shift))
        # bboxes = np.array([x_min, y_min, x_max, y_max])
        if self.inference:
            return (torch.tensor(img_1024).float(),torch.tensor(prompt_1024).float(),join(self.gt_path, img_name))
        else:

            return (
                torch.tensor(img_1024).float(),
                torch.tensor(gt).long(),
                torch.tensor(prompt_1024).float(),
                img_name,
            )