# author MengFanlin
# time 2024/7/2
# filename SAM_mask
# description: 一个专门输入mask训练的SAM模型

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

class PromptModule_noprompt(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375]
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # freeze prompt encoder parameters ()这个函数将可训练的参数调出来，单独拎出来是否冻结
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):  # 这里要改
        # 输入的应该都是1024*1024的
        # 注意输入进来的mask 要转成tensor
        image = (image - self.pixel_mean) / self.pixel_std  # 标准化
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            # 下面应该是想把box转成mask
            # box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            # if len(box_torch.shape) == 2:
            #     box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        ) # B*1*256*256
        return low_res_masks

class PromptModule(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375]
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # freeze prompt encoder parameters ()这个函数将可训练的参数调出来，单独拎出来是否冻结
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, mask):  # 这里要改
        # 输入的应该都是1024*1024的
        # 注意输入进来的mask 要转成tensor
        image = (image - self.pixel_mean) / self.pixel_std  # 标准化
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            # 下面应该是想把box转成mask
            # box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            # if len(box_torch.shape) == 2:
            #     box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=mask,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        ) # B*1*256*256
        return low_res_masks
        # ori_res_masks = F.interpolate(
        #     low_res_masks,
        #     size=(image.shape[2], image.shape[3]),
        #     mode="bilinear",
        #     align_corners=False,
        # )
        #
        # return ori_res_masks # B*1*1024*1024

class WithoutPromptModule(nn.Module):

    '''
        这是SPP无需提示，也不是黑色提示的版本
    '''


    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375]
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # freeze prompt encoder parameters ()这个函数将可训练的参数调出来，单独拎出来是否冻结
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image):  # 这里要改
        # 输入的应该都是1024*1024的
        # 注意输入进来的mask 要转成tensor
        image = (image - self.pixel_mean) / self.pixel_std  # 标准化
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            # 下面应该是想把box转成mask
            # box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            # if len(box_torch.shape) == 2:
            #     box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        ) # B*1*256*256
        return low_res_masks


class SamMask(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        pixel_mean: List[float] = [123.675, 116.28, 103.53],
        pixel_std: List[float] = [58.395, 57.12, 57.375]
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)
        # freeze prompt encoder parameters ()这个函数将可训练的参数调出来，单独拎出来是否冻结
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def forward(self, image, mask):  # 这里要改
        # 输入的应该都是1024*1024的
        # 注意输入进来的mask 要转成tensor
        image = (image - self.pixel_mean) / self.pixel_std  # 标准化
        image_embedding = self.image_encoder(image)  # (B, 256, 64, 64)
        # do not compute gradients for prompt encoder
        with torch.no_grad():
            # 下面应该是想把box转成mask
            # box_torch = torch.as_tensor(box, dtype=torch.float32, device=image.device)
            # if len(box_torch.shape) == 2:
            #     box_torch = box_torch[:, None, :]  # (B, 1, 4)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=mask,
            )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=image_embedding,  # (B, 256, 64, 64)
            image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
            sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
            dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
            multimask_output=False,
        )
        ori_res_masks = F.interpolate(
            low_res_masks,
            size=(image.shape[2], image.shape[3]),
            mode="bilinear",
            align_corners=False,
        )

        return ori_res_masks
        # return low_res_masks