# author MengFanlin
# time 2024/12/6
# filename prompt_water_net
# description:prompt_water_net的主框架
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from .uper_head import UPerHead

from .SAM_mask import PromptModule,PromptModule_noprompt
from SPP_model import sam_model_registry
from .swin_mmcv import SwinTransformer
from .cross_attention import WindowCrossAttention, FeatureFusion, BiWindowCrossAttention,FeatureFusion_conv2


class SPPromptWaterNet(nn.Module):
    def __init__(
        self,
        promptcheckpoint=None,
        swin_pretrained=None,
        freeze_prompt = True


    )-> None:
        super().__init__()
        self.promptcheckpoint = promptcheckpoint

        sam_model = sam_model_registry['vit_b'](checkpoint=self.promptcheckpoint)
        self.prompt_module = PromptModule(image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,)

        if freeze_prompt:
            for param in self.prompt_module.parameters():
                param.requires_grad = False



        self.swin = SwinTransformer(pretrained=swin_pretrained)
        self.uperhead=UPerHead(in_channels=[96, 192, 384, 768], num_classes=1,channels=512,
                               in_index=[0,1,2,3])

        self.conv_fusion = FeatureFusion()

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样

    def forward(self, image, prompt_mask):

        prompt_result = self.prompt_module(image, prompt_mask)  # B*1*256*256


        swin_result = self.swin(image)
        swin_result = self.uperhead(swin_result)  # B*1*256*256


        result = self.conv_fusion(swin_result, prompt_result)


        return result

