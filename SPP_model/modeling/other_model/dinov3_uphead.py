# author MengFanlin
# time 2024/12/25
# filename swin_net
# description:xxx
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from ..uper_head import UPerHead
from .dinov3 import DINOv3Backbone



class DINOv3Net(nn.Module):
    def __init__(
        self,
        pretrained=None
    ) -> None:
        super().__init__()
        self.dinov3 = DINOv3Backbone(adapt_patch_size='center_padding',
        frozen_stages=-1,
        img_size=1024,
        model_name='dinov3_vitl16',
        output_cls_token=False,
        repo_dir='D:\deeplearning\code\MedSAM-main\checkpoint\dinov3',
        type='DINOv3Backbone',
        weights='D:\deeplearning\code\MedSAM-main\checkpoint\dinov3\dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth') # mmcv
        # self.swin.init_weights()
        self.uperhead = UPerHead(in_channels=[96, 192, 384, 768], num_classes=1,channels=512,
                               in_index=[0,1,2,3])
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样

    def forward(self, image):
        #with torch.no_grad():

        x = self.dinov3(image)
        x = self.uperhead(x)  # 结果：B*1*256*256
        x = self.upsample(x)
        #x = self.crossattention(x, prompt_result)
        return x
