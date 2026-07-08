# author MengFanlin
# time 2024/12/25
# filename swin_net
# description:xxx
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple
from .uper_head import UPerHead
from .swin_mmcv import SwinTransformer


class SwinTransformerNet(nn.Module):
    def __init__(
        self,
        pretrained=None
    ) -> None:
        super().__init__()
        self.swin = SwinTransformer(pretrained=pretrained) # mmcv
        self.swin.init_weights()
        self.uperhead = UPerHead(in_channels=[96, 192, 384, 768], num_classes=1,channels=512,
                               in_index=[0,1,2,3])
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样

    def forward(self, image):
        #with torch.no_grad():

        x = self.swin(image)
        x = self.uperhead(x)  # 结果：B*1*256*256
        x = self.upsample(x)
        #x = self.crossattention(x, prompt_result)
        return x
