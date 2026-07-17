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
from .cross_attention import WindowCrossAttention, FeatureFusion, BiWindowCrossAttention,FeatureFusion_conv2, FeatureFusion_conv2_noloss


class EvidentialHead(nn.Module):
    """预测前景/背景的证据强度"""
    def __init__(self, in_ch=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 2, 1)  # ch0: fg evidence, ch1: bg evidence
        )
    
    def forward(self, x):
        evidence = F.relu(self.conv(x)) + 1e-6
        alpha = evidence + 1.0
        S = alpha[:, 0:1] + alpha[:, 1:2]
        prob = alpha[:, 0:1] / S
        uncertainty = 2.0 / S
        return evidence, prob, uncertainty


class FMAdapter(nn.Module):
    """将 FM 特征适配到分割空间"""
    def __init__(self, fm_dim=256, out_dim=64):
        super().__init__()
        self.compress = nn.Conv2d(fm_dim, out_dim, 1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
    
    def forward(self, x):
        x = self.compress(x)
        x = self.upsample(x)
        return x


class SemanticConsistencyHead(nn.Module):
    """分割特征 + FM 特征 -> 预测与 GT 的语义一致性"""
    def __init__(self, in_ch=65):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, seg_feat, fm_feat):
        x = torch.cat([seg_feat, fm_feat], dim=1)
        return self.conv(x)


class SPPromptWaterNet(nn.Module):
    def __init__(
        self,
        promptcheckpoint=None,
        swin_pretrained=None,
        freeze_prompt=False,
        use_uncertainty=True,
    )-> None:
        super().__init__()
        self.use_uncertainty = use_uncertainty
        self.promptcheckpoint = promptcheckpoint

        sam_model = sam_model_registry['vit_b'](checkpoint=self.promptcheckpoint)
        self.prompt_module = PromptModule(image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,)

        if freeze_prompt:
            for param in self.prompt_module.parameters():
                param.requires_grad = False

        self.swin = SwinTransformer(pretrained=swin_pretrained)
        self.swin.init_weights()
        self.uperhead=UPerHead(in_channels=[96, 192, 384, 768], num_classes=1,channels=512,
                               in_index=[0,1,2,3])

        self.conv_fusion = FeatureFusion()
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样

        # === 新增不确定性模块 ===
        if self.use_uncertainty:
            if self.promptcheckpoint is not None:
                fm_sam = sam_model_registry['vit_b'](checkpoint=self.promptcheckpoint)
                self.fm_encoder = fm_sam.image_encoder
            else:
                self.fm_encoder = sam_model.image_encoder
            for p in self.fm_encoder.parameters():
                p.requires_grad = False
            self.fm_encoder.eval()

            self.fusion_compress = nn.Conv2d(2, 1, 1)
            self.evidential_head = EvidentialHead(in_ch=1)
            self.fm_adapter = FMAdapter(fm_dim=256, out_dim=64)
            self.semantic_consistency = SemanticConsistencyHead(in_ch=65)

    def forward(self, image, prompt_mask):
        prompt_result = self.prompt_module(image, prompt_mask)  # B*1*256*256
        swin_result = self.swin(image)
        swin_result = self.uperhead(swin_result)  # B*1*256*256

        result = self.conv_fusion(swin_result, prompt_result)

        if not self.use_uncertainty:
            return result

        # --- 不确定性分支 (256x256 空间) ---
        fused_256 = torch.cat([swin_result, prompt_result], dim=1)
        fused_256 = self.fusion_compress(fused_256)  # B,1,256,256

        evidence, prob_256, uncertainty_256 = self.evidential_head(fused_256)

        # FM 语义一致性
        with torch.no_grad():
            fm_feat = self.fm_encoder(image)  # B,256,64,64
        fm_feat_adapted = self.fm_adapter(fm_feat)  # B,64,256,256
        consistency = self.semantic_consistency(fused_256, fm_feat_adapted)

        return result, evidence, prob_256, uncertainty_256, consistency
