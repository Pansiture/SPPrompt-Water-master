# author MengFanlin + EDL modification
# time 2024/12/6
# filename prompt_water_net_edl
# description: SPPromptWaterNet with EDL evidence output and SAM semantic referee
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Tuple

from .uper_head import UPerHead
from .SAM_mask import PromptModule, PromptModule_noprompt
from SPP_model import sam_model_registry
from .swin_mmcv import SwinTransformer


class SPPromptWaterNetEDL(nn.Module):
    """
    SPPrompt-Water with Evidential Deep Learning (EDL).
    Outputs 2-channel evidence (background, water) instead of 1-channel logits.
    Includes a frozen SAM semantic referee for global consistency.
    """

    def __init__(
        self,
        promptcheckpoint=None,
        swin_pretrained=None,
        freeze_prompt=False,
        use_referee=False,
    ) -> None:
        super().__init__()
        self.promptcheckpoint = promptcheckpoint
        self.use_referee = use_referee

        # SAM model (shared backbone for both branches)
        sam_model = sam_model_registry['vit_b'](checkpoint=self.promptcheckpoint)

        # Main branch: prompt-guided SAM (trainable mask decoder)
        self.prompt_module = PromptModule(
            image_encoder=sam_model.image_encoder,
            mask_decoder=sam_model.mask_decoder,
            prompt_encoder=sam_model.prompt_encoder,
        )

        # Semantic referee: optional frozen no-prompt SAM (can be disabled to save VRAM)
        self.sam_referee = None
        if use_referee:
            self.sam_referee = PromptModule_noprompt(
                image_encoder=sam_model.image_encoder,
                mask_decoder=sam_model.mask_decoder,
                prompt_encoder=sam_model.prompt_encoder,
            )
            # Freeze ALL referee parameters (foundation model as semantic judge)
            for param in self.sam_referee.parameters():
                param.requires_grad = False

        if freeze_prompt:
            for param in self.prompt_module.parameters():
                param.requires_grad = False

        # Swin Transformer + UPerHead (output 2 channels for evidence)
        self.swin = SwinTransformer(pretrained=swin_pretrained)
        self.swin.init_weights()
        self.uperhead = UPerHead(
            in_channels=[96, 192, 384, 768],
            num_classes=2,  # 2 channels: [evidence_bg, evidence_water]
            channels=512,
            in_index=[0, 1, 2, 3]
        )

        # Fusion: 2 (swin evidence raw) + 1 (sam mask) = 3 in, 2 (evidence) out
        self.conv_fusion = nn.Conv2d(3, 2, kernel_size=1, stride=1, padding=0)
        self.evidence_act = nn.Softplus()  # non-negative evidence

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, image, prompt_mask, return_evidence=False):
        """
        Args:
            image: (B, 3, 1024, 1024)
            prompt_mask: (B, 1, 256, 256)
            return_evidence: if True, return (B, 2, 1024, 1024) evidence; else return foreground prob
        Returns:
            evidence or foreground probability
        """
        # Main prompt branch (SAM with prompt)
        prompt_result = self.prompt_module(image, prompt_mask)  # B*1*256*256

        # Swin branch (multi-scale feature extraction)
        swin_result = self.swin(image)
        swin_result = self.uperhead(swin_result)  # B*2*256*256 (evidence raw)

        # Fusion: concatenate Swin evidence and SAM mask, then conv to 2-channel evidence
        x = torch.cat([swin_result, prompt_result], dim=1)  # B*3*256*256
        evidence = self.conv_fusion(x)  # B*2*256*256
        evidence = self.evidence_act(evidence)  # non-negative evidence e >= 0

        # Upsample to original image size
        evidence = self.upsample(evidence)  # B*2*1024*1024

        if return_evidence:
            return evidence

        # Convert to foreground probability for compatibility
        alpha = evidence + 1.0
        prob = alpha / alpha.sum(dim=1, keepdim=True)
        return prob[:, 1:2, :, :]  # foreground probability (B, 1, 1024, 1024)

    def forward_referee(self, image):
        """
        Frozen SAM semantic referee (no prompt).
        Returns None if use_referee=False.
        """
        if self.sam_referee is None:
            return None
        with torch.no_grad():
            sam_pred = self.sam_referee(image)  # B*1*256*256
        sam_pred = F.interpolate(
            sam_pred,
            size=(image.shape[2], image.shape[3]),
            mode='bilinear',
            align_corners=False
        )
        return torch.sigmoid(sam_pred)  # B*1*1024*1024
