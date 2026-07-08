# author MengFanlin
# time 2024/12/11
# filename cross_attention
# description:xxx

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class BiWindowCrossAttention(nn.Module):
    def __init__(self, window_size=64):
        super(BiWindowCrossAttention, self).__init__()
        self.window_size = window_size
        self.query_conv1 = nn.Conv2d(1, 1, kernel_size=1)  # 对于 x1 的 Query
        self.key_conv1 = nn.Conv2d(1, 1, kernel_size=1)    # 对于 x2 的 Key
        self.value_conv1 = nn.Conv2d(1, 1, kernel_size=1)  # 对于 x2 的 Value

        self.query_conv2 = nn.Conv2d(1, 1, kernel_size=1)  # 对于 x2 的 Query
        self.key_conv2 = nn.Conv2d(1, 1, kernel_size=1)    # 对于 x1 的 Key
        self.value_conv2 = nn.Conv2d(1, 1, kernel_size=1)  # 对于 x1 的 Value

        self.softmax = nn.Softmax(dim=-1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        ws = self.window_size

        # 分块
        x1_patches = x1.unfold(2, ws, ws).unfold(3, ws, ws)
        x2_patches = x2.unfold(2, ws, ws).unfold(3, ws, ws)
        B, C, num_h, num_w, ws, _ = x1_patches.shape

        fused_patches = torch.zeros_like(x1_patches)

        for i in range(num_h):
            for j in range(num_w):
                # 当前块
                x1_patch = x1_patches[:, :, i, j, :, :]
                x2_patch = x2_patches[:, :, i, j, :, :]

                # **第一步：x1 的 Query 和 x2 的 Key/Value**
                Q1 = self.query_conv1(x1_patch)
                K2 = self.key_conv1(x2_patch)
                V2 = self.value_conv1(x2_patch)

                Q1_flat = Q1.view(B, -1, ws * ws).permute(0, 2, 1)  # [B, ws*ws, C]
                K2_flat = K2.view(B, -1, ws * ws)                   # [B, C, ws*ws]
                att_weights1 = self.softmax(torch.bmm(Q1_flat, K2_flat) / (C ** 0.5))

                V2_flat = V2.view(B, -1, ws * ws).permute(0, 2, 1)  # [B, ws*ws, C]
                attended1 = torch.bmm(att_weights1, V2_flat)        # [B, ws*ws, C]
                attended1 = attended1.permute(0, 2, 1).view(B, C, ws, ws)

                # **第二步：x2 的 Query 和 x1 的 Key/Value**
                Q2 = self.query_conv2(x2_patch)
                K1 = self.key_conv2(x1_patch)
                V1 = self.value_conv2(x1_patch)

                Q2_flat = Q2.view(B, -1, ws * ws).permute(0, 2, 1)  # [B, ws*ws, C]
                K1_flat = K1.view(B, -1, ws * ws)                   # [B, C, ws*ws]
                att_weights2 = self.softmax(torch.bmm(Q2_flat, K1_flat) / (C ** 0.5))

                V1_flat = V1.view(B, -1, ws * ws).permute(0, 2, 1)  # [B, ws*ws, C]
                attended2 = torch.bmm(att_weights2, V1_flat)        # [B, ws*ws, C]
                attended2 = attended2.permute(0, 2, 1).view(B, C, ws, ws)

                # **融合两个方向的结果**
                fused_patch = (attended1 + attended2) / 2
                fused_patches[:, :, i, j, :, :] = fused_patch

        # 重建图像
        fused_feature = fused_patches.permute(0, 1, 4, 5, 2, 3).contiguous()
        fused_feature = fused_feature.view(B, C, H, W)
        fused_feature = self.upsample(fused_feature)  # 上采样到 1024x1024
        return fused_feature

class WindowCrossAttention(nn.Module):
    def __init__(self, window_size=64):
        super(WindowCrossAttention, self).__init__()
        self.window_size = window_size
        self.query_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.key_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.value_conv = nn.Conv2d(1, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        ws = self.window_size

        # 分块 (B, C, H, W) -> (B, C, num_windows, ws, ws)
        x1_patches = x1.unfold(2, ws, ws).unfold(3, ws, ws)  # 分块 x1
        x2_patches = x2.unfold(2, ws, ws).unfold(3, ws, ws)  # 分块 x2
        B, C, num_h, num_w, ws, _ = x1_patches.shape

        # 初始化存储交叉注意力结果
        fused_patches = torch.zeros_like(x1_patches)

        for i in range(num_h):
            for j in range(num_w):
                x1_patch = x1_patches[:, :, i, j, :, :]  # 当前窗口 x1
                x2_patch = x2_patches[:, :, i, j, :, :]  # 当前窗口 x2

                # 生成 Query, Key, Value
                Q1 = self.query_conv(x1_patch)
                K2 = self.key_conv(x2_patch)
                V2 = self.value_conv(x2_patch)

                # 计算注意力权重
                Q1_flat = Q1.view(B, -1, ws * ws).permute(0, 2, 1)  # [B, ws*ws, C]
                K2_flat = K2.view(B, -1, ws * ws)                   # [B, C, ws*ws]
                attention_weights = self.softmax(torch.bmm(Q1_flat, K2_flat) / (C ** 0.5))

                # 加权 x2 的 Value
                V2_flat = V2.view(B, -1, ws * ws).permute(0, 2, 1)  # [B, ws*ws, C]
                attended_patch = torch.bmm(attention_weights, V2_flat)  # [B, ws*ws, C]
                attended_patch = attended_patch.permute(0, 2, 1).view(B, C, ws, ws)

                # 保存结果
                fused_patches[:, :, i, j, :, :] = attended_patch

        # 将分块结果还原为完整图像
        fused_feature = fused_patches.permute(0, 1, 4, 5, 2, 3).contiguous()
        fused_feature = fused_feature.view(B, C, H, W)
        fused_feature = self.upsample(fused_feature)  # 上采样到 1024x1024
        return fused_feature


class CrossAttentionFusion(nn.Module):
    def __init__(self):
        super(CrossAttentionFusion, self).__init__()
        self.query_conv = nn.Conv2d(1, 1, kernel_size=1)  # 生成 Q
        self.key_conv = nn.Conv2d(1, 1, kernel_size=1)    # 生成 K
        self.value_conv = nn.Conv2d(1, 1, kernel_size=1)  # 生成 V
        self.softmax = nn.Softmax(dim=-1)  # 归一化
        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        # 1. 对 x1 生成 Query，对 x2 生成 Key 和 Value
        Q1 = self.query_conv(x1)  # x1 的 Query
        K2 = self.key_conv(x2)    # x2 的 Key
        V2 = self.value_conv(x2)  # x2 的 Value

        # 2. 计算 x1 对 x2 的注意力权重
        B, C, H, W = Q1.shape
        Q1_flat = Q1.view(B, -1, H * W).permute(0, 2, 1)  # 展平成 [B, HW, C]
        K2_flat = K2.view(B, -1, H * W)                  # 展平成 [B, C, HW]
        attention_weights = self.softmax(torch.bmm(Q1_flat, K2_flat) / (C ** 0.5))  # [B, HW, HW]

        # 3. 加权 x2 的 Value
        V2_flat = V2.view(B, -1, H * W).permute(0, 2, 1)  # 展平成 [B, HW, C]
        attended_x1 = torch.bmm(attention_weights, V2_flat)  # [B, HW, C]
        attended_x1 = attended_x1.permute(0, 2, 1).view(B, C, H, W)  # 恢复 [B, C, H, W]

        # 4. 对 x2 重复上述操作，计算 x2 对 x1 的关注
        Q2 = self.query_conv(x2)
        K1 = self.key_conv(x1)
        V1 = self.value_conv(x1)
        Q2_flat = Q2.view(B, -1, H * W).permute(0, 2, 1)
        K1_flat = K1.view(B, -1, H * W)
        attention_weights_2 = self.softmax(torch.bmm(Q2_flat, K1_flat) / (C ** 0.5))
        V1_flat = V1.view(B, -1, H * W).permute(0, 2, 1)
        attended_x2 = torch.bmm(attention_weights_2, V1_flat)
        attended_x2 = attended_x2.permute(0, 2, 1).view(B, C, H, W)

        # 5. 融合两个交叉注意力结果
        fused_feature = attended_x1 + attended_x2  # 简单相加
        fused_feature = self.upsample(fused_feature)  # 上采样到 1024x1024
        return fused_feature





class FeatureFusion(nn.Module):
    def __init__(self):
        super(FeatureFusion, self).__init__()
        self.conv_fusion = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)  # 融合层
        # self.conv_fusion2 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)  # 融合层

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # 通道拼接
        x = self.conv_fusion(x)  # 卷积融合

        # x = x1 + x2
        # x = self.conv_fusion2(x) # 相加版本

        x = self.upsample(x)  # 恢复到 1024x1024
        return x


class FeatureFusion_conv2(nn.Module):
    def __init__(self):
        super(FeatureFusion_conv2, self).__init__()
        self.conv_fusion1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)  # 融合层
        self.conv_fusion2 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)  # 融合层
        self.bn1 = LayerNorm2d(2)  # 归一化

        self.relu1 = nn.ReLU()  # 激活
        # self.conv_fusion2 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)  # 融合层

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # 通道拼接
        x = self.conv_fusion1(x)  # 卷积融合
        x = self.bn1(x)  # 归一化
        x = self.relu1(x)  # 激活函数
        x = self.conv_fusion2(x)



        x = self.upsample(x)  # 恢复到 1024x1024
        return x


class FeatureFusion_conv2_noloss(nn.Module):
    def __init__(self):
        super(FeatureFusion_conv2_noloss, self).__init__()
        self.conv_fusion1 = nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1)  # 融合层
        self.conv_fusion2 = nn.Conv2d(2, 1, kernel_size=1, stride=1, padding=0)  # 融合层
        self.bn1 = LayerNorm2d(2)  # 归一化

        self.relu1 = nn.ReLU()  # 激活
        # self.conv_fusion2 = nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0)  # 融合层

        self.upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)  # 上采样

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)  # 通道拼接
        x = self.conv_fusion1(x)  # 卷积融合
        x = self.bn1(x)  # 归一化
        x = self.relu1(x)  # 激活函数
        x = self.conv_fusion2(x)



        x = self.upsample(x)  # 恢复到 1024x1024
        return x