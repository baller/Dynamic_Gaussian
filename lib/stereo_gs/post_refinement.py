"""
渲染后精化网络 — 轻量 Conv 残差学习，用 FFS 高分辨率特征引导。

输入: 高斯光栅化渲染图 (3ch) + FFS 骨干特征 (Cch, 下采样到渲染分辨率)
输出: 精化后的最终图像 (3ch)

使用残差连接: output = rendered + refinement_net(cat(rendered, features))
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PostRefinement(nn.Module):
    """
    轻量渲染后精化网络。

    残差学习架构，仅学习渲染图与 GT 之间的差异，
    利用 FFS 高分辨率骨干特征提供几何+纹理引导信号。
    """

    def __init__(
        self,
        feat_channels: int = 224,
        hidden_channels: int = 32,
        num_layers: int = 3,
    ):
        """
        Args:
            feat_channels:   FFS 骨干特征通道 (1/4 分辨率, 会上采样到渲染分辨率)
            hidden_channels: 网络中间通道数
            num_layers:      Conv 层数
        """
        super().__init__()
        self.feat_compress = nn.Sequential(
            nn.Conv2d(feat_channels, hidden_channels, 1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        layers = []
        in_ch = 3 + hidden_channels
        for i in range(num_layers):
            out_ch = hidden_channels if i < num_layers - 1 else 3
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1))
            if i < num_layers - 1:
                layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.residual_net = nn.Sequential(*layers)

    def forward(
        self,
        rendered: torch.Tensor,
        ffs_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            rendered: (B, 3, H, W) 渲染图
            ffs_feat: (B, C, H_f, W_f) FFS 骨干特征 (通常 1/4 分辨率)

        Returns:
            refined: (B, 3, H, W) 精化后图像
        """
        if ffs_feat is not None:
            feat = self.feat_compress(ffs_feat)
            feat = F.interpolate(feat, size=rendered.shape[-2:], mode='bilinear', align_corners=True)
            x = torch.cat([rendered, feat], dim=1)
        else:
            x = torch.cat([rendered, torch.zeros_like(rendered[:, :1]).expand(-1, 32, -1, -1)], dim=1)

        residual = self.residual_net(x)
        refined = torch.clamp(rendered + residual, 0.0, 1.0)
        return refined
