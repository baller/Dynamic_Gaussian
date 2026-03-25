"""
跨视图特征融合模块 — 利用 FFS 视差将对侧视图特征 warp 到当前视图，
并与当前视图特征融合，为高斯解码器提供更丰富的逐像素信息。

提供三种可切换实现用于消融对比:
  1. NoFusion:              不做融合，直接返回单视图特征
  2. WarpAttentionFusion:   视差引导 warp + 差异 + 注意力加权
  3. OcclusionAwareFusion:  遮挡感知的分支融合
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def disparity_warp(feat: torch.Tensor, disp: torch.Tensor) -> torch.Tensor:
    """
    用视差将**右**视图特征 warp 到**左**视图坐标系。

    假设已校正的立体对，对应关系为 x_right = x_left - disp。

    Args:
        feat: (B, C, H, W) 右视图特征
        disp: (B, 1, H, W) 左视图处的视差 (正值)

    Returns:
        warped: (B, C, H, W) warp 后的右视图特征
    """
    B, C, H, W = feat.shape
    device = feat.device

    grid_y = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(B, H, W)
    grid_x = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)

    disp_2d = disp.squeeze(1)
    src_x = grid_x - disp_2d

    norm_x = 2.0 * src_x / (W - 1) - 1.0
    norm_y = 2.0 * grid_y / (H - 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)

    warped = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped


class NoFusion(nn.Module):
    """消融基线: 不做跨视图融合，直接返回单视图特征。"""

    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = in_channels

    def forward(
        self,
        feat_main: torch.Tensor,
        feat_other: torch.Tensor,
        disp: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return feat_main


class WarpAttentionFusion(nn.Module):
    """
    视差引导 warp + 差异计算 + 注意力加权融合。

    对侧特征通过视差 warp 到当前视图后，计算逐像素差异图，
    学习注意力权重来控制两侧特征的混合比例。
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.out_channels = in_channels
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat_main: torch.Tensor,
        feat_other: torch.Tensor,
        disp: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feat_main:  (B, C, H, W) 当前视图适配后特征
            feat_other: (B, C, H, W) 对侧视图适配后特征
            disp:       (B, 1, H, W) 当前视图视差
            confidence: (B, 1, H, W) 可选置信度

        Returns:
            fused: (B, C, H, W)
        """
        disp_scaled = F.interpolate(
            disp, size=feat_main.shape[-2:], mode='bilinear', align_corners=True,
        ) * (feat_main.shape[-1] / disp.shape[-1])

        warped = disparity_warp(feat_other, disp_scaled)
        diff = torch.abs(feat_main - warped)
        weight = self.gate(torch.cat([feat_main, warped, diff], dim=1))
        fused = weight * feat_main + (1 - weight) * warped
        return fused


class OcclusionAwareFusion(nn.Module):
    """
    遮挡感知的跨视图融合。

    利用代价体置信度检测遮挡区域:
    - 可见区域: 用注意力加权融合双视图特征
    - 遮挡区域: 仅使用当前视图特征，经自增强模块处理
    """

    def __init__(self, in_channels: int, occlusion_threshold: float = 0.3):
        super().__init__()
        self.out_channels = in_channels
        self.occlusion_threshold = occlusion_threshold

        self.cross_fuse = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.self_enhance = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.occlusion_head = nn.Sequential(
            nn.Conv2d(in_channels * 2 + 1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat_main: torch.Tensor,
        feat_other: torch.Tensor,
        disp: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        H, W = feat_main.shape[-2:]
        disp_scaled = F.interpolate(
            disp, size=(H, W), mode='bilinear', align_corners=True,
        ) * (W / disp.shape[-1])

        warped = disparity_warp(feat_other, disp_scaled)
        diff = torch.abs(feat_main - warped)
        diff_norm = diff.mean(dim=1, keepdim=True)

        if confidence is not None:
            conf_scaled = F.interpolate(confidence, size=(H, W), mode='bilinear', align_corners=True)
        else:
            conf_scaled = torch.ones(feat_main.shape[0], 1, H, W, device=feat_main.device)

        occ_input = torch.cat([diff, warped, 1.0 - conf_scaled], dim=1)
        occ_prob = self.occlusion_head(occ_input)

        visible_feat = self.cross_fuse(torch.cat([feat_main, warped], dim=1))
        occluded_feat = self.self_enhance(feat_main)

        fused = (1 - occ_prob) * visible_feat + occ_prob * occluded_feat
        return fused


def build_fusion_module(mode: str, in_channels: int, **kwargs) -> nn.Module:
    """工厂函数：根据配置字符串构建融合模块。"""
    if mode == 'none':
        return NoFusion(in_channels)
    elif mode == 'warp_attention':
        return WarpAttentionFusion(in_channels)
    elif mode == 'occlusion_aware':
        return OcclusionAwareFusion(in_channels, **kwargs)
    else:
        raise ValueError(f"Unknown fusion mode: {mode}")
