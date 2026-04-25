"""
跨视图颜色三角化 (Cross-View Color Triangulation, CVCT) 模块 — 用于 W-CVCT-GS。

提供:
  - VisibilityGate:   逐像素视见度门控 ω ∈ [0, 1]，表示逐高斯可见度
  - ResidualHead:     有界颜色残差 Δrgb ∈ [-ε, ε]
  - CVCTModule:       组装 ColorEvidence + ω + Δrgb，并支持恒等模式 (identity-mode) 切换
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisibilityGate(nn.Module):
    """逐像素视见度门控 ω ∈ [0, 1]。

    ω≈1 → 信任本视图像素;
    ω≈0 → 信任对侧视图经视差 warp 到本视图的像素;
    ω≈0.5 → 双视图可见区域 (两视图一致)。
    """

    def __init__(self, in_channels: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels + 1, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
        )

    def forward(self, fused_feat: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fused_feat: (B, in_channels, H, W) — CrossViewFusion 的输出。
            confidence: (B, 1, H, W) — ConfidenceExtractor 的输出。

        Returns:
            omega: (B, 1, H, W) ∈ [0, 1]。
        """
        x = torch.cat([fused_feat, confidence], dim=1)
        return torch.sigmoid(self.net(x))
