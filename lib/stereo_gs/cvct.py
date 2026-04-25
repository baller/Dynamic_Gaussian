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


class ResidualHead(nn.Module):
    """有界 RGB 残差 Δrgb ∈ [-ε, ε]。"""

    def __init__(self, in_channels: int, hidden: int = 16, bound: float = 0.05):
        super().__init__()
        self.bound = float(bound)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 3, kernel_size=1),
        )

    def forward(self, shared_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            shared_feat: (B, in_channels, H, W) — 来自 FullResGaussianHead 的共享特征。
        Returns:
            delta_rgb: (B, 3, H, W) ∈ [-bound, +bound]。
        """
        return torch.tanh(self.net(shared_feat)) * self.bound


def warp_with_disparity(
    img: torch.Tensor,
    disp: torch.Tensor,
    padding_mode: str = "border",
) -> torch.Tensor:
    """利用视差将*对侧视图*的图像 warp 到当前视图。

    与 lib.stereo_gs.cross_view_fusion.disparity_warp 行为一致，但作用于
    原始 RGB 图像。假设已校正立体对，对应关系为 x_other = x_self - disp。

    Args:
        img:  (B, 3, H, W) 对侧视图的 RGB。
        disp: (B, 1, H, W) 当前视图处的视差 (像素单位)。
        padding_mode: 'zeros' | 'border' | 'reflection'。

    Returns:
        (B, 3, H, W) warp 后的图像。
    """
    B, _, H, W = img.shape
    device = img.device
    grid_y = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(B, H, W)
    grid_x = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)
    src_x = grid_x - disp.squeeze(1)
    norm_x = 2.0 * src_x / (W - 1) - 1.0
    norm_y = 2.0 * grid_y / (H - 1) - 1.0
    grid = torch.stack([norm_x, norm_y], dim=-1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode=padding_mode, align_corners=True)


class CVCTModule(nn.Module):
    """跨视图颜色三角化: c_final = ω·c_self + (1-ω)·c_other_warp + Δrgb。

    `set_identity(True)` 将 ω 钳制为 0.5 且 Δrgb 置零 (用于训练计划的 Phase 1
    与 Phase 2，此时 CVCT 不应干扰 FDSG 的学习)。
    """

    def __init__(
        self,
        fused_channels: int,
        shared_channels: int,
        visibility_hidden: int = 32,
        residual_hidden: int = 16,
        residual_bound: float = 0.05,
    ):
        super().__init__()
        self.gate = VisibilityGate(in_channels=fused_channels, hidden=visibility_hidden)
        self.residual = ResidualHead(in_channels=shared_channels, hidden=residual_hidden, bound=residual_bound)
        self._identity_mode = False

    def set_identity(self, on: bool) -> None:
        self._identity_mode = bool(on)

    def is_identity(self) -> bool:
        return self._identity_mode

    def forward(
        self,
        fused_feat: torch.Tensor,
        shared_feat: torch.Tensor,
        confidence: torch.Tensor,
        c_self: torch.Tensor,
        c_other: torch.Tensor,
        disparity: torch.Tensor,
    ):
        """根据跨视图证据计算逐高斯颜色。

        Args:
            fused_feat:  (B, C_fused, H, W).
            shared_feat: (B, C_shared, H, W) 来自 FullResGaussianHead。
            confidence:  (B, 1, H, W).
            c_self:      (B, 3, H, W) 本视图 RGB，∈ [0, 1]。
            c_other:     (B, 3, H, W) 对侧视图 RGB，∈ [0, 1]。
            disparity:   (B, 1, H, W) 本视图视差 (正值)。

        Returns 字典，包含:
            'c_final':         (B, 3, H, W) ∈ [0, 1]
            'omega':           (B, 1, H, W)
            'delta_rgb':       (B, 3, H, W)
            'c_other_warped':  (B, 3, H, W) — 供 L_cycle 使用
        """
        c_other_warped = warp_with_disparity(c_other, disparity, padding_mode="border")

        if self._identity_mode:
            omega = torch.full_like(confidence, 0.5)
            delta_rgb = torch.zeros_like(c_self)
        else:
            omega = self.gate(fused_feat, confidence)
            delta_rgb = self.residual(shared_feat)

        c_final = omega * c_self + (1.0 - omega) * c_other_warped + delta_rgb
        c_final = c_final.clamp(0.0, 1.0)
        return {
            "c_final": c_final,
            "omega": omega,
            "delta_rgb": delta_rgb,
            "c_other_warped": c_other_warped,
        }
