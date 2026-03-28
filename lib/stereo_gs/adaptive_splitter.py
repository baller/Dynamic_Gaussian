"""
自适应高斯分裂 (CAGS) — 在边缘/纹理复杂区域将单个高斯分裂为多个子高斯。

打破像素级一对一高斯范式:
  - 平坦/高置信区域: 保持 1 个高斯 (w_k ≈ 0)
  - 边缘/遮挡/纹理区域: 分裂出 K-1 个子高斯，每个带独立残差

两种分裂判定模式:
  1. GradientSplitCriterion:  图像梯度驱动，无可学习参数
  2. LearnedSplitCriterion:   可学习网络预测分裂权重

子高斯属性 = 父级属性 + 残差 (SubGaussianResidualHead)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────
#  分裂判定: 图像梯度
# ──────────────────────────────────────────────────────────

class GradientSplitCriterion(nn.Module):
    """通过 Sobel 图像梯度幅值决定分裂强度，无可学习参数。"""

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                                dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (B, 3, H, W) RGB, range [-1, 1]

        Returns:
            split_score: (B, 1, H, W) in [0, 1]
        """
        gray = image.mean(dim=1, keepdim=True)
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

        B = mag.shape[0]
        flat = mag.view(B, -1)
        lo = flat.quantile(0.05, dim=1, keepdim=True).view(B, 1, 1, 1)
        hi = flat.quantile(0.95, dim=1, keepdim=True).view(B, 1, 1, 1)
        score = ((mag - lo) / (hi - lo + 1e-8)).clamp(0, 1)
        return score


# ──────────────────────────────────────────────────────────
#  分裂判定: 可学习网络
# ──────────────────────────────────────────────────────────

class LearnedSplitCriterion(nn.Module):
    """可学习的逐像素分裂权重预测，端到端训练。"""

    def __init__(self, in_channels: int, hidden_dim: int = 32, k_max: int = 4):
        """
        Args:
            in_channels: 输入特征通道 (upsampled_feat + RGB)
            hidden_dim:  中间层通道
            k_max:       最大子高斯数 (包括父级)，权重预测 k_max-1 个通道
        """
        super().__init__()
        self.k_sub = k_max - 1

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.weight_head = nn.Sequential(
            nn.Conv2d(hidden_dim, self.k_sub, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: (B, in_channels, H, W)

        Returns:
            weights: (B, k_sub, H, W) 每个子高斯的激活权重 ∈ [0, 1]
        """
        h = self.net(feat)
        return self.weight_head(h)


# ──────────────────────────────────────────────────────────
#  子高斯残差预测
# ──────────────────────────────────────────────────────────

class SubGaussianResidualHead(nn.Module):
    """预测每个子高斯相对于父级的属性残差。"""

    def __init__(self, in_channels: int, k_sub: int = 3, max_pos_offset: float = 0.002):
        """
        Args:
            in_channels:    输入特征通道 (shared_feat from FullResGaussianHead)
            k_sub:          子高斯数 (k_max - 1)
            max_pos_offset: 3D 位置偏移上限 (米)
        """
        super().__init__()
        self.k_sub = k_sub
        self.max_pos_offset = max_pos_offset

        out_per_sub = 3 + 4 + 3 + 1   # delta_xyz(3) + delta_rot(4) + delta_scale(3) + delta_opacity(1)
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, k_sub * out_per_sub, 1),
        )

    def forward(self, shared_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            shared_feat: (B, C, H, W)

        Returns:
            dict with per-sub-Gaussian residuals, each (B, k_sub, C_attr, H, W)
        """
        B, _, H, W = shared_feat.shape
        raw = self.head(shared_feat)                           # (B, k_sub*11, H, W)
        raw = raw.view(B, self.k_sub, 11, H, W)

        delta_xyz = torch.tanh(raw[:, :, :3]) * self.max_pos_offset
        delta_rot = raw[:, :, 3:7] * 0.1
        delta_scale = torch.sigmoid(raw[:, :, 7:10])
        delta_opacity = torch.sigmoid(raw[:, :, 10:11])

        return {
            'delta_xyz': delta_xyz,         # (B, k_sub, 3, H, W)
            'delta_rot': delta_rot,         # (B, k_sub, 4, H, W)
            'delta_scale': delta_scale,     # (B, k_sub, 3, H, W)  multiplicative [0,1]
            'delta_opacity': delta_opacity, # (B, k_sub, 1, H, W)  multiplicative [0,1]
        }


# ──────────────────────────────────────────────────────────
#  组合模块
# ──────────────────────────────────────────────────────────

class AdaptiveSplitter(nn.Module):
    """
    自适应高斯分裂: 判定 + 残差 → 子高斯。

    forward 接收父级高斯属性和特征，输出子高斯的完整属性 (flat tensor)。
    """

    def __init__(
        self,
        feat_channels: int,
        shared_feat_channels: int,
        split_mode: str = 'learned',
        k_max: int = 4,
        max_pos_offset: float = 0.002,
    ):
        super().__init__()
        self.split_mode = split_mode
        self.k_max = k_max
        self.k_sub = k_max - 1

        if split_mode == 'gradient':
            self.criterion = GradientSplitCriterion()
        elif split_mode == 'learned':
            self.criterion = LearnedSplitCriterion(
                in_channels=feat_channels,
                k_max=k_max,
            )
        else:
            raise ValueError(f"Unknown split_mode: {split_mode}")

        self.residual_head = SubGaussianResidualHead(
            in_channels=shared_feat_channels,
            k_sub=self.k_sub,
            max_pos_offset=max_pos_offset,
        )

    def forward(
        self,
        parent_xyz: torch.Tensor,
        parent_rot: torch.Tensor,
        parent_scale: torch.Tensor,
        parent_opacity: torch.Tensor,
        parent_rgb: torch.Tensor,
        parent_valid: torch.Tensor,
        shared_feat: torch.Tensor,
        image_or_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            parent_xyz:     (B, H*W, 3)
            parent_rot:     (B, 4, H, W)
            parent_scale:   (B, 3, H, W)
            parent_opacity: (B, 1, H, W)
            parent_rgb:     (B, 3, H, W)
            parent_valid:   (B, H*W)  bool
            shared_feat:    (B, C_shared, H, W) 来自 FullResGaussianHead
            image_or_feat:  gradient 模式传 image (B,3,H,W);
                            learned 模式传 concat feat (B, feat_ch, H, W)

        Returns:
            dict:
                'sub_xyz':     (B, N_sub, 3)
                'sub_rot':     (B, N_sub, 4)
                'sub_scale':   (B, N_sub, 3)
                'sub_opacity': (B, N_sub, 1)
                'sub_rgb':     (B, N_sub, 3)
                'sub_valid':   (B, N_sub)
                'split_weights': (B, k_sub, H, W) 用于稀疏损失
        """
        B, _, H, W = parent_rot.shape
        N = H * W

        # ── 分裂权重 ──
        if self.split_mode == 'gradient':
            score = self.criterion(image_or_feat)              # (B, 1, H, W)
            weights = score.expand(-1, self.k_sub, -1, -1)     # (B, k_sub, H, W)
        else:
            weights = self.criterion(image_or_feat)             # (B, k_sub, H, W)

        # ── 残差 ──
        residuals = self.residual_head(shared_feat)

        # ── 组装子高斯属性 ──
        parent_xyz_map = parent_xyz.view(B, H, W, 3).permute(0, 3, 1, 2)  # (B, 3, H, W)
        parent_rot_flat = parent_rot.unsqueeze(1).expand(-1, self.k_sub, -1, -1, -1)
        parent_scale_flat = parent_scale.unsqueeze(1).expand(-1, self.k_sub, -1, -1, -1)
        parent_opacity_flat = parent_opacity.unsqueeze(1).expand(-1, self.k_sub, -1, -1, -1)
        parent_xyz_exp = parent_xyz_map.unsqueeze(1).expand(-1, self.k_sub, -1, -1, -1)
        parent_rgb_exp = parent_rgb.unsqueeze(1).expand(-1, self.k_sub, -1, -1, -1)

        sub_xyz = parent_xyz_exp + residuals['delta_xyz']
        sub_rot = F.normalize(parent_rot_flat + residuals['delta_rot'], dim=2)
        sub_scale = parent_scale_flat * (0.5 + residuals['delta_scale'])
        sub_opacity = parent_opacity_flat * residuals['delta_opacity'] * weights.unsqueeze(2)

        parent_valid_map = parent_valid.view(B, 1, H, W).expand(-1, self.k_sub, -1, -1)

        sub_xyz = sub_xyz.permute(0, 1, 3, 4, 2).reshape(B, self.k_sub * N, 3)
        sub_rot = sub_rot.permute(0, 1, 3, 4, 2).reshape(B, self.k_sub * N, 4)
        sub_scale = sub_scale.permute(0, 1, 3, 4, 2).reshape(B, self.k_sub * N, 3)
        sub_opacity = sub_opacity.permute(0, 1, 3, 4, 2).reshape(B, self.k_sub * N, 1)
        sub_rgb = parent_rgb_exp.permute(0, 1, 3, 4, 2).reshape(B, self.k_sub * N, 3)
        sub_valid = parent_valid_map.reshape(B, self.k_sub * N)

        return {
            'sub_xyz': sub_xyz,
            'sub_rot': sub_rot,
            'sub_scale': sub_scale,
            'sub_opacity': sub_opacity,
            'sub_rgb': sub_rgb,
            'sub_valid': sub_valid,
            'split_weights': weights,
        }


def build_adaptive_splitter(
    feat_channels: int,
    shared_feat_channels: int,
    split_mode: str = 'learned',
    k_max: int = 4,
    max_pos_offset: float = 0.002,
) -> AdaptiveSplitter:
    return AdaptiveSplitter(
        feat_channels=feat_channels,
        shared_feat_channels=shared_feat_channels,
        split_mode=split_mode,
        k_max=k_max,
        max_pos_offset=max_pos_offset,
    )
