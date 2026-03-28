"""
全分辨率高斯属性预测头 — 在全分辨率上从上采样特征 + RGB + 置信度预测高斯属性。

替代原来的 GaussianDecoder (1/4) + GaussianUpsampler (1/4→full) 两阶段管线：
  - 深度直接来自 FFS 全分辨率视差，不再预测 depth_residual
  - rot / scale / opacity 由轻量 2-layer CNN 在全分辨率上一步到位预测
  - 置信度对 scale / opacity 做乘性调制（与原 GaussianDecoder 逻辑一致）
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FullResGaussianHead(nn.Module):

    def __init__(
        self,
        in_channels: int = 100,
        hidden_dim: int = 64,
        confidence_alpha: float = 0.3,
        confidence_beta: float = 0.3,
        max_scale: float = 0.003,
    ):
        """
        Args:
            in_channels:  输入通道 = adapted_feat(96) + RGB(3) + confidence(1)
            hidden_dim:   共享卷积的中间通道
            confidence_alpha / beta:  scale / opacity 的置信度调制下限
            max_scale:    scale 最大值
        """
        super().__init__()
        self.confidence_alpha = confidence_alpha
        self.confidence_beta = confidence_beta
        self.max_scale = max_scale

        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.rot_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 4, 1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 3, 1),
            nn.Softplus(beta=1),
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feat_fullres: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict:
        """
        Args:
            feat_fullres: (B, in_channels, H, W) 上采样特征 + RGB + confidence 拼接后的输入
            confidence:   (B, 1, H, W) 用于调制 scale/opacity

        Returns:
            dict with 'rot' (B,4,H,W), 'scale' (B,3,H,W), 'opacity' (B,1,H,W),
                       '_shared_feat' (B, hidden_dim, H, W)
        """
        shared = self.shared(feat_fullres)

        rot = F.normalize(self.rot_head(shared), dim=1)

        scale_base = torch.clamp_max(self.scale_head(shared), self.max_scale)
        alpha = self.confidence_alpha
        scale = scale_base * (alpha + (1 - alpha) * confidence)

        opacity_base = self.opacity_head(shared)
        beta = self.confidence_beta
        opacity = opacity_base * (beta + (1 - beta) * confidence)

        return {
            'rot': rot,
            'scale': scale,
            'opacity': opacity,
            '_shared_feat': shared,
        }
