"""
高斯解码器 — 在 1/4 分辨率从 FFS 适配特征预测高斯属性。

与 GPS_plus 原始 GSRegresser 的区别:
  1. 输入来自 FFS 适配后的多尺度特征，而非独立的 UNet 编码器
  2. 不再需要独立的 depth_encoder (FFS 特征已编码深度信息)
  3. 置信度调控 scale 和 opacity
  4. 工作在 1/4 分辨率而非全分辨率
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """简单的 2D 残差块。"""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class GaussianDecoder(nn.Module):
    """
    在 1/4 分辨率从 FFS 特征预测高斯属性。

    架构: 多尺度 FPN 式解码器 → 共享特征 → 四个预测头
    """

    def __init__(
        self,
        feat_dims: Tuple[int, ...] = (96, 96, 128),
        head_dim: int = 64,
        confidence_alpha: float = 0.3,
        confidence_beta: float = 0.3,
        max_scale: float = 0.003,
    ):
        """
        Args:
            feat_dims:   适配后的特征通道 [C_1/4, C_1/8, C_1/16]
            head_dim:    预测头的中间通道
            confidence_alpha: scale 置信度调控下限
            confidence_beta:  opacity 置信度调控下限
            max_scale:   scale 最大值 clamp
        """
        super().__init__()
        self.confidence_alpha = confidence_alpha
        self.confidence_beta = confidence_beta
        self.max_scale = max_scale

        self.decode_16 = ResBlock(feat_dims[2], feat_dims[1])
        self.decode_8 = ResBlock(feat_dims[1] * 2, feat_dims[0])

        in_ch_final = feat_dims[0] * 2 + 1 + 1
        self.fuse = nn.Sequential(
            ResBlock(in_ch_final, head_dim),
            ResBlock(head_dim, head_dim),
        )

        self.rot_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 4, 1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 3, 1),
            nn.Softplus(beta=1),
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 1, 1),
            nn.Sigmoid(),
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 1, 1),
            nn.Tanh(),
        )

    def forward(
        self,
        adapted_feats: list,
        disparity_1_4: torch.Tensor,
        confidence: torch.Tensor,
    ) -> dict:
        """
        Args:
            adapted_feats: [feat_1/4, feat_1/8, feat_1/16] 适配后特征
            disparity_1_4: (B, 1, H/4, W/4)
            confidence:    (B, 1, H/4, W/4)

        Returns:
            dict with rot, scale, opacity, depth_residual, all at 1/4 res
        """
        feat_4, feat_8, feat_16 = adapted_feats
        H4, W4 = feat_4.shape[-2:]

        up_16 = self.decode_16(feat_16)
        up_16 = F.interpolate(up_16, size=feat_8.shape[-2:], mode='bilinear', align_corners=True)
        up_8 = self.decode_8(torch.cat([up_16, feat_8], dim=1))
        up_8 = F.interpolate(up_8, size=(H4, W4), mode='bilinear', align_corners=True)

        disp_norm = disparity_1_4 / (disparity_1_4.max() + 1e-6)
        x = torch.cat([up_8, feat_4, disp_norm, confidence], dim=1)
        shared = self.fuse(x)

        rot = self.rot_head(shared)
        rot = F.normalize(rot, dim=1)

        scale_base = self.scale_head(shared)
        scale_base = torch.clamp_max(scale_base, self.max_scale)

        opacity_base = self.opacity_head(shared)

        alpha = self.confidence_alpha
        beta = self.confidence_beta
        scale = scale_base * (alpha + (1 - alpha) * confidence)
        opacity = opacity_base * (beta + (1 - beta) * confidence)

        depth_residual = self.depth_head(shared) * 0.5

        return {
            'rot': rot,
            'scale': scale,
            'opacity': opacity,
            'depth_residual': depth_residual,
            '_shared_feat': shared,
        }
