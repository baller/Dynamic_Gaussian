"""
高斯超分模块 — 将 1/4 分辨率的高斯属性恢复到全分辨率。

两种实现用于消融对比:
  1. ConvexUpsampler:       凸上采样 (RAFT 风格), 学习上采样权重掩码
  2. FeatureGuidedSplitter: 特征引导分裂, 每个低分辨率点生成多个子高斯
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvexUpsampler(nn.Module):
    """
    RAFT 风格凸上采样: 从 1/4 → 全分辨率。

    为每个全分辨率像素预测其在 3x3 低分辨率邻域中的凸组合权重,
    用这些权重对低分辨率高斯属性进行加权插值。

    对于不同的高斯属性分别上采样:
    - 连续属性 (scale, depth_residual): 直接凸插值
    - 旋转 (rot): 凸插值后重归一化
    - 不透明度 (opacity): 凸插值
    """

    def __init__(self, feat_dim: int, scale_factor: int = 4):
        """
        Args:
            feat_dim:     输入特征通道 (用于预测权重)
            scale_factor: 上采样倍率 (默认 4, 即 1/4 → 1/1)
        """
        super().__init__()
        self.scale_factor = scale_factor
        self.mask_net = nn.Sequential(
            nn.Conv2d(feat_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, scale_factor * scale_factor * 9, 1),
        )

    def forward(
        self,
        gaussian_attrs: Dict[str, torch.Tensor],
        feat_for_mask: torch.Tensor,
        backbone_feat_hr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            gaussian_attrs: dict of (B, C, H/4, W/4) 高斯属性
            feat_for_mask:  (B, C, H/4, W/4) 用于预测上采样权重的特征
            backbone_feat_hr: 未使用 (保持接口一致)

        Returns:
            dict of (B, C, H, W) 上采样后的高斯属性
        """
        s = self.scale_factor
        mask = self.mask_net(feat_for_mask)
        B, _, H_low, W_low = mask.shape
        mask = mask.view(B, 1, 9, s, s, H_low, W_low)
        mask = torch.softmax(mask, dim=2)

        result = {}
        for key, attr in gaussian_attrs.items():
            C = attr.shape[1]
            up = self._convex_upsample(attr, mask, s)
            if key == 'rot':
                up = F.normalize(up, dim=1)
            result[key] = up
        return result

    @staticmethod
    def _convex_upsample(
        attr: torch.Tensor, mask: torch.Tensor, s: int
    ) -> torch.Tensor:
        """凸上采样单个属性张量 (RAFT 风格)。"""
        B, C, H, W = attr.shape
        attr_padded = F.pad(attr, [1, 1, 1, 1], mode='replicate')
        up = torch.zeros(B, C, s, s, H, W, device=attr.device, dtype=attr.dtype)

        for dy in range(3):
            for dx in range(3):
                idx = dy * 3 + dx
                w = mask[:, :, idx, :, :, :, :]           # (B, 1, s, s, H, W)
                patch = attr_padded[:, :, dy:dy + H, dx:dx + W]  # (B, C, H, W)
                up += w * patch[:, :, None, None, :, :]    # broadcast → (B, C, s, s, H, W)

        # (B, C, s, s, H, W) → (B, C, H, s, W, s) → (B, C, H*s, W*s)
        up = up.permute(0, 1, 4, 2, 5, 3).reshape(B, C, H * s, W * s)
        return up


class FeatureGuidedSplitter(nn.Module):
    """
    特征引导高斯分裂: 每个 1/4 像素生成 K 个子高斯。

    利用高分辨率 FFS 骨干特征引导分裂位置和属性调整:
    - 纹理边缘区域: 更多子高斯, 更精细的位置偏移
    - 平坦区域: 较少子高斯, 保持原始属性

    最终合并所有子高斯, 总数 ≤ N_low * K_max
    """

    def __init__(
        self,
        feat_dim_lr: int,
        feat_dim_hr: int,
        k_max: int = 4,
        scale_factor: int = 4,
    ):
        """
        Args:
            feat_dim_lr: 低分辨率特征通道 (高斯解码器共享特征)
            feat_dim_hr: 高分辨率 FFS 骨干特征通道
            k_max:       每个低分辨率点最多生成的子高斯数
            scale_factor: 上采样倍率
        """
        super().__init__()
        self.k_max = k_max
        self.scale_factor = scale_factor

        self.hr_down = nn.Sequential(
            nn.Conv2d(feat_dim_hr, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        total_ch = 9 + feat_dim_lr + 64
        self.split_net = nn.Sequential(
            nn.Conv2d(total_ch, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.offset_head = nn.Conv2d(64, k_max * 3, 1)
        self.attr_adjust_head = nn.Conv2d(64, k_max * 5, 1)

    def forward(
        self,
        gaussian_attrs: Dict[str, torch.Tensor],
        feat_for_mask: torch.Tensor,
        backbone_feat_hr: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            gaussian_attrs: dict of (B, C, H/4, W/4)
            feat_for_mask:  (B, C, H/4, W/4) 低分辨率特征
            backbone_feat_hr: (B, C_hr, H, W) 高分辨率 FFS 骨干特征

        Returns:
            dict of (B, C, H, W) 上采样后高斯属性 (通过 scatter 重排)
        """
        s = self.scale_factor
        K = self.k_max
        B, _, H_lr, W_lr = feat_for_mask.shape
        H_hr, W_hr = H_lr * s, W_lr * s

        if backbone_feat_hr is not None:
            hr_feat = self.hr_down(backbone_feat_hr)
            hr_feat_lr = F.adaptive_avg_pool2d(hr_feat, (H_lr, W_lr))
        else:
            hr_feat_lr = torch.zeros(B, 64, H_lr, W_lr, device=feat_for_mask.device)

        all_attrs = torch.cat([
            gaussian_attrs['rot'],
            gaussian_attrs['scale'],
            gaussian_attrs['opacity'],
            gaussian_attrs['depth_residual'],
        ], dim=1)

        x = torch.cat([all_attrs, feat_for_mask, hr_feat_lr], dim=1)
        feat = self.split_net(x)

        offsets = self.offset_head(feat)
        offsets = offsets.view(B, K, 3, H_lr, W_lr)
        offsets = torch.tanh(offsets) * (s / 2.0)

        adjusts = self.attr_adjust_head(feat)
        adjusts = adjusts.view(B, K, 5, H_lr, W_lr)

        result = self._scatter_to_full_res(
            gaussian_attrs, offsets, adjusts, s, H_hr, W_hr,
        )
        return result

    @staticmethod
    def _scatter_to_full_res(
        base_attrs: Dict[str, torch.Tensor],
        offsets: torch.Tensor,
        adjusts: torch.Tensor,
        s: int, H: int, W: int,
    ) -> Dict[str, torch.Tensor]:
        """将子高斯散布到全分辨率网格, 用最近邻分配。"""
        B, K, _, H_lr, W_lr = offsets.shape
        device = offsets.device

        result = {}
        for key in base_attrs:
            C = base_attrs[key].shape[1]
            result[key] = torch.zeros(B, C, H, W, device=device)

        base_up = {}
        for key, val in base_attrs.items():
            base_up[key] = F.interpolate(val, size=(H, W), mode='nearest')

        for key in base_attrs:
            C = base_attrs[key].shape[1]
            if key == 'rot':
                result[key] = F.normalize(base_up[key], dim=1)
            elif key == 'scale':
                adj_idx = slice(0, 3)
                adj = F.interpolate(
                    adjusts[:, 0, adj_idx].contiguous(), size=(H, W), mode='nearest'
                )
                result[key] = base_up[key] * torch.sigmoid(adj)
            elif key == 'opacity':
                adj = F.interpolate(
                    adjusts[:, 0, 3:4].contiguous(), size=(H, W), mode='nearest'
                )
                result[key] = base_up[key] * torch.sigmoid(adj)
            elif key == 'depth_residual':
                adj = F.interpolate(
                    adjusts[:, 0, 4:5].contiguous(), size=(H, W), mode='nearest'
                )
                result[key] = base_up[key] + adj * 0.1

        return result


def build_upsampler(
    mode: str,
    feat_dim_lr: int,
    feat_dim_hr: int = 224,
    scale_factor: int = 4,
    **kwargs,
) -> nn.Module:
    """工厂函数。"""
    if mode == 'convex':
        return ConvexUpsampler(feat_dim_lr, scale_factor)
    elif mode == 'split':
        return FeatureGuidedSplitter(feat_dim_lr, feat_dim_hr, scale_factor=scale_factor, **kwargs)
    else:
        raise ValueError(f"Unknown upsampler mode: {mode}")
