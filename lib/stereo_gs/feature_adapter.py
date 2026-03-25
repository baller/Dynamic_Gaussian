"""
特征适配层 — 将 FFS 多尺度骨干特征通道对齐到下游高斯解码器期望的维度。

FFS EdgeNeXt-small 输出: [224, 192, 320, 304] at 1/4, 1/8, 1/16, 1/32
目标对齐到:              [C_4, C_8, C_16]     (可配置, 默认 [96, 96, 128])
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    """
    为每个尺度的 FFS 骨干特征添加 1x1 Conv 适配器。

    额外支持融入 FFS 的 GRU 隐状态和上下文特征（均在 1/4 分辨率），
    作为对 1/4 尺度特征的增强信号。

    Args:
        ffs_feat_dims:  FFS 多尺度特征通道列表, e.g. [224, 192, 320, 304]
        ffs_hidden_dim: FFS GRU 隐状态通道, e.g. 128
        out_dims:       输出通道列表, 取前 3 个尺度, e.g. [96, 96, 128]
        use_context:    是否将 context_net + context_inp 拼接到 1/4 特征
        use_gru:        是否将 gru_hidden 拼接到 1/4 特征
    """

    def __init__(
        self,
        ffs_feat_dims: List[int],
        out_dims: Tuple[int, ...] = (96, 96, 128),
        use_context: bool = True,
        use_gru: bool = True,
        context_net_dim: int = 128,
        context_inp_dim: int = 128,
    ):
        super().__init__()
        self.use_context = use_context
        self.use_gru = use_gru

        extra_ch_4 = 0
        if use_context:
            extra_ch_4 += context_net_dim + context_inp_dim
        if use_gru:
            extra_ch_4 += context_net_dim

        self.adapt_4 = nn.Sequential(
            nn.Conv2d(ffs_feat_dims[0] + extra_ch_4, out_dims[0], 1, bias=False),
            nn.BatchNorm2d(out_dims[0]),
            nn.ReLU(inplace=True),
        )
        self.adapt_8 = nn.Sequential(
            nn.Conv2d(ffs_feat_dims[1], out_dims[1], 1, bias=False),
            nn.BatchNorm2d(out_dims[1]),
            nn.ReLU(inplace=True),
        )
        self.adapt_16 = nn.Sequential(
            nn.Conv2d(ffs_feat_dims[2], out_dims[2], 1, bias=False),
            nn.BatchNorm2d(out_dims[2]),
            nn.ReLU(inplace=True),
        )
        self.out_dims = list(out_dims)

    def forward(
        self,
        backbone_feats: List[torch.Tensor],
        context_net: torch.Tensor | None = None,
        context_inp: torch.Tensor | None = None,
        gru_hidden: torch.Tensor | None = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            backbone_feats: [feat_1/4, feat_1/8, feat_1/16, feat_1/32]
            context_net:    (B, hidden_dim, H/4, W/4)  可选
            context_inp:    (B, hidden_dim, H/4, W/4)  可选
            gru_hidden:     (B, hidden_dim, H/4, W/4)  可选

        Returns:
            [adapted_1/4, adapted_1/8, adapted_1/16]
        """
        feat_4 = backbone_feats[0]
        extras = []
        if self.use_context and context_net is not None and context_inp is not None:
            extras.extend([context_net, context_inp])
        if self.use_gru and gru_hidden is not None:
            extras.append(gru_hidden)
        if extras:
            feat_4 = torch.cat([feat_4, *extras], dim=1)

        out_4 = self.adapt_4(feat_4)
        out_8 = self.adapt_8(backbone_feats[1])
        out_16 = self.adapt_16(backbone_feats[2])
        return [out_4, out_8, out_16]
