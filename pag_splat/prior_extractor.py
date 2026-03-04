"""
Module 1 — Metric-Aligned Prior Extraction (尺度对齐的单目先验提取)

流程：
  1. 冻结的 Depth Anything V3 (DA3) backbone 提取单目特征 F_mono 和相对深度 D_rel
  2. ScaleAlignmentMLP 根据相机位姿预测全局尺度 S、偏移 T
  3. D_metric = S * D_rel + T

DA3 特征细节：
  - 使用 DINOv2 backbone 的指定中间层 (默认第 FEAT_LAYER=8 层)
  - 特征形状: (B, H/14, W/14, embed_dim) → 投影到 feat_channels → 双线性上采样到 H/feat_stride
  - 深度: DA3 直接输出相对深度 D_rel，形状 (B, H, W)，activation=exp 保正
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class MonoPriorExtractor(nn.Module):
    """
    冻结 DA3 的特征提取包装器，同时输出单目特征图和相对深度图。

    Args:
        da3_net:      DA3 核心网络 (DepthAnything3Net 或其嵌套版本 NestedDepthAnything3Net)
                      即 `DepthAnything3().net`，**不是** 高层 DepthAnything3 API 对象。
        embed_dim:    DA3 backbone (DINOv2) 的特征维度
                      - ViT-B: 768   ViT-L: 1024   ViT-G: 1536
        feat_channels:输出的统一特征通道数 (投影后)
        feat_stride:  输出特征相对于输入图像的下采样倍数 (默认 4，即 H/4)
        feat_layer:   从 DINOv2 哪一层提取中间特征 (从 0 开始)
                      - ViT-B/L (12 层): 推荐 6~9;  ViT-G (40 层): 推荐 15~25
    """

    PATCH_SIZE: int = 14  # DINOv2 固定 patch size

    def __init__(
        self,
        da3_net: nn.Module,
        embed_dim: int = 768,
        feat_channels: int = 256,
        feat_stride: int = 4,
        feat_layer: int = 8,
    ) -> None:
        super().__init__()

        # 冻结全部 DA3 参数，不参与梯度计算
        self.da3_net = da3_net
        for p in self.da3_net.parameters():
            p.requires_grad_(False)
        self.da3_net.eval()

        self.feat_channels = feat_channels
        self.feat_stride = feat_stride
        self.feat_layer = feat_layer

        # 将 DINOv2 raw patch 特征线性投影到统一维度
        # 形状变化: (B, Hp, Wp, embed_dim) -> (B, feat_channels, Hp, Wp) -> upsample
        self.feat_proj = nn.Sequential(
            nn.Linear(embed_dim, feat_channels),
            nn.GELU(),
        )
        # 轻量 1×1 conv 做通道细化 (投影后在空间维度上)
        self.feat_refine = nn.Conv2d(feat_channels, feat_channels, 1, bias=False)

    @staticmethod
    def _pad_to_patch_multiple(
        img: torch.Tensor, patch_size: int
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        将图像填充到 patch_size 整数倍，右下角 reflect 填充。

        Returns:
            img_padded: (B, C, H_pad, W_pad)
            (pad_h, pad_w): 底部/右侧填充像素数，用于后续裁剪
        """
        H, W = img.shape[-2:]
        pad_h = (patch_size - H % patch_size) % patch_size
        pad_w = (patch_size - W % patch_size) % patch_size
        if pad_h > 0 or pad_w > 0:
            # F.pad 顺序: (left, right, top, bottom)
            img = F.pad(img, (0, pad_w, 0, pad_h), mode="reflect")
        return img, (pad_h, pad_w)

    @torch.no_grad()
    def _forward_da3(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单图 DA3 推理，冻结梯度。

        DINOv2 要求输入 H、W 均为 patch_size(14) 的整数倍。
        若不满足则先 padding，推理后将深度裁回原始尺寸。

        Args:
            img: (B, 3, H, W)，已 ImageNet 归一化

        Returns:
            d_rel: (B, H, W)  相对深度 (exp-activated，正值)
            feat:  (B, Hp, Wp, embed_dim)  patch 特征  (Hp≈H/14, Wp≈W/14)
        """
        B, C, H, W = img.shape

        # ── 填充到 14 的整数倍 ──
        img_pad, (pad_h, pad_w) = self._pad_to_patch_multiple(img, self.PATCH_SIZE)
        H_pad, W_pad = img_pad.shape[-2:]

        # DA3 net 接受 (B, N, 3, H_pad, W_pad)，N=1
        x = img_pad.unsqueeze(1)
        output = self.da3_net(x, export_feat_layers=[self.feat_layer])

        # ── 深度: (B, N, H_pad, W_pad) → 裁回原始 (B, H, W) ──
        raw_depth = output.depth
        if raw_depth.dim() == 4:
            d_pad = raw_depth[:, 0]   # (B, H_pad, W_pad)
        else:
            d_pad = raw_depth         # (B, H_pad, W_pad)
        d_rel = d_pad[:, :H, :W]     # 裁剪掉 padding 区域

        # ── 特征: (B, N, Hp_pad, Wp_pad, C) → 裁回 ──
        key = f"feat_layer_{self.feat_layer}"
        feat_raw = output.aux[key]
        if feat_raw.dim() == 5:
            feat_pad = feat_raw[:, 0]  # (B, Hp_pad, Wp_pad, C)
        else:
            feat_pad = feat_raw

        # patch 级别对应的有效范围: 原始图像对应的 patch 数
        # 用 ceil 保留所有覆盖原始图像的 patch (最后一个 patch 可能跨越 padding 边界)
        Hp_orig = math.ceil(H / self.PATCH_SIZE)
        Wp_orig = math.ceil(W / self.PATCH_SIZE)
        feat = feat_pad[:, :Hp_orig, :Wp_orig, :]  # (B, Hp_orig, Wp_orig, C)

        return d_rel, feat

    def forward(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        提取两视图的单目先验特征和相对深度。

        Args:
            img1: (B, 3, H, W)  视图 1，已归一化
            img2: (B, 3, H, W)  视图 2，已归一化

        Returns:
            f_mono1: (B, feat_channels, H//feat_stride, W//feat_stride)
            f_mono2: (B, feat_channels, H//feat_stride, W//feat_stride)
            d_rel1:  (B, 1, H, W)  视图 1 相对深度
            d_rel2:  (B, 1, H, W)  视图 2 相对深度
        """
        H, W = img1.shape[-2:]
        feat_h, feat_w = H // self.feat_stride, W // self.feat_stride

        # -------- 冻结推理 --------
        d_rel1, feat1 = self._forward_da3(img1)
        d_rel2, feat2 = self._forward_da3(img2)

        # -------- 特征投影 + 上采样 --------
        # feat: (B, Hp, Wp, embed_dim) -> Linear -> (B, Hp, Wp, feat_channels)
        f1 = self.feat_proj(feat1).permute(0, 3, 1, 2)  # (B, feat_channels, Hp, Wp)
        f2 = self.feat_proj(feat2).permute(0, 3, 1, 2)

        # 双线性上采样到目标特征分辨率
        f1 = F.interpolate(f1, size=(feat_h, feat_w), mode="bilinear", align_corners=False)
        f2 = F.interpolate(f2, size=(feat_h, feat_w), mode="bilinear", align_corners=False)

        # 可训练细化
        f1 = self.feat_refine(f1)
        f2 = self.feat_refine(f2)

        # -------- 深度整形 --------
        d_rel1 = d_rel1.unsqueeze(1)  # (B, 1, H, W)
        d_rel2 = d_rel2.unsqueeze(1)

        # 确保深度与 img1 分辨率一致 (DA3 输出可能因 padding 稍有不同)
        if d_rel1.shape[-2:] != (H, W):
            d_rel1 = F.interpolate(d_rel1, size=(H, W), mode="bilinear", align_corners=False)
        if d_rel2.shape[-2:] != (H, W):
            d_rel2 = F.interpolate(d_rel2, size=(H, W), mode="bilinear", align_corners=False)

        return f1, f2, d_rel1, d_rel2
