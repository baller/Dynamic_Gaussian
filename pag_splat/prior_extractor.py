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
    def _resize_to_patch_multiple(
        img: torch.Tensor, patch_size: int
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        将图像 resize 到 patch_size 最近整数倍。

        相比 reflect-padding，resize 不会在边缘引入虚假镜像内容，
        对 ViT 全局注意力更友好，边缘区域（白墙、天花板）深度更一致。
        典型缩放量 < 1 个 patch 宽（< 14px，< 2%），畸变可忽略。

        Returns:
            img_resized: (B, C, H_r, W_r)   H_r/W_r 均为 patch_size 整数倍
            (H_orig, W_orig): 原始尺寸，推理后 resize 深度时使用
        """
        H, W = img.shape[-2:]
        H_r = max(round(H / patch_size), 1) * patch_size
        W_r = max(round(W / patch_size), 1) * patch_size
        if H_r != H or W_r != W:
            img = F.interpolate(img, size=(H_r, W_r), mode="bilinear", align_corners=False)
        return img, (H, W)

    @torch.no_grad()
    def _forward_da3(
        self, img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单图 DA3 推理，冻结梯度。

        DINOv2 要求输入 H、W 均为 patch_size(14) 的整数倍。
        采用 resize（而非 padding）预处理：
          - resize 是 DA3/DINOv2 训练时的标准预处理，属于 in-distribution 操作
          - padding 的镜像内容会通过全局 attention 污染边缘 patch 的特征，
            导致白墙/天花板区域深度不稳定
          - 推理完成后将深度 resize 回原始分辨率

        Args:
            img: (B, 3, H, W)，已 ImageNet 归一化

        Returns:
            d_rel: (B, H, W)  相对深度 (exp-activated，正值)
            feat:  (B, Hp, Wp, embed_dim)  patch 特征  (Hp = H_r/14, Wp = W_r/14)
        """
        B, C, H, W = img.shape

        # ── resize 到 14 的整数倍 ──
        img_r, (H_orig, W_orig) = self._resize_to_patch_multiple(img, self.PATCH_SIZE)
        # img_r: (B, 3, H_r, W_r)，H_r/W_r 已是 14 的倍数

        # DA3 net 接受 (B, N, 3, H_r, W_r)，N=1
        x = img_r.unsqueeze(1)
        output = self.da3_net(x, export_feat_layers=[self.feat_layer])

        # ── 深度: (B, N, H_r, W_r) → resize 回原始 (B, H, W) ──
        raw_depth = output.depth
        if raw_depth.dim() == 4:
            d_r = raw_depth[:, 0]   # (B, H_r, W_r)
        else:
            d_r = raw_depth
        # resize 回原始分辨率（比 pad 方案裁剪更平滑）
        if d_r.shape[-2:] != (H_orig, W_orig):
            d_rel = F.interpolate(
                d_r.unsqueeze(1), size=(H_orig, W_orig),
                mode="bilinear", align_corners=False,
            ).squeeze(1)  # (B, H, W)
        else:
            d_rel = d_r

        # ── 特征: (B, N, Hp, Wp, C)，无需裁剪，直接取 N=0 ──
        key = f"feat_layer_{self.feat_layer}"
        feat_raw = output.aux[key]
        feat = feat_raw[:, 0] if feat_raw.dim() == 5 else feat_raw
        # feat: (B, Hp, Wp, embed_dim)，Hp = H_r/14，Wp = W_r/14

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

        # _forward_da3 已在内部将深度 resize 回 (H, W)，此处做保险校验
        if d_rel1.shape[-2:] != (H, W):
            d_rel1 = F.interpolate(d_rel1, size=(H, W), mode="bilinear", align_corners=False)
        if d_rel2.shape[-2:] != (H, W):
            d_rel2 = F.interpolate(d_rel2, size=(H, W), mode="bilinear", align_corners=False)

        return f1, f2, d_rel1, d_rel2
