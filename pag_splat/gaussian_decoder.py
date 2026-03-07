"""
Module 3 — Uncertainty-Aware Gaussian Maps Decoding
       (不确定性感知的 2D 高斯图解码器)

输入: ΔF (B, 3C, Hf, Wf) + 原始 RGB I1 (B, 3, H, W)
      可选: img2_warped_feat (B, 3, Hf, Wf) — 扭曲到视图1 的 img2 颜色图

输出: 与原图等分辨率的像素级 2D 高斯参数图 + 颜色融合图

预测头 (共享 head_ch 特征):
  ┌─ rot_head           → 4ch  (L2 归一化四元数)
  ├─ scale_head         → 3ch  (Softplus → clamp ≤ scale_max)
  ├─ opacity_head       → 1ch  (Sigmoid)
  ├─ depth_head         → 1ch  (Tanh × 1.0, 度量深度残差)
  ├─ uncertainty_head   → 1ch  (Sigmoid, 不确定性权重)
  ├─ color_blend_head   → 2ch  (Softmax，双视图融合权重)
  └─ color_residual_head→ 3ch  (Tanh × 0.3，颜色残差修正)

颜色融合 (Splat-SAP 风格):
  blend_w = softmax(color_blend_head(feat))         # (B, 2, H, W)
  color_map = blend_w[:,0:1]*I1 + blend_w[:,1:2]*I2w + residual

架构设计：
  - Encoder: ΔF @ Hf/1 → Hf/2 → Hf/4 (3 级，每级 2×Conv2d+BN+GELU)
  - Decoder: 逐步上采样回 H，每级拼接对应 skip 连接
  - 最终在全分辨率 (H, W) 输出，最后拼接原始 RGB 精细化

尺寸约定：
  - 输入特征在 Hf = H // feat_stride (默认 H/4)
  - 解码器通过 3 次上采样 ×2 回到 H
  - 若 feat_stride ≠ 4，最后一步通过 AdaptiveUpsample 对齐到 (H, W)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ──────────────────────────────────────────────
#  基础构建块
# ──────────────────────────────────────────────

class ConvBnGELU(nn.Sequential):
    """Conv2d → BatchNorm2d → GELU"""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class ResBlock(nn.Module):
    """轻量残差块: 2×[Conv-BN-GELU] + shortcut"""
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            ConvBnGELU(ch, ch),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.body(x))


class DownBlock(nn.Module):
    """2× 下采样块: stride-2 Conv + ResBlock"""
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.down = ConvBnGELU(in_ch, out_ch, k=3, s=2, p=1)
        self.res = ResBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res(self.down(x))


class UpBlock(nn.Module):
    """
    2× 上采样块: 双线性上采样 + 与 skip 连接拼接 + ConvBnGELU
    skip_ch=0 表示无 skip 连接
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnGELU(in_ch + skip_ch, out_ch),
            ResBlock(out_ch),
        )

    def forward(
        self, x: torch.Tensor, skip: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        if skip is not None:
            # 确保空间尺寸精确对齐 (防止奇数分辨率的 off-by-one)
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────
#  预测头
# ──────────────────────────────────────────────

class GaussianHeads(nn.Module):
    """
    4 个预测头，共享输入的 head_ch 通道特征图。
    去掉了 uncertainty_head：其乘法调制 eff_opa = opa×(1-unc) 会导致梯度消失陷阱
    （浮块 unc→1 → eff_opa→0 → 梯度→0 → 无法被修正，即 StableGS 伪平衡现象）。
    可靠性控制改由 valid_mask（几何越界）和 opacity_head 直接承担。

    Outputs (all at full resolution H×W):
      rot:         (B, 4, H, W)  L2 归一化单位四元数 [w, x, y, z]
      scale:       (B, 3, H, W)  各向异性缩放，clamp ≤ scale_max
      opacity:     (B, 1, H, W)  Sigmoid 不透明度
      delta_depth: (B, 1, H, W)  Tanh×1.0 深度残差
    """

    def __init__(self, in_ch: int = 32, scale_max: float = 0.002) -> None:
        super().__init__()
        self.scale_max = scale_max

        self.rot_head     = nn.Conv2d(in_ch, 4, 1)
        self.scale_head   = nn.Conv2d(in_ch, 3, 1)
        self.opacity_head = nn.Conv2d(in_ch, 1, 1)
        self.depth_head   = nn.Conv2d(in_ch, 1, 1)

        for head in [self.rot_head, self.scale_head, self.opacity_head, self.depth_head]:
            nn.init.xavier_uniform_(head.weight, gain=0.1)
            nn.init.zeros_(head.bias)

    def forward(self, feat: torch.Tensor) -> dict[str, torch.Tensor]:
        rot = F.normalize(self.rot_head(feat), p=2, dim=1, eps=1e-6)
        scale = F.softplus(self.scale_head(feat)).clamp(max=self.scale_max)
        opacity = torch.sigmoid(self.opacity_head(feat))
        delta_depth = torch.tanh(self.depth_head(feat)) * 1.0

        return {
            "rot":         rot,          # (B, 4, H, W)
            "scale":       scale,        # (B, 3, H, W)
            "opacity":     opacity,      # (B, 1, H, W)
            "delta_depth": delta_depth,  # (B, 1, H, W)
        }


# ──────────────────────────────────────────────
#  主解码器
# ──────────────────────────────────────────────

class GaussianDecoder(nn.Module):
    """
    不确定性感知的轻量 U-Net 高斯图解码器。

    编码器：
      enc0 (input stem):  (3C + 3, Hf, Wf) → enc_dims[0]
      enc1 (×2 down):     enc_dims[0] → enc_dims[1]  @ Hf/2
      enc2 (×2 down):     enc_dims[1] → enc_dims[2]  @ Hf/4

    解码器：
      dec2 (×2 up):       enc_dims[2] → dec_dims[2]  + skip enc1 → Hf/2
      dec1 (×2 up):       dec_dims[2] → dec_dims[1]  + skip enc0 → Hf
      dec0 (to full res): dec_dims[1] → dec_dims[0]  + RGB concat (×feat_stride upsample)

    输出层：
      out_conv: dec_dims[0] → head_ch  (1×1 Conv)
      GaussianHeads: 5 个独立预测头

    Args:
        delta_f_channels: ΔF 的通道数，= 3 × feat_channels
        feat_stride:      特征相对原图的下采样倍数 (决定最终上采样幅度)
        enc_dims:         编码器各级输出通道数
        dec_dims:         解码器各级输出通道数
        head_ch:          预测头共享特征通道数
        scale_max:        高斯缩放上限 (复用 GPS+ 设定)
    """

    def __init__(
        self,
        delta_f_channels: int = 768,   # 3 × 256
        feat_stride: int = 4,
        enc_dims: List[int] = [128, 256, 512],
        dec_dims: List[int] = [128, 256, 512],
        head_ch: int = 32,
        scale_max: float = 0.002,
    ) -> None:
        super().__init__()
        self.feat_stride = feat_stride

        # RGB 在特征分辨率的通道 (concat 到 stem 输入)
        rgb_ch = 3

        # ──── Encoder ────
        self.enc0 = nn.Sequential(          # input stem
            ConvBnGELU(delta_f_channels + rgb_ch, enc_dims[0]),
            ResBlock(enc_dims[0]),
        )
        self.enc1 = DownBlock(enc_dims[0], enc_dims[1])
        self.enc2 = DownBlock(enc_dims[1], enc_dims[2])

        # ──── Bottleneck ────
        self.bottleneck = ResBlock(enc_dims[2])

        # ──── Decoder ────
        self.dec2 = UpBlock(enc_dims[2], enc_dims[1], dec_dims[2])
        self.dec1 = UpBlock(dec_dims[2], enc_dims[0], dec_dims[1])
        # dec0 上采样到全分辨率，额外拼接 原始 RGB (3ch)
        self.dec0 = UpBlock(dec_dims[1], rgb_ch, dec_dims[0])

        # 若 feat_stride > 4，还需要额外上采样步骤
        # 这里固定支持 feat_stride=4 (3× ×2 up = ×8，但我们只用 2 次 dec，最后一次对齐)
        # 若 feat_stride=2，dec0 的最终分辨率就已经等于全分辨率，OK
        # 若 feat_stride=8，需要在 dec0 后再额外 ×2，通过 extra_up 处理
        self.extra_up = None
        if feat_stride > 4:
            extra_ups = []
            extra_stride = feat_stride // 4
            in_ch = dec_dims[0]
            while extra_stride > 1:
                extra_ups.extend([
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                    ConvBnGELU(in_ch, in_ch),
                ])
                extra_stride //= 2
            self.extra_up = nn.Sequential(*extra_ups)

        # ──── Output Conv ────
        self.out_conv = nn.Sequential(
            ConvBnGELU(dec_dims[0], head_ch),
            nn.Conv2d(head_ch, head_ch, 1),
        )

        # ──── Prediction Heads ────
        self.heads = GaussianHeads(head_ch, scale_max=scale_max)

        # ──── 颜色融合头 (双视图加权融合 + 残差，参考 Splat-SAP Eq.11-15) ────
        # color_blend_head: 预测 img1 与 warped img2 的混合权重 (softmax)
        self.color_blend_head    = nn.Conv2d(head_ch, 2, 1)
        # color_residual_head: 小幅颜色残差修正 (tanh × 0.3)
        self.color_residual_head = nn.Conv2d(head_ch, 3, 1)
        nn.init.zeros_(self.color_blend_head.weight)
        nn.init.zeros_(self.color_blend_head.bias)
        nn.init.zeros_(self.color_residual_head.weight)
        nn.init.zeros_(self.color_residual_head.bias)

    def forward(
        self,
        delta_f: torch.Tensor,
        img1: torch.Tensor,
        img2_warped_feat: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        解码高斯参数图，可选预测双视图融合颜色图。

        Args:
            delta_f:          (B, 3C, Hf, Wf)  Module 2 的交互特征
            img1:             (B, 3, H,  W )   视图1 原始 RGB [-1,1]
            img2_warped_feat: (B, 3, Hf, Wf)   扭曲到视图1 的 img2（可为 None）

        Returns:
            dict 包含:
              "rot"         (B, 4, H, W)
              "scale"       (B, 3, H, W)
              "opacity"     (B, 1, H, W)
              "delta_depth" (B, 1, H, W)
              "uncertainty" (B, 1, H, W)
              "color_map"   (B, 3, H, W)  [0,1] （若 img2_warped_feat 非 None）
        """
        B, _, H, W = img1.shape
        Hf, Wf = delta_f.shape[-2:]

        # ── 将 RGB 下采样到特征分辨率 ──
        img1_feat = F.interpolate(
            img1, size=(Hf, Wf), mode="bilinear", align_corners=False
        )  # (B, 3, Hf, Wf)

        # ── Encoder ──
        x0 = self.enc0(torch.cat([delta_f, img1_feat], dim=1))  # (B, enc_dims[0], Hf, Wf)
        x1 = self.enc1(x0)                                       # (B, enc_dims[1], Hf/2)
        x2 = self.enc2(x1)                                       # (B, enc_dims[2], Hf/4)

        # ── Bottleneck ──
        x2 = self.bottleneck(x2)

        # ── Decoder ──
        d2 = self.dec2(x2, skip=x1)
        d1 = self.dec1(d2, skip=x0)
        d0 = self.dec0(d1, skip=img1)  # ×2 → 全分辨率 (B, dec_dims[0], H, W)

        if self.extra_up is not None:
            d0 = self.extra_up(d0)
        if d0.shape[-2:] != (H, W):
            d0 = F.interpolate(d0, size=(H, W), mode="bilinear", align_corners=False)

        # ── 高斯参数预测 ──
        feat_out = self.out_conv(d0)
        out = self.heads(feat_out)

        # ── 颜色融合预测（双视图加权 + 残差）──
        if img2_warped_feat is not None:
            # 上采样 warped img2 到全分辨率（img2_warped_feat 中非重叠区已被 valid_mask 置零）
            img2_w = F.interpolate(
                img2_warped_feat, size=(H, W), mode="bilinear", align_corners=False
            )  # (B, 3, H, W)  [-1,1]，非重叠区≈0

            # 检测非重叠区（img2_warped 因 zeros-padding 全为 0）
            # cover2: (B, 1, H, W) 1=img2 有有效颜色, 0=无重叠
            cover2 = (img2_w.abs().sum(dim=1, keepdim=True) > 1e-4).float()

            # [-1,1] → [0,1]
            img1_01  = img1   * 0.5 + 0.5
            img2_w01 = img2_w * 0.5 + 0.5

            # 融合权重 (softmax 保证和为 1)
            blend_w = torch.softmax(
                self.color_blend_head(feat_out), dim=1
            )  # (B, 2, H, W)  sum=1

            # 非重叠区强制 w2=0, w1=1，避免 img2 零值污染颜色
            w1 = blend_w[:, 0:1] + blend_w[:, 1:2] * (1.0 - cover2)
            w2 = blend_w[:, 1:2] * cover2

            # 颜色残差（小幅修正，非重叠区残差也应贡献更少 → 乘 cover2 的 smooth 版本）
            residual = torch.tanh(self.color_residual_head(feat_out)) * 0.3

            color_map = w1 * img1_01 + w2 * img2_w01 + residual
            out["color_map"] = color_map.clamp(0.0, 1.0)  # (B, 3, H, W)

        return out
