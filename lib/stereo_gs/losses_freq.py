"""FDSG (W-CVCT-GS) 的频域损失函数集合。

包含三个损失:
  - l_band:        多带重建损失 (L_band)
  - l_active:      GT 小波引导的子高斯稀疏损失 (L_active)
  - l_disentangle: 各层级子高斯的子带特化损失 (L_disentangle)
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from lib.stereo_gs.wavelet_ops import (
    dwt3,
    pad_to_multiple,
    band_energy,
    log_compress,
)


def _dwt3_padded(img: torch.Tensor):
    """先填充到 8 的整数倍，再做 3 级 DWT。返回 (dwt_dict, pad)。"""
    padded, pad = pad_to_multiple(img, 8)
    return dwt3(padded), pad


def l_band(
    pred: torch.Tensor,
    gt: torch.Tensor,
    ll_weight: float = 0.4,
    band_weights: Sequence[float] = (0.3, 0.2, 0.1),
    log_compress_k: float = 10.0,
) -> torch.Tensor:
    """多带 L1 重建损失。

    L_band = β·‖LL−LL*‖₁ + Σ_j α_j · ‖D_j − D_j*‖₁ (在对数压缩空间中计算)。

    Args:
        pred, gt: (B, 3, H, W)，数值范围保持一致即可；规范建议 [0, 1]。
        ll_weight: β，LL 低频子带权重。
        band_weights: (α_1, α_2, α_3)，各级细节子带权重。
        log_compress_k: 对数压缩强度系数。

    Returns:
        标量张量。
    """
    assert pred.shape == gt.shape
    assert len(band_weights) == 3
    pred_dwt, _ = _dwt3_padded(pred)
    gt_dwt, _ = _dwt3_padded(gt)

    loss = ll_weight * F.l1_loss(pred_dwt["LL"], gt_dwt["LL"])
    for j, w in enumerate(band_weights, start=1):
        if w == 0.0:
            continue
        for sub_pred, sub_gt in zip(pred_dwt[f"D{j}"], gt_dwt[f"D{j}"]):
            sp = log_compress(sub_pred, k=log_compress_k)
            sg = log_compress(sub_gt, k=log_compress_k)
            loss = loss + w * F.l1_loss(sp, sg)
    return loss


def l_active(
    split_weights: torch.Tensor,
    gt_image: torch.Tensor,
    gt_levels: int = 3,
    mode: str = "symmetric",
    min_activation: float = 0.0,
) -> torch.Tensor:
    """基于 GT 小波能量的子高斯分裂权重正则。

    支持两种模式:
      - "symmetric":    L_active = Σ_j ‖ w_j − E_j_norm ‖₁（对称对齐，推荐）
      - "penalty_only": L_active = Σ_j ‖ w_j · (1 − E_j_norm) ‖₁（legacy 单向惩罚）

    对称模式鼓励 w_j 跟踪 GT 频带能量分布：高频区激活、平坦区稀疏。
    min_activation > 0 时追加 hinge loss 防止权重全零塌缩。

    Args:
        split_weights: (B, k_sub, H, W)，取值范围 [0, 1]。
            k_sub 必须等于 `gt_levels`。
        gt_image: (B, 3, H, W)，GT 图像。
        gt_levels: 使用的小波层级数 (默认 3)。
        mode: "symmetric" | "penalty_only"
        min_activation: 全局最低平均激活率，0 表示不生效。

    Returns:
        标量张量。
    """
    B, k_sub, H, W = split_weights.shape
    assert k_sub == gt_levels, f"k_sub={k_sub} must match gt_levels={gt_levels}"

    gt_dwt, _ = _dwt3_padded(gt_image)
    loss = split_weights.new_zeros(())
    for j in range(1, gt_levels + 1):
        e_j = band_energy(gt_dwt[f"D{j}"])  # (B, 1, h_j, w_j)
        e_j_full = F.interpolate(e_j, size=(H, W), mode="bilinear", align_corners=False)
        e_max = e_j_full.amax(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        e_norm = (e_j_full / e_max).clamp(0, 1)
        w_j = split_weights[:, j - 1 : j]
        if mode == "penalty_only":
            loss = loss + (w_j * (1.0 - e_norm)).abs().mean()
        else:  # symmetric
            loss = loss + (w_j - e_norm).abs().mean()
    if min_activation > 0:
        loss = loss + min_activation * F.relu(
            min_activation - split_weights.mean())
    return loss


def l_disentangle(
    delta_images: Sequence[torch.Tensor],
    log_compress_k: float = 10.0,
) -> torch.Tensor:
    """逐层级子高斯频带特化损失。

    对每个 j ∈ {1, 2, 3}，第 j 层级子高斯的贡献图 ΔI_j 应当完全集中在
    细节子带 D_j 上；若其能量出现在 LL 或 D_{i≠j} 中则会被惩罚。

    L_disentangle = Σ_j  (‖LL_Δ_j‖₁ + Σ_{i≠j} ‖D_i_Δ_j‖₁)   (在对数压缩空间中计算)

    Args:
        delta_images: 长度为 3 的列表，每个元素形状为 (B, 3, H, W)，依次为 [ΔI_1, ΔI_2, ΔI_3]。
        log_compress_k: 对数压缩强度系数。

    Returns:
        标量张量。
    """
    assert len(delta_images) == 3
    loss = delta_images[0].new_zeros(())
    for j_idx, delta in enumerate(delta_images, start=1):
        dwt, _ = _dwt3_padded(delta)
        loss = loss + log_compress(dwt["LL"], k=log_compress_k).abs().mean()
        for i in (1, 2, 3):
            if i == j_idx:
                continue
            for sub in dwt[f"D{i}"]:
                loss = loss + log_compress(sub, k=log_compress_k).abs().mean()
    return loss


def l_depth_smooth(depth_residual: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    """Edge-aware TV smoothness on inverse-depth residual.

    Penalizes spatial variation in flat regions more heavily than at edges.

    Args:
        depth_residual: (B, 1, H, W) inverse-depth residual.
        image: (B, 3, H, W) RGB in [0, 1].

    Returns:
        scalar tensor.
    """
    dx = (depth_residual[..., :-1] - depth_residual[..., 1:]).abs().mean()
    dy = (depth_residual[..., :-1, :] - depth_residual[..., 1:, :]).abs().mean()
    img_gray = image.mean(dim=1, keepdim=True)
    img_dx = (img_gray[..., :-1] - img_gray[..., 1:]).abs()
    img_dy = (img_gray[..., :-1, :] - img_gray[..., 1:, :]).abs()
    wx = torch.exp(-img_dx * 5.0).detach()
    wy = torch.exp(-img_dy * 5.0).detach()
    return (dx * wx.mean() + dy * wy.mean()) * 0.5


def l_depth_anchor(depth: torch.Tensor, depth_ffs: torch.Tensor) -> torch.Tensor:
    """L2 anchor: keep refined depth close to FFS initial depth.

    Args:
        depth: (B, 1, H, W) refined inverse depth.
        depth_ffs: (B, 1, H, W) FFS initial inverse depth.

    Returns:
        scalar tensor.
    """
    return F.mse_loss(depth, depth_ffs)
