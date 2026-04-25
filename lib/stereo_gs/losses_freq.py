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
) -> torch.Tensor:
    """基于 GT 小波能量的子高斯分裂权重稀疏正则。

    对每一层级 j，第 j 个子高斯只允许在 GT 含有 D_j 子带能量的位置激活。
    L_active = Σ_j ‖ weights[:, j-1] · (1 − E_j_GT) ‖_1。

    Args:
        split_weights: (B, k_sub, H, W)，取值范围 [0, 1]，来自 LearnedSplitCriterion 的输出。
            k_sub 必须等于 `gt_levels`。
        gt_image: (B, 3, H, W)，GT 图像。
        gt_levels: 使用的小波层级数 (默认 3)。

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
        loss = loss + (w_j * (1.0 - e_norm)).abs().mean()
    return loss
