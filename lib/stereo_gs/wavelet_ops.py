"""
W-CVCT-GS 的 Haar 离散小波变换工具集。

以纯张量运算实现正交归一 Haar 二维 DWT，从而保持项目零新增外部依赖。
被 FDSG 系列损失用于将图像分解为 LL 低频子带 + 每层 3 个细节子带。
"""

from __future__ import annotations

from typing import Tuple, Dict

import torch
import torch.nn.functional as F


def haar_dwt2d_step(x: torch.Tensor):
    """单层正交归一 Haar 二维 DWT。

    Args:
        x: (B, C, H, W)，要求 H、W 为偶数。

    Returns:
        (LL, LH, HL, HH)，每个张量形状为 (B, C, H/2, W/2)。
    """
    if x.shape[-1] % 2 != 0 or x.shape[-2] % 2 != 0:
        raise ValueError(f"haar_dwt2d_step needs even H,W, got {tuple(x.shape)}")
    a = x[..., 0::2, 0::2]
    b = x[..., 0::2, 1::2]
    c = x[..., 1::2, 0::2]
    d = x[..., 1::2, 1::2]
    LL = (a + b + c + d) * 0.5
    LH = (a + b - c - d) * 0.5
    HL = (a - b + c - d) * 0.5
    HH = (a - b - c + d) * 0.5
    return LL, LH, HL, HH


def dwt3(x: torch.Tensor) -> Dict[str, object]:
    """3 级 Haar DWT 级联分解。

    Args:
        x: (B, C, H, W)，要求 H、W 能被 8 整除。

    Returns:
        dict，包含:
            'LL':  (B, C, H/8, W/8)，最低频残量
            'D1':  (LH_1, HL_1, HH_1)，分辨率 H/2, W/2
            'D2':  (LH_2, HL_2, HH_2)，分辨率 H/4, W/4
            'D3':  (LH_3, HL_3, HH_3)，分辨率 H/8, W/8
    """
    LL_1, LH_1, HL_1, HH_1 = haar_dwt2d_step(x)
    LL_2, LH_2, HL_2, HH_2 = haar_dwt2d_step(LL_1)
    LL_3, LH_3, HL_3, HH_3 = haar_dwt2d_step(LL_2)
    return {
        "LL": LL_3,
        "D1": (LH_1, HL_1, HH_1),
        "D2": (LH_2, HL_2, HH_2),
        "D3": (LH_3, HL_3, HH_3),
    }


def pad_to_multiple(x: torch.Tensor, m: int) -> Tuple[torch.Tensor, Tuple[int, int, int, int]]:
    """对空间维做反射填充，使 H、W 成为 m 的整数倍。

    Args:
        x: (B, C, H, W)。
        m: 对齐倍数 (3 级 DWT 取 8)。

    Returns:
        (padded, (pad_left, pad_right, pad_top, pad_bottom))
    """
    _, _, H, W = x.shape
    pad_h = (m - H % m) % m
    pad_w = (m - W % m) % m
    pad = (0, pad_w, 0, pad_h)
    if pad_h == 0 and pad_w == 0:
        return x, pad
    return F.pad(x, pad, mode="reflect"), pad


def band_energy(detail_band: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
    """计算 (LH, HL, HH) 三元组的逐像素 L2 幅值，并在通道维取均值。

    Args:
        detail_band: 三个张量组成的元组，每个形状 (B, C, h, w)。

    Returns:
        (B, 1, h, w) 能量图。
    """
    LH, HL, HH = detail_band
    sq = LH.pow(2) + HL.pow(2) + HH.pow(2)
    e = torch.sqrt(sq + 1e-12).mean(dim=1, keepdim=True)
    return e


def log_compress(x: torch.Tensor, k: float = 10.0) -> torch.Tensor:
    """保留符号的对数压缩: y = sign(x) * log1p(|x|*k)。

    用于稳定小幅值高频子带上的损失数值。
    """
    return torch.sign(x) * torch.log1p(x.abs() * k)
