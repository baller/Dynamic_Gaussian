"""Utilities for excluding invalid image regions from reconstruction losses."""

from __future__ import annotations

import torch


def build_border_ignore_mask(reference: torch.Tensor, border: int) -> torch.Tensor:
    """Return a (B, 1, H, W) mask that ignores a fixed image border.

    Args:
        reference: Tensor shaped (B, C, H, W). Device and dtype are reused.
        border: Number of pixels to ignore on each side. Non-positive values keep
            the whole image valid.
    """
    if reference.ndim != 4:
        raise ValueError(f"reference must have shape (B, C, H, W), got {reference.shape}")

    b, _, h, w = reference.shape
    mask = torch.ones((b, 1, h, w), device=reference.device, dtype=reference.dtype)
    border = int(border)
    if border <= 0:
        return mask

    if border * 2 >= h or border * 2 >= w:
        return torch.zeros_like(mask)

    mask[:, :, :border, :] = 0
    mask[:, :, -border:, :] = 0
    mask[:, :, :, :border] = 0
    mask[:, :, :, -border:] = 0
    return mask


def masked_l1_loss(
    pred: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L1 averaged only over valid pixels in ``mask``."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}")
    if mask.ndim != 4 or mask.shape[0] != pred.shape[0] or mask.shape[-2:] != pred.shape[-2:]:
        raise ValueError(f"mask must have shape (B, 1|C, H, W), got {mask.shape}")

    mask = mask.to(device=pred.device, dtype=pred.dtype)
    weighted_error = (pred - gt).abs() * mask
    if mask.shape[1] == 1:
        denom = mask.sum() * pred.shape[1]
    elif mask.shape[1] == pred.shape[1]:
        denom = mask.sum()
    else:
        raise ValueError(f"mask channel count must be 1 or {pred.shape[1]}, got {mask.shape[1]}")
    return weighted_error.sum() / denom.clamp_min(eps)


def apply_loss_mask(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Replace ignored prediction pixels with GT so dense losses see zero error there."""
    if pred.shape != gt.shape:
        raise ValueError(f"pred and gt must have the same shape, got {pred.shape} and {gt.shape}")
    mask = mask.to(device=pred.device, dtype=pred.dtype)
    return pred * mask + gt.detach() * (1.0 - mask)
