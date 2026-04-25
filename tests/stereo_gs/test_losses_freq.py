import torch
import pytest

from lib.stereo_gs.losses_freq import l_band


def test_l_band_zero_when_pred_equals_gt():
    img = torch.rand(1, 3, 32, 32)
    loss = l_band(img, img)
    assert loss.item() < 1e-6


def test_l_band_positive_when_inputs_differ():
    pred = torch.zeros(1, 3, 32, 32)
    gt = torch.ones(1, 3, 32, 32)
    loss = l_band(pred, gt)
    assert loss.item() > 0


def test_l_band_pads_unaligned_inputs():
    """Inputs whose H, W are not multiples of 8 must not raise."""
    pred = torch.rand(1, 3, 13, 11)
    gt = torch.rand(1, 3, 13, 11)
    loss = l_band(pred, gt)  # must not raise
    assert torch.isfinite(loss)


def test_l_band_high_freq_weight_dominates_when_specified():
    """If we set α_3 = 1 and others 0, only D3 mismatch contributes."""
    pred = torch.zeros(1, 3, 32, 32)
    gt = torch.zeros(1, 3, 32, 32)
    gt[0, 0, ::8, ::8] = 1.0  # high-freq spike pattern
    loss_hf_only = l_band(pred, gt, ll_weight=0.0, band_weights=(0.0, 0.0, 1.0))
    loss_lf_only = l_band(pred, gt, ll_weight=1.0, band_weights=(0.0, 0.0, 0.0))
    assert loss_hf_only.item() > 0
    assert loss_lf_only.item() > 0  # LL band picks up some energy too
