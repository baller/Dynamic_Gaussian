import torch
import pytest

from lib.stereo_gs.losses_freq import l_band, l_active, l_disentangle


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


def test_l_active_zero_when_weights_zero():
    weights = torch.zeros(1, 3, 16, 16)
    gt = torch.rand(1, 3, 16, 16)
    loss = l_active(weights, gt)
    assert loss.item() == 0.0


def test_l_active_zero_when_weights_only_in_high_freq_regions():
    """If split weights activate only where GT has band energy, loss should be small."""
    gt = torch.zeros(1, 3, 16, 16)
    gt[0, 0, 4:6, 4:6] = 1.0  # localized edge → band energy is local
    weights = torch.zeros(1, 3, 16, 16)
    weights[:, :, 4:6, 4:6] = 1.0
    loss_aligned = l_active(weights, gt)

    weights_misaligned = torch.zeros_like(weights)
    weights_misaligned[:, :, 0:2, 0:2] = 1.0
    loss_misaligned = l_active(weights_misaligned, gt)

    assert loss_misaligned.item() > loss_aligned.item()


def test_l_disentangle_zero_when_each_delta_is_in_its_own_band():
    """Construct ΔI_j with energy concentrated in band D_j only — loss should be small."""
    H = W = 32
    pred_full = torch.zeros(1, 3, H, W)
    # ΔI_1 is high-frequency only (alternating pixels)
    delta_1 = torch.zeros_like(pred_full)
    delta_1[0, 0, 0::2, 0::2] = 1.0
    delta_1[0, 0, 1::2, 1::2] = -1.0
    # ΔI_2 mid-freq (alternating 4-pixel blocks)
    delta_2 = torch.zeros_like(pred_full)
    delta_2[0, 0, 0:8, 0:8] = 1.0
    delta_2[0, 0, 8:16, 8:16] = 1.0
    delta_2[0, 0, 0:8, 8:16] = -1.0
    delta_2[0, 0, 8:16, 0:8] = -1.0
    delta_3 = torch.zeros_like(pred_full)
    delta_3[0, 0, :H // 2] = 1.0
    delta_3[0, 0, H // 2:] = -1.0

    loss_aligned = l_disentangle([delta_1, delta_2, delta_3])
    # Now swap the bands → loss should be larger
    loss_swapped = l_disentangle([delta_3, delta_1, delta_2])
    assert loss_aligned.item() < loss_swapped.item()


def test_l_disentangle_zero_for_all_zero_inputs():
    H = W = 32
    z = torch.zeros(1, 3, H, W)
    loss = l_disentangle([z, z, z])
    assert loss.item() < 1e-6
