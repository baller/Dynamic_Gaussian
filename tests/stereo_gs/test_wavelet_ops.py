import torch
import pytest

from lib.stereo_gs.wavelet_ops import (
    haar_dwt2d_step,
    dwt3,
    pad_to_multiple,
    band_energy,
    log_compress,
)


def test_haar_dwt2d_step_constant_input_zero_high_freq():
    x = torch.full((1, 1, 4, 4), 3.0)
    LL, LH, HL, HH = haar_dwt2d_step(x)
    assert LL.shape == (1, 1, 2, 2)
    assert torch.allclose(LL, torch.full_like(LL, 6.0))  # (3+3+3+3)/2 = 6
    assert torch.allclose(LH, torch.zeros_like(LH))
    assert torch.allclose(HL, torch.zeros_like(HL))
    assert torch.allclose(HH, torch.zeros_like(HH))


def test_haar_dwt2d_step_impulse_spreads_to_all_bands():
    x = torch.zeros(1, 1, 2, 2)
    x[0, 0, 0, 0] = 1.0
    LL, LH, HL, HH = haar_dwt2d_step(x)
    for t in (LL, LH, HL, HH):
        assert torch.isclose(t.flatten()[0], torch.tensor(0.5))


def test_haar_dwt2d_step_horizontal_edge():
    """An input with a horizontal step edge should put energy in LH only."""
    x = torch.tensor([[[[1.0, 1.0], [-1.0, -1.0]]]])
    LL, LH, HL, HH = haar_dwt2d_step(x)
    assert torch.isclose(LL.flatten()[0], torch.tensor(0.0))
    assert torch.isclose(LH.flatten()[0], torch.tensor(2.0))
    assert torch.isclose(HL.flatten()[0], torch.tensor(0.0))
    assert torch.isclose(HH.flatten()[0], torch.tensor(0.0))


def test_dwt3_shapes():
    x = torch.randn(2, 3, 32, 24)
    out = dwt3(x)
    assert out["LL"].shape == (2, 3, 4, 3)
    assert out["D1"][0].shape == (2, 3, 16, 12)
    assert out["D2"][0].shape == (2, 3, 8, 6)
    assert out["D3"][0].shape == (2, 3, 4, 3)


def test_pad_to_multiple_pads_correctly():
    x = torch.zeros(1, 3, 13, 11)
    padded, pad = pad_to_multiple(x, 8)
    assert padded.shape == (1, 3, 16, 16)
    assert pad == (0, 5, 0, 3)  # left, right, top, bottom


def test_pad_to_multiple_no_op_when_already_aligned():
    x = torch.zeros(1, 3, 16, 16)
    padded, pad = pad_to_multiple(x, 8)
    assert padded.shape == (1, 3, 16, 16)
    assert pad == (0, 0, 0, 0)


def test_band_energy_returns_per_pixel_magnitude():
    LH = torch.tensor([[[[3.0]]]])
    HL = torch.tensor([[[[4.0]]]])
    HH = torch.tensor([[[[0.0]]]])
    e = band_energy((LH, HL, HH))
    assert torch.isclose(e.flatten()[0], torch.tensor(5.0))
    assert e.shape == (1, 1, 1, 1)


def test_log_compress_preserves_sign():
    x = torch.tensor([1.0, -1.0, 0.0])
    y = log_compress(x, k=10.0)
    assert y[0] > 0 and y[1] < 0 and y[2] == 0
    assert torch.isclose(y[0], -y[1])
