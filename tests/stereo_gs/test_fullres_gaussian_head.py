import torch

from lib.stereo_gs.fullres_gaussian_head import FullResGaussianHead


def test_fullres_gaussian_head_outputs_bounded_depth_residual():
    head = FullResGaussianHead(
        in_channels=100,
        hidden_dim=16,
        max_depth_residual=0.25,
    )
    feat = torch.randn(2, 100, 8, 12)
    confidence = torch.rand(2, 1, 8, 12)

    out = head(feat, confidence)

    assert out["depth_residual"].shape == (2, 1, 8, 12)
    assert out["depth_residual"].abs().max().item() <= 0.25 + 1e-6


def test_fullres_gaussian_head_depth_residual_starts_as_noop():
    head = FullResGaussianHead(
        in_channels=100,
        hidden_dim=16,
        max_depth_residual=0.5,
    )
    feat = torch.randn(1, 100, 4, 5)
    confidence = torch.rand(1, 1, 4, 5)

    out = head(feat, confidence)

    assert torch.allclose(out["depth_residual"], torch.zeros_like(out["depth_residual"]))
