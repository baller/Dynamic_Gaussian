import torch

from lib.ffs_depth import smooth_ffs_depth_edges


def test_smoothing_disabled_returns_input_unchanged():
    depth = torch.rand(2, 1, 8, 8, dtype=torch.float32)

    out = smooth_ffs_depth_edges(depth, enable=False)

    assert out is depth
    assert torch.equal(out, depth)


def test_smoothing_reduces_step_edge_without_changing_far_pixels():
    depth = torch.full((1, 1, 9, 9), 0.2)
    depth[:, :, :, 5:] = 0.8

    out = smooth_ffs_depth_edges(
        depth,
        enable=True,
        kernel_size=3,
        band_size=3,
        strength=1.0,
        quantile=0.8,
        min_gradient=1e-6,
    )

    before_jump = depth[0, 0, 4, 5] - depth[0, 0, 4, 4]
    after_jump = out[0, 0, 4, 5] - out[0, 0, 4, 4]

    assert after_jump.abs() < before_jump.abs()
    assert torch.equal(out[:, :, :, :2], depth[:, :, :, :2])
    assert torch.equal(out[:, :, :, 7:], depth[:, :, :, 7:])


def test_iterative_smoothing_creates_wider_edge_transition():
    depth = torch.full((1, 1, 15, 15), 0.2)
    depth[:, :, :, 8:] = 0.8

    one_pass = smooth_ffs_depth_edges(
        depth,
        enable=True,
        kernel_size=3,
        band_size=7,
        strength=0.7,
        quantile=0.8,
        min_gradient=1e-6,
        iterations=1,
    )
    multi_pass = smooth_ffs_depth_edges(
        depth,
        enable=True,
        kernel_size=3,
        band_size=7,
        strength=0.7,
        quantile=0.8,
        min_gradient=1e-6,
        iterations=4,
    )

    one_pass_max_jump = (one_pass[:, :, :, 1:] - one_pass[:, :, :, :-1]).abs().max()
    multi_pass_max_jump = (multi_pass[:, :, :, 1:] - multi_pass[:, :, :, :-1]).abs().max()

    assert multi_pass_max_jump < one_pass_max_jump
    assert torch.equal(multi_pass[:, :, :, :3], depth[:, :, :, :3])
    assert torch.equal(multi_pass[:, :, :, 12:], depth[:, :, :, 12:])


def test_mask_boundary_does_not_smooth_when_depth_gradient_is_weak():
    depth = torch.full((1, 1, 9, 9), 0.4)
    depth[:, :, :, 5:] = 0.42
    mask = torch.zeros_like(depth)
    mask[:, :, :, :5] = 1.0

    out = smooth_ffs_depth_edges(
        depth,
        mask=mask,
        enable=True,
        kernel_size=3,
        band_size=3,
        strength=1.0,
        quantile=1.0,
        min_gradient=10.0,
    )

    assert torch.equal(out, depth)


def test_smoothing_preserves_tensor_properties_and_clamps_depth():
    depth = torch.full((1, 1, 5, 5), 0.0, dtype=torch.float64)
    depth[:, :, :, 3:] = 1.0

    out = smooth_ffs_depth_edges(
        depth,
        enable=True,
        kernel_size=3,
        band_size=3,
        strength=0.5,
        quantile=0.8,
        min_gradient=1e-6,
    )

    assert out.shape == depth.shape
    assert out.dtype == depth.dtype
    assert out.device == depth.device
    assert torch.all(out >= 1e-6)
