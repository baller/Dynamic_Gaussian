import torch
import pytest

from lib.stereo_gs.cvct import VisibilityGate, ResidualHead, CVCTModule, warp_with_disparity


def test_visibility_gate_output_shape_and_range():
    gate = VisibilityGate(in_channels=64, hidden=32)
    fused_feat = torch.randn(2, 64, 16, 16)
    confidence = torch.rand(2, 1, 16, 16)
    omega = gate(fused_feat, confidence)
    assert omega.shape == (2, 1, 16, 16)
    assert (omega >= 0).all() and (omega <= 1).all()


def test_visibility_gate_param_count_under_5k():
    gate = VisibilityGate(in_channels=64, hidden=32)
    n = sum(p.numel() for p in gate.parameters())
    assert n < 5000, f"VisibilityGate has {n} params, expected < 5K"


def test_visibility_gate_gradient_flows():
    gate = VisibilityGate(in_channels=32, hidden=16)
    fused = torch.randn(1, 32, 8, 8, requires_grad=True)
    conf = torch.rand(1, 1, 8, 8)
    out = gate(fused, conf)
    out.sum().backward()
    assert fused.grad is not None and fused.grad.abs().sum() > 0


def test_residual_head_output_within_bound():
    head = ResidualHead(in_channels=64, hidden=16, bound=0.05)
    feat = torch.randn(1, 64, 16, 16) * 100  # large input
    delta = head(feat)
    assert delta.shape == (1, 3, 16, 16)
    assert delta.abs().max().item() <= 0.05 + 1e-6


def test_residual_head_param_count_under_5k():
    head = ResidualHead(in_channels=64, hidden=16, bound=0.05)
    n = sum(p.numel() for p in head.parameters())
    assert n < 5000


def test_residual_head_zero_bound_returns_zero():
    head = ResidualHead(in_channels=32, hidden=8, bound=0.0)
    feat = torch.randn(1, 32, 4, 4)
    delta = head(feat)
    assert delta.abs().max().item() == 0.0


def test_warp_with_disparity_identity_when_zero():
    img = torch.randn(1, 3, 8, 8)
    disp = torch.zeros(1, 1, 8, 8)
    warped = warp_with_disparity(img, disp)
    assert torch.allclose(warped, img, atol=1e-5)


def test_cvct_module_identity_mode_returns_unmixed_color():
    """In identity mode, c_final = 0.5·c_left + 0.5·c_right_warped + 0."""
    mod = CVCTModule(fused_channels=32, shared_channels=32, residual_bound=0.05)
    mod.set_identity(True)
    B, C, H, W = 1, 32, 8, 8
    fused = torch.randn(B, C, H, W)
    shared = torch.randn(B, C, H, W)
    conf = torch.rand(B, 1, H, W)
    c_self = torch.rand(B, 3, H, W)
    c_other = torch.rand(B, 3, H, W)
    disp = torch.zeros(B, 1, H, W)  # zero-disp warp = identity

    out = mod(fused, shared, conf, c_self, c_other, disp)
    expected = 0.5 * c_self + 0.5 * c_other
    assert torch.allclose(out["c_final"], expected, atol=1e-5)
    assert torch.allclose(out["omega"], torch.full_like(out["omega"], 0.5))
    assert torch.allclose(out["delta_rgb"], torch.zeros_like(out["delta_rgb"]))


def test_cvct_module_active_mode_omega_in_range():
    mod = CVCTModule(fused_channels=32, shared_channels=32, residual_bound=0.05)
    mod.set_identity(False)
    B, C, H, W = 1, 32, 8, 8
    fused = torch.randn(B, C, H, W)
    shared = torch.randn(B, C, H, W)
    conf = torch.rand(B, 1, H, W)
    c_self = torch.rand(B, 3, H, W)
    c_other = torch.rand(B, 3, H, W)
    disp = torch.rand(B, 1, H, W) * 2.0
    out = mod(fused, shared, conf, c_self, c_other, disp)
    assert (out["omega"] >= 0).all() and (out["omega"] <= 1).all()
    assert out["delta_rgb"].abs().max() <= 0.05 + 1e-6
