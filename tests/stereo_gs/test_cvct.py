import torch
import pytest

from lib.stereo_gs.cvct import VisibilityGate


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
