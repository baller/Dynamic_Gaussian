import torch
import pytest

from lib.stereo_gs.losses_cvct import l_cycle, l_omega_align, build_omega_target


def test_l_cycle_zero_when_evidences_match():
    c1 = torch.rand(1, 3, 8, 8)
    c2 = c1.clone()
    omega = torch.full((1, 1, 8, 8), 0.5)
    assert l_cycle(c1, c2, omega).item() < 1e-6


def test_l_cycle_zero_when_omega_at_endpoints():
    """At ω=0 or ω=1 the visibility mask 4·ω·(1-ω) = 0, so any mismatch is ignored."""
    c1 = torch.zeros(1, 3, 4, 4)
    c2 = torch.ones(1, 3, 4, 4)
    omega_zero = torch.zeros(1, 1, 4, 4)
    omega_one = torch.ones(1, 1, 4, 4)
    assert l_cycle(c1, c2, omega_zero).item() < 1e-6
    assert l_cycle(c1, c2, omega_one).item() < 1e-6


def test_l_cycle_positive_at_dual_visible_with_mismatch():
    c1 = torch.zeros(1, 3, 4, 4)
    c2 = torch.ones(1, 3, 4, 4)
    omega = torch.full((1, 1, 4, 4), 0.5)
    assert l_cycle(c1, c2, omega).item() > 0


def test_build_omega_target_central_value_in_dual_visible():
    """High confidence and matching c_self/c_other should give ω_target ≈ 0.5."""
    c_self = torch.full((1, 3, 4, 4), 0.5)
    c_other_warped = torch.full((1, 3, 4, 4), 0.5)  # match
    confidence = torch.ones(1, 1, 4, 4)
    target = build_omega_target(c_self, c_other_warped, confidence)
    assert target.shape == (1, 1, 4, 4)
    assert (target >= 0).all() and (target <= 1).all()
    assert torch.allclose(target, torch.full_like(target, 0.5), atol=0.1)


def test_l_omega_align_zero_when_omega_matches_target():
    omega = torch.full((1, 1, 4, 4), 0.5)
    target = torch.full((1, 1, 4, 4), 0.5)
    assert l_omega_align(omega, target).item() < 1e-6
