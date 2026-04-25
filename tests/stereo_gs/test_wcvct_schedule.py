import torch
from lib.stereo_gs.wcvct_schedule import WCVCTSchedule, PhaseState


def test_phase1_active_terms():
    sch = WCVCTSchedule(phase1_end=5000, phase2_end=30000)
    s = sch.state(step=100)
    assert s.cvct_identity_mode is True
    assert s.lambda_band == 0.0
    assert s.lambda_active == 0.0
    assert s.lambda_disentangle == 0.0
    assert s.lambda_cycle == 0.0
    assert s.lambda_omega == 0.0


def test_phase2_enables_band_and_active_only():
    sch = WCVCTSchedule(
        phase1_end=5000, phase2_end=30000,
        lambda_band=0.3, lambda_active=0.05,
        lambda_disentangle=0.5, lambda_cycle=0.2, lambda_omega=0.1,
    )
    s = sch.state(step=10000)
    assert s.cvct_identity_mode is True
    assert s.lambda_band == 0.3
    assert s.lambda_active == 0.05
    assert s.lambda_disentangle == 0.0
    assert s.lambda_cycle == 0.0
    assert s.lambda_omega == 0.0


def test_phase3_unlocks_all_with_disentangle_warmup():
    sch = WCVCTSchedule(
        phase1_end=5000, phase2_end=30000,
        lambda_band=0.3, lambda_active=0.05,
        lambda_disentangle=0.5, lambda_disentangle_warmup=0.1,
        lambda_disentangle_warmup_steps=5000,
        lambda_cycle=0.2, lambda_omega=0.1,
    )
    # Just past phase2_end → still in warmup
    s_warm = sch.state(step=30001)
    assert s_warm.cvct_identity_mode is False
    assert abs(s_warm.lambda_disentangle - 0.1) < 1e-6
    # After warmup
    s_full = sch.state(step=35001)
    assert abs(s_full.lambda_disentangle - 0.5) < 1e-6
    assert s_full.lambda_cycle == 0.2
    assert s_full.lambda_omega == 0.1


def test_state_is_deterministic_per_step():
    sch = WCVCTSchedule(phase1_end=10, phase2_end=20)
    s1 = sch.state(step=15)
    s2 = sch.state(step=15)
    assert s1 == s2
