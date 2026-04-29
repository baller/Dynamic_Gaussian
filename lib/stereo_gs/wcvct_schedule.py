"""W-CVCT-GS 三阶段训练调度器。

阶段 1 [0, phase1_end):           仅基础损失（RGB + Chamfer）；CVCT 处于恒等模式。
阶段 2 [phase1_end, phase2_end):  额外启用 L_band 与 L_active；CVCT 仍为恒等模式。
阶段 3 [phase2_end, ∞):           额外启用 L_disentangle（带 warmup）、L_cycle、L_omega；CVCT 激活。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseState:
    """单个训练步的阶段状态快照。

    字段说明：
        phase: 当前所处阶段编号（1、2 或 3）。
        cvct_identity_mode: CVCT 模块是否处于恒等模式（True 表示不进行小波系数变换）。
        lambda_band: 频带损失 L_band 的权重。
        lambda_active: 激活损失 L_active 的权重。
        lambda_disentangle: 解耦损失 L_disentangle 的权重（阶段 3 内带 warmup）。
        lambda_cycle: 循环一致性损失 L_cycle 的权重。
        lambda_omega: ω 正则项 L_omega 的权重。
    """

    phase: int
    cvct_identity_mode: bool
    cvct_blend: float       # 1.0 = identity, 0.0 = full CVCT (warmup 插值用)
    lambda_band: float
    lambda_active: float
    lambda_disentangle: float
    lambda_cycle: float
    lambda_omega: float


class WCVCTSchedule:
    """W-CVCT-GS 的三阶段训练调度器。

    根据当前训练步数返回 PhaseState，包含 CVCT 是否恒等以及各损失项权重，
    供训练循环按阶段控制损失组合与 CVCT 行为。
    """

    def __init__(
        self,
        phase1_end: int = 5000,
        phase2_end: int = 30000,
        lambda_band: float = 0.3,
        lambda_active: float = 0.05,
        lambda_disentangle: float = 0.5,
        lambda_disentangle_warmup: float = 0.1,
        lambda_disentangle_warmup_steps: int = 5000,
        lambda_cycle: float = 0.2,
        lambda_omega: float = 0.1,
        cvct_warmup_steps: int = 2000,
    ):
        assert phase1_end < phase2_end
        self.phase1_end = phase1_end
        self.phase2_end = phase2_end
        self.lambda_band = lambda_band
        self.lambda_active = lambda_active
        self.lambda_disentangle = lambda_disentangle
        self.lambda_disentangle_warmup = lambda_disentangle_warmup
        self.lambda_disentangle_warmup_steps = lambda_disentangle_warmup_steps
        self.lambda_cycle = lambda_cycle
        self.lambda_omega = lambda_omega
        self.cvct_warmup_steps = cvct_warmup_steps

    def state(self, step: int) -> PhaseState:
        if step < self.phase1_end:
            return PhaseState(
                phase=1, cvct_identity_mode=True, cvct_blend=1.0,
                lambda_band=0.0, lambda_active=0.0,
                lambda_disentangle=0.0, lambda_cycle=0.0, lambda_omega=0.0,
            )
        if step < self.phase2_end:
            return PhaseState(
                phase=2, cvct_identity_mode=True, cvct_blend=1.0,
                lambda_band=self.lambda_band, lambda_active=self.lambda_active,
                lambda_disentangle=0.0, lambda_cycle=0.0, lambda_omega=0.0,
            )
        # Phase 3
        steps_in_p3 = step - self.phase2_end
        if steps_in_p3 < self.lambda_disentangle_warmup_steps:
            ld = self.lambda_disentangle_warmup
        else:
            ld = self.lambda_disentangle
        cvct_blend = max(0.0, 1.0 - steps_in_p3 / max(1, self.cvct_warmup_steps))
        return PhaseState(
            phase=3, cvct_identity_mode=False, cvct_blend=cvct_blend,
            lambda_band=self.lambda_band, lambda_active=self.lambda_active,
            lambda_disentangle=ld,
            lambda_cycle=self.lambda_cycle, lambda_omega=self.lambda_omega,
        )
