"""W-CVCT-GS 的 CVCT 损失函数集合。

包含两个核心损失:
  - l_cycle:        双视图可见区域中的跨视图光度一致性损失
  - l_omega_align:  将可见性 ω 软对齐到由置信度与 c-self / c-other 一致性
                    导出的目标值
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _visibility_mask(omega: torch.Tensor) -> torch.Tensor:
    """4·ω·(1−ω) — 在 ω=0.5 处取得峰值, 在端点 (ω=0 或 ω=1) 处为零。"""
    return 4.0 * omega * (1.0 - omega)


def l_cycle(
    c_self: torch.Tensor,
    c_other_warped: torch.Tensor,
    omega: torch.Tensor,
) -> torch.Tensor:
    """限定在双视图可见区域内的跨视图光度一致性损失。

    L_cycle = mean( visibility_mask · |c_self − c_other_warped| )

    Args:
        c_self, c_other_warped: (B, 3, H, W), 数值范围 [0, 1]。
        omega: (B, 1, H, W)。

    Returns:
        标量张量。
    """
    mask = _visibility_mask(omega)
    diff = (c_self - c_other_warped).abs().mean(dim=1, keepdim=True)
    return (mask * diff).mean()


def build_omega_target(
    c_self: torch.Tensor,
    c_other_warped: torch.Tensor,
    confidence: torch.Tensor,
    agreement_sigma: float = 0.1,
) -> torch.Tensor:
    """根据跨视图匹配度与置信度构造 ω 的软目标。

    启发式策略:
      - 高匹配度 (低 |c_self - c_other_warped|) 且高置信度 → 目标值 = 0.5
      - 低匹配度 或 低置信度 → 目标值偏向 1 (更信任 c_self)

    Args:
        c_self, c_other_warped: (B, 3, H, W), 数值范围 [0, 1]。
        confidence:             (B, 1, H, W), 数值范围 [0, 1]。
        agreement_sigma:        控制匹配度指数项的陡峭程度。

    Returns:
        omega_target: (B, 1, H, W), 数值范围 [0, 1], 已 detach。
    """
    diff = (c_self - c_other_warped).abs().mean(dim=1, keepdim=True)
    agreement = torch.exp(-(diff / agreement_sigma).pow(2))
    dual_visible = (agreement * confidence).clamp(0.0, 1.0)
    target = 0.5 + (1.0 - dual_visible) * 0.5
    return target.detach()


def l_omega_align(
    omega: torch.Tensor,
    omega_target: torch.Tensor,
    entropy_weight: float = 0.0,
) -> torch.Tensor:
    """对 ω 与软目标进行 L2 对齐, 并可选地附加防塌缩熵项。

    Args:
        omega:        (B, 1, H, W)。
        omega_target: (B, 1, H, W) (通常已 detach)。
        entropy_weight: 可选的 −H(ω) 项系数, 用于防止 ω 塌缩到 0/1 端点。

    Returns:
        标量张量。
    """
    align = F.mse_loss(omega, omega_target)
    if entropy_weight > 0:
        eps = 1e-6
        ent = -(omega.clamp(eps, 1 - eps) * omega.clamp(eps, 1 - eps).log()
                + (1 - omega).clamp(eps, 1 - eps) * (1 - omega).clamp(eps, 1 - eps).log())
        align = align - entropy_weight * ent.mean()
    return align
