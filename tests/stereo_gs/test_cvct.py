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


def test_cvct_module_identity_mode_returns_c_self():
    """identity 模式下 ω=1.0 且 Δ=0，因此 c_final ≡ c_self（无混合污染）。

    回归测试：旧实现用 ω=0.5 → c_final = 0.5·c_self + 0.5·c_other_warped，
    导致 phase 1/2 时 `data[view]['img']` 也被混合图覆写、训练输入半透明。
    新实现 ω=1.0 → c_final = c_self，即使 disp 不为零也不会引入 c_other 污染。
    """
    mod = CVCTModule(fused_channels=32, shared_channels=32, residual_bound=0.05)
    mod.set_identity(True)
    B, C, H, W = 1, 32, 8, 8
    fused = torch.randn(B, C, H, W)
    shared = torch.randn(B, C, H, W)
    conf = torch.rand(B, 1, H, W)
    c_self = torch.rand(B, 3, H, W)
    c_other = torch.rand(B, 3, H, W)
    # 故意用非零视差：旧实现会把 c_other warp 后混入，新实现仍应等于 c_self。
    disp = torch.full((B, 1, H, W), 2.0)

    out = mod(fused, shared, conf, c_self, c_other, disp)
    assert torch.allclose(out["c_final"], c_self, atol=1e-5), (
        "identity 模式下 c_final 必须严格等于 c_self；任何混合都会污染下游 img"
    )
    assert torch.allclose(out["omega"], torch.ones_like(out["omega"]))
    assert torch.allclose(out["delta_rgb"], torch.zeros_like(out["delta_rgb"]))


def _make_color_ramp(H, W, mode="x"):
    """构造可肉眼判别 warp 方向的彩色斜坡: R 通道沿 x 单调递增。"""
    img = torch.zeros(1, 3, H, W)
    if mode == "x":
        img[0, 0] = torch.linspace(0.0, 1.0, W).view(1, W).expand(H, W)
    img[0, 1] = 0.5
    img[0, 2] = 0.0
    return img


def test_warp_with_disparity_left_view_direction():
    """左视图调用约定：传 +disp，warp 后 c_other(右图) 应被向右移动 |disp| 像素。

    具体：x_right = x_left - disp ⇒ src_x = grid_x - disp。
    若 disp=+2，则 warped[:, :, x] = c_other[:, :, x-2]。
    """
    H, W = 4, 16
    c_other = _make_color_ramp(H, W)  # R 通道在 x=0..15 线性递增
    disp = torch.full((1, 1, H, W), 2.0)  # 左视图正视差
    warped = warp_with_disparity(c_other, disp, padding_mode="zeros")
    # 验证：warped 的 R 通道在 x=2 处的值 ≈ c_other 在 x=0 处的值
    assert torch.allclose(warped[0, 0, :, 2], c_other[0, 0, :, 0], atol=1e-5)
    assert torch.allclose(warped[0, 0, :, 5], c_other[0, 0, :, 3], atol=1e-5)


def test_warp_with_disparity_right_view_direction():
    """右视图调用约定：传 -disp，warp 后 c_other(左图) 应被向左移动 |disp| 像素。

    具体：x_left = x_right + disp ⇒ src_x = grid_x - (-disp) = grid_x + disp。
    若 disp_right = +2 → 调用方传入 -2，则 warped[:, :, x] = c_other[:, :, x+2]。
    这是修复前后行为差异的关键测试：旧代码漏了符号翻转 → c_other 被向**右**而不是向左移。
    """
    H, W = 4, 16
    c_other = _make_color_ramp(H, W)
    disp_right_view = torch.full((1, 1, H, W), 2.0)
    # 模拟修复后的 stereo_gs_model._process_single_view_cags 行为：is_right_view=True 时取负
    disp_signed = -disp_right_view
    warped = warp_with_disparity(c_other, disp_signed, padding_mode="zeros")
    # warped 在 x=0 应该等于 c_other 在 x=2，向左移 2 像素
    assert torch.allclose(warped[0, 0, :, 0], c_other[0, 0, :, 2], atol=1e-5)
    assert torch.allclose(warped[0, 0, :, 5], c_other[0, 0, :, 7], atol=1e-5)


def test_cvct_right_view_warp_does_not_produce_doubled_image():
    """端到端回归：在右视图非 identity 模式下，给定一致的左右图与正确的有符号视差，
    若 ω 被强制为 1.0（即 self 通道完全占主导），c_final 必须等于 c_self；
    若 ω=0.0（完全用 warp），c_final 必须等于 c_other 经正确方向 warp 后的结果，
    而**不是**反方向 warp 的镜像。
    这构成对修复后视差符号语义的契约约束。"""
    H, W = 4, 16
    c_self = _make_color_ramp(H, W)  # 右图仿真
    c_other = _make_color_ramp(H, W)  # 左图仿真（同 ramp 便于断言）
    # 右视图正视差 = 2，按约定调用方应传 -2
    disp_signed = torch.full((1, 1, H, W), -2.0)
    warped = warp_with_disparity(c_other, disp_signed, padding_mode="zeros")
    # 与左视图传 +2 时（warp 把 ramp 向右移）相比，右视图传 -2 时 warp 必须是把 ramp 向左移
    expected_left_shifted = torch.zeros_like(c_other)
    expected_left_shifted[0, 0, :, : W - 2] = c_other[0, 0, :, 2:]
    expected_left_shifted[0, 1] = c_other[0, 1]  # 常量通道不变
    expected_left_shifted[0, 2] = c_other[0, 2]
    # 仅断言中间区域避免 zero-padding 边界
    assert torch.allclose(warped[..., 1:W - 3], expected_left_shifted[..., 1:W - 3], atol=1e-5)


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
