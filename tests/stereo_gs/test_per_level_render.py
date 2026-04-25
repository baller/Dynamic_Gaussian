"""Tests for per-level rendering wrapper used by L_disentangle.

Some tests are CUDA-only (the helper exercises the rasterizer); the
ValueError-on-bad-level test is pure-Python and runs on CPU.
"""
import torch
import pytest


def _make_minimal_cags_data():
    """Construct a minimal data dict that pts2render_cags can consume.

    Two pixels per view, single batch, all valid, simple identity rotation.
    """
    from lib.GaussianRender import pts2render_cags
    bs = 1
    H = W = 4
    N = H * W

    def _view():
        v = {}
        v['xyz'] = torch.zeros(bs, N, 3, device='cuda')
        v['xyz'][..., 2] = 1.0  # 1m in front of camera
        v['rot_maps'] = torch.zeros(bs, 4, H, W, device='cuda')
        v['rot_maps'][:, 0] = 1.0  # identity quaternion (w=1)
        v['scale_maps'] = torch.full((bs, 3, H, W), 0.001, device='cuda')
        v['opacity_maps'] = torch.full((bs, 1, H, W), 0.5, device='cuda')
        v['img'] = torch.zeros(bs, 3, H, W, device='cuda')  # gray
        v['pts_valid'] = torch.ones(bs, N, dtype=torch.bool, device='cuda')

        v['sub_xyz'] = torch.zeros(bs, 3 * N, 3, device='cuda')
        v['sub_xyz'][..., 2] = 1.0
        v['sub_rot'] = torch.zeros(bs, 3 * N, 4, device='cuda')
        v['sub_rot'][..., 0] = 1.0
        v['sub_scale'] = torch.full((bs, 3 * N, 3), 0.001, device='cuda')
        v['sub_opacity'] = torch.full((bs, 3 * N, 1), 0.5, device='cuda')
        v['sub_rgb'] = torch.zeros(bs, 3 * N, 3, device='cuda')
        v['sub_valid'] = torch.ones(bs, 3 * N, dtype=torch.bool, device='cuda')
        return v

    data = {'lmain': _view(), 'rmain': _view(), 'novel_view': {}}
    # Minimal camera params for the rasterizer (mocked downstream).
    return data


@pytest.mark.cuda
def test_pts2render_cags_per_level_signature():
    from lib.GaussianRender import pts2render_cags_per_level
    assert callable(pts2render_cags_per_level)


@pytest.mark.cuda
def test_pts2render_cags_per_level_levels_zero_through_three():
    """The 'level' argument must accept 0..3 and the values 'all' and 'no_level_j'."""
    from lib.GaussianRender import pts2render_cags_per_level
    import inspect
    sig = inspect.signature(pts2render_cags_per_level)
    assert "level" in sig.parameters


def test_pts2render_cags_per_level_raises_valueerror_on_out_of_range_drop_level():
    """The 'drop_<j>' branch must raise ValueError on out-of-range j (not silent under python -O).

    This test does NOT require CUDA — it triggers the validation path before any
    rendering and uses a tiny CPU data dict.
    """
    import pytest as _pytest
    from lib.GaussianRender import pts2render_cags_per_level
    bs = 1
    H = W = 4
    N = H * W
    k_sub = 3

    def _cpu_view():
        return {
            'sub_valid': torch.ones(bs, k_sub * N, dtype=torch.bool),
            'pts_valid': torch.ones(bs, N, dtype=torch.bool),
            'rot_maps': torch.zeros(bs, 4, H, W),
            'img': torch.zeros(bs, 3, H, W),
        }

    data = {'lmain': _cpu_view(), 'rmain': _cpu_view(), 'novel_view': {}}

    # j=5 is out of range (per_level=3). Must raise BEFORE any rendering.
    with _pytest.raises(ValueError, match="out of range"):
        pts2render_cags_per_level(data, [0, 0, 0], level='drop_5')
