"""
Fast-FoundationStereo 深度估计模块

将 FFS 的立体匹配集成到 GPS_plus 管线中，替换 RAFT-Stereo 的视差估计，
同时保留 GPS_plus 原有的高斯参数预测（GSRegresser）流程。

核心难点：GPS_plus 的立体校正产生的左右主点差异很大（cx_R - cx_L ≈ 250-265 像素），
导致实际的像素视差 d = f*B/Z + (cx_L - cx_R) 在正常深度范围内为负数。
而 FFS 只搜索 [0, max_disp] 的正视差，因此需要先补偿主点偏移。

解决方案：
- 左视图：水平平移右图以对齐主点 → FFS 只看到纯几何视差 d_geom = |Tf_x|/Z
- 右视图：水平翻转两张图 + 同样的主点对齐 + FFS → 翻转结果

FFS 输出几何视差 → inv_depth = d_geom / |Tf_x| → 送入 GSRegresser
"""

import sys
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_odd(value):
    value = int(value)
    if value < 1:
        return 1
    return value if value % 2 == 1 else value + 1


def _depth_gradient(depth):
    grad = torch.zeros_like(depth)
    grad[..., :, :-1] = grad[..., :, :-1] + (depth[..., :, 1:] - depth[..., :, :-1]).abs()
    grad[..., :, 1:] = grad[..., :, 1:] + (depth[..., :, 1:] - depth[..., :, :-1]).abs()
    grad[..., :-1, :] = grad[..., :-1, :] + (depth[..., 1:, :] - depth[..., :-1, :]).abs()
    grad[..., 1:, :] = grad[..., 1:, :] + (depth[..., 1:, :] - depth[..., :-1, :]).abs()
    return grad


def _batch_quantile_threshold(gradient, valid, quantile, min_gradient):
    flat_grad = gradient.flatten(1)
    flat_valid = valid.flatten(1)
    thresholds = []
    for grad_i, valid_i in zip(flat_grad, flat_valid):
        valid_grad = grad_i[valid_i]
        if valid_grad.numel() == 0:
            threshold = grad_i.new_tensor(float(min_gradient))
        else:
            threshold = torch.quantile(valid_grad.float(), float(quantile)).to(dtype=grad_i.dtype)
            threshold = torch.maximum(threshold, grad_i.new_tensor(float(min_gradient)))
        thresholds.append(threshold)
    return torch.stack(thresholds).view(-1, 1, 1, 1).to(device=gradient.device, dtype=gradient.dtype)


def smooth_ffs_depth_edges(
    depth,
    mask=None,
    *,
    enable=True,
    kernel_size=7,
    band_size=5,
    strength=0.35,
    quantile=0.95,
    min_gradient=1e-4,
    iterations=1,
):
    """Smooth only the narrow inverse-depth band around strong FFS edges."""
    if not enable:
        return depth

    kernel_size = _make_odd(kernel_size)
    band_size = _make_odd(band_size)
    strength = float(strength)
    quantile = min(max(float(quantile), 0.0), 1.0)
    iterations = max(1, int(iterations))

    valid = torch.isfinite(depth) & (depth > 1e-6)
    write_valid = valid if mask is None else valid & (mask > 0.5)

    safe_depth = torch.where(valid, depth, torch.zeros_like(depth))
    gradient = _depth_gradient(safe_depth)
    threshold = _batch_quantile_threshold(gradient, valid.flatten(1), quantile, min_gradient)
    edge = (gradient >= threshold) & write_valid

    edge_band = F.max_pool2d(
        edge.to(dtype=depth.dtype),
        kernel_size=band_size,
        stride=1,
        padding=band_size // 2,
    ) > 0

    smooth_den = F.avg_pool2d(
        valid.to(dtype=depth.dtype),
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
        count_include_pad=False,
    ).clamp_min(1e-6)

    out = depth
    for _ in range(iterations):
        safe_out = torch.where(valid, out, torch.zeros_like(out))
        smooth_num = F.avg_pool2d(
            safe_out,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            count_include_pad=False,
        )
        smooth_depth = smooth_num / smooth_den
        out = torch.where(edge_band & write_valid, out.lerp(smooth_depth, strength), out)

    return out.clamp_min(1e-6)


class _InputPadder:
    """将图像填充到 divis_by 的整数倍（自包含版本，避免跨仓库导入）"""

    def __init__(self, dims, divis_by=32, force_square=False):
        self.ht, self.wd = dims[-2:]
        if force_square:
            max_side = max(self.ht, self.wd)
            pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
            pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
            pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
            pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2,
                     pad_ht // 2, pad_ht - pad_ht // 2]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def _load_ffs_model(model_path, ffs_root):
    """
    加载 FFS 模型权重，处理与 GPS_plus 的 core/ 模块命名冲突。

    FFS 的序列化模型引用 core.foundation_stereo / foundation_stereo_ori 等模块路径，
    但 GPS_plus 也有自己的 core/ 目录。此函数通过临时交换 sys.modules 来安全加载。
    """
    saved_core = {}
    for k in list(sys.modules):
        if k == 'core' or k.startswith('core.'):
            saved_core[k] = sys.modules.pop(k)

    saved_utils = sys.modules.pop('Utils', None)

    sys.path.insert(0, ffs_root)

    try:
        import core.foundation_stereo  # noqa: F401  triggers alias setup
        model = torch.load(model_path, map_location='cpu', weights_only=False)
    finally:
        for k in list(sys.modules):
            if k == 'core' or k.startswith('core.'):
                del sys.modules[k]
        if 'Utils' in sys.modules:
            del sys.modules['Utils']
        sys.modules.update(saved_core)
        if saved_utils is not None:
            sys.modules['Utils'] = saved_utils
        if ffs_root in sys.path:
            sys.path.remove(ffs_root)

    return model


class FFSDepthEstimator(nn.Module):
    """
    Fast-FoundationStereo 深度估计器

    接口与 DA3 深度估计器一致：
        data, loss, metrics = estimator(data, is_train=True/False)

    调用后 data['lmain']['depth'] 和 data['rmain']['depth'] 包含逆深度 [B,1,H,W]，
    可直接送入 GPS_plus 的 depth2gsparms() 方法。
    """

    def __init__(self, cfg):
        super().__init__()
        ffs_cfg = cfg.ffs

        ffs_root = ffs_cfg.ffs_root
        model_path = ffs_cfg.model_path

        logging.info(f"[FFS] 加载模型: {model_path}")
        logging.info(f"[FFS] FFS 仓库路径: {ffs_root}")

        self.ffs_model = _load_ffs_model(model_path, ffs_root)

        self.ffs_model.args.valid_iters = getattr(ffs_cfg, 'valid_iters', 8)
        self.ffs_model.args.max_disp = getattr(ffs_cfg, 'max_disp', 320)

        self.valid_iters = self.ffs_model.args.valid_iters
        self.max_disp = self.ffs_model.args.max_disp
        self.use_hiera = getattr(ffs_cfg, 'use_hiera', False)
        self.finetune = getattr(ffs_cfg, 'finetune', False)
        self.edge_smooth_enable = getattr(ffs_cfg, 'edge_smooth_enable', False)
        self.edge_smooth_kernel = getattr(ffs_cfg, 'edge_smooth_kernel', 7)
        self.edge_smooth_band = getattr(ffs_cfg, 'edge_smooth_band', 5)
        self.edge_smooth_strength = getattr(ffs_cfg, 'edge_smooth_strength', 0.35)
        self.edge_smooth_quantile = getattr(ffs_cfg, 'edge_smooth_quantile', 0.95)
        self.edge_smooth_min_gradient = getattr(ffs_cfg, 'edge_smooth_min_gradient', 1e-4)
        self.edge_smooth_iterations = getattr(ffs_cfg, 'edge_smooth_iterations', 1)

        if not self.finetune:
            self.ffs_model.eval()
            for param in self.ffs_model.parameters():
                param.requires_grad = False
            logging.info("[FFS] 模型已冻结（不参与训练）")

        self.ffs_model.cuda()
        logging.info(
            f"[FFS] valid_iters={self.valid_iters}, max_disp={self.max_disp}, "
            f"hiera={self.use_hiera}, finetune={self.finetune}, "
            f"edge_smooth={self.edge_smooth_enable}"
        )

    def freeze_bn(self):
        if not self.finetune:
            self.ffs_model.eval()

    @staticmethod
    def _gps_to_ffs_image(img):
        """GPS_plus [-1, 1] → FFS [0, 255]"""
        return (img + 1.0) * 127.5

    @staticmethod
    def _shift_image_left(img, shift_px):
        """
        将图像内容向左平移 shift_px 个像素，右侧填零。

        Args:
            img: [B, C, H, W]
            shift_px: int, 平移像素数 (> 0 向左移)
        """
        if shift_px <= 0:
            return img
        W = img.shape[-1]
        if shift_px >= W:
            return torch.zeros_like(img)
        out = torch.zeros_like(img)
        out[..., :W - shift_px] = img[..., shift_px:]
        return out

    @staticmethod
    def _shift_image_right(img, shift_px):
        """
        将图像内容向右平移 shift_px 个像素，左侧填零。

        Args:
            img: [B, C, H, W]
            shift_px: int, 平移像素数 (> 0 向右移)
        """
        if shift_px <= 0:
            return img
        W = img.shape[-1]
        if shift_px >= W:
            return torch.zeros_like(img)
        out = torch.zeros_like(img)
        out[..., shift_px:] = img[..., :W - shift_px]
        return out

    def _run_ffs(self, left_img, right_img):
        """
        运行 FFS 获取视差。

        Args:
            left_img, right_img: [B, 3, H, W], range [0, 255]

        Returns:
            disp: [B, 1, H, W], 像素视差 (≥ 0)
        """
        padder = _InputPadder(left_img.shape, divis_by=32, force_square=False)
        left_padded, right_padded = padder.pad(left_img, right_img)

        with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
            if self.use_hiera:
                disp = self.ffs_model.run_hierachical(
                    left_padded, right_padded,
                    iters=self.valid_iters, test_mode=True, small_ratio=0.5,
                )
            else:
                disp = self.ffs_model.forward(
                    left_padded, right_padded,
                    iters=self.valid_iters, test_mode=True,
                    optimize_build_volume='pytorch1',
                )

        disp = padder.unpad(disp.float())
        disp = disp.clamp(min=1e-6)
        return disp

    def forward(self, data, is_train=True):
        """
        从立体图像对估计深度。

        GPS_plus 立体校正后 cx_L ≠ cx_R，实际像素视差 = f*B/Z + (cx_L - cx_R)，
        当 cx_L << cx_R 时为负数，FFS 无法处理。

        解决方法：平移参考图以消除主点偏移，使 FFS 只看到纯几何视差 d_geom = |Tf_x|/Z。

        左视图：右图左移 cx_shift → FFS(left, shifted_right) → d_geom
        右视图：翻转两图 + 左移翻转后的左图 → FFS(flipped_right, shifted_flipped_left)
                → 翻转结果 → d_geom

        最终：inv_depth = d_geom / |Tf_x|

        Returns:
            data: 更新后的数据字典
            loss: None
            metrics: {}
        """
        left_img = self._gps_to_ffs_image(data['lmain']['img'])
        right_img = self._gps_to_ffs_image(data['rmain']['img'])

        cx_L = data['lmain']['intr'][:, 0, 2]
        cx_R = data['lmain']['ref_intr'][:, 0, 2]
        cx_shift = int(torch.round(cx_R - cx_L).item())

        ctx = torch.no_grad() if not self.finetune else torch.enable_grad()
        with ctx:
            right_shifted = self._shift_image_left(right_img, cx_shift)
            disp_left = self._run_ffs(left_img, right_shifted)

            right_flipped = torch.flip(right_img, dims=[-1])
            left_flipped = torch.flip(left_img, dims=[-1])
            left_flipped_shifted = self._shift_image_left(left_flipped, cx_shift)
            disp_right_flipped = self._run_ffs(right_flipped, left_flipped_shifted)
            disp_right = torch.flip(disp_right_flipped, dims=[-1])

        Tf_x_abs = data['lmain']['Tf_x'].abs()
        while Tf_x_abs.dim() < 4:
            Tf_x_abs = Tf_x_abs.unsqueeze(-1)

        raw_left_depth = disp_left / Tf_x_abs
        raw_right_depth = disp_right / Tf_x_abs

        data['lmain']['depth_raw'] = raw_left_depth.detach().clone()
        data['rmain']['depth_raw'] = raw_right_depth.detach().clone()

        data['lmain']['depth'] = smooth_ffs_depth_edges(
            raw_left_depth,
            mask=data['lmain'].get('mask'),
            enable=self.edge_smooth_enable,
            kernel_size=self.edge_smooth_kernel,
            band_size=self.edge_smooth_band,
            strength=self.edge_smooth_strength,
            quantile=self.edge_smooth_quantile,
            min_gradient=self.edge_smooth_min_gradient,
            iterations=self.edge_smooth_iterations,
        )
        data['rmain']['depth'] = smooth_ffs_depth_edges(
            raw_right_depth,
            mask=data['rmain'].get('mask'),
            enable=self.edge_smooth_enable,
            kernel_size=self.edge_smooth_kernel,
            band_size=self.edge_smooth_band,
            strength=self.edge_smooth_strength,
            quantile=self.edge_smooth_quantile,
            min_gradient=self.edge_smooth_min_gradient,
            iterations=self.edge_smooth_iterations,
        )

        return data, None, {}
