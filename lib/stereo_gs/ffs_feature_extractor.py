"""
FFS 特征提取器 — 封装 Fast-FoundationStereo，返回中间特征而非仅视差。

输出字典包含:
  backbone_feats_left:  list[Tensor], 多尺度左视图特征 (1/4 ~ 1/32)
  backbone_feats_right: list[Tensor], 多尺度右视图特征
  disparity:            (B,1,H,W), 全分辨率视差
  disparity_1_4:        (B,1,H/4,W/4), 1/4 分辨率精化后视差
  cost_prob:            (B,D/4,H/4,W/4), 代价体 softmax 概率分布
  context_net:          (B,C,H/4,W/4), 上下文网络输出 (net)
  context_inp:          (B,C,H/4,W/4), 上下文网络输入特征 (inp)
  gru_hidden:           (B,C,H/4,W/4), GRU 最终隐藏状态
  mask_feat_4:          (B,C,H/4,W/4), 上采样掩码特征
  stem_2x:              (B,32,H/2,W/2), 2x 下采样的早期特征
"""

from __future__ import annotations

import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class FFSFeatures:
    """FFS 前向传播提取的全部中间特征。"""
    backbone_feats_left: List[torch.Tensor] = field(default_factory=list)
    backbone_feats_right: List[torch.Tensor] = field(default_factory=list)
    disparity: Optional[torch.Tensor] = None
    disparity_1_4: Optional[torch.Tensor] = None
    cost_prob: Optional[torch.Tensor] = None
    context_net: Optional[torch.Tensor] = None
    context_inp: Optional[torch.Tensor] = None
    gru_hidden: Optional[torch.Tensor] = None
    mask_feat_4: Optional[torch.Tensor] = None
    stem_2x: Optional[torch.Tensor] = None


def _load_ffs_model_and_funcs(model_path: str, ffs_root: str):
    """
    加载 FFS 模型并缓存所需的独立函数引用。

    FFS 的 core.submodule / core.geometry 在加载后会被 GPS_plus 的 core/ 覆盖，
    因此必须在命名空间切换期间把需要的函数对象保存下来。
    """
    saved_core = {}
    for k in list(sys.modules):
        if k == 'core' or k.startswith('core.'):
            saved_core[k] = sys.modules.pop(k)

    saved_utils = sys.modules.pop('Utils', None)
    sys.path.insert(0, ffs_root)

    try:
        import core.foundation_stereo  # noqa: F401
        model = torch.load(model_path, map_location='cpu', weights_only=False)

        from core.submodule import (
            build_gwc_volume_optimized_pytorch1,
            build_concat_volume_optimized_pytorch1,
            disparity_regression,
        )
        from core.geometry import Combined_Geo_Encoding_Volume

        ffs_funcs = {
            'build_gwc': build_gwc_volume_optimized_pytorch1,
            'build_concat': build_concat_volume_optimized_pytorch1,
            'disp_regression': disparity_regression,
            'geo_encoding_volume': Combined_Geo_Encoding_Volume,
        }
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

    return model, ffs_funcs


class _InputPadder:
    """将图像填充到 divis_by 的整数倍。"""

    def __init__(self, dims, divis_by=32):
        self.ht, self.wd = dims[-2:]
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


class FFSFeatureExtractor(nn.Module):
    """
    封装 Fast-FoundationStereo 模型，执行前向传播并捕获中间特征。

    FFS 模型完全冻结，不参与梯度计算。
    输入要求: GPS_plus 格式 [-1, 1] 的图像，内部转为 [0, 255]。
    """

    def __init__(self, cfg):
        super().__init__()
        ffs_cfg = cfg.ffs
        ffs_root = ffs_cfg.ffs_root
        model_path = ffs_cfg.model_path

        logging.info(f"[FFSFeatureExtractor] 加载模型: {model_path}")
        self.ffs, self._ffs_funcs = _load_ffs_model_and_funcs(model_path, ffs_root)
        self.ffs.args.valid_iters = getattr(ffs_cfg, 'valid_iters', 8)
        self.ffs.args.max_disp = getattr(ffs_cfg, 'max_disp', 320)
        self.valid_iters = self.ffs.args.valid_iters
        self.max_disp = self.ffs.args.max_disp

        self.ffs.eval()
        for p in self.ffs.parameters():
            p.requires_grad = False
        self.ffs.cuda()

        self._feat_dims = self.ffs.feature.d_out
        self._volume_dim = getattr(self.ffs, 'volume_dim', 28)

        self._context_net_dim, self._context_inp_dim = self._probe_context_dims()
        self._hidden_dim = self._context_net_dim

        logging.info(
            f"[FFSFeatureExtractor] feat_dims={self._feat_dims}, "
            f"context_net_dim={self._context_net_dim}, context_inp_dim={self._context_inp_dim}, "
            f"volume_dim={self._volume_dim}, "
            f"iters={self.valid_iters}, max_disp={self.max_disp}"
        )

    def _probe_context_dims(self):
        """通过小张量前向推理来检测剪枝后 cnet 的实际输出维度。"""
        ffs = self.ffs
        d0 = self._feat_dims[0]
        d1 = self._feat_dims[1]
        d2 = self._feat_dims[2]
        device = next(ffs.parameters()).device
        with torch.no_grad():
            dummy0 = torch.zeros(1, d0, 4, 4, device=device)
            dummy1 = torch.zeros(1, d1, 2, 2, device=device)
            dummy2 = torch.zeros(1, d2, 1, 1, device=device)
            cnet_out = list(ffs.cnet(dummy0, dummy1, dummy2))
            net_ch = cnet_out[0][0].shape[1]
            inp_ch = cnet_out[0][1].shape[1]
        return net_ch, inp_ch

    @property
    def feat_dims(self) -> list:
        return self._feat_dims

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def context_net_dim(self) -> int:
        return self._context_net_dim

    @property
    def context_inp_dim(self) -> int:
        return self._context_inp_dim

    def freeze_bn(self):
        self.ffs.eval()

    @staticmethod
    def _gps_to_ffs_image(img: torch.Tensor) -> torch.Tensor:
        """GPS_plus [-1, 1] → FFS [0, 255]"""
        return (img + 1.0) * 127.5

    @staticmethod
    def _shift_image_left(img: torch.Tensor, shift_px: int) -> torch.Tensor:
        if shift_px <= 0:
            return img
        W = img.shape[-1]
        if shift_px >= W:
            return torch.zeros_like(img)
        out = torch.zeros_like(img)
        out[..., :W - shift_px] = img[..., shift_px:]
        return out

    def _forward_ffs_with_features(
        self, image1: torch.Tensor, image2: torch.Tensor
    ) -> FFSFeatures:
        """
        镜像 FFS 的 forward 方法，但保留所有中间特征。

        Args:
            image1, image2: [B,3,H,W] in [0,255]
        """
        ffs = self.ffs
        B = image1.shape[0]
        result = FFSFeatures()

        # -- normalize --
        mean = image1.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = image1.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        img1_norm = (image1 / 255.0 - mean) / std
        img2_norm = (image2 / 255.0 - mean) / std

        with torch.amp.autocast('cuda', enabled=ffs.args.mixed_precision, dtype=torch.float16):
            out = ffs.feature(torch.cat([img1_norm, img2_norm], dim=0))
            features_left = [o[:B] for o in out]
            features_right = [o[B:] for o in out]
            stem_2x = ffs.stem_2(img1_norm)

            result.backbone_feats_left = [f.detach().float() for f in features_left]
            result.backbone_feats_right = [f.detach().float() for f in features_right]
            result.stem_2x = stem_2x.detach().float()

            # -- cost volume --
            gwc_volume = self._build_gwc_volume(features_left[0], features_right[0])
            left_tmp = ffs.proj_cmb(features_left[0])
            right_tmp = ffs.proj_cmb(features_right[0])
            concat_volume = self._build_concat_volume(left_tmp, right_tmp)
            comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
            del concat_volume, gwc_volume, left_tmp, right_tmp

            comb_volume = ffs.corr_stem(comb_volume)
            comb_volume = ffs.corr_feature_att(comb_volume, features_left[0])
            comb_volume = ffs.cost_agg(comb_volume, features_left)

            # -- initial disparity from cost volume --
            logits = ffs.classifier(comb_volume).squeeze(1)
            prob = F.softmax(logits, dim=1)
            result.cost_prob = prob.detach()

        init_disp = self._ffs_funcs['disp_regression'](prob.float(), ffs.args.max_disp // 4)

        # -- context net --
        with torch.amp.autocast('cuda', enabled=ffs.args.mixed_precision, dtype=torch.float16):
            cnet_list = ffs.cnet(features_left[0], features_left[1], features_left[2])
            cnet_list = list(cnet_list)
            net_list = [torch.tanh(x[0]) for x in cnet_list]
            inp_list = [torch.relu(x[1]) for x in cnet_list]
            inp_list = [ffs.cam(x) * x for x in inp_list]
            att = [ffs.sam(x) for x in inp_list]

        result.context_net = net_list[0].detach().float()
        result.context_inp = inp_list[0].detach().float()

        # -- geometry encoding volume + GRU refinement --
        dtype = ffs.dtype
        geo_fn = self._ffs_funcs['geo_encoding_volume'](
            features_left[0].to(dtype), features_right[0].to(dtype),
            comb_volume.to(dtype), num_levels=ffs.args.corr_levels,
        )
        b, c, h, w = features_left[0].shape
        coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1, 1, w, 1).repeat(b, h, 1, 1)
        disp = init_disp.to(dtype)

        del comb_volume, features_left, features_right, cnet_list

        for itr in range(self.valid_iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp, coords, dx=ffs.dx, low_memory=True)
            with torch.amp.autocast('cuda', enabled=ffs.args.mixed_precision, dtype=torch.float16):
                net_list, mask_feat_4, delta_disp = ffs.update_block(
                    net_list, inp_list, geo_feat.to(dtype), disp, att
                )
            disp = disp + delta_disp.to(dtype)

        result.gru_hidden = net_list[0].detach().float()
        result.mask_feat_4 = mask_feat_4.detach().float()
        result.disparity_1_4 = disp.detach().float()

        disp_up = ffs.upsample_disp(disp.to(dtype), mask_feat_4.to(dtype), stem_2x.to(dtype))
        result.disparity = disp_up.detach().float().clamp(min=1e-6)

        return result

    def _build_gwc_volume(self, feat_l, feat_r):
        """构建 GWC 代价体（使用 pytorch1 优化路径）。"""
        ffs = self.ffs
        return self._ffs_funcs['build_gwc'](
            feat_l, feat_r, ffs.args.max_disp // 4,
            ffs.cv_group, normalize=ffs.args.normalize,
        )

    def _build_concat_volume(self, left_tmp, right_tmp):
        ffs = self.ffs
        return self._ffs_funcs['build_concat'](
            left_tmp, right_tmp, maxdisp=ffs.args.max_disp // 4,
        )

    def forward(
        self, data: dict, *, return_both_views: bool = True
    ) -> Dict[str, FFSFeatures]:
        """
        从 GPS_plus 数据字典提取 FFS 特征。

        处理主点偏移问题（与 ffs_depth.py 逻辑一致）:
        - 左视图: 右图左移 cx_shift 后送入 FFS
        - 右视图: 翻转 + 左移 + FFS + 翻转回来

        Returns:
            dict with 'left' and optionally 'right' FFSFeatures
        """
        left_img = self._gps_to_ffs_image(data['lmain']['img'])
        right_img = self._gps_to_ffs_image(data['rmain']['img'])

        cx_L = data['lmain']['intr'][:, 0, 2]
        cx_R = data['lmain']['ref_intr'][:, 0, 2]
        cx_shift = int(torch.round(cx_R - cx_L).item())

        padder = _InputPadder(left_img.shape, divis_by=32)
        left_padded, right_padded = padder.pad(left_img, right_img)

        right_shifted = self._shift_image_left(right_padded, cx_shift)
        with torch.no_grad():
            left_feats = self._forward_ffs_with_features(left_padded, right_shifted)

        left_feats.disparity = padder.unpad(left_feats.disparity).clamp(min=1e-6)

        output = {'left': left_feats}

        if return_both_views:
            right_flipped = torch.flip(right_padded, dims=[-1])
            left_flipped = torch.flip(left_padded, dims=[-1])
            left_flipped_shifted = self._shift_image_left(left_flipped, cx_shift)
            with torch.no_grad():
                right_feats = self._forward_ffs_with_features(right_flipped, left_flipped_shifted)

            right_feats.disparity = padder.unpad(
                torch.flip(right_feats.disparity, dims=[-1])
            ).clamp(min=1e-6)
            right_feats.backbone_feats_left = [
                torch.flip(f, dims=[-1]) for f in right_feats.backbone_feats_left
            ]
            right_feats.backbone_feats_right = [
                torch.flip(f, dims=[-1]) for f in right_feats.backbone_feats_right
            ]
            right_feats.disparity_1_4 = torch.flip(right_feats.disparity_1_4, dims=[-1])
            if right_feats.cost_prob is not None:
                right_feats.cost_prob = torch.flip(right_feats.cost_prob, dims=[-1])
            if right_feats.gru_hidden is not None:
                right_feats.gru_hidden = torch.flip(right_feats.gru_hidden, dims=[-1])
            if right_feats.context_net is not None:
                right_feats.context_net = torch.flip(right_feats.context_net, dims=[-1])
            if right_feats.context_inp is not None:
                right_feats.context_inp = torch.flip(right_feats.context_inp, dims=[-1])
            output['right'] = right_feats

        return output
