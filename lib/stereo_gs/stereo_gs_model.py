"""
StereoGS 主模型 — 基于 FFS 特征复用的前馈式人体场景高斯泼溅。

七阶段管线:
  Stage 1: FFS 前向 (冻结) → 骨干特征 + 视差 + 代价体 + 上下文
  Stage 2: 特征适配 (1x1 conv) → 通道对齐
  Stage 3: 跨视图融合 (可插拔)
  Stage 4: 高斯解码 (1/4 res) + 置信度调控
  Stage 5: 高斯超分
  Stage 6: 高斯光栅化 (复用 GPS_plus 渲染管线)
  Stage 7: 渲染后精化 (可选)
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import depth2pc
from lib.stereo_gs.ffs_feature_extractor import FFSFeatureExtractor, FFSFeatures
from lib.stereo_gs.feature_adapter import FeatureAdapter
from lib.stereo_gs.cross_view_fusion import build_fusion_module
from lib.stereo_gs.confidence_extractor import ConfidenceExtractor
from lib.stereo_gs.gaussian_decoder import GaussianDecoder
from lib.stereo_gs.gaussian_upsampler import build_upsampler
from lib.stereo_gs.post_refinement import PostRefinement


class StereoGSModel(nn.Module):
    """
    StereoGS: 基于立体基础模型特征复用的前馈式人体场景高斯泼溅。

    FFS 冻结，可学习部分:
      - FeatureAdapter
      - CrossViewFusion (可切换)
      - ConfidenceExtractor (若 mode='learned')
      - GaussianDecoder
      - GaussianUpsampler (可切换)
      - PostRefinement (可选)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        stereo_gs_cfg = cfg.stereo_gs

        # ── Stage 1: FFS 特征提取 (冻结) ──
        self.ffs_extractor = FFSFeatureExtractor(cfg)
        ffs_dims = self.ffs_extractor.feat_dims
        ffs_hidden = self.ffs_extractor.hidden_dim
        ctx_net_dim = self.ffs_extractor.context_net_dim
        ctx_inp_dim = self.ffs_extractor.context_inp_dim
        logging.info(
            f"[StereoGS] FFS feat dims: {ffs_dims}, hidden: {ffs_hidden}, "
            f"context_net: {ctx_net_dim}, context_inp: {ctx_inp_dim}"
        )

        # ── Stage 2: 特征适配 ──
        adapt_dims = tuple(stereo_gs_cfg.adapt_dims)
        self.adapter = FeatureAdapter(
            ffs_feat_dims=ffs_dims,
            out_dims=adapt_dims,
            use_context=stereo_gs_cfg.use_context,
            use_gru=stereo_gs_cfg.use_gru,
            context_net_dim=ctx_net_dim,
            context_inp_dim=ctx_inp_dim,
        )

        # ── Stage 3: 跨视图融合 ──
        fusion_mode = stereo_gs_cfg.fusion_mode
        self.fusion = build_fusion_module(fusion_mode, adapt_dims[0])
        logging.info(f"[StereoGS] Fusion mode: {fusion_mode}")

        # ── Stage 4: 置信度提取 + 高斯解码 ──
        conf_mode = getattr(stereo_gs_cfg, 'confidence_mode', 'peak')
        self.confidence = ConfidenceExtractor(mode=conf_mode)
        self.decoder = GaussianDecoder(
            feat_dims=adapt_dims,
            head_dim=stereo_gs_cfg.head_dim,
            confidence_alpha=stereo_gs_cfg.confidence_alpha,
            confidence_beta=stereo_gs_cfg.confidence_beta,
            max_scale=stereo_gs_cfg.max_scale,
        )

        # ── Stage 5: 高斯超分 ──
        sr_mode = stereo_gs_cfg.sr_mode
        self.upsampler = build_upsampler(
            mode=sr_mode,
            feat_dim_lr=stereo_gs_cfg.head_dim,
            feat_dim_hr=ffs_dims[0],
            scale_factor=4,
        )
        logging.info(f"[StereoGS] SR mode: {sr_mode}")

        # ── Stage 7: 渲染后精化 (可选) ──
        self.use_post_refine = stereo_gs_cfg.use_post_refine
        if self.use_post_refine:
            self.post_refine = PostRefinement(
                feat_channels=ffs_dims[0],
                hidden_channels=stereo_gs_cfg.refine_hidden,
                num_layers=stereo_gs_cfg.refine_layers,
            )
            logging.info("[StereoGS] Post-refinement enabled")

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"[StereoGS] params total={total:,} trainable={trainable:,}")

    def freeze_bn(self):
        self.ffs_extractor.freeze_bn()

    def _process_single_view(
        self,
        ffs_feat: FFSFeatures,
        ffs_feat_other: FFSFeatures,
        view_data: dict,
        Tf_x_abs: torch.Tensor,
        is_right_view: bool = False,
    ) -> dict:
        """
        处理单个视图: 适配 → 融合 → 解码 → 超分。

        Args:
            ffs_feat:       当前视图的 FFS 特征
            ffs_feat_other: 对侧视图的 FFS 特征
            view_data:      data['lmain'] 或 data['rmain']
            Tf_x_abs:       |Tf_x| (B,1,1,1)
            is_right_view:  是否为右视图 (影响 warp 方向)

        Returns:
            dict with depth, rot_maps, scale_maps, opacity_maps
        """
        # Stage 2: 适配
        adapted = self.adapter(
            ffs_feat.backbone_feats_left,
            context_net=ffs_feat.context_net,
            context_inp=ffs_feat.context_inp,
            gru_hidden=ffs_feat.gru_hidden,
        )
        adapted_other = self.adapter(
            ffs_feat_other.backbone_feats_left,
            context_net=ffs_feat_other.context_net,
            context_inp=ffs_feat_other.context_inp,
            gru_hidden=ffs_feat_other.gru_hidden,
        )

        # Stage 3: 融合 (仅 1/4 分辨率)
        confidence = self.confidence(ffs_feat.cost_prob)
        disp_for_warp = -ffs_feat.disparity if is_right_view else ffs_feat.disparity
        fused_4 = self.fusion(
            adapted[0], adapted_other[0],
            disp_for_warp, confidence,
        )
        adapted_fused = [fused_4, adapted[1], adapted[2]]

        # Stage 4: 高斯解码 (1/4 res)
        decoder_out = self.decoder(adapted_fused, ffs_feat.disparity_1_4, confidence)
        shared_feat = decoder_out.pop('_shared_feat')

        # Stage 5: 高斯超分 (1/4 → full)
        gaussian_hr = self.upsampler(
            decoder_out,
            shared_feat,
            backbone_feat_hr=ffs_feat.backbone_feats_left[0],
        )

        # 视差 → 逆深度 + 残差
        depth = ffs_feat.disparity / Tf_x_abs
        depth = depth + gaussian_hr['depth_residual']

        return {
            'depth': depth,
            'rot_maps': gaussian_hr['rot'],
            'scale_maps': gaussian_hr['scale'],
            'opacity_maps': gaussian_hr['opacity'],
            'confidence': confidence,
        }

    def forward(self, data: dict, is_train: bool = True):
        """
        StereoGS 前向传播。

        Args:
            data: GPS_plus 格式数据字典
            is_train: 是否训练模式

        Returns:
            data: 更新后的数据字典 (包含 depth, xyz, rot_maps 等)
            loss: None (损失在训练脚本中计算)
            metrics: {}
        """
        bs = data['lmain']['img'].shape[0]

        # ── Stage 1: FFS 特征提取 ──
        ffs_output = self.ffs_extractor(data, return_both_views=True)
        ffs_left = ffs_output['left']
        ffs_right = ffs_output['right']

        Tf_x_abs = data['lmain']['Tf_x'].abs()
        while Tf_x_abs.dim() < 4:
            Tf_x_abs = Tf_x_abs.unsqueeze(-1)

        # ── Stage 2-5: 逐视图处理 ──
        left_result = self._process_single_view(ffs_left, ffs_right, data['lmain'], Tf_x_abs, is_right_view=False)
        right_result = self._process_single_view(ffs_right, ffs_left, data['rmain'], Tf_x_abs, is_right_view=True)

        # ── 填充 data 字典 (与 GPS_plus 渲染管线对齐) ──
        for view_key, result in [('lmain', left_result), ('rmain', right_result)]:
            data[view_key]['depth_init'] = result['depth'].detach().clone()
            data[view_key]['depth'] = result['depth']
            data[view_key]['rot_maps'] = result['rot_maps']
            data[view_key]['scale_maps'] = result['scale_maps']
            data[view_key]['opacity_maps'] = result['opacity_maps']

            data[view_key]['xyz'] = depth2pc(
                result['depth'], data[view_key]['extr'], data[view_key]['intr'],
            ).view(bs, -1, 3)
            valid = data[view_key]['mask'][:, :1, :, :] > 0.5
            data[view_key]['pts_valid'] = valid.view(bs, -1)

        data['novel_view']['scale_regular'] = torch.mean(
            torch.stack([left_result['scale_maps'].mean(), right_result['scale_maps'].mean()])
        )
        data['_stereo_gs_extras'] = {
            'ffs_feat_left_1_4': ffs_left.backbone_feats_left[0],
            'ffs_feat_right_1_4': ffs_right.backbone_feats_left[0],
            'confidence_left': left_result['confidence'],
            'confidence_right': right_result['confidence'],
        }

        return data, None, {}

    def refine_rendered(self, data: dict) -> dict:
        """渲染后精化（可选），在 pts2render 之后调用。"""
        if not self.use_post_refine:
            return data
        rendered = data['novel_view']['img_pred']
        extras = data.get('_stereo_gs_extras', {})
        ffs_feat = extras.get('ffs_feat_left_1_4', None)
        data['novel_view']['img_pred'] = self.post_refine(rendered, ffs_feat)
        return data
