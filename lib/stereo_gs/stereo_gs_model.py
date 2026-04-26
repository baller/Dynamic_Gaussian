"""
StereoGS 主模型 — 基于 FFS 特征复用的前馈式人体场景高斯泼溅。

支持两种管线 (通过 use_cags 配置切换):

Legacy (use_cags=False):
  Stage 1: FFS 前向 (冻结) → 骨干特征 + 视差 + 代价体 + 上下文
  Stage 2: 特征适配 (1x1 conv)
  Stage 3: 跨视图融合
  Stage 4: 高斯解码 (1/4 res)
  Stage 5: 高斯超分 (1/4 → full)
  Stage 6: 高斯光栅化
  Stage 7: 渲染后精化

CAGS (use_cags=True):
  Stage 1: FFS 前向 (冻结) → 骨干特征 + 视差 + 代价体
  Stage 2: 特征适配 (1x1 conv)
  Stage 3: 跨视图融合
  Stage 4: 全分辨率高斯属性预测 (FullResGaussianHead + depth residual)
  Stage 5: 自适应高斯分裂 (AdaptiveSplitter)
  Stage 6: 高斯光栅化 (pts2render_cags)
  Stage 7: 渲染后精化
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
from lib.stereo_gs.post_refinement import PostRefinement


class StereoGSModel(nn.Module):
    """
    StereoGS: 基于立体基础模型特征复用的前馈式人体场景高斯泼溅。

    FFS 冻结，可学习部分:
      - FeatureAdapter, CrossViewFusion, ConfidenceExtractor
      - [Legacy] GaussianDecoder, GaussianUpsampler
      - [CAGS]   FullResGaussianHead, AdaptiveSplitter
      - PostRefinement (可选)
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        stereo_gs_cfg = cfg.stereo_gs
        self.use_cags = getattr(stereo_gs_cfg, 'use_cags', False)

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
        warp_padding_mode = getattr(stereo_gs_cfg, 'warp_padding_mode', 'zeros')
        self.fusion = build_fusion_module(fusion_mode, adapt_dims[0], warp_padding_mode=warp_padding_mode)
        logging.info(f"[StereoGS] Fusion mode: {fusion_mode}, warp_padding: {warp_padding_mode}")

        # ── Stage 4: 置信度提取 ──
        conf_mode = getattr(stereo_gs_cfg, 'confidence_mode', 'peak')
        self.confidence = ConfidenceExtractor(mode=conf_mode)

        if self.use_cags:
            self._init_cags(stereo_gs_cfg, adapt_dims, ffs_dims)
        else:
            self._init_legacy(stereo_gs_cfg, adapt_dims, ffs_dims)

        # ── Stage 7: 渲染后精化 (可选) ──
        self.use_post_refine = stereo_gs_cfg.use_post_refine
        if self.use_post_refine:
            self.post_refine = PostRefinement(
                feat_channels=ffs_dims[0],
                hidden_channels=stereo_gs_cfg.refine_hidden,
                num_layers=stereo_gs_cfg.refine_layers,
            )
            logging.info("[StereoGS] Post-refinement enabled")

        # ── W-CVCT-GS: CVCT color head ──
        wcvct_cfg = getattr(cfg, 'wcvct', None)
        self.wcvct_enabled = bool(getattr(wcvct_cfg, 'enable', False)) if wcvct_cfg else False
        self.cvct = None
        if self.wcvct_enabled and self.use_cags:
            from lib.stereo_gs.cvct import CVCTModule
            cvct_cfg = wcvct_cfg.cvct
            self.cvct = CVCTModule(
                fused_channels=adapt_dims[0],
                shared_channels=stereo_gs_cfg.head_dim,
                visibility_hidden=cvct_cfg.visibility_hidden,
                residual_hidden=cvct_cfg.residual_hidden,
                residual_bound=cvct_cfg.residual_bound,
            )
            self.cvct.set_identity(True)  # default to identity until trainer flips
            logging.info(f"[W-CVCT-GS] CVCT enabled (residual_bound={cvct_cfg.residual_bound})")

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logging.info(f"[StereoGS] params total={total:,} trainable={trainable:,}")

    # ── 初始化: Legacy (1/4 解码 + 上采样) ──

    def _init_legacy(self, stereo_gs_cfg, adapt_dims, ffs_dims):
        from lib.stereo_gs.gaussian_decoder import GaussianDecoder
        from lib.stereo_gs.gaussian_upsampler import build_upsampler

        self.decoder = GaussianDecoder(
            feat_dims=adapt_dims,
            head_dim=stereo_gs_cfg.head_dim,
            confidence_alpha=stereo_gs_cfg.confidence_alpha,
            confidence_beta=stereo_gs_cfg.confidence_beta,
            max_scale=stereo_gs_cfg.max_scale,
            max_depth_residual=getattr(stereo_gs_cfg, 'cags_depth_residual_bound', 0.5),
        )
        sr_mode = stereo_gs_cfg.sr_mode
        self.upsampler = build_upsampler(
            mode=sr_mode,
            feat_dim_lr=stereo_gs_cfg.head_dim,
            feat_dim_hr=ffs_dims[0],
            scale_factor=4,
        )
        logging.info(f"[StereoGS] Legacy mode: SR={sr_mode}")

    # ── 初始化: CAGS (全分辨率头 + 自适应分裂) ──

    def _init_cags(self, stereo_gs_cfg, adapt_dims, ffs_dims):
        from lib.stereo_gs.fullres_gaussian_head import FullResGaussianHead
        from lib.stereo_gs.adaptive_splitter import build_adaptive_splitter

        cags_in_ch = adapt_dims[0] + 3 + 1       # adapted_feat + RGB + confidence
        cags_head_dim = stereo_gs_cfg.head_dim
        self.fullres_head = FullResGaussianHead(
            in_channels=cags_in_ch,
            hidden_dim=cags_head_dim,
            confidence_alpha=stereo_gs_cfg.confidence_alpha,
            confidence_beta=stereo_gs_cfg.confidence_beta,
            max_scale=stereo_gs_cfg.max_scale,
        )

        split_mode = getattr(stereo_gs_cfg, 'cags_split_mode', 'learned')
        k_max = getattr(stereo_gs_cfg, 'cags_k_max', 4)
        max_offset = getattr(stereo_gs_cfg, 'cags_max_pos_offset', 0.002)

        if split_mode == 'learned':
            split_feat_ch = adapt_dims[0] + 3     # adapted_feat + RGB
        else:
            split_feat_ch = 3                     # gradient 模式只用 image

        self.splitter = build_adaptive_splitter(
            feat_channels=split_feat_ch,
            shared_feat_channels=cags_head_dim,
            split_mode=split_mode,
            k_max=k_max,
            max_pos_offset=max_offset,
        )
        self._cags_split_mode = split_mode
        logging.info(f"[StereoGS] CAGS mode: split={split_mode}, k_max={k_max}")

    # ──────────────────────────────────────────────────────
    #  公共方法
    # ──────────────────────────────────────────────────────

    def freeze_bn(self):
        self.ffs_extractor.freeze_bn()

    def set_cvct_identity(self, on: bool) -> None:
        """切换 CVCT 的 identity 模式（由训练器按阶段调用）。"""
        if self.cvct is not None:
            self.cvct.set_identity(on)

    # ──────────────────────────────────────────────────────
    #  Legacy 管线
    # ──────────────────────────────────────────────────────

    def _process_single_view_legacy(
        self,
        ffs_feat: FFSFeatures,
        ffs_feat_other: FFSFeatures,
        view_data: dict,
        Tf_x_abs: torch.Tensor,
        is_right_view: bool = False,
    ) -> dict:
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

        confidence = self.confidence(ffs_feat.cost_prob)
        disp_for_warp = -ffs_feat.disparity if is_right_view else ffs_feat.disparity
        fused_4 = self.fusion(
            adapted[0], adapted_other[0],
            disp_for_warp, confidence,
        )
        adapted_fused = [fused_4, adapted[1], adapted[2]]

        decoder_out = self.decoder(adapted_fused, ffs_feat.disparity_1_4, confidence)
        shared_feat = decoder_out.pop('_shared_feat')

        gaussian_hr = self.upsampler(
            decoder_out,
            shared_feat,
            backbone_feat_hr=ffs_feat.backbone_feats_left[0],
        )

        depth = ffs_feat.disparity / Tf_x_abs
        depth = depth + gaussian_hr['depth_residual']

        return {
            'depth': depth,
            'rot_maps': gaussian_hr['rot'],
            'scale_maps': gaussian_hr['scale'],
            'opacity_maps': gaussian_hr['opacity'],
            'confidence': confidence,
        }

    # ──────────────────────────────────────────────────────
    #  CAGS 管线
    # ──────────────────────────────────────────────────────

    def _process_single_view_cags(
        self,
        ffs_feat: FFSFeatures,
        ffs_feat_other: FFSFeatures,
        view_data: dict,
        Tf_x_abs: torch.Tensor,
        is_right_view: bool = False,
    ) -> dict:
        """CAGS 管线: 全分辨率头 + 自适应分裂。"""

        # Stage 2: 适配 (1/4 res)
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

        # Stage 3: 融合 (1/4 res)
        confidence_1_4 = self.confidence(ffs_feat.cost_prob)
        disp_for_warp = -ffs_feat.disparity if is_right_view else ffs_feat.disparity
        fused_4 = self.fusion(
            adapted[0], adapted_other[0],
            disp_for_warp, confidence_1_4,
        )

        # 上采样到全分辨率
        B, _, H_full, W_full = view_data['img'].shape
        fused_fullres = F.interpolate(
            fused_4, size=(H_full, W_full), mode='bilinear', align_corners=True)
        conf_fullres = F.interpolate(
            confidence_1_4, size=(H_full, W_full), mode='bilinear', align_corners=True)

        # 归一化 RGB: [-1,1] → [0,1] 用于拼接
        img_01 = view_data['img'] * 0.5 + 0.5

        # Stage 4: 全分辨率高斯属性预测
        head_input = torch.cat([fused_fullres, img_01, conf_fullres], dim=1)
        head_out = self.fullres_head(head_input, conf_fullres)
        shared_feat = head_out.pop('_shared_feat')

        # FFS 输出 inverse depth；残差也在 inverse-depth 空间中学习。
        depth = ffs_feat.disparity / Tf_x_abs
        depth = (depth + head_out['depth_residual']).clamp(min=1e-6)

        # Stage 5: 自适应高斯分裂
        bs = B
        xyz = depth2pc(
            depth, view_data['extr'], view_data['intr'],
        ).view(bs, -1, 3)
        valid = view_data['mask'][:, :1, :, :] > 0.5
        pts_valid = valid.view(bs, -1)

        if self._cags_split_mode == 'learned':
            split_feat = torch.cat([fused_fullres, img_01], dim=1)
        else:
            split_feat = view_data['img']

        split_out = self.splitter(
            parent_xyz=xyz,
            parent_rot=head_out['rot'],
            parent_scale=head_out['scale'],
            parent_opacity=head_out['opacity'],
            parent_rgb=view_data['img'],
            parent_valid=pts_valid,
            shared_feat=shared_feat,
            image_or_feat=split_feat,
        )

        # ── W-CVCT-GS: CVCT 颜色三角化 ──
        if self.cvct is not None:
            c_self = (view_data['img'] * 0.5 + 0.5).clamp(0, 1)
            other_view_data = view_data['_cvct_other_view']  # 由 _forward_cags 注入
            c_other = (other_view_data['img'] * 0.5 + 0.5).clamp(0, 1)
            # 视差符号约定与 cross_view_fusion 完全一致：
            #   - 左视图 (self=L, other=R)，warp R→L 用 +disp（x_R = x_L - disp）
            #   - 右视图 (self=R, other=L)，warp L→R 需要 -disp（x_L = x_R + disp）
            # 旧代码 disp_self = ffs_feat.disparity 没翻转符号，导致右视图 c_other_warped
            # 朝错误方向位移，c_final = ω·c_self + (1-ω)·c_other_warped 出现重影。
            disp_self = -ffs_feat.disparity if is_right_view else ffs_feat.disparity
            cvct_out = self.cvct(
                fused_feat=fused_fullres, shared_feat=shared_feat,
                confidence=conf_fullres,
                c_self=c_self, c_other=c_other,
                disparity=disp_self,
            )
            cvct_color_self = cvct_out['c_final']
        else:
            cvct_color_self = None
            cvct_out = None

        return {
            'depth': depth,
            'rot_maps': head_out['rot'],
            'scale_maps': head_out['scale'],
            'opacity_maps': head_out['opacity'],
            'depth_residual': head_out['depth_residual'],
            'confidence': confidence_1_4,
            'xyz': xyz,
            'pts_valid': pts_valid,
            'sub_xyz': split_out['sub_xyz'],
            'sub_rot': split_out['sub_rot'],
            'sub_scale': split_out['sub_scale'],
            'sub_opacity': split_out['sub_opacity'],
            'sub_rgb': split_out['sub_rgb'],
            'sub_valid': split_out['sub_valid'],
            'split_weights': split_out['split_weights'],
            'cvct_color': cvct_color_self,        # (B, 3, H, W) or None
            'cvct_out': cvct_out,                 # full dict for losses, or None
        }

    # ──────────────────────────────────────────────────────
    #  Forward
    # ──────────────────────────────────────────────────────

    def forward(self, data: dict, is_train: bool = True):
        bs = data['lmain']['img'].shape[0]

        # ── Stage 1: FFS 特征提取 ──
        ffs_output = self.ffs_extractor(data, return_both_views=True)
        ffs_left = ffs_output['left']
        ffs_right = ffs_output['right']

        Tf_x_abs = data['lmain']['Tf_x'].abs()
        while Tf_x_abs.dim() < 4:
            Tf_x_abs = Tf_x_abs.unsqueeze(-1)

        if self.use_cags:
            return self._forward_cags(data, ffs_left, ffs_right, Tf_x_abs, bs)
        else:
            return self._forward_legacy(data, ffs_left, ffs_right, Tf_x_abs, bs)

    def _forward_legacy(self, data, ffs_left, ffs_right, Tf_x_abs, bs):
        left_result = self._process_single_view_legacy(
            ffs_left, ffs_right, data['lmain'], Tf_x_abs, is_right_view=False)
        right_result = self._process_single_view_legacy(
            ffs_right, ffs_left, data['rmain'], Tf_x_abs, is_right_view=True)

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

    def _forward_cags(self, data, ffs_left, ffs_right, Tf_x_abs, bs):
        # 注入对侧视图引用 (供 CVCT 取 c_other 使用)
        data['lmain']['_cvct_other_view'] = data['rmain']
        data['rmain']['_cvct_other_view'] = data['lmain']

        left_result = self._process_single_view_cags(
            ffs_left, ffs_right, data['lmain'], Tf_x_abs, is_right_view=False)
        right_result = self._process_single_view_cags(
            ffs_right, ffs_left, data['rmain'], Tf_x_abs, is_right_view=True)

        # 清理临时键
        del data['lmain']['_cvct_other_view']
        del data['rmain']['_cvct_other_view']

        for view_key, result in [('lmain', left_result), ('rmain', right_result)]:
            data[view_key]['depth_init'] = result['depth'].detach().clone()
            data[view_key]['depth'] = result['depth']
            data[view_key]['rot_maps'] = result['rot_maps']
            data[view_key]['scale_maps'] = result['scale_maps']
            data[view_key]['opacity_maps'] = result['opacity_maps']
            data[view_key]['depth_residual'] = result['depth_residual']
            data[view_key]['xyz'] = result['xyz']
            data[view_key]['pts_valid'] = result['pts_valid']

            data[view_key]['sub_xyz'] = result['sub_xyz']
            data[view_key]['sub_rot'] = result['sub_rot']
            data[view_key]['sub_scale'] = result['sub_scale']
            data[view_key]['sub_opacity'] = result['sub_opacity']
            data[view_key]['sub_rgb'] = result['sub_rgb']
            data[view_key]['sub_valid'] = result['sub_valid']

            if result.get('cvct_color') is not None:
                # 在用 CVCT 颜色覆写 img 之前保留原图，供 L_active/L_cycle/L_omega 用作 GT
                data[view_key]['img_orig'] = data[view_key]['img']
                # pts2render_cags expects img in [-1, 1] (它会用 *0.5+0.5 缩放)
                data[view_key]['img'] = (result['cvct_color'] * 2.0 - 1.0).clamp(-1, 1)
            # 子高斯 RGB 不变（继承父级颜色，由 splitter 已经填好）

        data['novel_view']['scale_regular'] = torch.mean(
            torch.stack([left_result['scale_maps'].mean(), right_result['scale_maps'].mean()])
        )
        data['_stereo_gs_extras'] = {
            'ffs_feat_left_1_4': ffs_left.backbone_feats_left[0],
            'ffs_feat_right_1_4': ffs_right.backbone_feats_left[0],
            'confidence_left': left_result['confidence'],
            'confidence_right': right_result['confidence'],
            'split_weights_left': left_result['split_weights'],
            'split_weights_right': right_result['split_weights'],
        }
        if self.cvct is not None and left_result.get('cvct_out') is not None:
            data['_stereo_gs_extras']['cvct_left']  = left_result['cvct_out']
            data['_stereo_gs_extras']['cvct_right'] = right_result['cvct_out']
        return data, None, {}

    def refine_rendered(self, data: dict) -> dict:
        """渲染后精化（可选），在 pts2render / pts2render_cags 之后调用。"""
        if not self.use_post_refine:
            return data
        rendered = data['novel_view']['img_pred']
        extras = data.get('_stereo_gs_extras', {})
        ffs_feat = extras.get('ffs_feat_left_1_4', None)
        data['novel_view']['img_pred'] = self.post_refine(rendered, ffs_feat)
        return data
