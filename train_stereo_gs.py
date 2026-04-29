"""
StereoGS 训练脚本

基于 GPS+ train_ffs.py 重构，适配 StereoGS 框架:
  - FFS 模型全程冻结，不参与梯度更新
  - 可学习: 特征适配器 + 融合模块 + 置信度解码器 + 高斯解码器 + 超分模块 + 精化网络
  - 损失 = 0.8×L1 + 0.2×(1-SSIM)
  - 支持渲染后精化 (post_refine)

用法:
    python train_stereo_gs.py --config config/stereo_gs_stage.yaml
    python train_stereo_gs.py --config config/stereo_gs_stage.yaml --auto_resume
"""

from __future__ import print_function, division

import argparse
import io
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import warnings
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.stereo_human_config import ConfigStereoHuman
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.train_recoder import Logger
from lib.GaussianRender import pts2render, pts2render_cags
from lib.GaussianRender import pts2render_cags_per_level
from lib.stereo_gs.wcvct_schedule import WCVCTSchedule
from lib.stereo_gs.losses_freq import l_band, l_active, l_disentangle
from lib.stereo_gs.losses_freq import l_depth_smooth, l_depth_anchor
from lib.stereo_gs.losses_cvct import l_cycle, l_omega_align, build_omega_target
from lib.stereo_gs.loss_masks import (
    apply_loss_mask,
    build_border_ignore_mask,
    masked_l1_loss,
)
from lib.gs_utils.loss_utils import ssim
from lib.gs_utils.image_utils import psnr
from pytorch3d.loss import chamfer_distance

warnings.filterwarnings("ignore", category=UserWarning)

FFS_PREFIX = "stereo_gs_model.ffs_extractor."


def stereo_gs_file_backup(exp_path: str, cfg, train_script: str) -> None:
    os.makedirs(exp_path, exist_ok=True)
    shutil.copy(train_script, exp_path)
    for subdir in ["config", "gaussian_renderer"]:
        dst = os.path.join(exp_path, subdir)
        if os.path.isdir(subdir):
            shutil.copytree(subdir, dst, dirs_exist_ok=True)
    for subdir in ["lib", "lib/stereo_gs"]:
        dst_dir = os.path.join(exp_path, subdir)
        Path(dst_dir).mkdir(exist_ok=True, parents=True)
        for fname in os.listdir(subdir):
            if fname.endswith(".py"):
                shutil.copy(os.path.join(subdir, fname), dst_dir)
    import json
    with open(os.path.join(exp_path, "cfg.json"), "w") as f:
        json.dump(dict(cfg), f, indent=2, default=str)


class StereoGSTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        # ── W-CVCT-GS: 调度器与损失开关 ──
        wcvct_cfg = getattr(cfg, 'wcvct', None)
        self.wcvct_enabled = bool(getattr(wcvct_cfg, 'enable', False)) if wcvct_cfg else False
        if self.wcvct_enabled:
            self.wcvct_cfg = wcvct_cfg
            self.wcvct_schedule = WCVCTSchedule(
                phase1_end=wcvct_cfg.schedule.phase1_end,
                phase2_end=wcvct_cfg.schedule.phase2_end,
                lambda_band=wcvct_cfg.fdsg.lambda_band,
                lambda_active=wcvct_cfg.fdsg.lambda_active,
                lambda_disentangle=wcvct_cfg.fdsg.lambda_disentangle,
                lambda_disentangle_warmup=wcvct_cfg.fdsg.lambda_disentangle_warmup,
                lambda_disentangle_warmup_steps=wcvct_cfg.fdsg.lambda_disentangle_warmup_steps,
                lambda_cycle=wcvct_cfg.cvct.lambda_cycle,
                lambda_omega=wcvct_cfg.cvct.lambda_omega_align,
                cvct_warmup_steps=getattr(wcvct_cfg.schedule, 'cvct_warmup_steps', 2000),
            )
            if wcvct_cfg.override_cags_sparsity:
                logging.info('[W-CVCT-GS] overriding cags_sparsity_weight to 0 (replaced by L_active)')
        else:
            self.wcvct_schedule = None
        self.bs = cfg.batch_size

        logging.info("=== StereoGS Trainer 初始化 ===")

        # ── 模型 ──
        self.model = RtStereoHumanModel(cfg, with_gs_render=True)
        self.model.cuda()
        logging.info("模型构建完成")

        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        logging.info(f"  总参数:     {total_params:,}")
        logging.info(f"  可训练:     {trainable_params:,}")
        logging.info(f"  冻结(FFS):  {frozen_params:,}")

        # ── 数据集 ──
        self.train_set = StereoHumanDataset(cfg.dataset, phase='train')
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.bs,
            shuffle=True, num_workers=8, pin_memory=True)
        self.train_iterator = iter(self.train_loader)

        self.val_set = StereoHumanDataset(cfg.dataset, phase='val')
        self.val_loader = DataLoader(
            self.val_set, batch_size=1,
            shuffle=False, num_workers=8, pin_memory=True)
        self.len_val = max(1, int(len(self.val_loader) / self.val_set.val_boost))
        self.val_iterator = iter(self.val_loader)

        # ── 优化器 ──
        params_to_train = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            params_to_train, lr=cfg.lr, weight_decay=cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=cfg.lr,
            total_steps=cfg.num_steps + 200,
            pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')
        self.scaler = GradScaler(enabled=cfg.raft.mixed_precision)

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        if cfg.restore_ckpt:
            self.load_ckpt(cfg.restore_ckpt)
        elif cfg.stage1_ckpt:
            logging.info("从 stage1 checkpoint 加载部分权重")
            self.load_ckpt(cfg.stage1_ckpt, load_optimizer=False, strict=False)

        self.model.train()
        self._freeze_bn()

    def _freeze_bn(self):
        if hasattr(self.model, 'stereo_gs_model'):
            self.model.stereo_gs_model.freeze_bn()

    def train(self):
        use_post_refine = getattr(self.cfg.stereo_gs, 'use_post_refine', False)
        use_cags = getattr(self.cfg.stereo_gs, 'use_cags', False)
        sparsity_weight = getattr(self.cfg.stereo_gs, 'cags_sparsity_weight', 0.01)
        chamfer_weight = getattr(self.cfg.stereo_gs, 'chamfer_weight', 0.0)
        n_chamfer = getattr(self.cfg.stereo_gs, 'chamfer_n_samples', 10000)
        novel_loss_ignore_border = getattr(self.cfg.stereo_gs, 'novel_loss_ignore_border', 0)
        render_fn = pts2render_cags if use_cags else pts2render
        log = dict(l1=0.0, ssim=0.0)
        if use_cags:
            log['sparse'] = 0.0
        if chamfer_weight > 0:
            log['chamfer'] = 0.0
        LOG_PERIOD = 100

        for itr in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data('train')

            phase_state = (
                self.wcvct_schedule.state(self.total_steps)
                if self.wcvct_schedule is not None else None
            )
            if phase_state is not None and hasattr(self.model, 'stereo_gs_model'):
                gs_model = self.model.stereo_gs_model
                if hasattr(gs_model, 'set_cvct_identity'):
                    gs_model.set_cvct_identity(phase_state.cvct_identity_mode)
                if hasattr(gs_model, 'set_cvct_blend'):
                    gs_model.set_cvct_blend(phase_state.cvct_blend)

            data, _, metrics = self.model(data, is_train=True)
            data = render_fn(data, bg_color=self.cfg.dataset.bg_color)

            if use_post_refine:
                data = self.model.stereo_gs_model.refine_rendered(data)

            render_novel = data['novel_view']['img_pred']
            gt_novel = data['novel_view']['img'].cuda()
            novel_loss_mask = build_border_ignore_mask(
                render_novel, novel_loss_ignore_border)
            render_novel_loss = apply_loss_mask(render_novel, gt_novel, novel_loss_mask)

            Ll1 = masked_l1_loss(render_novel, gt_novel, novel_loss_mask)
            Lssim = 1.0 - ssim(render_novel_loss, gt_novel)
            loss = 0.8 * Ll1 + 0.2 * Lssim

            # ── 原有 CAGS 稀疏正则（如启用 W-CVCT-GS 的 override_cags_sparsity 则跳过）──
            sparsity_active = use_cags and not (
                self.wcvct_enabled and self.wcvct_cfg.override_cags_sparsity)
            if sparsity_active:
                extras = data.get('_stereo_gs_extras', {})
                wl = extras.get('split_weights_left')
                wr = extras.get('split_weights_right')
                if wl is not None and wr is not None:
                    L_sparse = (wl.mean() + wr.mean()) * 0.5
                    loss = loss + sparsity_weight * L_sparse
                    log['sparse'] += sparsity_weight * L_sparse.item()

            # ── 原有 Chamfer 距离损失 ──
            Lcd = torch.tensor(0.0, device='cuda')
            if chamfer_weight > 0:
                for b_i in range(self.bs):
                    l_valid = data['lmain']['pts_valid'][b_i]
                    r_valid = data['rmain']['pts_valid'][b_i]
                    l_xyz_v = data['lmain']['xyz'][b_i][l_valid].unsqueeze(0).contiguous()
                    r_xyz_v = data['rmain']['xyz'][b_i][r_valid].unsqueeze(0).contiguous()
                    n_sample = min(n_chamfer, l_xyz_v.shape[1], r_xyz_v.shape[1])
                    if n_sample < 100:
                        continue
                    idx_l = np.random.choice(l_xyz_v.shape[1], n_sample, replace=False)
                    idx_r = np.random.choice(r_xyz_v.shape[1], n_sample, replace=False)
                    cd_i, _ = chamfer_distance(l_xyz_v[:, idx_l], r_xyz_v[:, idx_r])
                    Lcd = Lcd + cd_i
                Lcd = Lcd / self.bs
                loss = loss + chamfer_weight * Lcd
                log['chamfer'] += chamfer_weight * Lcd.item()

            # ── W-CVCT-GS 新增损失 ──
            wcvct_logs = {}
            if phase_state is not None:
                loss, wcvct_logs = self._add_wcvct_losses(
                    loss, data, phase_state, novel_loss_mask=novel_loss_mask)
            log.setdefault('band', 0.0); log['band'] += wcvct_logs.get('band', 0.0)
            log.setdefault('active', 0.0); log['active'] += wcvct_logs.get('active', 0.0)
            log.setdefault('dis', 0.0); log['dis'] += wcvct_logs.get('disentangle', 0.0)
            log.setdefault('cycle', 0.0); log['cycle'] += wcvct_logs.get('cycle', 0.0)
            log.setdefault('omega', 0.0); log['omega'] += wcvct_logs.get('omega', 0.0)

            # ── 深度残差正则（全阶段生效，独立于 W-CVCT 阶段调度）──
            depth_reg_cfg = getattr(self.cfg.stereo_gs, 'depth_regularization', None)
            if depth_reg_cfg is not None and getattr(depth_reg_cfg, 'enable', False):
                log.setdefault('depth_reg', 0.0)
                for view_key in ('lmain', 'rmain'):
                    if 'depth_residual' not in data[view_key]:
                        continue
                    d_res = data[view_key]['depth_residual']
                    d_full = data[view_key]['depth']
                    d_init = data[view_key].get('depth_init')
                    img_src = data[view_key].get('img_orig', data[view_key]['img'])
                    img_01 = img_src * 0.5 + 0.5
                    if getattr(depth_reg_cfg, 'lambda_smooth', 0.0) > 0:
                        L_ds = l_depth_smooth(d_res, img_01.clamp(0, 1))
                        loss = loss + depth_reg_cfg.lambda_smooth * L_ds
                        log['depth_reg'] += depth_reg_cfg.lambda_smooth * L_ds.item()
                    if d_init is not None and getattr(depth_reg_cfg, 'lambda_anchor', 0.0) > 0:
                        L_da = l_depth_anchor(d_full, d_init)
                        loss = loss + depth_reg_cfg.lambda_anchor * L_da
                        log['depth_reg'] += depth_reg_cfg.lambda_anchor * L_da.item()

            log['l1'] += 0.8 * Ll1.item()
            log['ssim'] += 0.2 * Lssim.item()

            if metrics is None:
                metrics = {}
            metrics.update({'l1': Ll1.item(), 'ssim': Lssim.item(),
                            'chamfer': Lcd.item(),
                            'novel_loss_valid_ratio': novel_loss_mask.mean().item()})
            metrics.update({f'wcvct_{k}': v for k, v in wcvct_logs.items()})
            self.logger.push(metrics)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0)
            self.scaler.step(self.optimizer)
            if self.scheduler.last_epoch < self.cfg.num_steps + 99:
                self.scheduler.step()
            self.scaler.update()

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(
                    'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                if phase_state is not None:
                    self.logger.writer.add_scalar('wcvct/phase', phase_state.phase, self.total_steps)

            if self.total_steps and self.total_steps % LOG_PERIOD == 0:
                msg = "  ".join(f"{k}={v / LOG_PERIOD:.4f}" for k, v in log.items())
                logging.info(f"[step {self.total_steps}] {msg}")
                for k in log:
                    log[k] = 0.0

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()
                self._freeze_bn()

            if self.total_steps > 0 and self.total_steps % self.cfg.record.save_iter == 0:
                self.save_ckpt(
                    Path(f"{self.cfg.record.ckpt_path}/iter{self.total_steps}.pth"))

            self.total_steps += 1

        logging.info("训练完成！")
        self.logger.close()
        self.save_ckpt(
            Path(f"{self.cfg.record.ckpt_path}/{self.cfg.name}_final.pth"))

    def _add_wcvct_losses(self, loss, data, phase_state, novel_loss_mask=None):
        """计算并加上 W-CVCT-GS 各项损失；返回 (新 loss, 日志字典)。"""
        logs = {}
        if phase_state.lambda_band == 0.0 and phase_state.lambda_disentangle == 0.0 \
                and phase_state.lambda_active == 0.0 and phase_state.lambda_cycle == 0.0 \
                and phase_state.lambda_omega == 0.0:
            return loss, logs

        def _src_img(view_key):
            """读取该视图的原始输入图像 (CVCT 启用时为 img_orig，否则为 img)。"""
            v = data[view_key]
            return v.get('img_orig', v['img'])

        gt = data['novel_view']['img'].cuda()
        pred = data['novel_view']['img_pred']
        if novel_loss_mask is None:
            novel_loss_mask = build_border_ignore_mask(
                pred, getattr(self.cfg.stereo_gs, 'novel_loss_ignore_border', 0))
        pred_recon = apply_loss_mask(pred, gt, novel_loss_mask)

        # L_band: 多带重建损失
        if phase_state.lambda_band > 0:
            Lb = l_band(
                pred_recon, gt,
                ll_weight=self.wcvct_cfg.fdsg.ll_weight,
                band_weights=tuple(self.wcvct_cfg.fdsg.band_weights),
                log_compress_k=self.wcvct_cfg.fdsg.log_compress_k,
            )
            loss = loss + phase_state.lambda_band * Lb
            logs['band'] = phase_state.lambda_band * Lb.item()

        # L_active: GT 小波引导稀疏（替代原 cags_sparsity）
        if phase_state.lambda_active > 0:
            extras = data.get('_stereo_gs_extras', {})
            wl = extras.get('split_weights_left')
            wr = extras.get('split_weights_right')
            if wl is not None and wr is not None:
                gt_l = (_src_img('lmain') * 0.5 + 0.5).clamp(0, 1)
                gt_r = (_src_img('rmain') * 0.5 + 0.5).clamp(0, 1)
                active_mode = getattr(self.wcvct_cfg.fdsg, 'active_mode', 'symmetric')
                active_min = getattr(self.wcvct_cfg.fdsg, 'active_min_activation', 0.0)
                La = 0.5 * (
                    l_active(wl, gt_l, mode=active_mode, min_activation=active_min) +
                    l_active(wr, gt_r, mode=active_mode, min_activation=active_min))
                loss = loss + phase_state.lambda_active * La
                logs['active'] = phase_state.lambda_active * La.item()

        # L_disentangle: 子高斯频带特化损失（需要逐级渲染）
        if phase_state.lambda_disentangle > 0:
            deltas = self._compute_per_level_deltas(data)
            if deltas is not None:
                Ld = l_disentangle(
                    deltas, log_compress_k=self.wcvct_cfg.fdsg.log_compress_k)
                loss = loss + phase_state.lambda_disentangle * Ld
                logs['disentangle'] = phase_state.lambda_disentangle * Ld.item()

        # L_cycle 和 L_omega_align: 跨视图一致性 + 视见度软对齐
        if phase_state.lambda_cycle > 0 or phase_state.lambda_omega > 0:
            extras = data.get('_stereo_gs_extras', {})
            for view_key, prefix in (('cvct_left', 'l'), ('cvct_right', 'r')):
                cvct_out = extras.get(view_key)
                if cvct_out is None:
                    continue
                src_view = 'lmain' if prefix == 'l' else 'rmain'
                if phase_state.lambda_cycle > 0:
                    Lc = l_cycle(
                        c_self=(_src_img(src_view) * 0.5 + 0.5).clamp(0, 1),
                        c_other_warped=cvct_out['c_other_warped'],
                        omega=cvct_out['omega'],
                    )
                    loss = loss + phase_state.lambda_cycle * Lc
                    logs[f'cycle_{prefix}'] = phase_state.lambda_cycle * Lc.item()
                if phase_state.lambda_omega > 0:
                    conf_key = 'confidence_left' if prefix == 'l' else 'confidence_right'
                    conf = extras.get(conf_key)
                    if conf is not None:
                        H, W = cvct_out['omega'].shape[-2:]
                        if conf.shape[-1] != W:
                            conf = F.interpolate(conf, size=(H, W),
                                                 mode='bilinear', align_corners=False)
                        c_self = (_src_img(src_view) * 0.5 + 0.5).clamp(0, 1)
                        target = build_omega_target(c_self, cvct_out['c_other_warped'], conf)
                        Lo = l_omega_align(
                            cvct_out['omega'], target,
                            entropy_weight=self.wcvct_cfg.cvct.lambda_omega_entropy,
                        )
                        loss = loss + phase_state.lambda_omega * Lo
                        logs[f'omega_{prefix}'] = phase_state.lambda_omega * Lo.item()
        # 聚合左右视图各自的 cycle / omega 日志
        for agg_key in ('cycle', 'omega'):
            keys = [k for k in logs if k.startswith(agg_key + '_')]
            if keys:
                logs[agg_key] = sum(logs[k] for k in keys) / len(keys)
        return loss, logs

    def _compute_per_level_deltas(self, data):
        """通过 4 次渲染获取各子高斯层级的贡献图 ΔI_j（用于 L_disentangle）。

        设计说明: 这 4 次渲染**不**经过 PostRefinement (后精化网络)，仅使用光栅化器
        原始输出。原因: L_disentangle 作用于差值 ΔI_j = I_full - I_drop_j，
        如果 refine_rendered 是非线性 UNet，差值上的 refine 偏置不会精确抵消，
        会引入与频带无关的失真。保持 raw render 让 L_disentangle 仅约束高斯
        参数本身的频域分工，与 refine 解耦。

        Phase A naive: 每步 4 次渲染。返回 [ΔI_1, ΔI_2, ΔI_3] 或 None（若失败）。
        """
        bg = self.cfg.dataset.bg_color
        # 暂存原渲染结果，避免被 4-render 流程覆盖
        original_pred = data['novel_view']['img_pred']
        try:
            data = pts2render_cags_per_level(data, bg, level='all')
            I_full = data['novel_view']['img_pred']
            deltas = []
            for j in (1, 2, 3):
                data = pts2render_cags_per_level(data, bg, level=f'drop_{j}')
                I_drop = data['novel_view']['img_pred']
                deltas.append(I_full - I_drop)
            data['novel_view']['img_pred'] = original_pred  # 恢复
            return deltas
        except RuntimeError as e:
            # 仅捕获 RuntimeError（涵盖 torch.cuda.OutOfMemoryError 等可恢复的运行时错误）
            # KeyError / AttributeError / AssertionError 等编程错误应当让其传播
            logging.warning(f"[W-CVCT-GS] L_disentangle render failed: {e}; skipping")
            data['novel_view']['img_pred'] = original_pred
            return None

    def _log_wcvct_diagnostics(self, data):
        """记录 ω 直方图与子高斯激活率（W-CVCT-GS 诊断指标）。"""
        if not self.wcvct_enabled:
            return
        extras = data.get('_stereo_gs_extras', {})

        # ω 直方图（左右两侧）
        for view_key, prefix in (('cvct_left', 'L'), ('cvct_right', 'R')):
            cvct_out = extras.get(view_key)
            if cvct_out is not None:
                omega = cvct_out['omega'].detach().flatten()
                self.logger.writer.add_histogram(
                    f'wcvct/omega_{prefix}', omega, self.total_steps)
                self.logger.writer.add_scalar(
                    f'wcvct/omega_{prefix}_mean', omega.mean().item(), self.total_steps)

        # 各层级子高斯激活率
        for view_key, prefix in (('split_weights_left', 'L'), ('split_weights_right', 'R')):
            w = extras.get(view_key)
            if w is None:
                continue
            for j in range(w.shape[1]):
                rate = (w[:, j] > 0.1).float().mean().item()
                self.logger.writer.add_scalar(
                    f'wcvct/active_rate_{prefix}_k{j+1}', rate, self.total_steps)

        # 深度残差统计
        for view_key, prefix in (('lmain', 'L'), ('rmain', 'R')):
            d_res = data[view_key].get('depth_residual')
            if d_res is not None:
                vals = d_res.detach().flatten()
                self.logger.writer.add_histogram(
                    f'depth_residual/dist_{prefix}', vals, self.total_steps)
                self.logger.writer.add_scalar(
                    f'depth_residual/abs_mean_{prefix}',
                    vals.abs().mean().item(), self.total_steps)

    @staticmethod
    def _val_group_of(sample_name: str) -> str:
        """Classify a val sample into a dataset group for per-group visualization.

        - sample names like 's1a1_s1_0034' (head 's1aN' where N is digits) → 'official'
        - sample names like 's1_0034'      (head 's1', i.e. no numeric letter) → 'hias'
        - everything else falls back to 'other'.
        """
        head = sample_name.split('_')[0]
        if head.startswith('s1a') and head[3:].isdigit():
            return 'official'
        if head == 's1':
            return 'hias'
        return 'other'

    def run_eval(self):
        use_post_refine = getattr(self.cfg.stereo_gs, 'use_post_refine', False)
        use_cags = getattr(self.cfg.stereo_gs, 'use_cags', False)
        render_fn = pts2render_cags if use_cags else pts2render
        logging.info(f"[step {self.total_steps}] 开始验证 ...")
        torch.cuda.empty_cache()
        psnr_list, ssim_list = [], []
        per_group_psnr: dict = {}

        # Build groups → list of val-loader indices, one show idx per non-empty group
        val_names = self.val_set.sample_list[:self.len_val]
        groups: dict = {}
        for i, n in enumerate(val_names):
            groups.setdefault(self._val_group_of(n), []).append(i)
        show_idx_per_group = {g: int(np.random.choice(idxs))
                              for g, idxs in groups.items() if idxs}
        idx_to_group = {sidx: g for g, sidx in show_idx_per_group.items()}
        logging.info(f"[step {self.total_steps}] val groups = "
                     f"{ {g: len(v) for g, v in groups.items()} };  "
                     f"show idx per group = {show_idx_per_group}")

        for idx in range(self.len_val):
            data = self.fetch_data('val')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                data = render_fn(data, bg_color=self.cfg.dataset.bg_color)
                if use_post_refine:
                    data = self.model.stereo_gs_model.refine_rendered(data)

                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].cuda()
                psnr_val = psnr(render_novel, gt_novel).mean().double()
                ssim_val = ssim(render_novel, gt_novel)
                psnr_list.append(psnr_val.item())
                ssim_list.append(ssim_val.item())

                this_name = data['novel_view'].get('sample_name', val_names[idx]) \
                    if idx < len(val_names) else val_names[idx]
                if isinstance(this_name, (list, tuple)):
                    this_name = this_name[0]
                this_group = self._val_group_of(this_name)
                per_group_psnr.setdefault(this_group, []).append(psnr_val.item())

                if idx in idx_to_group:
                    self._save_eval_visuals(data, suffix=idx_to_group[idx])

        val_psnr = float(np.mean(psnr_list))
        val_ssim = float(np.mean(ssim_list))
        logging.info(
            f"[step {self.total_steps}] Val PSNR={val_psnr:.4f}  SSIM={val_ssim:.4f}")
        log_dict = {'val_psnr': val_psnr, 'val_ssim': val_ssim}
        for g, vs in per_group_psnr.items():
            mean_g = float(np.mean(vs))
            log_dict[f'val_psnr_{g}'] = mean_g
            logging.info(f"  └ {g}: PSNR={mean_g:.4f}  (n={len(vs)})")
        self.logger.write_dict(log_dict, write_step=self.total_steps)
        # ── W-CVCT-GS 诊断: 使用最后一次验证迭代的 data 记录 ω 直方图与激活率 ──
        try:
            self._log_wcvct_diagnostics(data)
        except Exception as e:
            logging.warning(f"[W-CVCT-GS] diagnostic log failed: {e}")
        torch.cuda.empty_cache()

    @staticmethod
    def _t2np(t):
        return (t[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)

    @staticmethod
    def _depth_to_color(depth, label=""):
        import matplotlib.cm as cm
        d = depth[0, 0].float().cpu().numpy()
        z = 1.0 / (d + 1e-8)
        z = np.clip(z, 0, np.percentile(z[z > 0], 98) if (z > 0).any() else 10)
        z_min, z_max = float(z.min()), float(z.max())
        d_norm = (z - z_min) / (z_max - z_min + 1e-6)
        rgba = cm.inferno(d_norm)
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        text = f"{label} {z_min:.2f}~{z_max:.2f}m".strip()
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _conf_to_color(conf, label="Confidence"):
        import matplotlib.cm as cm
        c = conf[0, 0].float().cpu().numpy()
        rgba = cm.viridis(np.clip(c, 0, 1))
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _opacity_to_color(opacity_map, label="Opacity"):
        """将不透明度图 (B,1,H,W) 可视化为热力图，范围 [0, 1]。"""
        import matplotlib.cm as cm
        o = opacity_map[0, 0].float().cpu().numpy()
        o_min, o_max = float(o.min()), float(o.max())
        rgba = cm.plasma(np.clip(o, 0, 1))
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        text = f"{label} [{o_min:.2f}, {o_max:.2f}]"
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _label(img, text, pos="bottom"):
        out = img.copy()
        H = out.shape[0]
        y = H - 10 if pos == "bottom" else 22
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    @staticmethod
    def _split_weights_to_color(weights_tensor, label="Split Weights"):
        """分裂权重 (B, K_sub, H_lr, W_lr) → 求和后可视化为热力图。"""
        import matplotlib.cm as cm
        w_sum = weights_tensor[0].sum(dim=0).float().cpu().numpy()  # (H_lr, W_lr)
        hi = max(w_sum.max(), 1e-6)
        w_norm = np.clip(w_sum / hi, 0, 1)
        rgba = cm.hot(w_norm)
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        cv2.putText(img, f"{label} max={hi:.3f}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, f"{label} max={hi:.3f}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _project_xyz_to_screen(
        xyz: torch.Tensor,
        extr: torch.Tensor,
        FovX: float,
        FovY: float,
        H: int,
        W: int,
    ):
        """将世界坐标系 3D 点投影到图像像素坐标。

        Args:
            xyz:  (N, 3) 世界空间点云
            extr: (4, 4) world-to-camera 外参矩阵 [R|t]
            FovX, FovY: 水平/垂直视场角 (radians)
            H, W: 图像高宽

        Returns:
            u (N,) int64, v (N,) int64, valid (N,) bool
        """
        import math
        fx = W / (2.0 * math.tan(FovX * 0.5))
        fy = H / (2.0 * math.tan(FovY * 0.5))
        cx, cy = W / 2.0, H / 2.0

        R = extr[:3, :3].float()   # (3, 3)
        t = extr[:3, 3].float()    # (3,)
        pts_cam = (R @ xyz.float().T).T + t    # (N, 3)

        z = pts_cam[:, 2]
        front = z > 0.1
        z_s = z.clamp(min=1e-6)
        u = (fx * pts_cam[:, 0] / z_s + cx).long()
        v = (fy * pts_cam[:, 1] / z_s + cy).long()
        valid = front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return u, v, valid

    @staticmethod
    def _gs_density_image(
        u: torch.Tensor,
        v: torch.Tensor,
        valid: torch.Tensor,
        H: int,
        W: int,
        label: str = "GS Density",
    ) -> np.ndarray:
        """把投影到屏幕的高斯中心做像素级计数，返回 log 尺度 plasma 热力图。"""
        import matplotlib.cm as cm
        density = np.zeros((H, W), dtype=np.float32)
        vu = u[valid].cpu().numpy()
        vv = v[valid].cpu().numpy()
        np.add.at(density, (vv, vu), 1.0)
        density = np.log1p(density)
        d_max = density.max()
        d_norm = density / (d_max + 1e-6)
        rgba = cm.plasma(d_norm)
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        n_total = int(valid.sum().item())
        text = f"{label}  N={n_total}"
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _pcl_proj_image(
        xyz_list: list,
        color_list: list,
        extr: torch.Tensor,
        FovX: float,
        FovY: float,
        H: int,
        W: int,
        label: str = "",
    ) -> np.ndarray:
        """将多组点云用不同颜色投影到同一张图上（黑底彩点）。

        Args:
            xyz_list:   list of (N_i, 3) tensors
            color_list: list of (R, G, B) tuples in [0, 255]
            extr:       (4, 4) world-to-camera 外参
        """
        import math
        canvas = np.zeros((H, W, 3), dtype=np.uint8)
        fx = W / (2.0 * math.tan(FovX * 0.5))
        fy = H / (2.0 * math.tan(FovY * 0.5))
        cx, cy = W / 2.0, H / 2.0
        R = extr[:3, :3].float()
        t = extr[:3, 3].float()

        for xyz, color in zip(xyz_list, color_list):
            if xyz is None or xyz.shape[0] == 0:
                continue
            pts_cam = (R @ xyz.float().T).T + t
            z = pts_cam[:, 2]
            fv = (z > 0.1).cpu().numpy()
            z_s = z.clamp(min=1e-6)
            u = (fx * pts_cam[:, 0] / z_s + cx).long().cpu().numpy()
            v = (fy * pts_cam[:, 1] / z_s + cy).long().cpu().numpy()
            mask = fv & (u >= 0) & (u < W) & (v >= 0) & (v < H)
            canvas[v[mask], u[mask]] = color

        if label:
            cv2.putText(canvas, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 1, cv2.LINE_AA)
        return canvas

    @staticmethod
    def _density_map_to_color(data_view, k_sub, label="Density"):
        """高斯密度图: base(1) + active sub-Gaussians per pixel → 伪彩色。"""
        import matplotlib.cm as cm
        mask = data_view['pts_valid'][0].float().cpu()       # (H*W,)
        if 'sub_valid' not in data_view:
            return None
        sub_valid = data_view['sub_valid'][0].float().cpu()  # (k_sub*H*W,)
        sub_opacity = data_view['sub_opacity'][0, :, 0].float().cpu()  # (k_sub*H*W,)
        N = mask.shape[0]
        sub_active = (sub_valid * (sub_opacity > 0.05).float()).view(k_sub, N).sum(dim=0)
        density = mask + sub_active                            # 1 + active subs
        H = W = int(N ** 0.5)
        d_map = density.view(H, W).numpy()
        k_max = k_sub + 1
        d_norm = np.clip(d_map / k_max, 0, 1)
        rgba = cm.turbo(d_norm)
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        n_base = int(mask.sum().item())
        n_sub = int(sub_active.sum().item())
        text = f"{label}: base={n_base} sub={n_sub} total={n_base + n_sub}"
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    def _save_pcl_density_visuals(self, data, out_dir: str, H: int, W: int):
        """保存高斯密度投影图 + 每视图 / 合并点云投影图（投影到 novel view）。

        输出文件:
          gs_density.jpg   — 所有高斯中心投影到 novel view 的像素密度热力图
          pcl_per_view.jpg — 左视图(蓝)、右视图(橙红) 分别投影的拼图
          pcl_merged.jpg   — 左右（含子高斯）合并投影
        """
        nv = data['novel_view']
        idx = 0
        extr = nv['extr'][idx].cuda()                   # (4, 4)
        FovX = float(nv['FovX'][idx])
        FovY = float(nv['FovY'][idx])
        use_cags = getattr(self.cfg.stereo_gs, 'use_cags', False)

        def _get_base_xyz(view_key):
            xyz = data[view_key]['xyz'][idx]              # (N, 3)
            valid = data[view_key]['pts_valid'][idx]      # (N,) bool
            return xyz[valid]

        xyz_l = _get_base_xyz('lmain')
        xyz_r = _get_base_xyz('rmain')
        xyz_groups = [xyz_l, xyz_r]

        # CAGS 子高斯
        if use_cags and 'sub_xyz' in data.get('lmain', {}):
            def _get_sub_xyz(view_key):
                s_xyz = data[view_key]['sub_xyz'][idx]    # (N_sub, 3)
                s_valid = data[view_key]['sub_valid'][idx]
                return s_xyz[s_valid]
            sub_l = _get_sub_xyz('lmain')
            sub_r = _get_sub_xyz('rmain')
            xyz_groups_all = [xyz_l, xyz_r, sub_l, sub_r]
        else:
            xyz_groups_all = xyz_groups

        xyz_combined = torch.cat(xyz_groups_all, dim=0)  # (N_total, 3)

        # ── 1. 高斯密度图 ──
        u_all, v_all, valid_all = self._project_xyz_to_screen(
            xyz_combined, extr, FovX, FovY, H, W)
        dens_img = self._gs_density_image(u_all, v_all, valid_all, H, W,
                                          label="GS Density")
        cv2.imwrite(os.path.join(out_dir, "gs_density.jpg"),
                    dens_img[:, :, ::-1])

        # ── 2. 每视图点云投影图 ──
        COLOR_L   = (100, 160, 255)   # 左基础: 蓝
        COLOR_R   = (255, 110,  70)   # 右基础: 橙红
        COLOR_SL  = (170, 220, 255)   # 左子高斯: 浅蓝
        COLOR_SR  = (255, 200, 130)   # 右子高斯: 浅橙

        pcl_l = self._pcl_proj_image(
            [xyz_l], [COLOR_L], extr, FovX, FovY, H, W,
            label=f"Left  base N={xyz_l.shape[0]}")
        pcl_r = self._pcl_proj_image(
            [xyz_r], [COLOR_R], extr, FovX, FovY, H, W,
            label=f"Right base N={xyz_r.shape[0]}")

        if use_cags and 'sub_xyz' in data.get('lmain', {}):
            n_sub_l = sub_l.shape[0]
            n_sub_r = sub_r.shape[0]
            pcl_l_sub = self._pcl_proj_image(
                [sub_l], [COLOR_SL], extr, FovX, FovY, H, W,
                label=f"Left  sub  N={n_sub_l}")
            pcl_r_sub = self._pcl_proj_image(
                [sub_r], [COLOR_SR], extr, FovX, FovY, H, W,
                label=f"Right sub  N={n_sub_r}")
            per_view_row = np.concatenate(
                [pcl_l, pcl_l_sub, pcl_r, pcl_r_sub], axis=1)
        else:
            per_view_row = np.concatenate([pcl_l, pcl_r], axis=1)

        cv2.imwrite(os.path.join(out_dir, "pcl_per_view.jpg"),
                    per_view_row[:, :, ::-1])

        # ── 3. 合并点云投影图 ──
        if use_cags and 'sub_xyz' in data.get('lmain', {}):
            color_groups = [COLOR_L, COLOR_R, COLOR_SL, COLOR_SR]
        else:
            color_groups = [COLOR_L, COLOR_R]

        pcl_merged = self._pcl_proj_image(
            xyz_groups_all, color_groups, extr, FovX, FovY, H, W,
            label=f"Merged N={xyz_combined.shape[0]}")
        cv2.imwrite(os.path.join(out_dir, "pcl_merged.jpg"),
                    pcl_merged[:, :, ::-1])

        # ── 4. 保存点云 PLY 文件 ──
        def _get_rgb(view_key):
            # 取原始输入颜色（CVCT 启用时 img 被覆写，img_orig 保留输入）
            v = data[view_key]
            img = v.get('img_orig', v['img'])[idx]     # (3, H, W) in [-1, 1]
            valid = v['pts_valid'][idx]                # (H*W,)
            rgb = (img * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).reshape(-1, 3)
            return rgb[valid]

        rgb_l = _get_rgb('lmain')
        rgb_r = _get_rgb('rmain')

        # self._save_ply(xyz_l, rgb_l, os.path.join(out_dir, "pcl_left.ply"))
        # self._save_ply(xyz_r, rgb_r, os.path.join(out_dir, "pcl_right.ply"))

        merged_xyz_list = [xyz_l, xyz_r]
        merged_rgb_list = [rgb_l, rgb_r]

        if use_cags and 'sub_xyz' in data.get('lmain', {}):
            def _get_sub_rgb(view_key):
                s_rgb = data[view_key]['sub_rgb'][idx]
                s_valid = data[view_key]['sub_valid'][idx]
                return (s_rgb[s_valid] * 0.5 + 0.5).clamp(0, 1)
            sub_rgb_l = _get_sub_rgb('lmain')
            sub_rgb_r = _get_sub_rgb('rmain')
            merged_xyz_list.extend([sub_l, sub_r])
            merged_rgb_list.extend([sub_rgb_l, sub_rgb_r])

        # self._save_ply(
        #     torch.cat(merged_xyz_list, dim=0),
        #     torch.cat(merged_rgb_list, dim=0),
        #     os.path.join(out_dir, "pcl_merged.ply"),
        # )

        # ── 5. 保存完整高斯属性 PLY (3DGS 标准格式) ──
        def _get_gaussian_attrs(view_key):
            valid = data[view_key]['pts_valid'][idx]
            rot = data[view_key]['rot_maps'][idx].permute(1, 2, 0).reshape(-1, 4)
            scl = data[view_key]['scale_maps'][idx].permute(1, 2, 0).reshape(-1, 3)
            opa = data[view_key]['opacity_maps'][idx].permute(1, 2, 0).reshape(-1, 1)
            return rot[valid], scl[valid], opa[valid]

        rot_l, scl_l, opa_l = _get_gaussian_attrs('lmain')
        rot_r, scl_r, opa_r = _get_gaussian_attrs('rmain')

        merged_rot = [rot_l, rot_r]
        merged_scl = [scl_l, scl_r]
        merged_opa = [opa_l, opa_r]

        if use_cags and 'sub_xyz' in data.get('lmain', {}):
            def _get_sub_attrs(view_key):
                s_valid = data[view_key]['sub_valid'][idx]
                s_rot = data[view_key]['sub_rot'][idx][s_valid]
                s_scl = data[view_key]['sub_scale'][idx][s_valid]
                s_opa = data[view_key]['sub_opacity'][idx][s_valid]
                return s_rot, s_scl, s_opa

            s_rot_l, s_scl_l, s_opa_l = _get_sub_attrs('lmain')
            s_rot_r, s_scl_r, s_opa_r = _get_sub_attrs('rmain')
            merged_rot.extend([s_rot_l, s_rot_r])
            merged_scl.extend([s_scl_l, s_scl_r])
            merged_opa.extend([s_opa_l, s_opa_r])

        # self._save_gaussian_ply(
        #     torch.cat(merged_xyz_list, dim=0),
        #     torch.cat(merged_rgb_list, dim=0),
        #     torch.cat(merged_rot, dim=0),
        #     torch.cat(merged_scl, dim=0),
        #     torch.cat(merged_opa, dim=0),
        #     os.path.join(out_dir, "gaussians.ply"),
        # )

    @staticmethod
    def _save_ply(xyz: torch.Tensor, rgb: torch.Tensor, path: str):
        """保存带颜色的点云为 PLY 文件。

        Args:
            xyz: (N, 3) float, 世界坐标
            rgb: (N, 3) float in [0, 1]
            path: 输出路径
        """
        pts = xyz.detach().cpu().numpy()
        colors = (rgb.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        n = pts.shape[0]
        with open(path, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {n}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
            f.write("end_header\n")
            for i in range(n):
                f.write(f"{pts[i,0]:.6f} {pts[i,1]:.6f} {pts[i,2]:.6f} "
                        f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n")

    @staticmethod
    def _save_gaussian_ply(
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        rot: torch.Tensor,
        scale: torch.Tensor,
        opacity: torch.Tensor,
        path: str,
    ):
        """保存完整 3DGS 高斯属性为 PLY 文件（兼容 SuperSplat / antimatter15 等查看器）。

        Args:
            xyz:     (N, 3) 世界坐标
            rgb:     (N, 3) in [0, 1]
            rot:     (N, 4) 四元数 (wxyz or xyzw 取决于光栅化器)
            scale:   (N, 3) 缩放
            opacity: (N, 1) 不透明度 (sigmoid 后的值)
        """
        import struct
        xyz_np = xyz.detach().cpu().float().numpy()
        rgb_np = rgb.detach().cpu().float().numpy()
        rot_np = rot.detach().cpu().float().numpy()
        scale_np = scale.detach().cpu().float().numpy()
        op_np = opacity.detach().cpu().float().numpy().reshape(-1, 1)
        n = xyz_np.shape[0]

        log_scale = np.log(np.clip(scale_np, 1e-8, None))
        logit_op = np.log(np.clip(op_np, 1e-7, 1 - 1e-7) / (1 - np.clip(op_np, 1e-7, 1 - 1e-7)))
        sh_dc = (rgb_np - 0.5) / 0.2820947917738781

        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "property float nx\n"
            "property float ny\n"
            "property float nz\n"
            "property float f_dc_0\n"
            "property float f_dc_1\n"
            "property float f_dc_2\n"
            "property float opacity\n"
            "property float scale_0\n"
            "property float scale_1\n"
            "property float scale_2\n"
            "property float rot_0\n"
            "property float rot_1\n"
            "property float rot_2\n"
            "property float rot_3\n"
            "end_header\n"
        )
        with open(path, 'wb') as f:
            f.write(header.encode('ascii'))
            for i in range(n):
                f.write(struct.pack('<3f', *xyz_np[i]))
                f.write(struct.pack('<3f', 0.0, 0.0, 0.0))
                f.write(struct.pack('<3f', *sh_dc[i]))
                f.write(struct.pack('<f', logit_op[i, 0]))
                f.write(struct.pack('<3f', *log_scale[i]))
                f.write(struct.pack('<4f', *rot_np[i]))

    def _save_cags_visuals(self, data, extras, out_dir, H, W):
        """保存 CAGS 专属可视化: 分裂权重 + 高斯密度。"""
        k_sub = extras['split_weights_left'].shape[1]

        def _resize(img_np, h, w):
            if img_np.shape[0] == h and img_np.shape[1] == w:
                return img_np
            return cv2.resize(img_np, (w, h), interpolation=cv2.INTER_NEAREST)

        sw_l = _resize(self._split_weights_to_color(
            extras['split_weights_left'], "SplitW L"), H, W)
        sw_r = _resize(self._split_weights_to_color(
            extras['split_weights_right'], "SplitW R"), H, W)
        cv2.imwrite(os.path.join(out_dir, "split_weights.jpg"),
                    np.concatenate([sw_l, sw_r], axis=1)[:, :, ::-1])

        dens_l = self._density_map_to_color(data['lmain'], k_sub, "Density L")
        dens_r = self._density_map_to_color(data['rmain'], k_sub, "Density R")
        if dens_l is not None and dens_r is not None:
            dens_l = _resize(dens_l, H, W)
            dens_r = _resize(dens_r, H, W)
            cv2.imwrite(os.path.join(out_dir, "density.jpg"),
                        np.concatenate([dens_l, dens_r], axis=1)[:, :, ::-1])

    def _save_eval_visuals(self, data, suffix: str = ''):
        step = self.total_steps
        if suffix:
            out_dir = os.path.join(self.cfg.record.show_path, str(step), suffix)
        else:
            out_dir = os.path.join(self.cfg.record.show_path, str(step))
        os.makedirs(out_dir, exist_ok=True)

        lm, rm, nv = data['lmain'], data['rmain'], data['novel_view']
        render_novel = nv['img_pred']
        gt_novel = nv['img'].cuda() if not nv['img'].is_cuda else nv['img']
        H, W = render_novel.shape[-2:]

        def _resize(img_np, h, w):
            if img_np.shape[0] == h and img_np.shape[1] == w:
                return img_np
            return cv2.resize(img_np, (w, h), interpolation=cv2.INTER_LINEAR)

        def _pad_w(row, target_w):
            pad = target_w - row.shape[1]
            if pad <= 0:
                return row
            left = pad // 2
            return np.pad(row, ((0, 0), (left, pad - left), (0, 0)),
                          mode='constant', constant_values=255)

        # 使用 img_orig（若 CVCT 覆写过 img）保证显示真正的输入而不是 CVCT 输出
        lm_input = lm.get('img_orig', lm['img'])
        rm_input = rm.get('img_orig', rm['img'])
        img_l = self._t2np(lm_input * 0.5 + 0.5)
        img_r = self._t2np(rm_input * 0.5 + 0.5)
        render_np = self._t2np(render_novel)
        gt_np = self._t2np(gt_novel)

        diff_np = np.abs(render_np.astype(np.float32) - gt_np.astype(np.float32))
        diff_np = np.clip(diff_np * 3, 0, 255).astype(np.uint8)

        row_top = np.concatenate([
            self._label(img_l, "Input L"), self._label(img_r, "Input R"),
        ], axis=1)
        row_bot = np.concatenate([
            self._label(render_np, "Render"), self._label(gt_np, "GT"),
            self._label(diff_np, "Diff x3"),
        ], axis=1)
        tw = max(row_top.shape[1], row_bot.shape[1])
        overview = np.concatenate([_pad_w(row_top, tw), _pad_w(row_bot, tw)], axis=0)
        cv2.imwrite(os.path.join(out_dir, "overview.jpg"), overview[:, :, ::-1])

        panels = [
            _resize(self._depth_to_color(lm['depth'], "Depth L"), H, W),
            _resize(self._depth_to_color(rm['depth'], "Depth R"), H, W),
        ]
        cv2.imwrite(os.path.join(out_dir, "depth.jpg"),
                    np.concatenate(panels, axis=1)[:, :, ::-1])

        if 'opacity_maps' in lm and 'opacity_maps' in rm:
            opa_panels = [
                _resize(self._opacity_to_color(lm['opacity_maps'], "Opacity L"), H, W),
                _resize(self._opacity_to_color(rm['opacity_maps'], "Opacity R"), H, W),
            ]
            cv2.imwrite(os.path.join(out_dir, "opacity.jpg"),
                        np.concatenate(opa_panels, axis=1)[:, :, ::-1])

        extras = data.get('_stereo_gs_extras', {})
        if 'confidence_left' in extras:
            conf_panels = [
                _resize(self._conf_to_color(extras['confidence_left'], "Conf L"), H, W),
                _resize(self._conf_to_color(extras['confidence_right'], "Conf R"), H, W),
            ]
            cv2.imwrite(os.path.join(out_dir, "confidence.jpg"),
                        np.concatenate(conf_panels, axis=1)[:, :, ::-1])

        # ── CAGS 可视化: 分裂权重图 + 高斯密度图 ──
        if 'split_weights_left' in extras:
            self._save_cags_visuals(data, extras, out_dir, H, W)

        # ── 高斯密度投影图 + 点云投影图 ──
        try:
            self._save_pcl_density_visuals(data, out_dir, H, W)
        except Exception as e:
            logging.warning(f"[pcl_density] 可视化失败: {e}")

        snap_name = f"{step}_{suffix}.jpg" if suffix else f"{step}.jpg"
        cv2.imwrite(os.path.join(self.cfg.record.show_path, snap_name),
                    np.concatenate([render_np, gt_np], axis=1)[:, :, ::-1])

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        else:
            try:
                data = next(self.val_iterator)
            except StopIteration:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def _get_saveable_state_dict(self):
        full_sd = self.model.state_dict()
        return {k: v for k, v in full_sd.items() if not k.startswith(FFS_PREFIX)}

    def load_ckpt(self, load_path, load_optimizer=True, strict=False):
        assert os.path.exists(load_path), f"Checkpoint 不存在: {load_path}"
        logging.info(f"加载 checkpoint: {load_path}")
        ckpt = torch.load(load_path, map_location='cuda', weights_only=False)

        missing, unexpected = self.model.load_state_dict(ckpt['network'], strict=False)
        ffs_missing = [k for k in missing if k.startswith(FFS_PREFIX)]
        other_missing = [k for k in missing if not k.startswith(FFS_PREFIX)]
        if ffs_missing:
            logging.info(f"  FFS 权重从模型文件加载（跳过 {len(ffs_missing)} 个键）")
        if other_missing:
            logging.warning(f"  缺少非FFS权重 ({len(other_missing)} 个): {other_missing[:5]}...")
        if unexpected:
            logging.warning(f"  多余权重 ({len(unexpected)} 个): {unexpected[:5]}...")

        self.total_steps = ckpt.get('total_steps', 0) + 1
        self.logger.total_steps = self.total_steps

        if load_optimizer and 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                if 'scheduler' in ckpt:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                if 'scaler' in ckpt:
                    self.scaler.load_state_dict(ckpt['scaler'])
            except (ValueError, RuntimeError) as e:
                logging.warning(f"  optimizer 加载失败: {e}，使用新初始化")

        logging.info(f"  从 step {self.total_steps} 继续训练")

    def save_ckpt(self, save_path, show_log=True):
        save_path = Path(save_path)
        if show_log:
            logging.info(f"保存 checkpoint → {save_path}")
        payload = {
            'total_steps': self.total_steps,
            'exp_name': self.cfg.exp_name,
            'network': self._get_saveable_state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
        }
        torch.save(payload, save_path)
        latest = save_path.parent / "latest.pth"
        torch.save(payload, latest)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = argparse.ArgumentParser(description='StereoGS Training')
    parser.add_argument('--config', default='config/stereo_gs_stage.yaml')
    parser.add_argument('--restore_ckpt', default=None)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--exp_name', default=None)
    parser.add_argument('--num_steps', type=int, default=None)
    args = parser.parse_args()

    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(args.config)
    cfg = cfg_obj.get_cfg()
    cfg.defrost()

    if args.num_steps is not None:
        cfg.num_steps = args.num_steps

    restore_path = None
    dt = datetime.today()
    default_exp = f"{cfg.name}_{str(dt.month).zfill(2)}{str(dt.day).zfill(2)}"
    resolved_exp = args.exp_name

    if args.restore_ckpt:
        restore_path = args.restore_ckpt
        try:
            _meta = torch.load(restore_path, map_location='cpu', weights_only=False)
            saved_exp = _meta.get('exp_name', None)
            if saved_exp and resolved_exp is None:
                resolved_exp = saved_exp
        except Exception:
            pass
    elif args.auto_resume:
        _search_exp = resolved_exp or default_exp
        _ckpt_dir = Path(f"experiments/{_search_exp}/ckpt")
        _latest = _ckpt_dir / "latest.pth"
        if _latest.exists():
            restore_path = str(_latest)
            resolved_exp = _search_exp
            logging.info(f"[auto_resume] 找到: {_latest}")
        else:
            for _exp_dir in sorted(Path("experiments").glob("*stereo_gs*"),
                                   key=lambda p: p.stat().st_mtime, reverse=True):
                _best = _exp_dir / "ckpt" / "latest.pth"
                if _best.exists():
                    restore_path = str(_best)
                    resolved_exp = _exp_dir.name
                    logging.info(f"[auto_resume] 找到: {_best}")
                    break

    final_exp = resolved_exp or default_exp
    cfg.exp_name = final_exp
    if restore_path:
        cfg.restore_ckpt = restore_path

    cfg.record.ckpt_path = f"experiments/{cfg.exp_name}/ckpt"
    cfg.record.show_path = f"experiments/{cfg.exp_name}/show"
    cfg.record.logs_path = f"experiments/{cfg.exp_name}/logs"
    cfg.record.file_path = f"experiments/{cfg.exp_name}/file"
    cfg.freeze()

    for p in [cfg.record.ckpt_path, cfg.record.show_path,
              cfg.record.logs_path, cfg.record.file_path]:
        Path(p).mkdir(exist_ok=True, parents=True)

    stereo_gs_file_backup(cfg.record.file_path, cfg, train_script=__file__)

    torch.manual_seed(1314)
    np.random.seed(1314)

    logging.info(f"实验名: {cfg.exp_name}")
    use_cags = getattr(cfg.stereo_gs, 'use_cags', False)
    logging.info(f"配置: fusion={cfg.stereo_gs.fusion_mode}, "
                 f"confidence={cfg.stereo_gs.confidence_mode}, "
                 f"cags={use_cags}, "
                 f"refine={cfg.stereo_gs.use_post_refine}")
    if use_cags:
        logging.info(f"  CAGS: split_mode={cfg.stereo_gs.cags_split_mode}, "
                     f"k_max={cfg.stereo_gs.cags_k_max}, "
                     f"sparsity_w={cfg.stereo_gs.cags_sparsity_weight}")

    trainer = StereoGSTrainer(cfg)
    trainer.train()
