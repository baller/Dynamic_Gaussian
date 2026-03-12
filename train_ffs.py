"""
GPS+ (FFS) 训练脚本

基于 GPS+ train.py 重构，适配 Fast-FoundationStereo 深度估计：
  - FFS 模型全程冻结，不参与梯度更新 → checkpoint 中不保存 FFS 权重
  - 只训练 img_encoder + loftr_coarse + gs_parm_regresser
  - 损失 = 0.8×L1 + 0.2×(1-SSIM) + 0.5×Chamfer（可选）
  - 完整的 resume / auto_resume 支持
  - eval 时保存深度图、点云投影图、输入输出对比图

用法:
    python train_ffs.py --config config/ffs_stage.yaml
    python train_ffs.py --config config/ffs_stage.yaml --auto_resume
    python train_ffs.py --config config/ffs_stage.yaml --restore_ckpt experiments/gps_ffs_0312/ckpt/latest.pth
"""

from __future__ import print_function, division

import argparse
import io
import logging
import os
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.stereo_human_config import ConfigStereoHuman
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.train_recoder import Logger
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr

warnings.filterwarnings("ignore", category=UserWarning)

FFS_PREFIX = "depth_model.ffs_model."


# ──────────────────────────────────────────────────────────
#  文件备份
# ──────────────────────────────────────────────────────────

def ffs_file_backup(exp_path: str, cfg, train_script: str) -> None:
    os.makedirs(exp_path, exist_ok=True)
    shutil.copy(train_script, exp_path)
    for subdir in ["config", "gaussian_renderer"]:
        dst = os.path.join(exp_path, subdir)
        if os.path.isdir(subdir):
            shutil.copytree(subdir, dst, dirs_exist_ok=True)
    for subdir in ["lib"]:
        dst_dir = os.path.join(exp_path, subdir)
        Path(dst_dir).mkdir(exist_ok=True, parents=True)
        for fname in os.listdir(subdir):
            if fname.endswith(".py"):
                shutil.copy(os.path.join(subdir, fname), dst_dir)
    import json
    with open(os.path.join(exp_path, "cfg.json"), "w") as f:
        json.dump(dict(cfg), f, indent=2, default=str)


# ──────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────

class FFSTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bs = cfg.batch_size
        self.depth_mode = getattr(cfg, 'depth_mode', 'ffs')

        logging.info("=== GPS+ (FFS) Trainer 初始化 ===")
        logging.info(f"深度估计模式: {self.depth_mode}")

        # ── 模型 ──
        self.model = RtStereoHumanModel(cfg, with_gs_render=True)
        self.model.cuda()
        logging.info("模型构建完成")

        # 参数量统计
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        frozen_params = total_params - trainable_params
        logging.info(f"  总参数:   {total_params:,}")
        logging.info(f"  可训练:   {trainable_params:,}")
        logging.info(f"  冻结(FFS): {frozen_params:,}")

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

        # ── 优化器（只优化可训练参数，跳过冻结的 FFS）──
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

        # ── 恢复训练 ──
        if cfg.restore_ckpt:
            self.load_ckpt(cfg.restore_ckpt)
        elif cfg.stage1_ckpt:
            logging.info("从 stage1 checkpoint 加载部分权重")
            self.load_ckpt(cfg.stage1_ckpt, load_optimizer=False, strict=False)

        self.model.train()
        self._freeze_bn()

    def _freeze_bn(self):
        if hasattr(self.model, 'depth_model') and self.model.depth_model is not None:
            self.model.depth_model.freeze_bn()

    # ──────────────────────────────────────────────
    #  训练主循环
    # ──────────────────────────────────────────────

    def train(self):
        log = dict(l1=0.0, ssim=0.0, chamfer=0.0, scale=0.0)
        LOG_PERIOD = 100

        for itr in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data('train')

            data, _, metrics = self.model(data, is_train=True)
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

            render_novel = data['novel_view']['img_pred']
            gt_novel = data['novel_view']['img'].cuda()

            Ll1 = l1_loss(render_novel, gt_novel)
            Lssim = 1.0 - ssim(render_novel, gt_novel)
            loss = 0.8 * Ll1 + 0.2 * Lssim

            log['l1'] += 0.8 * Ll1.item()
            log['ssim'] += 0.2 * Lssim.item()

            if metrics is None:
                metrics = {}
            metrics.update({'l1': Ll1.item(), 'ssim': Lssim.item()})
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

    # ──────────────────────────────────────────────
    #  验证
    # ──────────────────────────────────────────────

    def run_eval(self):
        logging.info(f"[step {self.total_steps}] 开始验证 ...")
        torch.cuda.empty_cache()
        psnr_list, ssim_list = [], []
        show_idx = np.random.choice(list(range(self.len_val)), 1)

        for idx in range(self.len_val):
            data = self.fetch_data('val')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].cuda()
                psnr_val = psnr(render_novel, gt_novel).mean().double()
                ssim_val = ssim(render_novel, gt_novel)
                psnr_list.append(psnr_val.item())
                ssim_list.append(ssim_val.item())

                if idx == show_idx:
                    self._save_eval_visuals(data)

        val_psnr = float(np.mean(psnr_list))
        val_ssim = float(np.mean(ssim_list))
        logging.info(
            f"[step {self.total_steps}] Val PSNR={val_psnr:.4f}  SSIM={val_ssim:.4f}")
        self.logger.write_dict(
            {'val_psnr': val_psnr, 'val_ssim': val_ssim},
            write_step=self.total_steps)

        if val_psnr < 10.0:
            logging.warning("PSNR < 10，训练可能崩溃，请检查配置后重新训练。")

        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    #  可视化辅助
    # ──────────────────────────────────────────────

    @staticmethod
    def _t2np(t):
        """(B,3,H,W) [0,1] → (H,W,3) uint8 RGB"""
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
    def _diff_to_color(img_a, img_b, label="Diff"):
        import matplotlib.cm as cm
        diff = (img_a - img_b)[0].float().abs().mean(dim=0).cpu().numpy()
        d_max = float(diff.max())
        d_norm = np.clip(diff / (d_max + 1e-6), 0, 1)
        rgba = cm.hot(d_norm)
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        text = f"{label} max={d_max:.3f}"
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, text, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _opacity_to_color(opa_map, label="Opacity"):
        import matplotlib.cm as cm
        o = opa_map[0, 0].float().cpu().numpy()
        rgba = cm.viridis(np.clip(o, 0, 1))
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, label, (6, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 1, cv2.LINE_AA)
        return img

    @staticmethod
    def _pts_scatter(xyz, H, W, title="", max_pts=80000):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as mcm

        pts = xyz[0].float().detach().cpu().numpy()
        finite = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite]
        if len(pts) < 10:
            return np.ones((H, W, 3), dtype=np.uint8) * 255

        n_total = len(pts)
        z_vals = pts[:, 2]
        z_lo, z_hi = np.percentile(z_vals, [2, 98])
        z_norm = np.clip((z_vals - z_lo) / (z_hi - z_lo + 1e-8), 0, 1)
        colors = mcm.turbo(z_norm)[:, :3]

        x_lo, x_hi = np.percentile(pts[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(pts[:, 1], [1, 99])
        in_range = ((pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi) &
                    (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi))
        pts_r, colors_r = pts[in_range], colors[in_range]

        if len(pts_r) > max_pts:
            idx = np.random.choice(len(pts_r), max_pts, replace=False)
            pts_r, colors_r = pts_r[idx], colors_r[idx]

        pt_size = max(1.5, min(8.0, H * W * 0.35 / max(len(pts_r), 1)))
        dpi = 150
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.scatter(pts_r[:, 0], pts_r[:, 1], c=colors_r, s=pt_size,
                   alpha=0.7, linewidths=0, rasterized=True)
        ax.set_title(f"{title} ({n_total:,}pts)" if title else f"({n_total:,}pts)",
                     fontsize=8)
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0.1)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return bgr[:, :, ::-1]

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

    def _save_eval_visuals(self, data):
        step = self.total_steps
        out_dir = os.path.join(self.cfg.record.show_path, str(step))
        os.makedirs(out_dir, exist_ok=True)

        lm, rm, nv = data['lmain'], data['rmain'], data['novel_view']
        render_novel = nv['img_pred']
        gt_novel = nv['img'].cuda() if not nv['img'].is_cuda else nv['img']
        H, W = render_novel.shape[-2], render_novel.shape[-1]

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

        img_l = self._t2np(lm['img'] * 0.5 + 0.5)
        img_r = self._t2np(rm['img'] * 0.5 + 0.5)
        render_np = self._t2np(render_novel)
        gt_np = self._t2np(gt_novel)
        diff_np = self._diff_to_color(render_novel, gt_novel)
        diff_np = _resize(diff_np, H, W)

        row_top = np.concatenate([
            self._label(img_l, "Input L"),
            self._label(img_r, "Input R"),
        ], axis=1)
        row_bot = np.concatenate([
            self._label(render_np, "Render"),
            self._label(gt_np, "GT"),
            self._label(diff_np, "Diff"),
        ], axis=1)
        tw = max(row_top.shape[1], row_bot.shape[1])
        overview = np.concatenate([_pad_w(row_top, tw), _pad_w(row_bot, tw)], axis=0)
        cv2.imwrite(os.path.join(out_dir, "overview.jpg"), overview[:, :, ::-1])

        panels = []
        if 'depth_init' in lm:
            panels.append(_resize(self._depth_to_color(lm['depth_init'], "Init L"), H, W))
        panels.append(_resize(self._depth_to_color(lm['depth'], "Final L"), H, W))
        if 'depth_init' in rm:
            panels.append(_resize(self._depth_to_color(rm['depth_init'], "Init R"), H, W))
        panels.append(_resize(self._depth_to_color(rm['depth'], "Final R"), H, W))
        cv2.imwrite(os.path.join(out_dir, "depth.jpg"),
                    np.concatenate(panels, axis=1)[:, :, ::-1])

        pts_l = _resize(self._pts_scatter(lm['xyz'], H, W, title="Left"), H, W)
        pts_r = _resize(self._pts_scatter(rm['xyz'], H, W, title="Right"), H, W)
        pts_m = _resize(self._pts_scatter(
            torch.cat([lm['xyz'], rm['xyz']], dim=1), H, W, title="Merged"), H, W)
        cv2.imwrite(os.path.join(out_dir, "pts.jpg"),
                    np.concatenate([pts_l, pts_r, pts_m], axis=1)[:, :, ::-1])

        if 'opacity_maps' in lm:
            opa_l = _resize(self._opacity_to_color(lm['opacity_maps'], "Opacity L"), H, W)
            opa_r = _resize(self._opacity_to_color(rm['opacity_maps'], "Opacity R"), H, W)
            cv2.imwrite(os.path.join(out_dir, "opacity.jpg"),
                        np.concatenate([opa_l, opa_r], axis=1)[:, :, ::-1])

        cv2.imwrite(os.path.join(self.cfg.record.show_path, f"{step}.jpg"),
                    np.concatenate([render_np, gt_np], axis=1)[:, :, ::-1])

    # ──────────────────────────────────────────────
    #  数据获取
    # ──────────────────────────────────────────────

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

    # ──────────────────────────────────────────────
    #  Checkpoint 管理（排除冻结的 FFS 模型权重）
    # ──────────────────────────────────────────────

    def _get_saveable_state_dict(self):
        """
        获取可保存的 state_dict：排除冻结的 FFS 模型权重。

        FFS 模型参数量大（~300MB），且每次从 model_path 加载，
        不保存到 checkpoint 中可以：
        - 大幅缩减 checkpoint 体积
        - 避免 max_disp 等配置变更导致的 checkpoint 损坏
        """
        full_sd = self.model.state_dict()
        return {k: v for k, v in full_sd.items() if not k.startswith(FFS_PREFIX)}

    def load_ckpt(self, load_path, load_optimizer=True, strict=False):
        assert os.path.exists(load_path), f"Checkpoint 不存在: {load_path}"
        logging.info(f"加载 checkpoint: {load_path}")
        ckpt = torch.load(load_path, map_location='cuda', weights_only=False)

        # 模型权重（strict=False 因为 FFS 权重不在 checkpoint 中）
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

        optimizer_loaded = False
        scheduler_loaded = False
        scaler_loaded = False

        if load_optimizer and 'optimizer' in ckpt:
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                optimizer_loaded = True
            except ValueError as e:
                logging.warning(f"  optimizer 参数组不匹配，跳过加载（{e}）")
                logging.warning("  可能是旧 checkpoint 包含 FFS 参数，将使用新初始化的 optimizer")

            if optimizer_loaded and 'scheduler' in ckpt:
                try:
                    self.scheduler.load_state_dict(ckpt['scheduler'])
                    scheduler_loaded = True
                except Exception as e:
                    logging.warning(f"  scheduler 加载失败，跳过（{e}）")

            if 'scaler' in ckpt:
                try:
                    self.scaler.load_state_dict(ckpt['scaler'])
                    scaler_loaded = True
                except Exception as e:
                    logging.warning(f"  scaler 加载失败，跳过（{e}）")

        logging.info("=" * 50)
        logging.info("  Resume 状态:")
        logging.info(f"    checkpoint:   {load_path}")
        logging.info(f"    exp_name:     {ckpt.get('exp_name', 'N/A')}")
        logging.info(f"    total_steps:  {ckpt.get('total_steps', 'N/A')} → 从 {self.total_steps} 继续")
        logging.info(f"    optimizer:    {'已加载' if optimizer_loaded else '跳过（参数组不匹配），使用新 optimizer'}")
        logging.info(f"    scheduler:    {'已加载' if scheduler_loaded else '使用新 scheduler'}")
        logging.info(f"    scaler:       {'已加载' if scaler_loaded else '使用新 scaler'}")
        logging.info(f"    lr:           {self.optimizer.param_groups[0]['lr']:.2e}")
        logging.info(f"    剩余步数:     {self.cfg.num_steps - self.total_steps}")
        logging.info("=" * 50)

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


# ──────────────────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = argparse.ArgumentParser(description='GPS+ (FFS) Training')
    parser.add_argument('--config', default='config/ffs_stage.yaml',
                        help='YAML 配置文件路径')
    parser.add_argument('--restore_ckpt', default=None,
                        help='恢复训练的 checkpoint 路径')
    parser.add_argument('--auto_resume', action='store_true',
                        help='自动从实验目录的 latest.pth 恢复')
    parser.add_argument('--exp_name', default=None,
                        help='指定实验名（覆盖自动生成的 name_MMDD）')
    parser.add_argument('--num_steps', type=int, default=None,
                        help='覆盖 YAML 中的 num_steps（如 --num_steps 200000）')
    args = parser.parse_args()

    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(args.config)
    cfg = cfg_obj.get_cfg()
    cfg.defrost()

    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
        logging.info(f"[CLI] 覆盖 num_steps = {cfg.num_steps}")

    # ── 确定实验名 & 恢复路径 ──
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
                logging.info(f"[resume] 从 checkpoint 读取到实验名: {saved_exp}")
        except Exception as e:
            logging.warning(f"[resume] 读取 checkpoint 元数据失败: {e}")

    elif args.auto_resume:
        _search_exp = resolved_exp or default_exp
        _ckpt_dir = Path(f"experiments/{_search_exp}/ckpt")

        def _find_best_ckpt(ckpt_dir):
            """在 ckpt_dir 中按优先级查找: latest.pth > *latest.pth > 最新 iter*.pth"""
            if not ckpt_dir.exists():
                return None
            latest = ckpt_dir / "latest.pth"
            if latest.exists():
                return latest
            star_latest = sorted(ckpt_dir.glob("*latest.pth"),
                                 key=lambda p: p.stat().st_mtime, reverse=True)
            if star_latest:
                return star_latest[0]
            iters = sorted(ckpt_dir.glob("iter*.pth"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            if iters:
                return iters[0]
            return None

        _best = _find_best_ckpt(_ckpt_dir)
        if _best:
            restore_path = str(_best)
            resolved_exp = _search_exp
            logging.info(f"[auto_resume] 在 {_search_exp} 找到: {_best.name}")
        else:
            _all_exp_dirs = sorted(
                Path("experiments").glob(f"*ffs*"),
                key=lambda p: p.stat().st_mtime, reverse=True)
            for _exp_dir in _all_exp_dirs:
                _best = _find_best_ckpt(_exp_dir / "ckpt")
                if _best:
                    restore_path = str(_best)
                    resolved_exp = _exp_dir.name
                    logging.info(f"[auto_resume] 在 {_exp_dir.name} 找到: {_best.name}")
                    break
            if not restore_path:
                logging.warning("[auto_resume] 未找到任何可用 checkpoint，从头开始训练")

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

    ffs_file_backup(cfg.record.file_path, cfg, train_script=__file__)

    torch.manual_seed(1314)
    np.random.seed(1314)

    logging.info(f"实验名: {cfg.exp_name}")
    logging.info(f"配置文件: {args.config}")
    if restore_path:
        logging.info(f"恢复训练: {restore_path}")

    trainer = FFSTrainer(cfg)
    trainer.train()
