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
        log = dict(l1=0.0, ssim=0.0)
        LOG_PERIOD = 100

        for itr in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data('train')

            data, _, metrics = self.model(data, is_train=True)
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

            if use_post_refine:
                data = self.model.stereo_gs_model.refine_rendered(data)

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

    def run_eval(self):
        use_post_refine = getattr(self.cfg.stereo_gs, 'use_post_refine', False)
        logging.info(f"[step {self.total_steps}] 开始验证 ...")
        torch.cuda.empty_cache()
        psnr_list, ssim_list = [], []
        show_idx = np.random.choice(list(range(self.len_val)), 1)

        for idx in range(self.len_val):
            data = self.fetch_data('val')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)
                if use_post_refine:
                    data = self.model.stereo_gs_model.refine_rendered(data)

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

        img_l = self._t2np(lm['img'] * 0.5 + 0.5)
        img_r = self._t2np(rm['img'] * 0.5 + 0.5)
        render_np = self._t2np(render_novel)
        gt_np = self._t2np(gt_novel)

        row_top = np.concatenate([
            self._label(img_l, "Input L"), self._label(img_r, "Input R"),
        ], axis=1)
        row_bot = np.concatenate([
            self._label(render_np, "Render"), self._label(gt_np, "GT"),
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

        extras = data.get('_stereo_gs_extras', {})
        if 'confidence_left' in extras:
            conf_panels = [
                _resize(self._conf_to_color(extras['confidence_left'], "Conf L"), H, W),
                _resize(self._conf_to_color(extras['confidence_right'], "Conf R"), H, W),
            ]
            cv2.imwrite(os.path.join(out_dir, "confidence.jpg"),
                        np.concatenate(conf_panels, axis=1)[:, :, ::-1])

        cv2.imwrite(os.path.join(self.cfg.record.show_path, f"{step}.jpg"),
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
    logging.info(f"配置: fusion={cfg.stereo_gs.fusion_mode}, "
                 f"confidence={cfg.stereo_gs.confidence_mode}, "
                 f"sr={cfg.stereo_gs.sr_mode}, "
                 f"refine={cfg.stereo_gs.use_post_refine}")

    trainer = StereoGSTrainer(cfg)
    trainer.train()
