from __future__ import print_function, division

import argparse
import io
import logging

import numpy as np
import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr

import trimesh 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore", category=UserWarning)


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size
        self.depth_mode = getattr(self.cfg, 'depth_mode', 'raft')
        
        logging.info(f"深度估计模式: {self.depth_mode}")

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        self.train_set = StereoHumanDataset(self.cfg.dataset, phase='train')
        self.train_loader = DataLoader(self.train_set, batch_size=self.bs, shuffle=True, num_workers=8, pin_memory=True)
        self.train_iterator = iter(self.train_loader)
        self.val_set = StereoHumanDataset(self.cfg.dataset, phase='val')
        self.val_loader = DataLoader(self.val_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.stage1_ckpt:
            logging.info(f"Using checkpoint from stage1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        self.model.train()
        
        # 根据深度模式冻结BN
        self._freeze_bn()
        
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)

    def _freeze_bn(self):
        """根据深度模式冻结BatchNorm层"""
        if self.depth_mode == 'raft':
            if hasattr(self.model, 'raft_stereo') and self.model.raft_stereo is not None:
                self.model.raft_stereo.freeze_bn()
                logging.info("已冻结RAFT-Stereo的BatchNorm层")
        elif self.depth_mode == 'da3':
            if hasattr(self.model, 'depth_model') and self.model.depth_model is not None:
                self.model.depth_model.freeze_bn()
                logging.info("已冻结DA3的BatchNorm层")
        elif self.depth_mode == 'ffs':
            if hasattr(self.model, 'depth_model') and self.model.depth_model is not None:
                self.model.depth_model.freeze_bn()
                logging.info("已冻结FFS的BatchNorm层")

    def train(self):
        log_l1 = 0
        log_ssim = 0
        log_chamfer = 0
        log_scale = 0
        if_chamfer = False   
        if_scale = False  
        iter_from = -1 
        for itr_ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            # 深度估计（RAFT-Stereo 或 DA3）
            data, _, metrics = self.model(data, is_train=True)
            
            # 高斯渲染
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

            # 计算损失
            render_novel = data['novel_view']['img_pred']
            gt_novel = data['novel_view']['img'].cuda()

            l_xyz = data['lmain']['xyz']
            r_xyz = data['rmain']['xyz']
            chamfer_loss = 0
            if if_chamfer and itr_ > iter_from:
                for b_i in range(self.bs):
                    l_valid_i = data['lmain']['pts_valid'][b_i, :]  # [S*S]
                    l_xyz_i = l_xyz[b_i, :, :]
                    l_xyz_i = l_xyz_i[l_valid_i].view(1, -1, 3).contiguous()
                    
                    r_valid_i = data['rmain']['pts_valid'][b_i, :]  # [S*S]
                    r_xyz_i = r_xyz[b_i, :, :]
                    r_xyz_i = r_xyz_i[r_valid_i].view(1, -1, 3).contiguous()
                    
                    sample_l = np.random.choice(l_xyz_i.shape[1], 10000, replace=False)
                    sample_r = np.random.choice(r_xyz_i.shape[1], 10000, replace=False)
                    chamfer_loss_i, _ = chamfer_distance(l_xyz_i[:, sample_l], r_xyz_i[:, sample_r])
                    chamfer_loss += chamfer_loss_i
                
                chamfer_loss /= self.bs

            Ll1 = l1_loss(render_novel, gt_novel)
            Lssim = 1.0 - ssim(render_novel, gt_novel)
            loss = 0.8 * Ll1 + 0.2 * Lssim + 0.5 * chamfer_loss 

            log_l1 += 0.8 * Ll1.item()
            log_ssim += 0.2 * Lssim.item()
            log_chamfer += 0.5 * chamfer_loss.item() if if_chamfer and itr_ > iter_from else 0 
            log_scale += 0.5 * data['novel_view']['scale_regular'].item() if if_scale else 0

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
            metrics.update({
                'l1': Ll1.item(),
                'ssim': Lssim.item(),
                'chamfer': 0.5 * chamfer_loss.item() if if_chamfer and itr_ > iter_from else 0
            })
            self.logger.push(metrics)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()
                # 重新冻结BN
                self._freeze_bn()
                
            if self.total_steps % self.cfg.record.save_iter == 0:
                self.save_ckpt(save_path=Path('%s/iter%d.pth' % (cfg.record.ckpt_path, self.total_steps)))

            self.total_steps += 1
            if self.total_steps % 100 == 99:
                print(
                    'l1 ', log_l1 / 100,
                    'ssim', log_ssim / 100,
                    'chamfer', log_chamfer / 100,
                    'scale', log_scale / 100,
                )
                log_l1 = 0
                log_ssim = 0
                log_chamfer = 0
                log_scale = 0

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_eval(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        psnr_list = []
        ssim_list = []
        show_idx = np.random.choice(list(range(self.len_val)), 1)
 
        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].cuda()
                psnr_value = psnr(render_novel, gt_novel).mean().double()
                ssim_value = ssim(render_novel, gt_novel)
                psnr_list.append(psnr_value.item())
                ssim_list.append(ssim_value.item())

                if idx == show_idx:
                    self._save_eval_visuals(data)

        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        val_ssim = np.round(np.mean(np.array(ssim_list)), 4)
        if val_psnr < 10:
            print('something wrong during training, please change random seed and re-train')
            exit()

        logging.info(f"Validation ({self.total_steps}): PSNR={val_psnr}  SSIM={val_ssim}")
        self.logger.write_dict({'val_psnr': val_psnr, 'val_ssim': val_ssim},
                               write_step=self.total_steps)
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
        """
        逆深度 (B,1,H,W) → 彩色 (H,W,3) uint8。
        先转真实深度 z=1/(inv+eps)，再 inferno colormap。
        """
        import matplotlib.cm as cm
        d = depth[0, 0].float().cpu().numpy()
        z = 1.0 / (d + 1e-8)
        z = np.clip(z, 0, np.percentile(z[z > 0], 98) if (z > 0).any() else 10)
        z_min, z_max = float(z.min()), float(z.max())
        if z_max - z_min < 1e-6:
            d_norm = np.zeros_like(z)
        else:
            d_norm = (z - z_min) / (z_max - z_min)
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
        """两张 [0,1] 图像差值 → hot colormap (H,W,3) uint8"""
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
        """opacity_maps (B,1,H,W) → viridis colormap (H,W,3) uint8"""
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
        """
        世界坐标点云 (B,N,3) → XY 散点图 (H,W,3) uint8。
        turbo colormap 按 Z 深度着色，白色背景。
        """
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
        """在图像底部/顶部叠加标签"""
        out = img.copy()
        H = out.shape[0]
        y = H - 10 if pos == "bottom" else 22
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    def _save_eval_visuals(self, data):
        """
        保存完整 eval 可视化到 show/{step}/ 目录:

          overview.jpg      — Input_L | Input_R (上行)  Render | GT | Diff (下行)
          depth.jpg         — Depth_init_L | Depth_final_L | Depth_init_R | Depth_final_R
          pts.jpg           — Pts_L | Pts_R | Pts_Merged
          opacity.jpg       — Opacity_L | Opacity_R
        """
        step = self.total_steps
        out_dir = os.path.join(cfg.record.show_path, str(step))
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

        # ── 输入图像 ──
        img_l = self._t2np(lm['img'] * 0.5 + 0.5)
        img_r = self._t2np(rm['img'] * 0.5 + 0.5)
        render_np = self._t2np(render_novel)
        gt_np = self._t2np(gt_novel)
        diff_np = self._diff_to_color(render_novel, gt_novel)
        diff_np = _resize(diff_np, H, W)

        # ── overview.jpg: 上行=输入, 下行=渲染/GT/差值 ──
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
        row_top = _pad_w(row_top, tw)
        row_bot = _pad_w(row_bot, tw)
        overview = np.concatenate([row_top, row_bot], axis=0)
        cv2.imwrite(os.path.join(out_dir, "overview.jpg"), overview[:, :, ::-1])

        # ── depth.jpg ──
        panels = []
        if 'depth_init' in lm:
            panels.append(_resize(self._depth_to_color(lm['depth_init'], "Init L"), H, W))
        panels.append(_resize(self._depth_to_color(lm['depth'], "Final L"), H, W))
        if 'depth_init' in rm:
            panels.append(_resize(self._depth_to_color(rm['depth_init'], "Init R"), H, W))
        panels.append(_resize(self._depth_to_color(rm['depth'], "Final R"), H, W))
        depth_vis = np.concatenate(panels, axis=1)
        cv2.imwrite(os.path.join(out_dir, "depth.jpg"), depth_vis[:, :, ::-1])

        # ── pts.jpg: 点云散点图 ──
        pts_l = self._pts_scatter(lm['xyz'], H, W, title="Left")
        pts_r = self._pts_scatter(rm['xyz'], H, W, title="Right")
        xyz_merged = torch.cat([lm['xyz'], rm['xyz']], dim=1)
        pts_m = self._pts_scatter(xyz_merged, H, W, title="Merged")
        pts_l = _resize(pts_l, H, W)
        pts_r = _resize(pts_r, H, W)
        pts_m = _resize(pts_m, H, W)
        pts_vis = np.concatenate([pts_l, pts_r, pts_m], axis=1)
        cv2.imwrite(os.path.join(out_dir, "pts.jpg"), pts_vis[:, :, ::-1])

        # ── opacity.jpg ──
        if 'opacity_maps' in lm:
            opa_l = _resize(self._opacity_to_color(lm['opacity_maps'], "Opacity L"), H, W)
            opa_r = _resize(self._opacity_to_color(rm['opacity_maps'], "Opacity R"), H, W)
            opa_vis = np.concatenate([opa_l, opa_r], axis=1)
            cv2.imwrite(os.path.join(out_dir, "opacity.jpg"), opa_vis[:, :, ::-1])

        # 快速预览：渲染+GT 横向拼接
        cv2.imwrite(os.path.join(cfg.record.show_path, f"{step}.jpg"),
                    np.concatenate([render_np, gt_np], axis=1)[:, :, ::-1])

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                data[view][item] = data[view][item].cuda()
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda', weights_only=False)
        self.model.load_state_dict(ckpt['network'], strict=strict)
        logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load("config/stage.yaml")
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    # 在实验名称中添加深度模式标识
    depth_mode = getattr(cfg, 'depth_mode', 'raft')
    cfg.exp_name = '%s_%s_%s%s' % (cfg.name, depth_mode, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()

    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)
    
    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()
