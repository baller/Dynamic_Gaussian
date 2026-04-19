"""
PAG-Splat 简化训练脚本

保留:
  - train_pag.py / pag_stage.yaml 入口
  - DA3 冻结先验 + ScaleAlignmentMLP

回退:
  - GS 参数预测改回 GPS+ 风格 GSRegresser
  - 渲染改回 lib.GaussianRender.pts2render
  - 训练损失以 GPS+ 风格重建损失为主，辅以 scale_regular + 3D chamfer
"""

from __future__ import print_function, division

import argparse
import logging
import os
import shutil
import sys
from copy import deepcopy
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
from lib.pag_multi_loader import build_pag_dataset
from lib.train_recoder import Logger
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr
from pag_splat.model import build_pag_splat
from pag_splat.render import move_data_to_cuda

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────
#  P1: Chamfer Distance 点云几何一致性
# ──────────────────────────────────────────────────────────

def chamfer_distance_loss(
    xyz1: torch.Tensor,
    xyz2: torch.Tensor,
    valid1: torch.Tensor | None = None,
    valid2: torch.Tensor | None = None,
    n_samples: int = 5000,
    chunk: int = 512,
) -> torch.Tensor:
    """
    双向 3D 倒角距离，仅在有效点上计算。

    Args:
        xyz1, xyz2: (B, N, 3)  世界坐标点云
        valid1, valid2: (B, N) 有效点掩码，可为 None
        n_samples:  每方向随机采样数
        chunk:      分块大小（控制显存）

    Returns:
        标量倒角距离（两方向平均）
    """
    bsz = xyz1.shape[0]
    losses = []

    for batch_idx in range(bsz):
        pts1 = xyz1[batch_idx]
        pts2 = xyz2[batch_idx]

        if valid1 is not None:
            pts1 = pts1[valid1[batch_idx].bool()]
        if valid2 is not None:
            pts2 = pts2[valid2[batch_idx].bool()]

        if pts1.shape[0] < 2 or pts2.shape[0] < 2:
            continue

        n1 = min(n_samples, pts1.shape[0])
        n2 = min(n_samples, pts2.shape[0])

        idx1 = torch.randperm(pts1.shape[0], device=pts1.device)[:n1]
        idx2 = torch.randperm(pts2.shape[0], device=pts2.device)[:n2]
        p1 = pts1[idx1].unsqueeze(0)
        p2 = pts2[idx2].unsqueeze(0)

        min_sq_12 = []
        for i in range(0, n1, chunk):
            sub = p1[:, i:i + chunk]
            d2 = ((sub.unsqueeze(2) - p2.unsqueeze(1)) ** 2).sum(-1)
            min_sq_12.append(d2.min(dim=2).values)

        min_sq_21 = []
        for j in range(0, n2, chunk):
            sub = p2[:, j:j + chunk]
            d2 = ((sub.unsqueeze(2) - p1.unsqueeze(1)) ** 2).sum(-1)
            min_sq_21.append(d2.min(dim=2).values)

        cd_12 = torch.cat(min_sq_12, dim=1).mean()
        cd_21 = torch.cat(min_sq_21, dim=1).mean()
        losses.append((cd_12 + cd_21) * 0.5)

    if not losses:
        return xyz1.new_tensor(0.0)
    return torch.stack(losses).mean()


# ──────────────────────────────────────────────────────────
#  文件备份 (扩展 GPS+ file_backup，增加 pag_splat 目录)
# ──────────────────────────────────────────────────────────

def pag_file_backup(exp_path: str, cfg, train_script: str) -> None:
    """备份训练相关源码到实验目录。"""
    os.makedirs(exp_path, exist_ok=True)
    shutil.copy(train_script, exp_path)
    # 核心目录
    for subdir in ["pag_splat", "config", "gaussian_renderer"]:
        dst = os.path.join(exp_path, subdir)
        if os.path.isdir(subdir):
            shutil.copytree(subdir, dst, dirs_exist_ok=True)
    # lib 目录仅复制 .py 文件
    for subdir in ["lib"]:
        dst_dir = os.path.join(exp_path, subdir)
        Path(dst_dir).mkdir(exist_ok=True, parents=True)
        for fname in os.listdir(subdir):
            if fname.endswith(".py"):
                shutil.copy(os.path.join(subdir, fname), dst_dir)
    # 保存配置
    import json
    with open(os.path.join(exp_path, "cfg.json"), "w") as f:
        json.dump(dict(cfg), f, indent=2, default=str)


# ──────────────────────────────────────────────────────────
#  Trainer
# ──────────────────────────────────────────────────────────

class PAGSplatTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bs = cfg.batch_size
        pag_cfg = cfg.pagsplat

        logging.info("=== PAG-Splat Trainer 初始化 ===")

        # ── 模型 ──
        logging.info(f"正在加载 DA3 ({pag_cfg.da3_checkpoint}) ...")
        # restore_ckpt 用于继续训练时自动检测旧/新 checkpoint 的 t12_mode
        _ckpt_for_detect = cfg.restore_ckpt or cfg.stage1_ckpt or None
        self.model = build_pag_splat(
            da3_checkpoint=pag_cfg.da3_checkpoint,
            feat_channels=pag_cfg.feat_channels,
            feat_stride=pag_cfg.feat_stride,
            feat_layer=pag_cfg.feat_layer,
            mlp_hidden=pag_cfg.mlp_hidden,
            enc_dims=list(cfg.gsnet.encoder_dims),
            dec_dims=list(cfg.gsnet.decoder_dims),
            head_ch=cfg.gsnet.parm_head_dim,
            scale_max=pag_cfg.scale_max,
            device="cuda",
            ckpt_path=_ckpt_for_detect,
            raft_encoder_dims=list(cfg.raft.encoder_dims),
        )
        logging.info("模型构建完成")

        # 参数量统计
        for name, cnt in self.model.count_parameters().items():
            logging.info(f"  {name:<40s}: {cnt:,}")

        # ── 数据集 ──
        # 若 cfg.dataset.multi_train_roots 非空则走多数据集模式，否则退回旧 StereoHumanDataset
        _use_multi = bool(getattr(cfg.dataset, "multi_train_roots", None))

        if _use_multi:
            logging.info("使用 PAGMultiDataset（多数据集混合训练）")
            self.train_set = build_pag_dataset(cfg.dataset, phase="train")
            self.val_set   = build_pag_dataset(cfg.dataset, phase="val")
            # val_boost 供计算 len_val 用
            _val_boost = getattr(cfg.dataset, "val_boost", 200)
        else:
            logging.info("使用 StereoHumanDataset（GPS+ legacy 模式）")
            self.train_set = StereoHumanDataset(cfg.dataset, phase="train")
            self.val_set   = StereoHumanDataset(cfg.dataset, phase="val")
            _val_boost = self.val_set.val_boost

        self.train_loader = DataLoader(
            self.train_set, batch_size=self.bs,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        self.train_iterator = iter(self.train_loader)

        self.val_loader = DataLoader(
            self.val_set, batch_size=1,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        self.len_val = max(1, len(self.val_loader) // _val_boost)
        self.val_iterator = iter(self.val_loader)

        # ── 优化器 (只优化可训练参数，跳过冻结的 DA3) ──
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        logging.info(f"可训练参数总量: {sum(p.numel() for p in trainable_params):,}")

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=cfg.lr,
            weight_decay=cfg.wdecay,
            eps=1e-8,
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.lr,
            total_steps=cfg.num_steps + 200,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy="linear",
        )

        self.scaler = GradScaler(enabled=pag_cfg.mixed_precision)
        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        # ── 恢复训练 ──
        if cfg.restore_ckpt:
            self.load_ckpt(cfg.restore_ckpt)
        elif cfg.stage1_ckpt:
            logging.info("从 stage1 checkpoint 加载简化模型权重")
            self.load_ckpt(cfg.stage1_ckpt, load_optimizer=False, strict=True)

        self.model.train()
        # DA3 始终保持 eval 模式 (已在 build_pag_splat 中冻结)
        self.model.prior_extractor.da3_net.eval()

    # ──────────────────────────────────────────────
    #  训练主循环
    # ──────────────────────────────────────────────

    def train(self):
        pag_cfg = self.cfg.pagsplat
        bg = self.cfg.dataset.bg_color

        # 累积日志变量
        log = dict(l1=0.0, ssim=0.0, scale=0.0, chamfer=0.0)
        LOG_PERIOD = 100

        for itr in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()

            # ── 取数据 ──
            data = self.fetch_data("train")

            # ── 前向传播 ──
            with torch.autocast(device_type="cuda", enabled=pag_cfg.mixed_precision):
                data = self.model(data, is_train=True)

                # ── 渲染 ──
                data = pts2render(data, bg_color=bg)

                render_novel = data["novel_view"]["img_pred"]
                gt_novel = data["novel_view"]["img"]  # 已由 fetch_data 移至 CUDA

                # ── 主重建损失 ──
                Ll1 = l1_loss(render_novel, gt_novel)
                Lssim = 1.0 - ssim(render_novel, gt_novel)
                loss  = 0.8 * Ll1 + 0.2 * Lssim

                # ── 轻量几何正则：scale_regular + 3D chamfer ──
                Lscale = data["novel_view"]["scale_regular"]
                loss = loss + pag_cfg.loss_scale * Lscale

                Lchamfer = chamfer_distance_loss(
                    data["lmain"]["xyz"],
                    data["rmain"]["xyz"],
                    valid1=data["lmain"]["pts_valid"],
                    valid2=data["rmain"]["pts_valid"],
                    n_samples=pag_cfg.chamfer_samples,
                )
                loss = loss + pag_cfg.loss_chamfer * Lchamfer

            # ── 反向传播 ──
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.scaler.step(self.optimizer)
            # OneCycleLR 有硬性 total_steps 上限，超出后不再 step 以避免 ValueError
            if self.scheduler.last_epoch < self.cfg.num_steps + 99:
                self.scheduler.step()
            self.scaler.update()

            # ── 日志 ──
            log["l1"]      += 0.8 * Ll1.item()
            log["ssim"]    += 0.2 * Lssim.item()
            log["scale"]   += pag_cfg.loss_scale   * Lscale.item()
            log["chamfer"] += pag_cfg.loss_chamfer * Lchamfer.item()

            metrics = {
                "l1":      Ll1.item(),
                "ssim":    Lssim.item(),
                "scale":   Lscale.item(),
                "chamfer": Lchamfer.item(),
            }
            self.logger.push(metrics)

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], self.total_steps
                )

            if self.total_steps and self.total_steps % LOG_PERIOD == 0:
                msg = "  ".join(f"{k}={v/LOG_PERIOD:.4f}" for k, v in log.items())
                logging.info(f"[step {self.total_steps}] {msg}")
                for k in log:
                    log[k] = 0.0

            # ── 验证 ──
            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.model.prior_extractor.da3_net.eval()
                self.run_eval()
                self.model.train()
                self.model.prior_extractor.da3_net.eval()  # 保持 DA3 为 eval

            # ── 定期保存 ──
            if self.total_steps % self.cfg.record.save_iter == 0 and self.total_steps > 0:
                self.save_ckpt(
                    Path(f"{self.cfg.record.ckpt_path}/iter{self.total_steps}.pth")
                )

            self.total_steps += 1

        logging.info("训练完成！")
        self.logger.close()
        self.save_ckpt(
            Path(f"{self.cfg.record.ckpt_path}/{self.cfg.name}_final.pth")
        )

    # ──────────────────────────────────────────────
    #  验证
    # ──────────────────────────────────────────────

    @staticmethod
    def _dataset_tag_from_sample(sample_name: str) -> str:
        """
        从 sample_name 推断数据集标识，用于 eval 可视化子目录命名。
        - mini 格式：sample_name = "actor1_4_0042"  → "actor1_4"
        - legacy 格式：sample_name = "s1a1_s1_0000" → "s1a1_s1"
        规则：去掉最后一段 _XXXX（纯数字后缀）
        """
        parts = sample_name.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0]
        return sample_name

    def run_eval(self):
        logging.info(f"[step {self.total_steps}] 开始验证 ...")
        torch.cuda.empty_cache()

        psnr_list, ssim_list = [], []
        bg = self.cfg.dataset.bg_color

        # 记录已为哪些数据集保存过可视化，每个数据集只保存第一张
        saved_datasets: set[str] = set()

        for idx in range(self.len_val):
            data = self.fetch_data("val")
            with torch.no_grad():
                data = self.model(data, is_train=False)
                data = pts2render(data, bg_color=bg)
                render_novel = data["novel_view"]["img_pred"]
                gt_novel     = data["novel_view"]["img"]

                cover_mask = None
                psnr_val = psnr(render_novel, gt_novel).mean().item()
                ssim_val = ssim(render_novel, gt_novel).item()
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

                # 每个数据集保存第一张遇到的样本可视化
                sample_name  = data["novel_view"].get("sample_name", [""])[0]
                ds_tag = self._dataset_tag_from_sample(
                    sample_name if isinstance(sample_name, str) else str(sample_name)
                )
                if ds_tag not in saved_datasets:
                    saved_datasets.add(ds_tag)
                    self._save_eval_visuals(
                        render_novel, gt_novel, data,
                        cover_mask=cover_mask,
                        dataset_tag=ds_tag,
                    )

        val_psnr = float(np.mean(psnr_list))
        val_ssim = float(np.mean(ssim_list))

        logging.info(
            f"[step {self.total_steps}] Val PSNR={val_psnr:.4f}  SSIM={val_ssim:.4f}"
        )
        self.logger.write_dict(
            {"val_psnr": val_psnr, "val_ssim": val_ssim},
            write_step=self.total_steps,
        )

        if val_psnr < 10.0:
            logging.warning(
                "PSNR < 10，训练可能崩溃，请检查配置后重新训练。"
            )

        torch.cuda.empty_cache()

    # ──────────────────────────────────────────────
    #  评估可视化
    # ──────────────────────────────────────────────

    @staticmethod
    def _depth_to_color(depth: torch.Tensor, label: str = "") -> np.ndarray:
        """
        将深度图 (B,1,H,W) 或 (1,H,W) 转为彩色 numpy uint8 (H,W,3)。
        使用 inferno colormap：近→黄，远→紫黑。
        左上角叠加 label 和真实深度范围 [min, max]（单位：米）。
        """
        import matplotlib.cm as cm
        d = depth[0] if depth.dim() == 4 else depth
        d = d[0].float().cpu().numpy()          # (H, W)
        d_min, d_max = float(d.min()), float(d.max())
        if d_max - d_min < 1e-6:
            d_norm = np.zeros_like(d)
        else:
            d_norm = (d - d_min) / (d_max - d_min)
        rgba = cm.inferno(d_norm)               # (H, W, 4)  [0,1]
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        # 叠加范围文字
        range_text = f"{d_min:.2f}~{d_max:.2f}m"
        full_text  = f"{label} {range_text}".strip() if label else range_text
        cv2.putText(img, full_text, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, full_text, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),       1, cv2.LINE_AA)
        return img

    @staticmethod
    def _diff_to_color(diff: torch.Tensor, label: str = "") -> np.ndarray:
        """差值图：绝对值 → hot colormap，越亮误差越大。左上角标注范围。"""
        import matplotlib.cm as cm
        if diff.dim() == 4:
            d = diff[0, 0].float().abs().cpu().numpy()
        elif diff.dim() == 3:
            d = diff[0].float().abs().cpu().numpy()
        else:
            d = diff.float().abs().cpu().numpy()
        d_max = float(d.max())
        d_norm = np.clip(d / (d_max + 1e-6), 0, 1)
        rgba = cm.hot(d_norm)
        img = (rgba[:, :, :3] * 255).astype(np.uint8)
        range_text = f"max {d_max:.3f}"
        full_text  = f"{label} {range_text}".strip() if label else range_text
        cv2.putText(img, full_text, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(img, full_text, (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),       1, cv2.LINE_AA)
        return img

    @staticmethod
    def _mask_to_color(mask: torch.Tensor) -> np.ndarray:
        """二值掩码 → 灰度 uint8 (H,W,3)。"""
        if mask.dim() == 4:
            m = mask[0, 0].float().cpu().numpy()
        elif mask.dim() == 3:
            m = mask[0].float().cpu().numpy()
        else:
            m = mask.float().cpu().numpy()
        gray = (m * 255).astype(np.uint8)
        return np.stack([gray, gray, gray], axis=-1)

    @staticmethod
    def _opacity_to_color(opa: torch.Tensor, H: int, W: int) -> np.ndarray:
        """
        opacity (B, H*W, 1) → 彩色 (H,W,3)。
        使用 viridis colormap。
        """
        import matplotlib.cm as cm
        if opa.dim() == 4:
            o = opa[0, 0].float().cpu().numpy()
        else:
            o = opa[0, :, 0].float().cpu().numpy().reshape(H, W)
        rgba = cm.viridis(o)
        return (rgba[:, :, :3] * 255).astype(np.uint8)

    @staticmethod
    def _pts_scatter(
        xyz:   torch.Tensor,   # (B, N, 3)  世界坐标
        H: int, W: int,
        title: str = "",
        max_pts: int = 100_000,
    ) -> np.ndarray:
        """
        与参考实现对齐的点云散点图（白色背景，turbo colormap 按 Z 深度着色）：
          - 绘制世界坐标 XY 平面的 2D 散点（X→横轴，-Y→纵轴）
          - 颜色 = Z 深度归一化（2%~98% 分位，turbo colormap）
          - 随机下采样至 max_pts 点
          - 白色背景，标题居中，显示有效点总数
          - 返回 (H, W, 3) uint8 RGB numpy 数组
        """
        import io
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as mcm

        pts = xyz[0].float().detach().cpu().numpy()   # (N, 3)

        # 过滤 NaN / Inf
        finite = np.all(np.isfinite(pts), axis=1)
        pts = pts[finite]

        n_total = len(pts)
        if n_total < 10:
            return np.ones((H, W, 3), dtype=np.uint8) * 255

        # Z 值按分位数归一化 → turbo colormap
        z_vals = pts[:, 2]
        z_lo, z_hi = np.percentile(z_vals, [2, 98])
        z_norm = np.clip((z_vals - z_lo) / (z_hi - z_lo + 1e-8), 0, 1)
        colors = mcm.turbo(z_norm)[:, :3]   # (N, 3) RGB [0,1]

        # X / Y 按分位数裁剪离群点，避免坐标范围过大压缩有效显示区
        x_lo, x_hi = np.percentile(pts[:, 0], [1, 99])
        y_lo, y_hi = np.percentile(pts[:, 1], [1, 99])
        in_range = (
            (pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi) &
            (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi)
        )
        pts_r    = pts[in_range]
        colors_r = colors[in_range]
        n_range  = len(pts_r)

        # 随机下采样
        if n_range > max_pts:
            idx = np.random.choice(n_range, max_pts, replace=False)
            pts_show    = pts_r[idx]
            colors_show = colors_r[idx]
        else:
            pts_show    = pts_r
            colors_show = colors_r

        # 点大小自适应：根据下采样后点数估算合适尺寸
        # 目标：让画布上约 30% 的像素被覆盖，点径 ≈ sqrt(area * 0.3 / n)
        fig_px   = H * W
        pt_area  = max(1.5, fig_px * 0.35 / max(len(pts_show), 1))
        pt_size  = min(pt_area, 8.0)    # 上限 8 pt²，防止过大

        # 绘制散点（X 横轴，-Y 纵轴，与图像坐标系一致）
        dpi = 150
        fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi)
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        ax.scatter(pts_show[:, 0], pts_show[:, 1],
                   c=colors_show, s=pt_size, alpha=0.7,
                   linewidths=0, rasterized=True)

        title_text = f"{title} ({n_total:,} pts)" if title else f"({n_total:,} pts)"
        ax.set_title(title_text, fontsize=8, color="#222222")
        ax.set_aspect("equal")
        ax.axis("off")
        fig.tight_layout(pad=0.1)

        # figure → numpy RGB
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                    facecolor="white")
        plt.close(fig)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        img_bgr = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)   # BGR
        img_rgb = img_bgr[:, :, ::-1]                        # → RGB
        return img_rgb

    @staticmethod
    def _img_tensor_to_np(t: torch.Tensor) -> np.ndarray:
        """(B,3,H,W) [0,1] → (H,W,3) uint8"""
        return (t[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
                ).astype(np.uint8)

    @staticmethod
    def _add_label(img: np.ndarray, text: str, font_scale: float = 0.55) -> np.ndarray:
        """在图像左上角叠加文字标签。"""
        out = img.copy()
        cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, text, (6, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        return out

    def _save_eval_visuals(
        self,
        render_novel: torch.Tensor,
        gt_novel:     torch.Tensor,
        data:         dict,
        cover_mask:   torch.Tensor | None = None,
        dataset_tag:  str = "",
    ) -> None:
        """
        分开保存各可视化项到 show/{step:06d}/ 子目录，并额外生成对比拼图：

        单图：
          render.jpg / gt.jpg / diff.jpg
          d_metric_l/r.jpg  d_final_l/r.jpg  d_delta_l/r.jpg
          pts_valid.jpg  opacity_l.jpg
          pts_lmain/rmain/merged.jpg

        对比拼图（横向拼接，已标注内容和数值范围）：
          cmp_render.jpg       — Render | GT | Diff
          cmp_psnr.jpg         — Render(masked) | GT(masked) | Diff(masked) | PSNR_Mask
          cmp_color.jpg        — Img_L(ref) | Img_L | Img_R
          cmp_depth_l.jpg      — d_metric_l | d_final_l | d_delta_l
          cmp_depth_r.jpg      — d_metric_r | d_final_r | d_delta_r
          cmp_depth_lr.jpg     — d_metric_l | d_metric_r（双视图尺度一致性）
          cmp_pts.jpg          — pts_lmain | pts_rmain | pts_merged
          cmp_geom.jpg         — pts_valid | opacity_l
        """
        step   = self.total_steps
        # 目录结构: show/{step:06d}/{dataset_tag}/
        # dataset_tag 为空时直接放在步数目录下
        step_dir = os.path.join(self.cfg.record.show_path, f"{step:06d}")
        subdir   = os.path.join(step_dir, dataset_tag) if dataset_tag else step_dir
        os.makedirs(subdir, exist_ok=True)

        H = render_novel.shape[-2]
        W = render_novel.shape[-1]
        lm = data["lmain"]
        rm = data["rmain"]
        nv = data["novel_view"]

        def resize_h(img_np: np.ndarray, h: int) -> np.ndarray:
            """仅统一高度，宽度等比例缩放。"""
            if img_np.shape[0] == h:
                return img_np
            scale = h / img_np.shape[0]
            new_w = max(1, int(img_np.shape[1] * scale))
            return cv2.resize(img_np, (new_w, h), interpolation=cv2.INTER_LINEAR)

        def save(name: str, img_np: np.ndarray) -> None:
            """单图：强制对齐到 H×W。"""
            p = os.path.join(subdir, name)
            arr = img_np
            if arr.shape[:2] != (H, W):
                arr = cv2.resize(arr, (W, H), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(p, arr[:, :, ::-1])

        def save_cmp(name: str, *panels: np.ndarray) -> None:
            """对比拼图：各面板统一高度后横向拼接，宽度不限。"""
            p = os.path.join(subdir, name)
            resized = [resize_h(panel, H) for panel in panels]
            combined = np.concatenate(resized, axis=1)
            cv2.imwrite(p, combined[:, :, ::-1])


        def labeled(img: np.ndarray, text: str) -> np.ndarray:
            """在图像左下角加标签，黑色描边+白色字，在任意背景下都清晰可见。"""
            if not text:
                return img
            out = img.copy()
            y = H - 8
            # 黑色厚描边（在白背景上显）
            cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 0, 0), 4, cv2.LINE_AA)
            # 白色主体字（在黑背景上显）
            cv2.putText(out, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 1, cv2.LINE_AA)
            return out

        # ── 渲染质量 ──
        render_np = self._img_tensor_to_np(render_novel)
        gt_np     = self._img_tensor_to_np(gt_novel)
        diff_np   = self._diff_to_color(render_novel - gt_novel, label="Diff")
        img_l_vis = self._img_tensor_to_np(lm["img"] * 0.5 + 0.5)
        img_r_vis = self._img_tensor_to_np(rm["img"] * 0.5 + 0.5)
        save("render.jpg", render_np)
        save("gt.jpg",     gt_np)

        # 视角编号标签（来自 loader 输出的 cam_id_*，回退到 L/R/N）
        def _cam_id(key, default):
            v = data.get(key, default)
            return v.item() if isinstance(v, torch.Tensor) else v
        cam_id_l = _cam_id("cam_id_l", "L")
        cam_id_r = _cam_id("cam_id_r", "R")
        cam_id_n = _cam_id("cam_id_n", "N")
        label_l = f"Input #{cam_id_l}"
        label_r = f"Input #{cam_id_r}"
        label_n = f"Cam #{cam_id_n}"

        # cmp_render: 上行=两个输入视角，下行=Render|GT|Diff
        # 上行居中对齐（填充到与下行等宽），下行各格宽度与上行一格等宽
        row_top = np.concatenate([
            labeled(img_l_vis, label_l),
            labeled(img_r_vis, label_r),
        ], axis=1)   # (H, 2W, 3)
        row_bot = np.concatenate([
            labeled(render_np, f"Render #{cam_id_n}"),
            labeled(gt_np,     f"GT #{cam_id_n}"),
            labeled(diff_np,   ""),
        ], axis=1)   # (H, 3W, 3)
        # 宽度对齐：将较窄的行居中填充白色到较宽行宽度
        tw = max(row_top.shape[1], row_bot.shape[1])
        def _pad_width(img: np.ndarray, target_w: int) -> np.ndarray:
            pad = target_w - img.shape[1]
            if pad <= 0:
                return img
            left  = pad // 2
            right = pad - left
            return np.pad(img, ((0, 0), (left, right), (0, 0)),
                          mode="constant", constant_values=255)
        row_top = _pad_width(row_top, tw)
        row_bot = _pad_width(row_bot, tw)
        cmp_render = np.concatenate([row_top, row_bot], axis=0)
        cv2.imwrite(os.path.join(subdir, "cmp_render.jpg"),
                    cmp_render[:, :, ::-1])

        # ── PSNR 有效区域可视化 ──
        # 若外部未传 cover_mask，则在内部按相同规则重新计算
        if cover_mask is None:
            bg_vals = self.cfg.dataset.bg_color
            bg_t2   = torch.tensor(bg_vals, device=render_novel.device).view(1, 3, 1, 1)
            cover_mask = ((render_novel - bg_t2).abs().sum(dim=1, keepdim=True) > 1e-3)

        # cover_mask: (1,1,H,W) bool
        cm_float = cover_mask.float()                    # (1,1,H,W) [0,1]
        cover_ratio = cm_float.mean().item() * 100       # 覆盖率 %

        # 有效区域遮罩的可视化（白=纳入PSNR, 黑=背景排除）
        mask_vis_np = (cm_float[0, 0].cpu().numpy() * 255).astype(np.uint8)
        mask_vis_np = np.stack([mask_vis_np] * 3, axis=-1)
        cv2.putText(mask_vis_np,
                    f"Coverage {cover_ratio:.1f}%", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0),   2, cv2.LINE_AA)
        cv2.putText(mask_vis_np,
                    f"Coverage {cover_ratio:.1f}%", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 80), 1, cv2.LINE_AA)

        # 仅对有效区域的渲染 / GT / Diff（背景置白，突出覆盖区）
        white_bg = torch.ones_like(render_novel)
        render_masked = render_novel * cm_float + white_bg * (1 - cm_float)
        gt_masked     = gt_novel     * cm_float + white_bg * (1 - cm_float)
        diff_masked   = self._diff_to_color(
            (render_novel - gt_novel) * cm_float, label="Diff(masked)"
        )
        render_masked_np = self._img_tensor_to_np(render_masked)
        gt_masked_np     = self._img_tensor_to_np(gt_masked)

        save_cmp("cmp_psnr.jpg",
            labeled(render_masked_np, "Render(masked)"),
            labeled(gt_masked_np,     "GT(masked)"),
            labeled(diff_masked,      ""),
            labeled(mask_vis_np,      "PSNR Mask"),
        )

        # ── 输入视图颜色 ──
        cmap_np = self._img_tensor_to_np(lm["img"] * 0.5 + 0.5)
        img_l_np = self._img_tensor_to_np(lm["img"] * 0.5 + 0.5)
        img_r_np = self._img_tensor_to_np(rm["img"] * 0.5 + 0.5)
        save_cmp("cmp_color.jpg",
            labeled(cmap_np,  "Img_L(ref)"),
            labeled(img_l_np, "Img_L"),
            labeled(img_r_np, "Img_R"),
        )

        # ── 深度：尺度对齐效果 ──
        log_s_l = data["log_scale_l"][0].item()
        log_s_r = data["log_scale_r"][0].item()
        d_metric_l = self._depth_to_color(data["metric_depth_l"],
                                          label=f"Metric_L logS={log_s_l:.3f}")
        d_metric_r = self._depth_to_color(data["metric_depth_r"],
                                          label=f"Metric_R logS={log_s_r:.3f}")
        d_final_l  = self._depth_to_color(data["final_depth_l"],  label="Final_L")
        d_final_r  = self._depth_to_color(data["final_depth_r"],  label="Final_R")
        d_delta_l  = self._diff_to_color(
            data["final_depth_l"] - data["metric_depth_l"], label="Delta_L"
        )
        d_delta_r  = self._diff_to_color(
            data["final_depth_r"] - data["metric_depth_r"], label="Delta_R"
        )
        save_cmp("cmp_depth_l.jpg",  d_metric_l, d_final_l, d_delta_l)
        save_cmp("cmp_depth_r.jpg",  d_metric_r, d_final_r, d_delta_r)
        save_cmp("cmp_depth_lr.jpg", d_metric_l, d_metric_r)

        # ── 几何覆盖 ──
        pts_valid_map = lm["pts_valid"][0].view(H, W)
        mask_np = self._mask_to_color(pts_valid_map)
        opacity_np = self._opacity_to_color(lm["opacity_maps"], H, W)
        save_cmp("cmp_geom.jpg",
            labeled(mask_np,    "PtsValid"),
            labeled(opacity_np, "Opacity_L"),
        )

        # ── 点云散点图（世界坐标 XY 平面，turbo 深度着色，白色背景）──
        xyz_merged = torch.cat([lm["xyz"], rm["xyz"]], dim=1)

        pts_l = self._pts_scatter(lm["xyz"],  H, W, title="Left Point Cloud")
        pts_r = self._pts_scatter(rm["xyz"],  H, W, title="Right Point Cloud")
        pts_m = self._pts_scatter(xyz_merged, H, W, title="Merged Point Cloud")
        save_cmp("cmp_pts.jpg", pts_l, pts_r, pts_m)

        # 步数目录根部：每个数据集各生成一张 overview_{tag}.jpg 供快速浏览
        tag_safe = dataset_tag.replace("/", "_") if dataset_tag else "default"
        quick_path = os.path.join(step_dir, f"overview_{tag_safe}.jpg")
        cv2.imwrite(quick_path,
                    np.concatenate([render_np, gt_np], axis=1)[:, :, ::-1])

    # ──────────────────────────────────────────────
    #  数据获取
    # ──────────────────────────────────────────────

    def fetch_data(self, phase: str) -> dict:
        """从 DataLoader 取一个 batch，并将所有 Tensor 移至 CUDA。"""
        if phase == "train":
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

        # 将 lmain / rmain / novel_view 中的所有 Tensor 移至 CUDA
        return move_data_to_cuda(data)

    # ──────────────────────────────────────────────
    #  Checkpoint 管理
    # ──────────────────────────────────────────────

    def load_ckpt(
        self,
        load_path: str,
        load_optimizer: bool = True,
        strict: bool = True,
    ) -> None:
        assert os.path.exists(load_path), f"Checkpoint 不存在: {load_path}"
        logging.info(f"加载 checkpoint: {load_path}")
        ckpt = torch.load(load_path, map_location="cuda", weights_only=False)

        if "network" not in ckpt:
            raise KeyError("Checkpoint 缺少 'network' 键，无法加载简化模型")

        missing, unexpected = self.model.load_state_dict(ckpt["network"], strict=strict)
        if missing or unexpected:
            raise RuntimeError(
                "Checkpoint 与当前简化模型不兼容："
                f"missing={missing[:5]} unexpected={unexpected[:5]}"
            )

        if load_optimizer and "optimizer" in ckpt:
            # ── 训练步数 ──
            self.total_steps = ckpt.get("total_steps", 0) + 1
            self.logger.total_steps = self.total_steps

            # ── 优化器 ──
            self.optimizer.load_state_dict(ckpt["optimizer"])

            # ── 调度器：检测是否已耗尽，若耗尽则重置以支持延长训练 ──
            sched_state = ckpt.get("scheduler", {})
            ckpt_sched_total = sched_state.get("total_steps", 0)
            ckpt_step_count  = sched_state.get("_step_count", 0)
            new_sched_total  = self.scheduler.total_steps  # 当前新建 scheduler 的 total_steps
            if ckpt_step_count >= ckpt_sched_total - 1:
                # 旧 scheduler 已耗尽（或即将耗尽），直接使用新 scheduler（learning rate warm-restart）
                logging.info(
                    f"  旧 scheduler 已耗尽 (_step_count={ckpt_step_count}, "
                    f"total_steps={ckpt_sched_total})，重置为新 scheduler "
                    f"(total_steps={new_sched_total})，做 warm-restart"
                )
            elif new_sched_total != ckpt_sched_total:
                # total_steps 不一致，也跳过加载避免越界
                logging.warning(
                    f"  scheduler total_steps 不匹配：ckpt={ckpt_sched_total} "
                    f"vs new={new_sched_total}，跳过加载 scheduler state"
                )
            else:
                self.scheduler.load_state_dict(sched_state)

            # ── GradScaler (混合精度) ──
            if "scaler" in ckpt:
                self.scaler.load_state_dict(ckpt["scaler"])

            logging.info(
                f"  恢复到训练步 {self.total_steps}  "
                f"lr={self.optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            logging.info(f"  仅加载模型权重（不恢复训练状态），strict={strict}")

    def save_ckpt(self, save_path: Path, show_log: bool = True) -> None:
        save_path = Path(save_path)
        if show_log:
            logging.info(f"保存 checkpoint → {save_path}")
        payload = {
            "total_steps": self.total_steps,
            "exp_name":    self.cfg.exp_name,   # 保存实验名，resume 时恢复目录
            "network":     self.model.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "scheduler":   self.scheduler.state_dict(),
            "scaler":      self.scaler.state_dict(),
        }
        torch.save(payload, save_path)

        # 同时更新 latest.pth（方便 --auto_resume 直接找到最新断点）
        latest = save_path.parent / "latest.pth"
        torch.save(payload, latest)


# ──────────────────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    parser = argparse.ArgumentParser(description="PAG-Splat Training")
    parser.add_argument(
        "--config",
        default="pag_splat/pag_stage.yaml",
        help="YAML 配置文件路径",
    )
    parser.add_argument(
        "--restore_ckpt",
        default=None,
        help="恢复训练的 checkpoint 路径（绝对或相对路径）",
    )
    parser.add_argument(
        "--auto_resume",
        action="store_true",
        help="自动从当前实验目录的 latest.pth 恢复，无需手动指定路径",
    )
    parser.add_argument(
        "--exp_name",
        default=None,
        help="指定实验名（覆盖自动生成的 name_MMDD）",
    )
    parser.add_argument(
        "--da3_checkpoint",
        default=None,
        help="覆盖 YAML 中的 da3_checkpoint (本地路径或 HF repo id)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="覆盖 YAML 中的 num_steps，用于延长训练（如 --num_steps 150000）",
    )
    args = parser.parse_args()

    # ── 加载配置 ──
    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(args.config)
    cfg = cfg_obj.get_cfg()
    cfg.defrost()

    if args.da3_checkpoint:
        cfg.pagsplat.da3_checkpoint = args.da3_checkpoint

    if args.num_steps is not None:
        cfg.num_steps = args.num_steps
        logging.info(f"[CLI] 覆盖 num_steps = {cfg.num_steps}")

    # ── 确定实验名 & 恢复路径 ──
    # 优先级: --restore_ckpt > --auto_resume > 新建实验
    restore_path: str | None = None
    dt = datetime.today()
    default_exp  = f"{cfg.name}_{str(dt.month).zfill(2)}{str(dt.day).zfill(2)}"
    resolved_exp: str | None = args.exp_name  # 可能为 None

    if args.restore_ckpt:
        restore_path = args.restore_ckpt
        # 从 checkpoint 内读取 exp_name，使目录与原实验保持一致
        try:
            _meta = torch.load(restore_path, map_location="cpu", weights_only=False)
            saved_exp = _meta.get("exp_name", None)
            if saved_exp and resolved_exp is None:
                resolved_exp = saved_exp
                logging.info(f"[resume] 从 checkpoint 读取到实验名: {saved_exp}")
        except Exception as _e:
            logging.warning(f"[resume] 读取 checkpoint 元数据失败: {_e}")

    elif args.auto_resume:
        # 先用 --exp_name 或 today 确定目录，再找 latest.pth
        _search_exp = resolved_exp or default_exp
        _latest = Path(f"experiments/{_search_exp}/ckpt/latest.pth")
        if _latest.exists():
            restore_path = str(_latest)
            resolved_exp = _search_exp
            logging.info(f"[auto_resume] 找到 latest.pth: {_latest}")
        else:
            # 在 experiments/ 中搜索前缀匹配的所有 latest.pth，取最近修改的
            _found = sorted(
                Path("experiments").glob(f"{cfg.name}*/ckpt/latest.pth"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if _found:
                restore_path  = str(_found[0])
                resolved_exp  = _found[0].parts[1]  # experiments/<exp_name>/ckpt/latest.pth
                logging.info(f"[auto_resume] 自动选择最近实验: {restore_path}")
            else:
                logging.warning("[auto_resume] 未找到任何 latest.pth，将从头开始训练")

    # 最终实验名：已从 checkpoint 解析 > --exp_name > 日期自动生成
    final_exp = resolved_exp or default_exp
    cfg.exp_name = final_exp

    if restore_path:
        cfg.restore_ckpt = restore_path

    # ── 实验目录 ──
    cfg.record.ckpt_path = f"experiments/{cfg.exp_name}/ckpt"
    cfg.record.show_path = f"experiments/{cfg.exp_name}/show"
    cfg.record.logs_path = f"experiments/{cfg.exp_name}/logs"
    cfg.record.file_path = f"experiments/{cfg.exp_name}/file"
    cfg.freeze()

    for p in [
        cfg.record.ckpt_path,
        cfg.record.show_path,
        cfg.record.logs_path,
        cfg.record.file_path,
    ]:
        Path(p).mkdir(exist_ok=True, parents=True)

    # ── 备份源码 ──
    pag_file_backup(cfg.record.file_path, cfg, train_script=__file__)

    # ── 随机种子 ──
    torch.manual_seed(1314)
    np.random.seed(1314)

    logging.info(f"实验名: {cfg.exp_name}")
    logging.info(f"配置文件: {args.config}")
    if restore_path:
        logging.info(f"恢复训练: {restore_path}")

    # ── 启动训练 ──
    trainer = PAGSplatTrainer(cfg)
    trainer.train()
