"""
PAG-Splat 训练脚本

基于 GPS+ train.py 重构，适配 PAGSplat 三模块架构：
  - 使用 StereoHumanDataset (与 GPS+ 完全兼容)
  - DA3 backbone 全程冻结，不参与梯度更新
  - 损失 = 0.8×L1 + 0.2×(1-SSIM) + λs×Smooth + λw×WarpCons + λu×UncReg
  - 使用 pag_pts2render 替代 GPS+ 的 pts2render

用法:
    python train_pag.py --config pag_splat/pag_stage.yaml
    python train_pag.py --config pag_splat/pag_stage.yaml --restore_ckpt experiments/xxx/ckpt/latest.pth
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
from lib.train_recoder import Logger
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr
from pag_splat.model import build_pag_splat
from pag_splat.render import pag_pts2render, move_data_to_cuda

warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────
#  辅助损失函数
# ──────────────────────────────────────────────────────────

def edge_aware_smooth_loss(depth: torch.Tensor, img: torch.Tensor) -> torch.Tensor:
    """
    边缘感知深度平滑损失 (Monodepth2 风格)。

    在图像边缘区域不惩罚深度跳变，在平滑区域要求深度也平滑。

    Args:
        depth: (B, 1, H, W)  度量深度
        img:   (B, 3, H, W)  对应 RGB 图像

    Returns:
        标量损失值
    """
    # 均值归一化深度 (使损失与深度绝对值无关)
    mean_d = depth.mean(dim=[2, 3], keepdim=True).clamp(min=1e-5)
    d_norm = depth / mean_d

    # 深度梯度
    d_dx = torch.abs(d_norm[:, :, :, :-1] - d_norm[:, :, :, 1:])  # (B,1,H,W-1)
    d_dy = torch.abs(d_norm[:, :, :-1, :] - d_norm[:, :, 1:, :])  # (B,1,H-1,W)

    # 图像梯度 (取均值通道减少计算量)
    img_mean = img.mean(dim=1, keepdim=True)
    i_dx = torch.abs(img_mean[:, :, :, :-1] - img_mean[:, :, :, 1:])
    i_dy = torch.abs(img_mean[:, :, :-1, :] - img_mean[:, :, 1:, :])

    # 边缘权重: 边缘处权重趋近 0
    w_x = torch.exp(-i_dx)
    w_y = torch.exp(-i_dy)

    return (d_dx * w_x).mean() + (d_dy * w_y).mean()


def warp_consistency_loss(
    opacity: torch.Tensor, uncertainty: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    """
    扭曲一致性损失: 无效扭曲区域应具有高不确定性 (低有效 opacity)。

    Args:
        opacity:     (B, H*W, 1)  不透明度
        uncertainty: (B, H*W, 1)  不确定性
        valid_mask:  (B, 1, Hf, Wf) 有效扭曲区域掩码 (1=有效)

    Returns:
        标量损失值
    """
    # 将 valid_mask 上采样 / 展平到 H*W 分辨率
    B, _, Hf, Wf = valid_mask.shape
    BHW = opacity.shape[1]
    H = W = int(BHW ** 0.5)  # 假设正方形

    # 若特征分辨率不等于全分辨率，先上采样
    if Hf * Wf != H * W:
        vm = F.interpolate(
            valid_mask.float(), size=(H, W), mode="nearest"
        )
    else:
        vm = valid_mask.float()

    vm_flat = vm.reshape(B, H * W, 1)  # (B, HW, 1)

    # 无效区域掩码 (0=无效)
    invalid = 1.0 - vm_flat

    # 在无效区域，希望 opacity * (1 - uncertainty) ≈ 0
    # 即: 要么 opacity 低，要么 uncertainty 高
    eff_opa_invalid = opacity * (1.0 - uncertainty) * invalid
    return eff_opa_invalid.mean()


def uncertainty_sparsity_loss(uncertainty: torch.Tensor) -> torch.Tensor:
    """
    不确定性稀疏正则: 鼓励不确定性图稀疏 (大部分区域应有高确信度)。

    使用 L1 正则 (鼓励稀疏), 避免退化为全 1。
    同时加入熵项防止退化为全 0。

    Args:
        uncertainty: (B, H*W, 1) 不确定性，值域 [0,1]

    Returns:
        标量损失值
    """
    u = uncertainty.clamp(1e-6, 1 - 1e-6)
    # L1 稀疏性 (鼓励接近 0)
    l1_sparse = u.mean()
    # 熵项防止退化为全 0 (使网络保留一定不确定性表达能力)
    entropy = -(u * (u + 1e-6).log() + (1 - u) * (1 - u + 1e-6).log()).mean()
    # 净效果: 鼓励稀疏，但不完全消除不确定性
    return l1_sparse - 0.1 * entropy


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
        self.model = build_pag_splat(
            da3_checkpoint=pag_cfg.da3_checkpoint,
            feat_channels=pag_cfg.feat_channels,
            feat_stride=pag_cfg.feat_stride,
            feat_layer=pag_cfg.feat_layer,
            mlp_hidden=pag_cfg.mlp_hidden,
            enc_dims=list(pag_cfg.enc_dims),
            dec_dims=list(pag_cfg.dec_dims),
            head_ch=pag_cfg.head_ch,
            scale_max=pag_cfg.scale_max,
            device="cuda",
            # embed_dim 由 build_pag_splat 自动从模型权重检测，兼容不同 DA3 变体
        )
        logging.info("模型构建完成")

        # 参数量统计
        for name, cnt in self.model.count_parameters().items():
            logging.info(f"  {name:<40s}: {cnt:,}")

        # ── 数据集 ──
        self.train_set = StereoHumanDataset(cfg.dataset, phase="train")
        self.train_loader = DataLoader(
            self.train_set, batch_size=self.bs,
            shuffle=True, num_workers=4, pin_memory=True,
        )
        self.train_iterator = iter(self.train_loader)

        self.val_set = StereoHumanDataset(cfg.dataset, phase="val")
        self.val_loader = DataLoader(
            self.val_set, batch_size=1,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)
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
            total_steps=cfg.num_steps + 100,
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
            logging.info("从 stage1 checkpoint 加载部分权重")
            self.load_ckpt(cfg.stage1_ckpt, load_optimizer=False, strict=False)

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
        log = dict(l1=0.0, ssim=0.0, smooth=0.0, warp=0.0, unc=0.0)
        LOG_PERIOD = 100

        for itr in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()

            # ── 取数据 ──
            data = self.fetch_data("train")

            # ── 前向传播 ──
            with torch.autocast(device_type="cuda", enabled=pag_cfg.mixed_precision):
                data = self.model(data, is_train=True)

                # ── 渲染 ──
                data = pag_pts2render(
                    data, bg_color=bg, min_opacity=pag_cfg.min_opacity
                )

                render_novel = data["novel_view"]["img_pred"]
                gt_novel = data["novel_view"]["img"]  # 已由 fetch_data 移至 CUDA

                # ── 主重建损失 ──
                Ll1   = l1_loss(render_novel, gt_novel)
                Lssim = 1.0 - ssim(render_novel, gt_novel)
                loss  = 0.8 * Ll1 + 0.2 * Lssim

                # ── 辅助损失 1: 边缘感知深度平滑 ──
                d_metric_l = data["metric_depth_l"]  # (B, 1, H, W)
                Lsmooth = edge_aware_smooth_loss(d_metric_l, data["lmain"]["img"])
                loss = loss + pag_cfg.loss_smooth * Lsmooth

                # ── 辅助损失 2: 扭曲一致性 ──
                opa_l = data["lmain"]["opacity"]      # (B, H*W, 1)
                unc_l = data["lmain"]["uncertainty"]  # (B, H*W, 1)
                vm_l  = data["warp_valid_mask"]       # (B, 1, Hf, Wf)
                Lwarp = warp_consistency_loss(opa_l, unc_l, vm_l)
                loss  = loss + pag_cfg.loss_warp * Lwarp

                # ── 辅助损失 3: 不确定性稀疏正则 ──
                Lunc = (
                    uncertainty_sparsity_loss(unc_l)
                    + uncertainty_sparsity_loss(data["rmain"]["uncertainty"])
                ) * 0.5
                loss = loss + pag_cfg.loss_unc * Lunc

            # ── 反向传播 ──
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 1.0
            )
            self.scaler.step(self.optimizer)
            self.scheduler.step()
            self.scaler.update()

            # ── 日志 ──
            log["l1"]     += 0.8 * Ll1.item()
            log["ssim"]   += 0.2 * Lssim.item()
            log["smooth"] += pag_cfg.loss_smooth * Lsmooth.item()
            log["warp"]   += pag_cfg.loss_warp   * Lwarp.item()
            log["unc"]    += pag_cfg.loss_unc     * Lunc.item()

            metrics = {
                "l1": Ll1.item(),
                "ssim": Lssim.item(),
                "smooth": Lsmooth.item(),
                "warp": Lwarp.item(),
                "unc": Lunc.item(),
            }
            self.logger.push(metrics)

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(
                    "lr", self.optimizer.param_groups[0]["lr"], self.total_steps
                )
                self.save_ckpt(
                    Path(f"{self.cfg.record.ckpt_path}/{self.cfg.name}_latest.pth"),
                    show_log=False,
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

    def run_eval(self):
        logging.info(f"[step {self.total_steps}] 开始验证 ...")
        torch.cuda.empty_cache()

        psnr_list, ssim_list = [], []
        save_idx = np.random.randint(0, max(1, self.len_val))
        bg = self.cfg.dataset.bg_color

        for idx in range(self.len_val):
            data = self.fetch_data("val")
            with torch.no_grad():
                data = self.model(data, is_train=False)
                data = pag_pts2render(
                    data, bg_color=bg,
                    min_opacity=self.cfg.pagsplat.min_opacity,
                )
                render_novel = data["novel_view"]["img_pred"]
                gt_novel     = data["novel_view"]["img"]

                psnr_val = psnr(render_novel, gt_novel).mean().item()
                ssim_val = ssim(render_novel, gt_novel).item()
                psnr_list.append(psnr_val)
                ssim_list.append(ssim_val)

                # 随机保存一张对比图
                if idx == save_idx:
                    self._save_comparison(render_novel, gt_novel)

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

    def _save_comparison(
        self, render: torch.Tensor, gt: torch.Tensor
    ) -> None:
        """保存 渲染结果 | GT 对比图到 show 目录。"""
        def to_numpy(t: torch.Tensor) -> np.ndarray:
            img = t[0].detach().clamp(0, 1) * 255
            return img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        render_np = to_numpy(render)
        gt_np     = to_numpy(gt)
        # 并排拼接
        compare = np.concatenate([render_np, gt_np], axis=1)
        save_path = os.path.join(
            self.cfg.record.show_path, f"{self.total_steps:06d}.jpg"
        )
        cv2.imwrite(save_path, compare[:, :, ::-1])

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
        self.model.load_state_dict(ckpt["network"], strict=strict)
        if load_optimizer and "optimizer" in ckpt:
            self.total_steps = ckpt["total_steps"] + 1
            self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.scheduler.load_state_dict(ckpt["scheduler"])
            logging.info(f"  恢复到训练步 {self.total_steps}")

    def save_ckpt(self, save_path: Path, show_log: bool = True) -> None:
        if show_log:
            logging.info(f"保存 checkpoint → {save_path}")
        torch.save(
            {
                "total_steps": self.total_steps,
                "network":     self.model.state_dict(),
                "optimizer":   self.optimizer.state_dict(),
                "scheduler":   self.scheduler.state_dict(),
            },
            save_path,
        )


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
        help="恢复训练的 checkpoint 路径",
    )
    parser.add_argument(
        "--da3_checkpoint",
        default=None,
        help="覆盖 YAML 中的 da3_checkpoint (本地路径或 HF repo id)",
    )
    args = parser.parse_args()

    # ── 加载配置 ──
    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(args.config)
    cfg = cfg_obj.get_cfg()

    cfg.defrost()

    # 命令行参数覆盖
    if args.restore_ckpt:
        cfg.restore_ckpt = args.restore_ckpt
    if args.da3_checkpoint:
        cfg.pagsplat.da3_checkpoint = args.da3_checkpoint

    # 实验目录命名: pag_splat_MMDD
    dt = datetime.today()
    cfg.exp_name = f"{cfg.name}_{str(dt.month).zfill(2)}{str(dt.day).zfill(2)}"
    cfg.record.ckpt_path = f"experiments/{cfg.exp_name}/ckpt"
    cfg.record.show_path = f"experiments/{cfg.exp_name}/show"
    cfg.record.logs_path = f"experiments/{cfg.exp_name}/logs"
    cfg.record.file_path = f"experiments/{cfg.exp_name}/file"

    cfg.freeze()

    # ── 创建目录 ──
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

    # ── 启动训练 ──
    trainer = PAGSplatTrainer(cfg)
    trainer.train()
