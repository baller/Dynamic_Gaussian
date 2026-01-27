"""
GPS_plus多卡训练脚本 - 基于Hugging Face Accelerate
支持多GPU分布式训练、混合精度训练、梯度累积等功能

使用方法:
    # 单卡训练
    accelerate launch train_accelerate.py
    
    # 多卡训练 (使用配置文件)
    accelerate launch --config_file accelerate_config.yaml train_accelerate.py
    
    # 多卡训练 (直接指定)
    accelerate launch --multi_gpu --num_processes 4 train_accelerate.py
    
    # 带梯度累积的训练
    accelerate launch train_accelerate.py --gradient_accumulation_steps 4
"""

from __future__ import print_function, division

import argparse
import logging
import numpy as np
import cv2
import os
import random
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
from copy import deepcopy

from accelerate import Accelerator
from accelerate.utils import set_seed, DistributedDataParallelKwargs

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render, pts2render_moe
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr
from lib.loss import MoELoss, create_moe_loss

warnings.filterwarnings("ignore", category=UserWarning)


class AccelerateTrainer:
    """基于Accelerate的多卡训练器"""
    
    def __init__(self, cfg_file, accelerator: Accelerator):
        self.cfg = cfg_file
        self.accelerator = accelerator
        self.bs = self.cfg.batch_size
        self.depth_mode = getattr(self.cfg, 'depth_mode', 'raft')
        
        # MoE配置
        self.moe_cfg = getattr(self.cfg, 'moe', None)
        self.use_moe = self.moe_cfg is not None and getattr(self.moe_cfg, 'enabled', False)
        
        if self.accelerator.is_main_process:
            logging.info(f"深度估计模式: {self.depth_mode}")
            logging.info(f"使用设备数量: {self.accelerator.num_processes}")
            logging.info(f"混合精度: {self.accelerator.mixed_precision}")
            if self.use_moe:
                logging.info(f"MoE模式已启用")

        # 创建模型
        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)
        
        # 创建数据集和数据加载器
        self.train_set = StereoHumanDataset(self.cfg.dataset, phase='train')
        self.val_set = StereoHumanDataset(self.cfg.dataset, phase='val')
        
        # 使用DataLoader（accelerate会自动处理分布式采样）
        self.train_loader = DataLoader(
            self.train_set, 
            batch_size=self.bs, 
            shuffle=True, 
            num_workers=4,  # 多卡时减少worker数量避免内存问题
            pin_memory=True,
            drop_last=True  # 多卡训练时丢弃不完整的batch
        )
        self.val_loader = DataLoader(
            self.val_set, 
            batch_size=1, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True
        )
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)
        
        # 创建优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=self.cfg.lr, 
            weight_decay=self.cfg.wdecay, 
            eps=1e-8
        )
        
        # 创建学习率调度器
        # 注意：总步数需要根据梯度累积调整
        effective_batch_size = self.bs * self.accelerator.num_processes * self.accelerator.gradient_accumulation_steps
        num_update_steps = self.cfg.num_steps
        
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer, 
            self.cfg.lr, 
            num_update_steps + 100,
            pct_start=0.01, 
            cycle_momentum=False, 
            anneal_strategy='linear'
        )

        # MoE损失函数
        if self.use_moe:
            self.moe_loss_fn = create_moe_loss(self.cfg)
            self.prev_data = None
            
            training_cfg = getattr(self.moe_cfg, 'training', None)
            if training_cfg is not None:
                self.freeze_router_steps = getattr(training_cfg, 'freeze_router_epochs', 5) * len(self.train_loader)
                self.progressive_training = getattr(training_cfg, 'progressive', True)
            else:
                self.freeze_router_steps = 5 * len(self.train_loader)
                self.progressive_training = True
            
            if self.accelerator.is_main_process:
                logging.info(f"MoE渐进式训练: 前{self.freeze_router_steps}步冻结路由器")
        else:
            self.moe_loss_fn = None
            self.prev_data = None
        
        # 检查是否使用 Transformer MoE (DINO-based)
        model_unwrapped = self.model  # 在 prepare 之前
        self.use_transformer_moe = (
            hasattr(model_unwrapped, 'use_transformer_moe') and 
            model_unwrapped.use_transformer_moe
        )
        
        if self.use_transformer_moe:
            if self.accelerator.is_main_process:
                logging.info("[Curriculum] Transformer MoE 模式: 启用课程学习策略")
                # 打印课程学习配置
                curriculum_cfg = getattr(
                    getattr(self.moe_cfg, 'training', None), 
                    'curriculum', 
                    None
                )
                if curriculum_cfg is not None:
                    logging.info(f"  - Warmup: {getattr(curriculum_cfg, 'warmup_steps', 10000)} 步")
                    logging.info(f"  - Decay: {getattr(curriculum_cfg, 'decay_steps', 40000)} 步")
                    logging.info(f"  - Min BG Prob: {getattr(curriculum_cfg, 'min_bg_prob', 0.3)}")

        # 使用accelerate准备所有组件
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
            )
        
        self.train_iterator = iter(self.train_loader)
        self.val_iterator = iter(self.val_loader)
        
        self.total_steps = 0
        
        # 只在主进程创建Logger
        if self.accelerator.is_main_process:
            self.logger = Logger(self.scheduler, cfg.record)
        else:
            self.logger = None
        
        # 加载checkpoint
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.stage1_ckpt:
            if self.accelerator.is_main_process:
                logging.info(f"Using checkpoint from stage1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        
        self.model.train()
        
        # 冻结BN层
        self._freeze_bn()
        
        # MoE渐进式训练：初始阶段冻结路由器
        if self.use_moe and self.progressive_training:
            self._freeze_router()

    def _freeze_bn(self):
        """根据深度模式冻结BatchNorm层"""
        # 获取原始模型（去除DDP包装）
        model = self.accelerator.unwrap_model(self.model)
        
        if self.depth_mode == 'raft':
            if hasattr(model, 'raft_stereo') and model.raft_stereo is not None:
                model.raft_stereo.freeze_bn()
                if self.accelerator.is_main_process:
                    logging.info("已冻结RAFT-Stereo的BatchNorm层")
        elif self.depth_mode == 'da3':
            if hasattr(model, 'depth_model') and model.depth_model is not None:
                model.depth_model.freeze_bn()
                if self.accelerator.is_main_process:
                    logging.info("已冻结DA3的BatchNorm层")

    def _freeze_router(self):
        """冻结MoE路由器"""
        model = self.accelerator.unwrap_model(self.model)
        if hasattr(model, 'freeze_router'):
            model.freeze_router()
            if self.accelerator.is_main_process:
                logging.info("已冻结MoE路由器")

    def _unfreeze_router(self):
        """解冻MoE路由器"""
        model = self.accelerator.unwrap_model(self.model)
        if hasattr(model, 'unfreeze_router'):
            model.unfreeze_router()
            if self.accelerator.is_main_process:
                logging.info(f"步骤 {self.total_steps}: 解冻MoE路由器，开始联合训练")

    def train(self):
        log_l1 = 0
        log_lssim = 0
        log_chamfer = 0
        log_scale = 0
        log_moe = 0
        if_chamfer = False
        if_scale = False
        router_unfrozen = False
        
        progress_bar = tqdm(
            range(self.total_steps, self.cfg.num_steps), 
            disable=not self.accelerator.is_main_process,
            desc="Training"
        )
        
        for itr_ in progress_bar:
            # MoE渐进式训练：在指定步数后解冻路由器
            if self.use_moe and self.progressive_training and not router_unfrozen:
                if self.total_steps >= self.freeze_router_steps:
                    self._unfreeze_router()
                    router_unfrozen = True
            
            self.optimizer.zero_grad()
            
            # 使用accelerate的梯度累积上下文
            with self.accelerator.accumulate(self.model):
                data = self.fetch_data(phase='train')

                # 深度估计和高斯参数预测
                # Transformer MoE 模式: 传递 step 用于课程学习
                if self.use_transformer_moe:
                    # bg_update_signal=None 表示由模型内部的课程调度器决定
                    data, _, metrics = self.model(
                        data, 
                        is_train=True, 
                        bg_update_signal=None,
                        step=self.total_steps
                    )
                else:
                    data, _, metrics = self.model(data, is_train=True)
                
                # 高斯渲染
                if self.use_moe and 'router_weights' in data.get('lmain', {}):
                    data = pts2render_moe(data, bg_color=self.cfg.dataset.bg_color)
                else:
                    data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                # 计算损失
                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].to(self.accelerator.device)

                Ll1 = l1_loss(render_novel, gt_novel)
                Lssim = 1.0 - ssim(render_novel, gt_novel)
                loss = 0.8 * Ll1 + 0.2 * Lssim
                
                # MoE损失
                moe_loss = 0
                moe_loss_dict = {}
                if self.use_moe and self.moe_loss_fn is not None:
                    moe_loss, moe_loss_dict = self.moe_loss_fn(data, self.prev_data)
                    loss = loss + moe_loss
                    
                    self.prev_data = {
                        'lmain': {'router_weights': data['lmain'].get('router_weights', None)},
                        'rmain': {'router_weights': data['rmain'].get('router_weights', None)}
                    }
                    if self.prev_data['lmain']['router_weights'] is not None:
                        self.prev_data['lmain']['router_weights'] = self.prev_data['lmain']['router_weights'].detach()
                    if self.prev_data['rmain']['router_weights'] is not None:
                        self.prev_data['rmain']['router_weights'] = self.prev_data['rmain']['router_weights'].detach()

                # 反向传播（accelerate自动处理缩放）
                self.accelerator.backward(loss)
                
                # 梯度裁剪
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                
                self.optimizer.step()
                self.scheduler.step()

            # 累积日志（仅在主进程）
            if self.accelerator.is_main_process:
                log_l1 += 0.8 * Ll1.item()
                log_lssim += 0.2 * Lssim.item()
                log_moe += moe_loss.item() if isinstance(moe_loss, torch.Tensor) else moe_loss

                if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                    self.logger.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                    self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
                    
                    if self.use_moe and moe_loss_dict:
                        for key, value in moe_loss_dict.items():
                            self.logger.writer.add_scalar(f'moe/{key}', value, self.total_steps)
                    
                    # 记录课程学习指标
                    if self.use_transformer_moe:
                        if 'curriculum_bg_prob' in metrics:
                            self.logger.writer.add_scalar(
                                'curriculum/bg_signal_prob', 
                                metrics['curriculum_bg_prob'], 
                                self.total_steps
                            )
                        if 'bg_update_signal' in data:
                            self.logger.writer.add_scalar(
                                'curriculum/bg_update_signal',
                                float(data['bg_update_signal']),
                                self.total_steps
                            )
                            
                metrics.update({
                    'l1': Ll1.item(),
                    'L_ssim': Lssim.item(),
                })
                if self.use_moe:
                    metrics['moe_loss'] = moe_loss.item() if isinstance(moe_loss, torch.Tensor) else moe_loss
                self.logger.push(metrics)

            # 验证
            if self.total_steps and self.total_steps % self.cfg.record.eval_freq == 0:
                self.model.eval()
                self.run_eval()
                self.model.train()
                self._freeze_bn()
                
            # 保存checkpoint
            if self.total_steps % self.cfg.record.save_iter == 0 and self.total_steps > 0:
                if self.accelerator.is_main_process:
                    self.save_ckpt(save_path=Path('%s/iter%d.pth' % (cfg.record.ckpt_path, self.total_steps)))

            self.total_steps += 1
            
            # 打印日志
            if self.accelerator.is_main_process and self.total_steps % 100 == 99:
                progress_bar.set_postfix({
                    'l1': log_l1 / 100,
                    'ssim': log_lssim / 100,
                    'moe': log_moe / 100 if self.use_moe else 0,
                })
                log_l1 = 0
                log_lssim = 0
                log_moe = 0

        # 训练完成
        if self.accelerator.is_main_process:
            logging.info("FINISHED TRAINING")
            self.logger.close()
            self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_eval(self):
        if self.accelerator.is_main_process:
            logging.info(f"Doing validation ...")
        
        torch.cuda.empty_cache()
        psnr_list = []
        show_idx = np.random.choice(list(range(self.len_val)), 1)
 
        for idx in range(self.len_val):
            data = self.fetch_data(phase='val')
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].to(self.accelerator.device)
                psnr_value = psnr(render_novel, gt_novel).mean().double()
                psnr_list.append(psnr_value.item())

                if self.accelerator.is_main_process and idx == show_idx:
                    tmp_novel = data['novel_view']['img_pred'][0].detach()
                    tmp_novel *= 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    tmp_img_name = '%s/%s.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

        # 汇总所有进程的PSNR
        psnr_tensor = torch.tensor(psnr_list, device=self.accelerator.device)
        gathered_psnr = self.accelerator.gather(psnr_tensor)
        
        if self.accelerator.is_main_process:
            val_psnr = np.round(gathered_psnr.cpu().numpy().mean(), 4)
            if val_psnr < 10:
                logging.warning('something wrong during training, please change random seed and re-train')
            logging.info(f"Validation Metrics ({self.total_steps}): psnr {val_psnr}")
            self.logger.write_dict({'val_psnr': val_psnr}, write_step=self.total_steps)
        
        torch.cuda.empty_cache()

    def fetch_data(self, phase):
        if phase == 'train':
            try:
                data = next(self.train_iterator)
            except StopIteration:
                self.train_iterator = iter(self.train_loader)
                data = next(self.train_iterator)
        elif phase == 'val':
            try:
                data = next(self.val_iterator)
            except StopIteration:
                self.val_iterator = iter(self.val_loader)
                data = next(self.val_iterator)

        # 数据已经通过accelerate自动移动到正确的设备
        for view in ['lmain', 'rmain']:
            for item in data[view].keys():
                if isinstance(data[view][item], torch.Tensor):
                    data[view][item] = data[view][item].to(self.accelerator.device)
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        if not os.path.exists(load_path):
            if self.accelerator.is_main_process:
                logging.warning(f"Checkpoint not found: {load_path}")
            return
            
        if self.accelerator.is_main_process:
            logging.info(f"Loading checkpoint from {load_path} ...")
        
        # 使用accelerator加载checkpoint
        ckpt = torch.load(load_path, map_location=self.accelerator.device, weights_only=False)
        
        # 获取未包装的模型
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(ckpt['network'], strict=strict)
        
        if self.accelerator.is_main_process:
            logging.info(f"Parameter loading done")
        
        if load_optimizer and 'optimizer' in ckpt:
            self.total_steps = ckpt['total_steps'] + 1
            if self.logger is not None:
                self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            if 'scheduler' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler'])
            if self.accelerator.is_main_process:
                logging.info(f"Optimizer loading done, resuming from step {self.total_steps}")

    def save_ckpt(self, save_path, show_log=True):
        # 只在主进程保存
        if not self.accelerator.is_main_process:
            return
            
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        
        # 获取未包装的模型状态
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # 等待所有进程同步
        self.accelerator.wait_for_everyone()
        
        torch.save({
            'total_steps': self.total_steps,
            'network': unwrapped_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description='GPS_plus Accelerate Training')
    parser.add_argument('--config', type=str, default='config/stage.yaml',
                        help='Path to config file')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Number of gradient accumulation steps')
    parser.add_argument('--mixed_precision', type=str, default=None,
                        choices=['no', 'fp16', 'bf16', None],
                        help='Mixed precision training mode')
    parser.add_argument('--seed', type=int, default=1314,
                        help='Random seed')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    )

    # 设置随机种子
    set_seed(args.seed)
    
    # DDP配置：允许未使用的参数（对于MoE模式很重要）
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    
    # 创建Accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        kwargs_handlers=[ddp_kwargs]
    )

    # 加载配置
    cfg = config()
    cfg.load(args.config)
    cfg = cfg.get_cfg()

    cfg.defrost()
    dt = datetime.today()
    
    # 实验名称
    depth_mode = getattr(cfg, 'depth_mode', 'raft')
    moe_cfg = getattr(cfg, 'moe', None)
    use_moe = moe_cfg is not None and getattr(moe_cfg, 'enabled', False)
    moe_suffix = '_moe' if use_moe else ''
    
    # 添加多卡标识
    num_gpus = accelerator.num_processes
    multi_gpu_suffix = f'_{num_gpus}gpu' if num_gpus > 1 else ''
    
    # 实验名称格式: {name}_{depth_mode}{moe_suffix}{multi_gpu_suffix}_{月日_时分秒}
    timestamp = dt.strftime('%m%d_%H%M%S')
    cfg.exp_name = '%s_%s%s%s_%s' % (
        cfg.name, depth_mode, moe_suffix, multi_gpu_suffix, timestamp
    )
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()

    # 只在主进程创建目录和备份文件
    if accelerator.is_main_process:
        for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
            Path(path).mkdir(exist_ok=True, parents=True)
        
        file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))
        
        logging.info(f"="*60)
        logging.info(f"GPS_plus Accelerate Training")
        logging.info(f"="*60)
        logging.info(f"Number of GPUs: {accelerator.num_processes}")
        logging.info(f"Mixed Precision: {accelerator.mixed_precision}")
        logging.info(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
        logging.info(f"Experiment: {cfg.exp_name}")
        logging.info(f"="*60)

    # 等待主进程创建完目录
    accelerator.wait_for_everyone()

    # 创建训练器并开始训练
    trainer = AccelerateTrainer(cfg, accelerator)
    trainer.train()
