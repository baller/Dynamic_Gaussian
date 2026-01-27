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
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr
from lib.loss import CombinedLoss, DepthConsistencyLoss
from lib.chamfer_distance import chamfer_distance
from lib.visualization import TrainingVisualizer

import trimesh 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
import warnings
from copy import deepcopy
warnings.filterwarnings("ignore", category=UserWarning)

# Accelerate 支持
try:
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    ACCELERATE_AVAILABLE = True
except ImportError:
    ACCELERATE_AVAILABLE = False
    print("Warning: accelerate not installed, using single GPU mode")

class Trainer:
    def __init__(self, cfg_file, use_accelerate: bool = True):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size
        self.use_accelerate = use_accelerate and ACCELERATE_AVAILABLE
        
        # 初始化 Accelerator
        if self.use_accelerate:
            self.accelerator = Accelerator(
                mixed_precision='fp16' if self.cfg.raft.mixed_precision else 'no',
                gradient_accumulation_steps=1,
            )
            self.is_main_process = self.accelerator.is_main_process
            self.device = self.accelerator.device
            if self.is_main_process:
                print(f"Using Accelerate with {self.accelerator.num_processes} GPUs")
        else:
            self.accelerator = None
            self.is_main_process = True
            self.device = torch.device('cuda')

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

        # 只在主进程创建 logger
        if self.is_main_process:
            self.logger = Logger(self.scheduler, cfg.record)
        else:
            self.logger = None
        self.total_steps = 0
        
        # 获取深度模式
        self.depth_mode = getattr(self.cfg, 'depth_mode', 'raft')

        # 使用 Accelerator 准备模型和优化器
        if self.use_accelerate:
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = \
                self.accelerator.prepare(
                    self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
                )
            self.train_iterator = iter(self.train_loader)
            self.val_iterator = iter(self.val_loader)
        else:
            self.model.cuda()
        
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.stage1_ckpt:
            if self.is_main_process:
                logging.info(f"Using checkpoint from stage1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        self.model.train()
        self._freeze_bn()  # 冻结 BatchNorm
        
        # 不使用 accelerate 时才需要 GradScaler
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision) if not self.use_accelerate else None
        
        # 初始化组合损失函数
        self.combined_loss = CombinedLoss(self.cfg) if self.depth_mode == 'da3' else None
        self.depth_consistency_loss = DepthConsistencyLoss() if self.depth_mode == 'da3' else None
        
        # 从配置获取损失权重
        loss_cfg = getattr(self.cfg, 'loss', None)
        if loss_cfg is not None:
            self.l1_weight = getattr(loss_cfg, 'l1_weight', 0.8)
            self.ssim_weight = getattr(loss_cfg, 'ssim_weight', 0.2)
            self.chamfer_weight = getattr(loss_cfg, 'chamfer_weight', 0.5)
            self.depth_consistency_weight = getattr(loss_cfg, 'depth_consistency_weight', 0.1)
            self.moe_balance_weight = getattr(loss_cfg, 'moe_balance_weight', 0.01)
        else:
            self.l1_weight = 0.8
            self.ssim_weight = 0.2
            self.chamfer_weight = 0.5
            self.depth_consistency_weight = 0.1
            self.moe_balance_weight = 0.01
        
        # 初始化可视化器 (仅主进程)
        if self.is_main_process:
            vis_dir = Path(cfg.record.show_path) / 'visualizations'
            self.visualizer = TrainingVisualizer(self.cfg, str(vis_dir))
            logging.info(f"[Visualizer] 可视化输出目录: {vis_dir}")
            logging.info(f"[Visualizer] 可视化频率: 每 {self.visualizer.vis_freq} 步")
        else:
            self.visualizer = None
    
    def _freeze_bn(self):
        """根据深度模式冻结相应的 BatchNorm 层"""
        if self.depth_mode == 'da3':
            # DA3 模式: 冻结 DA3 backbone 和 feature adapter 的 BN
            if hasattr(self.model, 'da3_estimator') and self.model.da3_estimator.model is not None:
                for module in self.model.da3_estimator.model.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        module.eval()
            if hasattr(self.model, 'feature_adapter') and self.model.feature_adapter is not None:
                for module in self.model.feature_adapter.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                        module.eval()
        else:
            # RAFT 模式: 冻结 RAFT-Stereo 的 BN
            self.model.raft_stereo.freeze_bn()

    def train(self):
        log_l1 = 0
        log_ssim = 0
        log_chamfer = 0
        log_scale = 0
        log_depth_cons = 0
        log_moe_balance = 0
        if_chamfer = True   
        if_scale = False  
        iter_from = -1 
        for itr_ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            #  Raft Stereo / DA3 前向传播
            data, _, metrics = self.model(data, is_train=True)
            #  Gaussian Render
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

            # Loss
            render_novel = data['novel_view']['img_pred']
            gt_novel = data['novel_view']['img'].to(self.device)

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
                    
                    # 处理点数不足的情况
                    n_l = l_xyz_i.shape[1]
                    n_r = r_xyz_i.shape[1]
                    sample_size = min(10000, n_l, n_r)
                    
                    if sample_size > 0:
                        sample_l = np.random.choice(n_l, sample_size, replace=(n_l < sample_size))
                        sample_r = np.random.choice(n_r, sample_size, replace=(n_r < sample_size))
                        chamfer_loss_i, _ = chamfer_distance(l_xyz_i[:, sample_l], r_xyz_i[:, sample_r])
                        chamfer_loss += chamfer_loss_i
                
                chamfer_loss /= self.bs

            Ll1 = l1_loss(render_novel, gt_novel)
            Lssim = 1.0 - ssim(render_novel, gt_novel)
            
            # 基础损失
            loss = self.l1_weight * Ll1 + self.ssim_weight * Lssim + self.chamfer_weight * chamfer_loss 

            # DA3 模式的额外损失
            depth_cons_loss = torch.tensor(0.0, device=loss.device)
            moe_balance_loss = torch.tensor(0.0, device=loss.device)
            
            if self.depth_mode == 'da3':
                # 深度一致性损失
                if self.depth_consistency_loss is not None:
                    # 优先使用融合后的深度，否则使用原始深度
                    if 'fusion_aux' in data:
                        aux = data['fusion_aux']
                        depth_l = aux.get('depth_l_metric')
                        depth_r = aux.get('depth_r_metric')
                    else:
                        # 使用原始 DA3 深度
                        depth_l = data['lmain'].get('depth')
                        depth_r = data['rmain'].get('depth')
                    
                    if depth_l is not None and depth_r is not None:
                        baseline = self._compute_baseline(data['lmain']['extr'], data['rmain']['extr'])
                        depth_cons_loss = self.depth_consistency_loss(
                            depth_l, depth_r, data['lmain']['intr'], baseline
                        )
                        loss = loss + self.depth_consistency_weight * depth_cons_loss
                
                # MoE 负载均衡损失 (如果使用 Transformer+MoE)
                # 获取模型 (处理 accelerate 包装)
                model = self.accelerator.unwrap_model(self.model) if self.use_accelerate else self.model
                if hasattr(model, 'gs_parm_regresser'):
                    regresser = model.gs_parm_regresser
                    # 检查是否有存储的 MoE 损失
                    if hasattr(regresser, 'last_moe_balance_loss'):
                        lb_loss = regresser.last_moe_balance_loss
                        if lb_loss is not None and isinstance(lb_loss, torch.Tensor):
                            moe_balance_loss = lb_loss
                        elif lb_loss is not None and lb_loss > 0:
                            moe_balance_loss = torch.tensor(lb_loss, device=loss.device)
                        loss = loss + self.moe_balance_weight * moe_balance_loss

            log_l1 += self.l1_weight * Ll1.item()
            log_ssim += self.ssim_weight * Lssim.item()
            log_chamfer += self.chamfer_weight * chamfer_loss.item() if if_chamfer and itr_>iter_from else 0 
            log_scale += 0.5 * data['novel_view']['scale_regular'].item() if if_scale else 0
            log_depth_cons += depth_cons_loss.item() if self.depth_mode == 'da3' else 0
            log_moe_balance += moe_balance_loss.item() if self.depth_mode == 'da3' else 0

            if self.is_main_process and self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
            
            # 更新指标
            metrics.update({
                'l1': Ll1.item(),
                'ssim': Lssim.item(),
                'chamfer': self.chamfer_weight * chamfer_loss.item() if if_chamfer and itr_ > iter_from else 0
            })
            
            # DA3 模式的额外指标
            if self.depth_mode == 'da3':
                metrics['depth_consistency'] = depth_cons_loss.item()
                metrics['moe_balance'] = moe_balance_loss.item()
                
                # 记录融合模块的 scale 和 shift
                if 'fusion_aux' in data:
                    aux = data['fusion_aux']
                    if 'scale_l' in aux:
                        metrics['fusion_scale'] = aux['scale_l'].mean().item()
                    if 'shift_l' in aux:
                        metrics['fusion_shift'] = aux['shift_l'].mean().item()
            
            if self.is_main_process and self.logger is not None:
                self.logger.push(metrics)
            
            # 可视化中间结果 (仅主进程)
            if self.is_main_process and self.visualizer is not None:
                self.visualizer.visualize(data, self.total_steps, phase='train')

            # 反向传播 - accelerate 或 单卡模式
            if self.use_accelerate:
                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
            else:
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
                self._freeze_bn()  # 冻结 BatchNorm
                
            if self.is_main_process and self.total_steps % self.cfg.record.save_iter == 0 and self.total_steps != 0:
                self.save_ckpt(save_path=Path('%s/iter%d.pth' % (cfg.record.ckpt_path, self.total_steps)))


            self.total_steps += 1
            if self.is_main_process and self.total_steps % 100 == 99:
                print_items = [
                    f'l1: {log_l1/100:.4f}',
                    f'ssim: {log_ssim/100:.4f}',
                    f'chamfer: {log_chamfer/100:.4f}',
                    f'scale: {log_scale/100:.4f}',
                ]
                if self.depth_mode == 'da3':
                    print_items.extend([
                        f'depth_cons: {log_depth_cons/100:.6f}',
                        f'moe_bal: {log_moe_balance/100:.6f}',
                    ])
                print(' | '.join(print_items))
                
                log_l1 = 0
                log_ssim = 0
                log_chamfer = 0
                log_scale = 0
                log_depth_cons = 0
                log_moe_balance = 0

        if self.is_main_process:
            print("FINISHED TRAINING")
            if self.logger is not None:
                self.logger.close()
            self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))
    
    def _compute_baseline(self, extr_l: torch.Tensor, extr_r: torch.Tensor) -> torch.Tensor:
        """计算立体基线距离"""
        if extr_l.shape[1] == 4:
            t_l = extr_l[:, :3, 3]
            t_r = extr_r[:, :3, 3]
        else:
            t_l = extr_l[:, :, 3]
            t_r = extr_r[:, :, 3]
        return torch.norm(t_l - t_r, dim=1)

    def run_eval(self):
        # 只在主进程执行验证
        if not self.is_main_process:
            return
            
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
                gt_novel = data['novel_view']['img'].to(self.device)
                psnr_value = psnr(render_novel, gt_novel).mean().double()
                psnr_list.append(psnr_value.item())

                if idx == show_idx:
                    tmp_novel = data['novel_view']['img_pred'][0].detach()
                    tmp_novel *= 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    tmp_img_name = '%s/%s.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))
                    
                    # 验证阶段可视化
                    if self.visualizer is not None:
                        self.visualizer.visualize(data, self.total_steps, phase='val')

        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        if val_psnr < 10:
            print('something wrong during training, please change random seed and re-train')
            exit()

            
        logging.info(f"Validation Metrics ({self.total_steps}): psnr {val_psnr}")
        if self.logger is not None:
            self.logger.write_dict({'val_psnr': val_psnr}, write_step=self.total_steps)
        torch.cuda.empty_cache()

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
                data[view][item] = data[view][item].to(self.device)
        return data

    def load_ckpt(self, load_path, load_optimizer=True, strict=True):
        assert os.path.exists(load_path)
        if self.is_main_process:
            logging.info(f"Loading checkpoint from {load_path} ...")
        ckpt = torch.load(load_path, map_location='cuda')
        
        # 获取原始模型 (accelerate 包装后需要 unwrap)
        model = self.accelerator.unwrap_model(self.model) if self.use_accelerate else self.model
        model.load_state_dict(ckpt['network'], strict=strict)
        
        if self.is_main_process:
            logging.info(f"Parameter loading done")
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            if self.logger is not None:
                self.logger.total_steps = self.total_steps
            self.optimizer.load_state_dict(ckpt['optimizer'])
            self.scheduler.load_state_dict(ckpt['scheduler'])
            if self.is_main_process:
                logging.info(f"Optimizer loading done")

    def save_ckpt(self, save_path, show_log=True):
        # 只在主进程保存
        if not self.is_main_process:
            return
            
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        
        # 获取原始模型 (accelerate 包装后需要 unwrap)
        model = self.accelerator.unwrap_model(self.model) if self.use_accelerate else self.model
        
        torch.save({
            'total_steps': self.total_steps,
            'network': model.state_dict(),
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
    cfg.exp_name = '%s_%s%s' % (cfg.name, str(dt.month).zfill(2), str(dt.day).zfill(2))
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    cfg.freeze()

    # 创建目录 (所有进程都需要)
    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)
    
    # 文件备份只在主进程执行
    if not ACCELERATE_AVAILABLE or int(os.environ.get('LOCAL_RANK', 0)) == 0:
        file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg)
    trainer.train()
