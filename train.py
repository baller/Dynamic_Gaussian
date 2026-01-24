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
from lib.GaussianRender import pts2render, pts2render_moe
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr
from lib.loss import MoELoss, create_moe_loss
# from pytorch3d.loss import chamfer_distance

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
        
        # MoE配置
        self.moe_cfg = getattr(self.cfg, 'moe', None)
        self.use_moe = self.moe_cfg is not None and getattr(self.moe_cfg, 'enabled', False)
        
        logging.info(f"深度估计模式: {self.depth_mode}")
        if self.use_moe:
            logging.info(f"MoE模式已启用")

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
        
        # MoE损失函数
        if self.use_moe:
            self.moe_loss_fn = create_moe_loss(self.cfg)
            self.prev_data = None  # 用于时序一致性损失
            
            # 渐进式训练配置
            training_cfg = getattr(self.moe_cfg, 'training', None)
            if training_cfg is not None:
                self.freeze_router_steps = getattr(training_cfg, 'freeze_router_epochs', 5) * len(self.train_loader)
                self.progressive_training = getattr(training_cfg, 'progressive', True)
            else:
                self.freeze_router_steps = 5 * len(self.train_loader)
                self.progressive_training = True
            
            logging.info(f"MoE渐进式训练: 前{self.freeze_router_steps}步冻结路由器")
        else:
            self.moe_loss_fn = None
            self.prev_data = None

        self.model.cuda()
        if self.cfg.restore_ckpt:
            self.load_ckpt(self.cfg.restore_ckpt)
        elif self.cfg.stage1_ckpt:
            logging.info(f"Using checkpoint from stage1")
            self.load_ckpt(self.cfg.stage1_ckpt, load_optimizer=False, strict=False)
        self.model.train()
        
        # 根据深度模式冻结BN
        self._freeze_bn()
        
        # MoE渐进式训练：初始阶段冻结路由器
        if self.use_moe and self.progressive_training:
            self.model.freeze_router()
        
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)

    def _freeze_bn(self):
        """根据深度模式冻结BatchNorm层"""
        if self.depth_mode == 'raft':
            # RAFT模式：冻结RAFT-Stereo的BN
            if hasattr(self.model, 'raft_stereo') and self.model.raft_stereo is not None:
                self.model.raft_stereo.freeze_bn()
                logging.info("已冻结RAFT-Stereo的BatchNorm层")
        elif self.depth_mode == 'da3':
            # DA3模式：冻结DA3的BN（如果不微调的话）
            if hasattr(self.model, 'depth_model') and self.model.depth_model is not None:
                self.model.depth_model.freeze_bn()
                logging.info("已冻结DA3的BatchNorm层")

    def train(self):
        log_l1 = 0
        log_ssim = 0
        log_chamfer = 0
        log_scale = 0
        log_moe = 0
        if_chamfer = False   
        if_scale = False  
        iter_from = -1
        router_unfrozen = False  # 跟踪路由器是否已解冻
        
        for itr_ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            # MoE渐进式训练：在指定步数后解冻路由器
            if self.use_moe and self.progressive_training and not router_unfrozen:
                if self.total_steps >= self.freeze_router_steps:
                    self.model.unfreeze_router()
                    router_unfrozen = True
                    logging.info(f"步骤 {self.total_steps}: 解冻MoE路由器，开始联合训练")
            
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            # 深度估计（RAFT-Stereo 或 DA3）
            data, _, metrics = self.model(data, is_train=True)
            
            # 高斯渲染（MoE模式使用专门的渲染函数）
            if self.use_moe and 'router_weights' in data.get('lmain', {}):
                data = pts2render_moe(data, bg_color=self.cfg.dataset.bg_color)
            else:
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
            
            # MoE损失
            moe_loss = 0
            moe_loss_dict = {}
            if self.use_moe and self.moe_loss_fn is not None:
                moe_loss, moe_loss_dict = self.moe_loss_fn(data, self.prev_data)
                loss = loss + moe_loss
                
                # 更新上一帧数据（用于时序一致性损失）
                self.prev_data = {
                    'lmain': {'router_weights': data['lmain'].get('router_weights', None)},
                    'rmain': {'router_weights': data['rmain'].get('router_weights', None)}
                }
                if self.prev_data['lmain']['router_weights'] is not None:
                    self.prev_data['lmain']['router_weights'] = self.prev_data['lmain']['router_weights'].detach()
                if self.prev_data['rmain']['router_weights'] is not None:
                    self.prev_data['rmain']['router_weights'] = self.prev_data['rmain']['router_weights'].detach()

            log_l1 += 0.8 * Ll1.item()
            log_ssim += 0.2 * Lssim.item()
            log_chamfer += 0.5 * chamfer_loss.item() if if_chamfer and itr_ > iter_from else 0 
            log_scale += 0.5 * data['novel_view']['scale_regular'].item() if if_scale else 0
            log_moe += moe_loss.item() if isinstance(moe_loss, torch.Tensor) else moe_loss

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
                
                # 记录MoE相关损失
                if self.use_moe and moe_loss_dict:
                    for key, value in moe_loss_dict.items():
                        self.logger.writer.add_scalar(f'moe/{key}', value, self.total_steps)
                        
            metrics.update({
                'l1': Ll1.item(),
                'ssim': Lssim.item(),
                'chamfer': 0.5 * chamfer_loss.item() if if_chamfer and itr_ > iter_from else 0
            })
            if self.use_moe:
                metrics['moe_loss'] = moe_loss.item() if isinstance(moe_loss, torch.Tensor) else moe_loss
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
                
            if self.total_steps % self.cfg.record.save_iter == 0 and self.total_steps > 0:
                self.save_ckpt(save_path=Path('%s/iter%d.pth' % (cfg.record.ckpt_path, self.total_steps)))

            self.total_steps += 1
            if self.total_steps % 100 == 99:
                print(
                    'l1 ', log_l1 / 100,
                    'ssim', log_ssim / 100,
                    'chamfer', log_chamfer / 100,
                    'scale', log_scale / 100,
                    'moe', log_moe / 100 if self.use_moe else 0,
                )
                log_l1 = 0
                log_ssim = 0
                log_chamfer = 0
                log_scale = 0
                log_moe = 0

        print("FINISHED TRAINING")
        self.logger.close()
        self.save_ckpt(save_path=Path('%s/%s_final.pth' % (cfg.record.ckpt_path, cfg.name)))

    def run_eval(self):
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
                gt_novel = data['novel_view']['img'].cuda()
                psnr_value = psnr(render_novel, gt_novel).mean().double()
                psnr_list.append(psnr_value.item())

                if idx == show_idx:
                    tmp_novel = data['novel_view']['img_pred'][0].detach()
                    tmp_novel *= 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    tmp_img_name = '%s/%s.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))

        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        if val_psnr < 10:
            print('something wrong during training, please change random seed and re-train')
            exit()

        logging.info(f"Validation Metrics ({self.total_steps}): psnr {val_psnr}")
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
    # 在实验名称中添加深度模式和MoE标识
    depth_mode = getattr(cfg, 'depth_mode', 'raft')
    moe_cfg = getattr(cfg, 'moe', None)
    use_moe = moe_cfg is not None and getattr(moe_cfg, 'enabled', False)
    moe_suffix = '_moe' if use_moe else ''
    cfg.exp_name = '%s_%s%s_%s%s' % (cfg.name, depth_mode, moe_suffix, str(dt.month).zfill(2), str(dt.day).zfill(2))
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
