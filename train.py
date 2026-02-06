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
from lib.network import RtStereoHumanModel, create_model, DAV3_AVAILABLE
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr
from lib.visualization import create_visualization_grid
from pytorch3d.loss import chamfer_distance

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
    def __init__(self, cfg_file, use_dav3=False):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size
        self.use_dav3 = use_dav3

        # Create model (either RAFT-based or DAV3-based)
        if use_dav3:
            if not DAV3_AVAILABLE:
                raise ImportError("DAV3 model requested but not available. Check Depth-Anything-3 installation.")
            logging.info("Using DAV3-based depth estimator")
            self.model = create_model(self.cfg, with_gs_render=True, use_dav3=True)
        else:
            logging.info("Using RAFT-Stereo depth estimator")
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
        # Freeze BatchNorm for RAFT-Stereo (only if not using DAV3)
        if not self.use_dav3 and hasattr(self.model, 'raft_stereo'):
            self.model.raft_stereo.freeze_bn()
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)

    def train(self):
        log_l1 = 0
        log_ssim = 0
        log_chamfer = 0
        log_scale = 0
        log_xyz_res = 0
        if_chamfer = True   
        if_scale = False  
        iter_from = -1 
        for itr_ in tqdm(range(self.total_steps, self.cfg.num_steps)):
            self.optimizer.zero_grad()
            data = self.fetch_data(phase='train')

            #  Raft Stereo
            data, _, metrics = self.model(data, is_train=True)
            #  Gaussian Render
            data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

            # Loss
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
                    
                    sample_l = np.random.choice(l_xyz_i.shape[1], 100000, replace = False)
                    sample_r = np.random.choice(r_xyz_i.shape[1], 100000, replace = False)
                    chamfer_loss_i, _ = chamfer_distance(l_xyz_i[:, sample_l], r_xyz_i[:, sample_r])
                    chamfer_loss += chamfer_loss_i
                
                chamfer_loss /= self.bs

            Ll1 = l1_loss(render_novel, gt_novel)
            Lssim = 1.0 - ssim(render_novel, gt_novel)
            
            # xyz 残差正则化损失 - 防止点云漂移
            xyz_res_regular = data['novel_view'].get('xyz_res_regular', torch.tensor(0.0).cuda())
            xyz_res_loss = 10.0 * xyz_res_regular  # 权重 10.0
            
            loss = 0.8 * Ll1 + 0.2 * Lssim + 2.0 * chamfer_loss + xyz_res_loss

            log_l1 += 0.8 * Ll1.item()
            log_ssim += 0.2 * Lssim.item()
            log_chamfer += 0.5 * chamfer_loss.item() if if_chamfer and itr_>iter_from else 0 
            log_scale += 0.5 * data['novel_view']['scale_regular'].item() if if_scale else 0
            log_xyz_res += xyz_res_regular.item() if isinstance(xyz_res_regular, torch.Tensor) else xyz_res_regular

            if self.total_steps and self.total_steps % self.cfg.record.loss_freq == 0:
                self.logger.writer.add_scalar(f'lr', self.optimizer.param_groups[0]['lr'], self.total_steps)
                self.save_ckpt(save_path=Path('%s/%s_latest.pth' % (cfg.record.ckpt_path, cfg.name)), show_log=False)
            metrics.update({
                'l1': Ll1.item(),
                'ssim': Lssim.item(),
                'chamfer': 0.5*chamfer_loss.item() if if_chamfer and itr_>iter_from else 0,
                'xyz_res': xyz_res_regular.item() if isinstance(xyz_res_regular, torch.Tensor) else xyz_res_regular
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
                # Freeze BatchNorm for RAFT-Stereo (only if not using DAV3)
                if not self.use_dav3 and hasattr(self.model, 'raft_stereo'):
                    self.model.raft_stereo.freeze_bn()
                
            if self.total_steps in self.cfg.record.save_iter:
                self.save_ckpt(save_path=Path('%s/iter%d.pth' % (cfg.record.ckpt_path, self.total_steps)))


            self.total_steps += 1
            if self.total_steps % 100 == 99:
                print(
                    'l1 ', log_l1/ 100,
                    'ssim', log_ssim/100,
                    'chamfer', log_chamfer/100,
                    'scale', log_scale/100,
                    'xyz_res', log_xyz_res/100,
                    )
                log_l1 = 0
                log_ssim = 0
                log_chamfer = 0
                log_scale = 0
                log_xyz_res = 0

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
                data, _, metrics = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                render_novel = data['novel_view']['img_pred']
                gt_novel = data['novel_view']['img'].cuda()
                psnr_value = psnr(render_novel, gt_novel).mean().double()
                psnr_list.append(psnr_value.item())

                if idx == show_idx:
                    # Save original novel view image
                    tmp_novel = data['novel_view']['img_pred'][0].detach()
                    tmp_novel *= 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    tmp_img_name = '%s/%s.jpg' % (cfg.record.show_path, self.total_steps)
                    cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))
                    
                    # Create comprehensive visualization grid
                    try:
                        vis_paths = create_visualization_grid(
                            data, 
                            self.total_steps, 
                            cfg.record.show_path,
                            prefix=""
                        )
                        logging.info(f"Saved visualizations to {cfg.record.show_path}")
                    except Exception as e:
                        logging.warning(f"Visualization failed: {e}")

        val_psnr = np.round(np.mean(np.array(psnr_list)), 4)
        # if val_psnr < 10:
        #     print('something wrong during training, please change random seed and re-train')
        #     exit()

            
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
        ckpt = torch.load(load_path, map_location='cuda')
        
        # Try strict loading first, fall back to non-strict if needed
        try:
            self.model.load_state_dict(ckpt['network'], strict=strict)
            logging.info(f"Parameter loading done (strict={strict})")
        except RuntimeError as e:
            if strict:
                logging.warning(f"Strict loading failed: {e}")
                logging.info("Retrying with strict=False...")
                
                # Load with strict=False and report missing/unexpected keys
                model_state = self.model.state_dict()
                ckpt_state = ckpt['network']
                
                # Find missing and unexpected keys
                missing_keys = set(model_state.keys()) - set(ckpt_state.keys())
                unexpected_keys = set(ckpt_state.keys()) - set(model_state.keys())
                
                if missing_keys:
                    logging.warning(f"Missing keys ({len(missing_keys)}): {list(missing_keys)[:5]}...")
                if unexpected_keys:
                    logging.warning(f"Unexpected keys ({len(unexpected_keys)}): {list(unexpected_keys)[:5]}...")
                
                # Filter out unexpected keys and load
                filtered_state = {k: v for k, v in ckpt_state.items() if k in model_state}
                self.model.load_state_dict(filtered_state, strict=False)
                logging.info(f"Parameter loading done (strict=False, loaded {len(filtered_state)}/{len(model_state)} keys)")
            else:
                raise e
        
        if load_optimizer:
            self.total_steps = ckpt['total_steps'] + 1
            self.logger.total_steps = self.total_steps
            try:
                self.optimizer.load_state_dict(ckpt['optimizer'])
                self.scheduler.load_state_dict(ckpt['scheduler'])
                logging.info(f"Optimizer loading done, resuming from step {self.total_steps}")
            except Exception as e:
                logging.warning(f"Failed to load optimizer state: {e}")
                logging.warning("Optimizer will start fresh, but training will resume from the saved step")

    def save_ckpt(self, save_path, show_log=True):
        if show_log:
            logging.info(f"Save checkpoint to {save_path} ...")
        torch.save({
            'total_steps': self.total_steps,
            'network': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }, save_path)


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint in the directory."""
    if not os.path.exists(ckpt_dir):
        return None
    
    ckpt_files = []
    for f in os.listdir(ckpt_dir):
        if f.endswith('.pth'):
            ckpt_files.append(os.path.join(ckpt_dir, f))
    
    if not ckpt_files:
        return None
    
    # Prefer *_latest.pth if exists
    for f in ckpt_files:
        if '_latest.pth' in f:
            return f
    
    # Otherwise return the most recently modified checkpoint
    ckpt_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return ckpt_files[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train GPS+ model')
    parser.add_argument('--use_dav3', type=bool,
                        default=True,
                        help='Use Depth Anything V3 instead of RAFT-Stereo')
    parser.add_argument('--config', type=str, default='config/stage.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from. Use "auto" to find latest checkpoint, '
                             'or provide explicit path to .pth file')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name (required when resume="auto" to locate checkpoint)')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg = config()
    cfg.load(args.config)
    cfg = cfg.get_cfg()

    cfg.defrost()
    
    # Determine experiment name
    if args.exp_name:
        # Use provided experiment name (for resume)
        cfg.exp_name = args.exp_name
    elif args.resume and args.resume != 'auto' and os.path.exists(args.resume):
        # Extract exp_name from checkpoint path: experiments/{exp_name}/ckpt/*.pth
        try:
            ckpt_path = os.path.abspath(args.resume)
            exp_name = ckpt_path.split('/experiments/')[-1].split('/')[0]
            cfg.exp_name = exp_name
            logging.info(f"Extracted experiment name from checkpoint path: {exp_name}")
        except:
            # Fallback to default naming
            dt = datetime.today()
            model_suffix = '_dav3' if args.use_dav3 else ''
            cfg.exp_name = '%s%s_%s%s' % (cfg.name, model_suffix, str(dt.month).zfill(2), str(dt.day).zfill(2))
    else:
        # Create new experiment name
        dt = datetime.today()
        model_suffix = '_dav3' if args.use_dav3 else ''
        cfg.exp_name = '%s%s_%s%s' % (cfg.name, model_suffix, str(dt.month).zfill(2), str(dt.day).zfill(2))
    
    cfg.record.ckpt_path = "experiments/%s/ckpt" % cfg.exp_name
    cfg.record.show_path = "experiments/%s/show" % cfg.exp_name
    cfg.record.logs_path = "experiments/%s/logs" % cfg.exp_name
    cfg.record.file_path = "experiments/%s/file" % cfg.exp_name
    
    # Handle resume logic
    if args.resume:
        if args.resume == 'auto':
            # Auto-find latest checkpoint
            if not args.exp_name:
                logging.error("--exp_name is required when using --resume auto")
                exit(1)
            latest_ckpt = find_latest_checkpoint(cfg.record.ckpt_path)
            if latest_ckpt:
                cfg.restore_ckpt = latest_ckpt
                logging.info(f"Auto-detected checkpoint: {latest_ckpt}")
            else:
                logging.warning(f"No checkpoint found in {cfg.record.ckpt_path}, starting from scratch")
                cfg.restore_ckpt = None
        else:
            # Use provided checkpoint path
            if os.path.exists(args.resume):
                cfg.restore_ckpt = args.resume
                logging.info(f"Using provided checkpoint: {args.resume}")
            else:
                logging.error(f"Checkpoint not found: {args.resume}")
                exit(1)
    else:
        cfg.restore_ckpt = getattr(cfg, 'restore_ckpt', None)
    
    cfg.freeze()


    for path in [cfg.record.ckpt_path, cfg.record.show_path, cfg.record.logs_path, cfg.record.file_path]:
        Path(path).mkdir(exist_ok=True, parents=True)
    
    file_backup(cfg.record.file_path, cfg, train_script=os.path.basename(__file__))

    torch.manual_seed(1314)
    np.random.seed(1314)

    trainer = Trainer(cfg, use_dav3=args.use_dav3)
    trainer.train()
