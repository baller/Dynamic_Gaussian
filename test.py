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
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.train_recoder import Logger, file_backup
from lib.GaussianRender import pts2render
from lib.gs_utils.loss_utils import l1_loss, ssim
from lib.gs_utils.image_utils import psnr

from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

import warnings
import trimesh 
warnings.filterwarnings("ignore", category=UserWarning)


class Trainer:
    def __init__(self, cfg_file):
        self.cfg = cfg_file
        self.bs = self.cfg.batch_size
        self.depth_mode = getattr(self.cfg, 'depth_mode', 'raft')
        
        logging.info(f"深度估计模式: {self.depth_mode}")

        self.model = RtStereoHumanModel(self.cfg, with_gs_render=True)

        self.val_set = StereoHumanDataset(self.cfg.dataset, phase=cfg.seq_name)
        self.val_loader = DataLoader(self.val_set, batch_size=self.bs, shuffle=False, num_workers=8, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)  # real length of val set
        self.val_iterator = iter(self.val_loader)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.wdecay, eps=1e-8)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, self.cfg.lr, self.cfg.num_steps + 100,
                                                       pct_start=0.01, cycle_momentum=False, anneal_strategy='linear')

        self.logger = Logger(self.scheduler, cfg.record)
        self.total_steps = 0

        self.model.cuda()
        if self.cfg.restore_ckpt:
            print('load good ckpt')
            # DA3模式使用strict=False，因为DA3模型会单独从预训练权重加载
            strict = (self.depth_mode != 'da3')
            self.load_ckpt(self.cfg.restore_ckpt, load_optimizer=False, strict=strict)

        self.model.eval()
        
        # 根据深度模式冻结BN
        self._freeze_bn()
        
        self.scaler = GradScaler(enabled=self.cfg.raft.mixed_precision)

    def _freeze_bn(self):
        """根据深度模式冻结BatchNorm层"""
        if self.depth_mode == 'raft':
            # RAFT模式：冻结RAFT-Stereo的BN
            if hasattr(self.model, 'raft_stereo') and self.model.raft_stereo is not None:
                self.model.raft_stereo.freeze_bn()
                logging.info("已冻结RAFT-Stereo的BatchNorm层")
        elif self.depth_mode == 'da3':
            # DA3模式：冻结DA3的BN
            if hasattr(self.model, 'depth_model') and self.model.depth_model is not None:
                self.model.depth_model.freeze_bn()
                logging.info("已冻结DA3的BatchNorm层")

    def _detect_available_views(self, candidate_views=None):
        """
        检测数据集中可用的视角
        
        Args:
            candidate_views: 候选视角列表，None表示检测0-9的视角
            
        Returns:
            可用视角列表
        """
        if candidate_views is None:
            candidate_views = list(range(10))  # 检测0-9的视角
        
        seq_name = self.cfg.seq_name
        
        # 根据StereoHumanDataset的逻辑确定数据路径
        # 对于测试序列 (如 s1a6_process)，数据路径是 local_data_root/test/seq_name
        if seq_name not in ['train', 'val']:
            # 测试序列
            data_root = os.path.join(self.cfg.dataset.local_data_root, 'test', seq_name)
        elif seq_name == 'val':
            data_root = self.cfg.dataset.val_data_root
        else:
            data_root = self.cfg.dataset.train_data_root
        
        param_dir = os.path.join(data_root, 'parameter')
        
        # 获取第一个样本名称
        if os.path.exists(param_dir):
            sample_dirs = sorted(os.listdir(param_dir))
            if sample_dirs:
                sample_name = sample_dirs[0]
                sample_param_dir = os.path.join(param_dir, sample_name)
                
                available_views = []
                for view_id in candidate_views:
                    intr_file = os.path.join(sample_param_dir, f'{view_id}_intrinsic.npy')
                    if os.path.exists(intr_file):
                        available_views.append(view_id)
                
                if available_views:
                    logging.info(f"检测到可用视角: {available_views}")
                    return available_views
                else:
                    logging.warning(f"在 {sample_param_dir} 中未找到任何视角参数文件")
        else:
            logging.warning(f"参数目录不存在: {param_dir}")
        
        # 默认返回视角2和3（通常存在于StereoHuman数据集）
        logging.warning(f"无法自动检测视角，使用默认视角 [2, 3]")
        return [2, 3]

    def val(self):
        logging.info(f"Doing validation ...")
        torch.cuda.empty_cache()
        psnr_list = []
        for idx in tqdm(range(self.len_val)):
            data = self.fetch_data(phase='val')

            view_id = data['novel_view']['view_id'][0, 0].item()
            s_name = data['novel_view']['sample_name']
           
            with torch.no_grad():
                data, _, _ = self.model(data, is_train=False)
                data = pts2render(data, bg_color=self.cfg.dataset.bg_color)

                tmp_novel = data['novel_view']['img_pred'][0].detach()
                tmp_novel *= 255
                tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                tmp_img_name = '%s/%s_%02d.jpg' % (self.cfg.record.show_path, s_name[0], view_id)
                cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))
                
                # 保存深度图
                if 'depth' in data['lmain']:
                    # 从 (1, 1024, 1024) 形状的张量中获取深度图，并去除第一个维度，得到 (1024, 1024)
                    depth_lmain = data['lmain']['depth'][0].squeeze().detach().cpu().numpy()

                    # 检查以防止除以零的错误
                    min_depth = depth_lmain.min()
                    max_depth = depth_lmain.max()

                    if max_depth > min_depth:
                        # 归一化深度图到0-255范围，并转换为8位无符号整数类型
                        depth_normalized = ((depth_lmain - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
                    else:
                        # 如果深度图是平坦的（所有值相同），则创建一个全黑的图像
                        depth_normalized = np.zeros_like(depth_lmain, dtype=np.uint8)
                        
                    # 应用OpenCV的Colormap将灰度深度图转换为彩色深度图
                    colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)

                    # 构建彩色深度图的文件名
                    depth_img_name = '%s/%s_%02d_depth_lmain_color.png' % (self.cfg.record.show_path, s_name[0], view_id)
                    
                    # 保存彩色深度图
                    cv2.imwrite(depth_img_name, colored_depth_map)

    def val_dynamic(self, num_interp=30, views=None, loop=False):
        """
        动态视角测试 - 在多个视角之间进行平滑插值渲染
        
        对每一帧数据，生成从第一个视角到最后一个视角的平滑过渡
        
        Args:
            num_interp: 总共生成的视角数量（在所有关键视角之间均匀插值）
            views: 关键视角列表，None表示使用所有可用视角
            loop: 是否循环（从最后一个视角回到第一个视角）
        """
        logging.info(f"Doing smooth dynamic view validation ...")
        torch.cuda.empty_cache()
        
        if views is None:
            views = self._detect_available_views()
        else:
            # 过滤出实际可用的视角
            available = self._detect_available_views(views)
            views = [v for v in views if v in available]
            if len(views) < 2:
                logging.error("动态视角测试需要至少2个可用视角！")
                return
        
        logging.info(f"关键视角: {views}, 插值帧数: {num_interp}, 循环: {loop}")
        
        # 收集所有关键视角的相机参数
        view_cameras = {}
        for view_id in views:
            self.cfg.defrost()
            self.cfg.dataset.val_novel_id = [view_id]
            self.cfg.freeze()
            
            val_set_temp = StereoHumanDataset(self.cfg.dataset, phase=self.cfg.seq_name)
            val_loader_temp = DataLoader(val_set_temp, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
            
            for data in val_loader_temp:
                for view in ['lmain', 'rmain']:
                    for item in data[view].keys():
                        data[view][item] = data[view][item].cuda()
                
                # 保存相机参数 (使用实际的键名) - 确保都在CUDA上
                view_cameras[view_id] = {
                    'extr': data['novel_view']['extr'].clone().cuda(),
                    'world_view_transform': data['novel_view']['world_view_transform'].clone().cuda(),
                    'full_proj_transform': data['novel_view']['full_proj_transform'].clone().cuda(),
                    'camera_center': data['novel_view']['camera_center'].clone().cuda(),
                    'FovX': data['novel_view']['FovX'].clone() if isinstance(data['novel_view']['FovX'], torch.Tensor) else torch.tensor(data['novel_view']['FovX']),
                    'FovY': data['novel_view']['FovY'].clone() if isinstance(data['novel_view']['FovY'], torch.Tensor) else torch.tensor(data['novel_view']['FovY']),
                    'width': data['novel_view']['width'],
                    'height': data['novel_view']['height'],
                }
                break
            del val_set_temp, val_loader_temp
        
        logging.info(f"收集到 {len(view_cameras)} 个关键视角的相机参数")
        
        # 重新设置数据加载器（使用第一个视角）
        self.cfg.defrost()
        self.cfg.dataset.val_novel_id = [views[0]]
        self.cfg.freeze()
        self.val_set = StereoHumanDataset(self.cfg.dataset, phase=self.cfg.seq_name)
        self.val_loader = DataLoader(self.val_set, batch_size=self.bs, shuffle=False, num_workers=8, pin_memory=True)
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)
        self.val_iterator = iter(self.val_loader)
        
        # 计算插值路径
        # 如果loop=True: views[0] -> views[1] -> ... -> views[-1] -> views[0]
        # 如果loop=False: views[0] -> views[1] -> ... -> views[-1]
        num_segments = len(views) if loop else len(views) - 1
        frames_per_segment = num_interp // num_segments
        
        logging.info(f"总段数: {num_segments}, 每段帧数: {frames_per_segment}")
        
        # 对每一帧数据进行多视角渲染
        output_frame_idx = 0
        total_output = self.len_val * num_interp
        
        with tqdm(total=total_output, desc="平滑视角渲染") as pbar:
            for data_idx in range(self.len_val):
                data = self.fetch_data(phase='val')
                s_name = data['novel_view']['sample_name'][0]
                
                # 先进行一次模型前向传播，获取高斯参数
                with torch.no_grad():
                    data_with_gs, _, _ = self.model(data, is_train=False)
                
                # 对这一帧数据，生成所有插值视角
                for interp_idx in range(num_interp):
                    # 计算当前在哪个段以及段内位置
                    global_t = interp_idx / (num_interp - 1) if num_interp > 1 else 0
                    segment_idx = min(int(global_t * num_segments), num_segments - 1)
                    
                    # 段内位置 [0, 1)
                    segment_start_t = segment_idx / num_segments
                    segment_end_t = (segment_idx + 1) / num_segments
                    local_t = (global_t - segment_start_t) / (segment_end_t - segment_start_t) if segment_end_t > segment_start_t else 0
                    local_t = min(local_t, 1.0)
                    
                    # 获取当前段的起始和结束视角
                    view_start = views[segment_idx]
                    view_end = views[(segment_idx + 1) % len(views)]
                    
                    cam_start = view_cameras[view_start]
                    cam_end = view_cameras[view_end]
                    
                    # 线性插值相机参数
                    interp_world_view = (1 - local_t) * cam_start['world_view_transform'] + local_t * cam_end['world_view_transform']
                    interp_full_proj = (1 - local_t) * cam_start['full_proj_transform'] + local_t * cam_end['full_proj_transform']
                    interp_camera_center = (1 - local_t) * cam_start['camera_center'] + local_t * cam_end['camera_center']
                    interp_FovX = (1 - local_t) * cam_start['FovX'] + local_t * cam_end['FovX']
                    interp_FovY = (1 - local_t) * cam_start['FovY'] + local_t * cam_end['FovY']
                    
                    # 直接修改 data_with_gs 的 novel_view 参数
                    data_with_gs['novel_view']['world_view_transform'] = interp_world_view.unsqueeze(0) if interp_world_view.dim() == 2 else interp_world_view
                    data_with_gs['novel_view']['full_proj_transform'] = interp_full_proj.unsqueeze(0) if interp_full_proj.dim() == 2 else interp_full_proj
                    data_with_gs['novel_view']['camera_center'] = interp_camera_center.unsqueeze(0) if interp_camera_center.dim() == 1 else interp_camera_center
                    data_with_gs['novel_view']['FovX'] = interp_FovX.unsqueeze(0) if isinstance(interp_FovX, torch.Tensor) and interp_FovX.dim() == 0 else interp_FovX
                    data_with_gs['novel_view']['FovY'] = interp_FovY.unsqueeze(0) if isinstance(interp_FovY, torch.Tensor) and interp_FovY.dim() == 0 else interp_FovY
                    
                    with torch.no_grad():
                        render_result = pts2render(data_with_gs, bg_color=self.cfg.dataset.bg_color)
                        
                        # 保存渲染结果
                        tmp_novel = render_result['novel_view']['img_pred'][0].detach()
                        tmp_novel *= 255
                        tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                        
                        # 文件名格式: seq_dataframe_interpframe.jpg
                        tmp_img_name = '%s/%s_%04d_%04d.jpg' % (
                            self.cfg.record.show_path, s_name.split('_')[0], data_idx, interp_idx
                        )
                        cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))
                    
                    output_frame_idx += 1
                    pbar.update(1)
        
        logging.info(f"平滑视角渲染完成，共生成 {output_frame_idx} 帧")

    def val_all_views(self, views=None):
        """
        多视角测试 - 按帧顺序依次切换视角渲染
        
        对于每一帧数据，按照视角列表顺序依次生成渲染结果
        例如: 帧0-视角2, 帧0-视角3, 帧1-视角2, 帧1-视角3, ...
        输出文件名格式: {sample_name}_{frame_idx:04d}_v{view_id:02d}.jpg
        
        Args:
            views: 要渲染的视角列表，None表示使用所有可用视角
        """
        logging.info(f"Doing sequential views validation ...")
        torch.cuda.empty_cache()
        
        if views is None:
            views = self._detect_available_views()
        else:
            # 过滤出实际可用的视角
            available = self._detect_available_views(views)
            views = [v for v in views if v in available]
            if not views:
                logging.error("指定的视角都不可用！")
                return
        
        logging.info(f"使用视角: {views}")
        
        # 为每个视角创建数据加载器
        view_loaders = {}
        view_iterators = {}
        len_val = None
        
        for view_id in views:
            self.cfg.defrost()
            self.cfg.dataset.val_novel_id = [view_id]
            self.cfg.freeze()
            
            val_set_temp = StereoHumanDataset(self.cfg.dataset, phase=self.cfg.seq_name)
            val_loader_temp = DataLoader(val_set_temp, batch_size=self.bs, shuffle=False, num_workers=4, pin_memory=True)
            
            if len_val is None:
                len_val = int(len(val_loader_temp) / val_set_temp.val_boost)
            
            view_loaders[view_id] = val_loader_temp
            view_iterators[view_id] = iter(val_loader_temp)
        
        logging.info(f"总帧数: {len_val}, 视角数: {len(views)}, 总输出: {len_val * len(views)} 帧")
        
        # 按帧顺序，每帧依次渲染所有视角
        frame_idx = 0
        for data_idx in tqdm(range(len_val), desc="渲染帧"):
            for view_id in views:
                try:
                    data = next(view_iterators[view_id])
                except StopIteration:
                    view_iterators[view_id] = iter(view_loaders[view_id])
                    data = next(view_iterators[view_id])
                
                for view in ['lmain', 'rmain']:
                    for item in data[view].keys():
                        data[view][item] = data[view][item].cuda()
                
                s_name = data['novel_view']['sample_name'][0]
                
                with torch.no_grad():
                    data, _, _ = self.model(data, is_train=False)
                    data = pts2render(data, bg_color=self.cfg.dataset.bg_color)
                    
                    tmp_novel = data['novel_view']['img_pred'][0].detach()
                    tmp_novel *= 255
                    tmp_novel = tmp_novel.permute(1, 2, 0).cpu().numpy()
                    
                    # 文件名格式: sample_frame_view.jpg (便于按顺序排列生成视频)
                    tmp_img_name = '%s/%s_%04d_v%02d.jpg' % (self.cfg.record.show_path, s_name.split('_')[0], frame_idx, view_id)
                    cv2.imwrite(tmp_img_name, tmp_novel[:, :, ::-1].astype(np.uint8))
                    
                    # 保存深度图
                    if 'depth' in data['lmain']:
                        depth_lmain = data['lmain']['depth'][0].squeeze().detach().cpu().numpy()
                        min_depth = depth_lmain.min()
                        max_depth = depth_lmain.max()
                        
                        if max_depth > min_depth:
                            depth_normalized = ((depth_lmain - min_depth) / (max_depth - min_depth) * 255).astype(np.uint8)
                        else:
                            depth_normalized = np.zeros_like(depth_lmain, dtype=np.uint8)
                        
                        colored_depth_map = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                        depth_img_name = '%s/%s_%04d_v%02d_depth.png' % (self.cfg.record.show_path, s_name.split('_')[0], frame_idx, view_id)
                        cv2.imwrite(depth_img_name, colored_depth_map)
                
                frame_idx += 1
        
        # 清理
        for view_id in views:
            del view_loaders[view_id]
        
        logging.info(f"所有视角渲染完成，共生成 {frame_idx} 帧")

    def _interpolate_transform(self, transform_start, transform_end, t):
        """
        插值4x4变换矩阵 (使用SLERP进行旋转插值)
        
        Args:
            transform_start: 起始变换矩阵 (B, 4, 4) 或 (4, 4)
            transform_end: 结束变换矩阵 (B, 4, 4) 或 (4, 4)
            t: 插值参数 [0, 1]
            
        Returns:
            插值后的变换矩阵
        """
        # 处理不同的输入维度
        if transform_start.dim() == 2:
            # (4, 4) -> (1, 4, 4)
            transform_start = transform_start.unsqueeze(0)
            transform_end = transform_end.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size = transform_start.shape[0]
        result = torch.zeros_like(transform_start)
        
        for b in range(batch_size):
            # 提取旋转和平移
            R_start = transform_start[b, :3, :3].cpu().numpy()
            R_end = transform_end[b, :3, :3].cpu().numpy()
            t_start = transform_start[b, :3, 3].cpu().numpy()
            t_end = transform_end[b, :3, 3].cpu().numpy()
            
            # 使用球面线性插值 (SLERP) 进行旋转插值
            try:
                rot_start = R.from_matrix(R_start)
                rot_end = R.from_matrix(R_end)
                
                # 创建关键帧
                key_rots = R.from_quat([rot_start.as_quat(), rot_end.as_quat()])
                key_times = [0, 1]
                slerp = Slerp(key_times, key_rots)
                
                # 插值旋转
                rot_interp = slerp(t)
                R_interp = rot_interp.as_matrix()
            except Exception as e:
                # 如果SLERP失败，使用线性插值
                R_interp = (1 - t) * R_start + t * R_end
            
            # 线性插值平移
            t_interp = (1 - t) * t_start + t * t_end
            
            # 构建变换矩阵
            result[b, :3, :3] = torch.from_numpy(R_interp).float().to(transform_start.device)
            result[b, :3, 3] = torch.from_numpy(t_interp).float().to(transform_start.device)
            result[b, 3, 3] = transform_start[b, 3, 3]  # 保持最后一个元素
            
            # 复制第4行（如果有特殊值）
            result[b, 3, :3] = (1 - t) * transform_start[b, 3, :3] + t * transform_end[b, 3, :3]
        
        if squeeze_output:
            result = result.squeeze(0)
        
        return result

    def _interpolate_camera(self, extrinsic_start, extrinsic_end, t):
        """
        插值相机外参矩阵 (调用_interpolate_transform)
        """
        return self._interpolate_transform(extrinsic_start, extrinsic_end, t)

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
    # 用法示例:
    # 单视角测试:     python test.py -i s1a6 -v 2
    # 多视角测试:     python test.py -i s1a6 --all-views
    # 动态视角测试:   python test.py -i s1a6 --dynamic --interp 10
    # 指定视角范围:   python test.py -i s1a6 --dynamic --views 0,1,2,3 --interp 15
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    parser = argparse.ArgumentParser(description='GPS_plus 测试脚本')
    parser.add_argument('-i', '--input', type=str, default='s1a6', required=False, 
                        help='输入序列名称')
    parser.add_argument('-v', '--view', type=int, default=3, required=False, 
                        help='单视角测试时的目标视角 (默认: 3)')
    parser.add_argument('--all-views', action='store_true', 
                        help='对所有视角进行测试')
    parser.add_argument('--dynamic', action='store_true', 
                        help='启用动态视角测试（视角间插值）')
    parser.add_argument('--views', type=str, default='auto', 
                        help='要测试的视角列表，逗号分隔；"auto"表示自动检测 (默认: auto)')
    parser.add_argument('--interp', type=int, default=30, 
                        help='动态视角测试时总插值帧数 (默认: 30)')
    parser.add_argument('--loop', action='store_true',
                        help='动态视角测试时是否循环（从最后视角回到第一视角）')
    parser.add_argument('--ckpt', type=str, default=None, 
                        help='指定checkpoint路径 (默认使用配置文件中的路径)')
    arg = parser.parse_args()

    # 解析视角列表
    if arg.views.lower() == 'auto':
        views_list = None  # 让方法自动检测
    else:
        views_list = [int(v.strip()) for v in arg.views.split(',')]

    for seq_name in [arg.input + '_process']:
        cfg = config()
        cfg.load("config/stage.yaml")
        cfg = cfg.get_cfg()

        cfg.defrost()
        dt = datetime.today()
        # 根据深度模式设置实验名称
        depth_mode = getattr(cfg, 'depth_mode', 'raft')
        cfg.exp_name = f'gps_plus_{depth_mode}'
        
        # 根据测试模式设置输出路径
        if arg.dynamic:
            cfg.record.show_path = "experiments/%s/show_%s_dynamic" % (cfg.exp_name, seq_name)
        elif arg.all_views:
            cfg.record.show_path = "experiments/%s/show_%s_allviews" % (cfg.exp_name, seq_name)
        else:
            cfg.record.show_path = "experiments/%s/show_%s" % (cfg.exp_name, seq_name)
        
        cfg.seq_name = seq_name 
        cfg.dataset.val_novel_id = [arg.view]  # 初始视角
        
        # 设置checkpoint路径
        if arg.ckpt:
            cfg.restore_ckpt = arg.ckpt
        else:
            cfg.restore_ckpt = '/home/user_3/3DGS/GPS_plus/experiments/gps_plus_da3_0119/ckpt/iter10000.pth'

        cfg.freeze()
        
        print(f"=" * 50)
        print(f"GPS_plus 测试")
        print(f"=" * 50)
        print(f"检查点路径: {cfg.restore_ckpt}")
        print(f"深度模式: {depth_mode}")
        print(f"序列名称: {seq_name}")
        print(f"输出目录: {cfg.record.show_path}")
        
        if arg.dynamic:
            views_str = "自动检测" if views_list is None else str(views_list)
            loop_str = "是" if arg.loop else "否"
            print(f"测试模式: 平滑动态视角 (关键视角: {views_str}, 总帧数: {arg.interp}, 循环: {loop_str})")
        elif arg.all_views:
            views_str = "自动检测" if views_list is None else str(views_list)
            print(f"测试模式: 多视角 (视角: {views_str})")
        else:
            print(f"测试模式: 单视角 (视角: {arg.view})")
        print(f"=" * 50)

        Path(cfg.record.show_path).mkdir(exist_ok=True, parents=True)

        trainer = Trainer(cfg)
        
        # 根据参数选择测试模式
        if arg.dynamic:
            trainer.val_dynamic(num_interp=arg.interp, views=views_list, loop=arg.loop)
        elif arg.all_views:
            trainer.val_all_views(views=views_list)
        else:
            trainer.val()
