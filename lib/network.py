"""
GPS_plus 网络架构

支持三种深度估计模式:
1. RAFT-Stereo: 原始的立体匹配深度估计
2. DA3: 使用Depth-Anything-3进行深度估计
3. FFS: 使用Fast-FoundationStereo进行立体匹配深度估计
"""

import torch
import torch.nn.functional as F
from torch import nn
from core.raft_stereo_human import RAFTStereoHuman
from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser
from lib.loss import sequence_loss
from lib.utils import flow2depth, depth2pc
from lib.embedder import get_embedder
from lib.attention_module import LocalFeatureTransformer
from torch.cuda.amp import autocast as autocast


class RtStereoHumanModel(nn.Module):
    """
    GPS_plus主网络模型
    
    支持三种深度估计模式:
    - 'raft': 使用RAFT-Stereo进行立体匹配深度估计
    - 'da3': 使用Depth-Anything-3进行深度估计
    - 'ffs': 使用Fast-FoundationStereo进行立体匹配深度估计
    """
    
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        self.use_flow_init = self.cfg.dataset.use_depth_init
        self.train_iters = self.cfg.raft.train_iters
        self.val_iters = self.cfg.raft.val_iters
        
        # 获取深度估计模式
        self.depth_mode = getattr(self.cfg, 'depth_mode', 'raft')
        print(f"[Network] 深度估计模式: {self.depth_mode}")
        
        # 图像编码器 (stereo_gs 模式使用 FFS 特征，不需要独立编码器)
        if self.depth_mode != 'stereo_gs':
            self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        
        # 根据模式初始化深度估计模块
        if self.depth_mode == 'raft':
            # 原始RAFT-Stereo模式
            self.loftr_coarse = LocalFeatureTransformer()
            self.raft_stereo = RAFTStereoHuman(self.cfg.raft)
            self.depth_model = None
        elif self.depth_mode == 'da3':
            # DA3深度估计模式
            from lib.da3_depth import create_da3_depth_estimator
            da3_mode = getattr(self.cfg.da3, 'mode', 'single')
            self.depth_model = create_da3_depth_estimator(self.cfg, mode=da3_mode)
            self.raft_stereo = None
            self.loftr_coarse = None
            # DA3模式下仍需要特征匹配用于高斯参数预测
            if getattr(self.cfg.da3, 'use_loftr', True):
                self.loftr_coarse = LocalFeatureTransformer()
        elif self.depth_mode == 'ffs':
            # Fast-FoundationStereo 深度估计模式
            from lib.ffs_depth import FFSDepthEstimator
            self.depth_model = FFSDepthEstimator(self.cfg)
            self.raft_stereo = None
            self.loftr_coarse = None
            if getattr(self.cfg.ffs, 'use_loftr', True):
                self.loftr_coarse = LocalFeatureTransformer()
        elif self.depth_mode == 'stereo_gs':
            from lib.stereo_gs import StereoGSModel
            self.stereo_gs_model = StereoGSModel(self.cfg)
            self.depth_model = None
            self.raft_stereo = None
            self.loftr_coarse = None
        else:
            raise ValueError(f"未知的深度估计模式: {self.depth_mode}")
        
        # 高斯参数回归器 (stereo_gs 模式内置解码器，不需要 GSRegresser)
        if self.with_gs_render and self.depth_mode != 'stereo_gs':
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)

    def forward(self, data, is_train=True):
        """
        前向传播
        
        Args:
            data: 输入数据字典，包含 'lmain' 和 'rmain'
            is_train: 是否训练模式
            
        Returns:
            data: 更新后的数据字典
            flow_loss: 光流/深度损失
            metrics: 指标字典
        """
        bs = data['lmain']['img'].shape[0]

        if self.depth_mode == 'stereo_gs':
            return self._forward_stereo_gs(data, bs, is_train)
        
        # 合并左右图像用于批量处理
        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
        
        # 提取图像特征
        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)
        
        # 根据深度模式进行处理
        if self.depth_mode == 'raft':
            return self._forward_raft(data, image, img_feat, bs, is_train)
        elif self.depth_mode == 'da3':
            return self._forward_da3(data, image, img_feat, bs, is_train)
        elif self.depth_mode == 'ffs':
            return self._forward_ffs(data, image, img_feat, bs, is_train)
        else:
            raise ValueError(f"未知的深度估计模式: {self.depth_mode}")
    
    def _forward_raft(self, data, image, img_feat, bs, is_train):
        """
        RAFT-Stereo模式前向传播
        """
        flow_init = torch.cat([data['lmain']['flow_init'], data['rmain']['flow_init']], dim=0) if self.use_flow_init else None
        
        # 特征匹配
        (feat_c0, feat_c1) = img_feat[2].split(bs)
        mask_c0 = mask_c1 = None  
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_cs = torch.cat((feat_c0, feat_c1), 0)
        img_feat = img_feat[0], img_feat[1], feat_cs
        
        if is_train:
            flow_predictions = self.raft_stereo(feat_cs, flow_init=flow_init, iters=self.train_iters)
            flow_loss = None
            metrics = {}
            flow_pred_lmain, flow_pred_rmain = torch.split(flow_predictions[-1], [bs, bs])

            if not self.with_gs_render:
                data['lmain']['flow_pred'] = flow_pred_lmain.detach()
                data['rmain']['flow_pred'] = flow_pred_rmain.detach()
                return data, flow_loss, metrics

            data['lmain']['flow_pred'] = flow_pred_lmain
            data['rmain']['flow_pred'] = flow_pred_rmain
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics
        else:
            flow_up = self.raft_stereo(feat_cs, flow_init=flow_init, iters=self.val_iters, test_mode=True)
            flow_loss, metrics = None, None

            data['lmain']['flow_pred'] = flow_up[0]
            data['rmain']['flow_pred'] = flow_up[1]

            if not self.with_gs_render:
                return data, flow_loss, metrics
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics
    
    def _forward_da3(self, data, image, img_feat, bs, is_train):
        """
        DA3模式前向传播
        """
        # 使用DA3进行深度估计
        data, depth_loss, metrics = self.depth_model(data, is_train=is_train)
        
        # 如果使用LoFTR进行特征匹配（用于高斯参数预测）
        if self.loftr_coarse is not None:
            (feat_c0, feat_c1) = img_feat[2].split(bs)
            mask_c0 = mask_c1 = None
            feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
            feat_cs = torch.cat((feat_c0, feat_c1), 0)
            img_feat = img_feat[0], img_feat[1], feat_cs
        
        if not self.with_gs_render:
            return data, depth_loss, metrics
        
        # DA3已经直接输出深度，不需要从flow转换
        # 直接进行高斯参数回归
        data = self.depth2gsparms(image, img_feat, data, bs)
        
        return data, depth_loss, metrics

    def _forward_ffs(self, data, image, img_feat, bs, is_train):
        """
        Fast-FoundationStereo 模式前向传播

        FFS 提供高质量立体视差 → 转换为逆深度 → 复用原版 GSRegresser 预测高斯参数。
        """
        data, depth_loss, metrics = self.depth_model(data, is_train=is_train)

        if self.loftr_coarse is not None:
            (feat_c0, feat_c1) = img_feat[2].split(bs)
            mask_c0 = mask_c1 = None
            feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
            feat_cs = torch.cat((feat_c0, feat_c1), 0)
            img_feat = img_feat[0], img_feat[1], feat_cs

        if not self.with_gs_render:
            return data, depth_loss, metrics

        data = self.depth2gsparms(image, img_feat, data, bs)

        return data, depth_loss, metrics

    def _forward_stereo_gs(self, data, bs, is_train):
        """
        StereoGS 模式: FFS 特征驱动的端到端高斯预测。

        不使用 img_encoder / GSRegresser，完全由 StereoGSModel 内部处理。
        """
        data, loss, metrics = self.stereo_gs_model(data, is_train=is_train)
        return data, loss, metrics or {}

    def flow2gsparms(self, lr_img, lr_img_feat, data, bs):
        """
        从光流计算高斯参数（RAFT模式使用）
        
        Args:
            lr_img: 左右图像
            lr_img_feat: 图像特征
            data: 数据字典
            bs: batch size
            
        Returns:
            data: 更新后的数据字典
        """
        # 从光流计算深度
        for view in ['lmain', 'rmain']:
            data[view]['depth'] = flow2depth(data[view])
            
        l_depth = data['lmain']['depth']  
        r_depth = data['rmain']['depth'] 
        lr_depth = torch.concat([l_depth, r_depth], dim=0)
        
        # 回归高斯参数
        rot_maps, scale_maps, opacity_maps, depth_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)
        l_resdepth, r_resdepth = torch.split(depth_maps, [bs, bs])

        # 添加深度残差
        data['lmain']['depth'] += l_resdepth
        data['rmain']['depth'] += r_resdepth

        # 将深度转换为3D点
        for view in ['lmain', 'rmain']:
            data[view]['xyz'] = depth2pc(data[view]['depth'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)
            valid = data[view]['mask'][:, :1, :, :] > 0.5
            data[view]['pts_valid'] = valid.view(bs, -1)

        data['novel_view']['scale_regular'] = torch.mean(scale_maps)

        # 分配高斯参数
        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])

        return data
    
    def depth2gsparms(self, lr_img, lr_img_feat, data, bs):
        """
        从深度计算高斯参数（DA3/FFS模式使用）
        
        与flow2gsparms类似，但深度已经由外部模型直接提供。
        """
        l_depth = data['lmain']['depth']  
        r_depth = data['rmain']['depth'] 
        lr_depth = torch.concat([l_depth, r_depth], dim=0)

        # 保存初始深度用于可视化对比
        data['lmain']['depth_init'] = l_depth.detach().clone()
        data['rmain']['depth_init'] = r_depth.detach().clone()
        
        # 回归高斯参数
        rot_maps, scale_maps, opacity_maps, depth_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)
        l_resdepth, r_resdepth = torch.split(depth_maps, [bs, bs])

        # 添加深度残差
        data['lmain']['depth'] = data['lmain']['depth'] + l_resdepth
        data['rmain']['depth'] = data['rmain']['depth'] + r_resdepth

        # 将深度转换为3D点
        for view in ['lmain', 'rmain']:
            data[view]['xyz'] = depth2pc(data[view]['depth'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)
            valid = data[view]['mask'][:, :1, :, :] > 0.5
            data[view]['pts_valid'] = valid.view(bs, -1)

        data['novel_view']['scale_regular'] = torch.mean(scale_maps)

        # 分配高斯参数
        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])

        return data
