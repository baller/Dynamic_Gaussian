"""
GPS_plus 网络架构

支持两种深度估计模式:
1. RAFT-Stereo: 原始的立体匹配深度估计
2. DA3: 使用Depth-Anything-3进行深度估计

支持两种高斯生成模式:
1. 单流模式: 原始的统一高斯参数回归
2. MoE模式: 动静分离，背景和人体分别预测
"""

import torch
import torch.nn.functional as F
from torch import nn
from core.raft_stereo_human import RAFTStereoHuman
from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser, DualStreamGSRegresser, create_gs_regresser
from lib.loss import sequence_loss
from lib.utils import flow2depth, depth2pc
from lib.embedder import get_embedder
from lib.attention_module import LocalFeatureTransformer
from torch.cuda.amp import autocast as autocast


class RtStereoHumanModel(nn.Module):
    """
    GPS_plus主网络模型
    
    支持两种深度估计模式:
    - 'raft': 使用RAFT-Stereo进行立体匹配深度估计
    - 'da3': 使用Depth-Anything-3进行深度估计
    
    支持两种高斯生成模式:
    - 单流模式: 原始的统一高斯参数回归
    - MoE模式: 动静分离，背景和人体分别预测
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
        
        # 获取MoE模式配置
        self.moe_cfg = getattr(self.cfg, 'moe', None)
        self.use_moe = self.moe_cfg is not None and getattr(self.moe_cfg, 'enabled', False)
        if self.use_moe:
            print(f"[Network] MoE模式已启用，专家数量: {getattr(self.moe_cfg, 'num_experts', 2)}")
        
        # 图像编码器（两种模式都需要）
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
        else:
            raise ValueError(f"未知的深度估计模式: {self.depth_mode}")
        
        # 高斯参数回归器（根据MoE配置选择单流或双流）
        if self.with_gs_render:
            self.gs_parm_regresser = create_gs_regresser(self.cfg, rgb_dim=3, depth_dim=1)
            
            # MoE模式下初始化额外模块
            if self.use_moe:
                from lib.moe_router import create_moe_router
                from lib.gaussian_allocation import create_gaussian_allocator
                
                # MoE路由器
                router_in_channels = self.cfg.raft.encoder_dims[0]  # 使用第一层特征
                self.moe_router = create_moe_router(self.cfg, router_in_channels)
                
                # 高斯分配网络（如果配置为可学习）
                alloc_cfg = getattr(self.moe_cfg, 'allocation', None)
                if alloc_cfg is not None and getattr(alloc_cfg, 'learnable', False):
                    self.gaussian_allocator = create_gaussian_allocator(self.cfg)
                else:
                    self.gaussian_allocator = None

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
        
        # MoE模式处理
        if self.use_moe:
            return self._flow2gsparms_moe(lr_img, lr_img_feat, data, bs, lr_depth)
        
        # 原始单流模式
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
    
    def _compute_router_weights(self, router_input, depth):
        """
        计算路由权重，支持不同类型的路由器
        
        Args:
            router_input: 路由器输入特征 [2*bs, C, H, W]
            depth: 深度图 [2*bs, 1, H, W]（用于depth_aware路由器）
            
        Returns:
            router_weights: 路由权重 [2*bs, num_experts(+1), H, W]
            expert_info: 专家信息字典
        """
        from lib.moe_router import DepthAwareMoERouter
        
        if isinstance(self.moe_router, DepthAwareMoERouter):
            # depth_aware路由器需要深度信息
            # 下采样深度到与特征相同的尺寸
            depth_downsampled = F.interpolate(
                depth, size=router_input.shape[2:], 
                mode='bilinear', align_corners=False
            )
            router_weights, expert_info = self.moe_router(router_input, depth_downsampled)
        else:
            # basic或multiscale路由器
            router_weights, expert_info = self.moe_router(router_input)
        
        return router_weights, expert_info
    
    def _flow2gsparms_moe(self, lr_img, lr_img_feat, data, bs, lr_depth):
        """
        MoE模式的高斯参数计算（支持任意专家数量+共享专家）
        """
        # 计算路由权重
        router_input = lr_img_feat[0]  # [2*bs, C, H, W]
        router_weights, expert_info = self._compute_router_weights(router_input, lr_depth)
        
        # 上采样到原始分辨率
        target_size = lr_img.shape[2:]
        router_weights = F.interpolate(router_weights, size=target_size, mode='bilinear', align_corners=False)
        
        # 多专家高斯参数预测
        expert_params_list, fused_params = self.gs_parm_regresser.forward_with_routing(
            lr_img, lr_depth, lr_img_feat, router_weights, expert_info
        )
        
        # 使用融合后的深度残差
        depth_maps = fused_params['depth_residual']
        l_resdepth, r_resdepth = torch.split(depth_maps, [bs, bs])
        
        # 添加深度残差
        data['lmain']['depth'] = data['lmain']['depth'] + l_resdepth
        data['rmain']['depth'] = data['rmain']['depth'] + r_resdepth
        
        # 将深度转换为3D点
        for view in ['lmain', 'rmain']:
            data[view]['xyz'] = depth2pc(data[view]['depth'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)
            valid = data[view]['mask'][:, :1, :, :] > 0.5
            data[view]['pts_valid'] = valid.view(bs, -1)
        
        # 分配融合后的高斯参数
        rot_maps = fused_params['rot_maps']
        scale_maps = fused_params['scale_maps']
        opacity_maps = fused_params['opacity_maps']
        
        data['novel_view']['scale_regular'] = torch.mean(scale_maps)
        
        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])
        
        # 保存MoE相关信息
        l_router, r_router = torch.split(router_weights, [bs, bs])
        data['lmain']['router_weights'] = l_router
        data['rmain']['router_weights'] = r_router
        data['expert_info'] = expert_info
        
        # 保存各专家参数（用于可视化和分析）
        self._save_expert_params(data, expert_params_list, bs)
        
        # 如果有高斯分配网络，计算分配比例
        if self.gaussian_allocator is not None:
            alloc_ratio, _, _ = self.gaussian_allocator(lr_img_feat[0], router_weights)
            data['allocation_ratio'] = alloc_ratio
        
        return data
    
    def _save_expert_params(self, data, expert_params_list, bs):
        """
        保存各专家参数到data字典
        
        向后兼容：同时保存bg_params和human_params（如果存在）
        """
        # 保存专家参数列表
        l_expert_params = []
        r_expert_params = []
        
        for exp_params in expert_params_list:
            l_params = {}
            r_params = {}
            for key, val in exp_params.items():
                if isinstance(val, torch.Tensor):
                    l_params[key] = val[:bs]
                    r_params[key] = val[bs:]
                else:
                    l_params[key] = val
                    r_params[key] = val
            l_expert_params.append(l_params)
            r_expert_params.append(r_params)
        
        data['lmain']['expert_params_list'] = l_expert_params
        data['rmain']['expert_params_list'] = r_expert_params
        
        # 向后兼容：保存bg_params和human_params
        if len(expert_params_list) >= 1:
            bg_params = expert_params_list[0]
            data['lmain']['bg_params'] = {k: v[:bs] if isinstance(v, torch.Tensor) else v for k, v in bg_params.items()}
            data['rmain']['bg_params'] = {k: v[bs:] if isinstance(v, torch.Tensor) else v for k, v in bg_params.items()}
        
        if len(expert_params_list) >= 2:
            human_params = expert_params_list[1]
            data['lmain']['human_params'] = {k: v[:bs] if isinstance(v, torch.Tensor) else v for k, v in human_params.items()}
            data['rmain']['human_params'] = {k: v[bs:] if isinstance(v, torch.Tensor) else v for k, v in human_params.items()}
    
    def depth2gsparms(self, lr_img, lr_img_feat, data, bs):
        """
        从深度计算高斯参数（DA3模式使用）
        
        与flow2gsparms类似，但深度已经由DA3直接提供。
        
        Args:
            lr_img: 左右图像
            lr_img_feat: 图像特征
            data: 数据字典
            bs: batch size
            
        Returns:
            data: 更新后的数据字典
        """
        # DA3已经提供了深度，直接使用
        l_depth = data['lmain']['depth']  
        r_depth = data['rmain']['depth'] 
        lr_depth = torch.concat([l_depth, r_depth], dim=0)
        
        # MoE模式处理
        if self.use_moe:
            return self._depth2gsparms_moe(lr_img, lr_img_feat, data, bs, lr_depth)
        
        # 原始单流模式
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
    
    def _depth2gsparms_moe(self, lr_img, lr_img_feat, data, bs, lr_depth):
        """
        MoE模式的高斯参数计算（DA3模式使用，支持任意专家数量+共享专家）
        """
        # 计算路由权重
        router_input = lr_img_feat[0]  # [2*bs, C, H, W]
        router_weights, expert_info = self._compute_router_weights(router_input, lr_depth)
        
        # 上采样到原始分辨率
        target_size = lr_img.shape[2:]
        router_weights = F.interpolate(router_weights, size=target_size, mode='bilinear', align_corners=False)
        
        # 多专家高斯参数预测
        expert_params_list, fused_params = self.gs_parm_regresser.forward_with_routing(
            lr_img, lr_depth, lr_img_feat, router_weights, expert_info
        )
        
        # 使用融合后的深度残差
        depth_maps = fused_params['depth_residual']
        l_resdepth, r_resdepth = torch.split(depth_maps, [bs, bs])
        
        # 添加深度残差
        data['lmain']['depth'] = data['lmain']['depth'] + l_resdepth
        data['rmain']['depth'] = data['rmain']['depth'] + r_resdepth
        
        # 将深度转换为3D点
        for view in ['lmain', 'rmain']:
            data[view]['xyz'] = depth2pc(data[view]['depth'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)
            valid = data[view]['mask'][:, :1, :, :] > 0.5
            data[view]['pts_valid'] = valid.view(bs, -1)
        
        # 分配融合后的高斯参数
        rot_maps = fused_params['rot_maps']
        scale_maps = fused_params['scale_maps']
        opacity_maps = fused_params['opacity_maps']
        
        data['novel_view']['scale_regular'] = torch.mean(scale_maps)
        
        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])
        
        # 保存MoE相关信息
        l_router, r_router = torch.split(router_weights, [bs, bs])
        data['lmain']['router_weights'] = l_router
        data['rmain']['router_weights'] = r_router
        data['expert_info'] = expert_info
        
        # 保存各专家参数
        self._save_expert_params(data, expert_params_list, bs)
        
        # 如果有高斯分配网络，计算分配比例
        if self.gaussian_allocator is not None:
            alloc_ratio, _, _ = self.gaussian_allocator(lr_img_feat[0], router_weights)
            data['allocation_ratio'] = alloc_ratio
        
        return data
    
    def freeze_router(self):
        """冻结MoE路由器参数（渐进式训练阶段A使用）"""
        if self.use_moe and hasattr(self, 'moe_router'):
            for param in self.moe_router.parameters():
                param.requires_grad = False
            print("[Network] MoE路由器已冻结")
    
    def unfreeze_router(self):
        """解冻MoE路由器参数（渐进式训练阶段B使用）"""
        if self.use_moe and hasattr(self, 'moe_router'):
            for param in self.moe_router.parameters():
                param.requires_grad = True
            print("[Network] MoE路由器已解冻")
