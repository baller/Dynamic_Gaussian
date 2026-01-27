"""
GPS_plus 网络架构

支持两种深度估计模式:
1. RAFT-Stereo: 原始的立体匹配深度估计
2. DA3: 使用Depth-Anything-3进行深度估计

支持三种高斯生成模式:
1. 单流模式: 原始的统一高斯参数回归
2. MoE-CNN模式: 基于CNN的动静分离
3. MoE-Transformer模式: 基于DA3 DINO特征的Transformer MoE
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
from typing import Optional, Dict, Tuple


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
        
        # 检查是否使用 Transformer 专家类型
        self.expert_type = 'cnn'  # 默认CNN专家
        if self.use_moe and self.moe_cfg is not None:
            self.expert_type = getattr(self.moe_cfg, 'expert_type', 'cnn')
            print(f"[Network] MoE模式已启用，专家类型: {self.expert_type}，"
                  f"专家数量: {getattr(self.moe_cfg, 'num_experts', 2)}")
        
        # 是否使用 Transformer MoE (基于DINO特征)
        self.use_transformer_moe = self.use_moe and self.expert_type == 'transformer'
        
        # 根据模式初始化组件
        if self.use_transformer_moe and self.depth_mode == 'da3':
            # Transformer MoE 模式: 使用 DA3 DINO 特征，不需要 UNet
            self._init_transformer_moe_mode()
        else:
            # 传统模式: 需要 UNet 图像编码器
            self._init_traditional_mode()
    
    def _init_transformer_moe_mode(self):
        """
        初始化 Transformer MoE 模式
        
        特点:
        - 使用 DA3 作为统一编码器
        - 直接从 DA3 提取 DINO 特征
        - 不需要 UNet 和 LoFTR
        - 使用 Transformer 专家预测高斯参数
        """
        print("[Network] 初始化 Transformer MoE 模式 (DINO-based)")
        
        # DA3 深度估计器 (带 DINO 特征导出)
        from lib.da3_depth import create_da3_depth_estimator
        da3_mode = getattr(self.cfg.da3, 'mode', 'single')
        self.depth_model = create_da3_depth_estimator(
            self.cfg, 
            mode=da3_mode,
            export_features=True  # 启用 DINO 特征导出
        )
        
        # 不需要传统编码器
        self.img_encoder = None
        self.raft_stereo = None
        self.loftr_coarse = None
        
        if self.with_gs_render:
            # MoE Transformer 回归器
            from lib.moe_transformer import create_moe_transformer_regresser
            self.moe_transformer = create_moe_transformer_regresser(self.cfg)
            
            # 动态高斯分配器
            from lib.dynamic_gaussian_allocator import (
                create_dynamic_allocator, 
                create_curriculum_scheduler
            )
            self.dynamic_allocator = create_dynamic_allocator(self.cfg)
            self.curriculum_scheduler = create_curriculum_scheduler(self.cfg)
            
            # 保持向后兼容
            self.gs_parm_regresser = None
            self.moe_router = None
            self.gaussian_allocator = None
    
    def _init_traditional_mode(self):
        """
        初始化传统模式 (CNN-based MoE 或单流)
        """
        # 图像编码器
        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        
        # Transformer MoE 相关组件设为 None
        self.moe_transformer = None
        self.dynamic_allocator = None
        self.curriculum_scheduler = None
        
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
            # CNN MoE 模式不需要 DINO 特征
            export_features = False
            self.depth_model = create_da3_depth_estimator(
                self.cfg, 
                mode=da3_mode,
                export_features=export_features
            )
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

    def forward(self, data, is_train=True, bg_update_signal: Optional[bool] = None, step: int = 0):
        """
        前向传播
        
        Args:
            data: 输入数据字典，包含 'lmain' 和 'rmain'
            is_train: 是否训练模式
            bg_update_signal: 背景更新信号 (仅 Transformer MoE 模式使用)
                - None: 自动使用课程学习调度
                - True: 正常模式（更新背景）
                - False: 人体专注模式（使用背景缓存）
            step: 当前训练步数 (用于课程学习)
            
        Returns:
            data: 更新后的数据字典
            flow_loss: 光流/深度损失
            metrics: 指标字典
        """
        bs = data['lmain']['img'].shape[0]
        
        # Transformer MoE 模式: 使用 DA3 DINO 特征
        if self.use_transformer_moe:
            return self._forward_transformer_moe(data, bs, is_train, bg_update_signal, step)
        
        # 传统模式: 合并左右图像用于批量处理
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
    
    def _forward_transformer_moe(
        self, 
        data: Dict, 
        bs: int, 
        is_train: bool,
        bg_update_signal: Optional[bool],
        step: int
    ) -> Tuple[Dict, Optional[torch.Tensor], Dict]:
        """
        Transformer MoE 模式前向传播
        
        使用 DA3 提取深度和 DINO 特征，然后用 Transformer MoE 预测高斯参数。
        
        Args:
            data: 输入数据字典
            bs: batch size
            is_train: 是否训练模式
            bg_update_signal: 背景更新信号
            step: 当前训练步数
            
        Returns:
            data: 更新后的数据字典
            depth_loss: 深度损失
            metrics: 指标字典
        """
        metrics = {}
        
        # 1. 使用 DA3 提取深度和 DINO 特征
        data, depth_loss, da3_metrics = self.depth_model(data, is_train=is_train)
        if da3_metrics:
            metrics.update(da3_metrics)
        
        if not self.with_gs_render:
            return data, depth_loss, metrics
        
        # 2. 确定 bg_update_signal
        if bg_update_signal is None and is_train:
            # 训练时使用课程学习调度
            bg_update_signal = self.curriculum_scheduler.sample_bg_signal(step)
        elif bg_update_signal is None:
            # 推理时默认更新背景（可以外部控制）
            bg_update_signal = True
        
        # 记录当前模式
        data['bg_update_signal'] = bg_update_signal
        
        # 3. 分别处理左右视图
        for view in ['lmain', 'rmain']:
            view_data = data[view]
            
            # 检查 DINO 特征是否存在
            if 'dino_features' not in view_data:
                raise ValueError(f"[Network] {view} 缺少 dino_features，请确保 DA3 export_features=True")
            
            dino_features = view_data['dino_features']  # [B, N_patches, 1024]
            depth = view_data['depth']  # [B, 1, H, W]
            
            # 4. MoE Transformer 预测高斯参数
            moe_output = self.moe_transformer(
                dino_features,
                depth,
                bg_update_signal=bg_update_signal
            )
            
            # 5. 提取融合后的参数
            fused_params = moe_output['fused_params']
            router_weights = moe_output['router_weights']
            
            # 6. 将 patch 级参数转换为像素级
            # DINO patch size = 14, 需要上采样到原始分辨率
            H, W = depth.shape[-2:]
            view_data = self._patch_to_pixel(
                view_data, 
                fused_params, 
                router_weights,
                target_size=(H, W)
            )
            
            # 7. 添加深度残差
            depth_residual = view_data['depth_residual_map']  # 已上采样
            view_data['depth'] = view_data['depth'] + depth_residual
            
            # 8. 将深度转换为 3D 点
            view_data['xyz'] = depth2pc(
                view_data['depth'], 
                view_data['extr'], 
                view_data['intr']
            ).view(bs, -1, 3)
            valid = view_data['mask'][:, :1, :, :] > 0.5
            view_data['pts_valid'] = valid.view(bs, -1)
            
            # 9. 保存 MoE 相关信息
            view_data['router_weights'] = view_data['router_weights_map']
            view_data['expert_params'] = moe_output['expert_params']
            view_data['shared_params'] = moe_output.get('shared_params')
            
            # 更新 data
            data[view] = view_data
        
        # 10. 计算 scale_regular
        data['novel_view']['scale_regular'] = torch.mean(
            torch.cat([data['lmain']['scale_maps'], data['rmain']['scale_maps']], dim=0)
        )
        
        # 11. 添加 MoE 指标（只添加数值类型，字符串类型单独处理）
        metrics['bg_update_signal'] = float(bg_update_signal)
        if is_train:
            # curriculum_bg_prob 是数值，可以添加到 metrics
            metrics['curriculum_bg_prob'] = self.curriculum_scheduler.get_bg_signal_prob(step)
            # curriculum_phase 是字符串，存储在 data 中供单独记录
            data['curriculum_phase'] = self.curriculum_scheduler.get_phase(step)
        
        return data, depth_loss, metrics
    
    def _patch_to_pixel(
        self, 
        view_data: Dict,
        fused_params: Dict[str, torch.Tensor],
        router_weights: torch.Tensor,
        target_size: Tuple[int, int]
    ) -> Dict:
        """
        将 patch 级参数上采样到像素级
        
        Args:
            view_data: 视图数据字典
            fused_params: patch 级融合参数
            router_weights: [B, N_patches, num_experts] 路由权重
            target_size: (H, W) 目标尺寸
            
        Returns:
            view_data: 更新后的视图数据
        """
        B = router_weights.shape[0]
        H, W = target_size
        
        # 从实际的 patch 数量计算 patch 网格尺寸
        N_patches = router_weights.shape[1]
        
        # 尝试找到最接近的 H_patch 和 W_patch
        # 假设接近正方形，从 sqrt 开始找
        H_patch = int(N_patches ** 0.5)
        W_patch = N_patches // H_patch
        
        # 如果无法整除，尝试其他分解
        if H_patch * W_patch != N_patches:
            # 从 sqrt 向下找能整除的因子
            for h in range(int(N_patches ** 0.5), 0, -1):
                if N_patches % h == 0:
                    H_patch = h
                    W_patch = N_patches // h
                    break
        
        # 验证
        if H_patch * W_patch != N_patches:
            raise ValueError(
                f"无法将 {N_patches} 个 patch 分解为 H_patch × W_patch 的网格"
            )
        
        # 重塑并上采样每个参数
        for key, param in fused_params.items():
            # param: [B, N_patches, D]
            D = param.shape[-1]
            
            # 重塑为空间格式: [B, H_patch, W_patch, D]
            param_spatial = param.view(B, H_patch, W_patch, D)
            
            # 转换为 [B, D, H_patch, W_patch] 用于上采样
            param_spatial = param_spatial.permute(0, 3, 1, 2).contiguous()
            
            # 双线性上采样
            param_upsampled = F.interpolate(
                param_spatial,
                size=target_size,
                mode='bilinear',
                align_corners=False
            )
            
            # 根据参数类型存储
            if key == 'rotation':
                view_data['rot_maps'] = param_upsampled
            elif key == 'scale':
                view_data['scale_maps'] = param_upsampled
            elif key == 'opacity':
                view_data['opacity_maps'] = param_upsampled
            elif key == 'depth_residual':
                view_data['depth_residual_map'] = param_upsampled
        
        # 上采样路由权重
        num_experts = router_weights.shape[-1]
        router_spatial = router_weights.view(B, H_patch, W_patch, num_experts)
        router_spatial = router_spatial.permute(0, 3, 1, 2).contiguous()
        router_upsampled = F.interpolate(
            router_spatial,
            size=target_size,
            mode='bilinear',
            align_corners=False
        )
        view_data['router_weights_map'] = router_upsampled
        
        return view_data

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
        if not self.use_moe:
            return
            
        # Transformer MoE 模式：冻结 moe_transformer 中的路由器
        if self.use_transformer_moe and hasattr(self, 'moe_transformer') and self.moe_transformer is not None:
            if hasattr(self.moe_transformer, 'router'):
                for param in self.moe_transformer.router.parameters():
                    param.requires_grad = False
                print("[Network] MoE Transformer 路由器已冻结")
        # CNN MoE 模式：冻结独立的 moe_router
        elif hasattr(self, 'moe_router') and self.moe_router is not None:
            for param in self.moe_router.parameters():
                param.requires_grad = False
            print("[Network] MoE路由器已冻结")
    
    def unfreeze_router(self):
        """解冻MoE路由器参数（渐进式训练阶段B使用）"""
        if not self.use_moe:
            return
            
        # Transformer MoE 模式：解冻 moe_transformer 中的路由器
        if self.use_transformer_moe and hasattr(self, 'moe_transformer') and self.moe_transformer is not None:
            if hasattr(self.moe_transformer, 'router'):
                for param in self.moe_transformer.router.parameters():
                    param.requires_grad = True
                print("[Network] MoE Transformer 路由器已解冻")
        # CNN MoE 模式：解冻独立的 moe_router
        elif hasattr(self, 'moe_router') and self.moe_router is not None:
            for param in self.moe_router.parameters():
                param.requires_grad = True
            print("[Network] MoE路由器已解冻")
