"""
MoE (Mixture of Experts) 路由器模块

支持:
1. 任意数量的路由专家
2. 共享专家（所有输入都会经过，与路由专家输出融合）
3. 多种路由策略（basic, multiscale, depth_aware）

架构设计:
- num_experts: 路由专家数量（不包括共享专家）
- use_shared_expert: 是否使用共享专家
- 总专家数 = num_experts + 1 (如果use_shared_expert=True)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """
    MoE路由器 - 输出像素级的专家路由权重
    
    使用软路由（soft routing）而非硬分割，允许梯度反传。
    路由权重可作为"伪mask"用于分离特征。
    
    Args:
        in_channels: 输入特征通道数
        num_experts: 路由专家数量（不包括共享专家）
        hidden_channels: 隐藏层通道数
        temperature: softmax温度参数，控制分布锐度
        use_shared_expert: 是否使用共享专家
        shared_expert_weight: 共享专家的权重（相对于路由专家的权重和）
    """
    
    def __init__(self, in_channels, num_experts=2, hidden_channels=64, temperature=1.0,
                 use_shared_expert=True, shared_expert_weight=0.5):
        super().__init__()
        self.num_experts = num_experts  # 路由专家数量
        self.temperature = temperature
        self.use_shared_expert = use_shared_expert
        self.shared_expert_weight = shared_expert_weight
        
        # 总专家数（包括共享专家）
        self.total_experts = num_experts + 1 if use_shared_expert else num_experts
        
        # 路由网络 - 只为路由专家输出权重
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_experts, kernel_size=1)  # 只输出路由专家的权重
        )
        
        # 可学习的温度参数
        self.learnable_temperature = nn.Parameter(torch.ones(1) * temperature)
        
        # 可学习的共享专家权重
        if use_shared_expert:
            self.learnable_shared_weight = nn.Parameter(torch.ones(1) * shared_expert_weight)
        
    def forward(self, features, use_learnable_temp=True, return_full_weights=True):
        """
        前向传播
        
        Args:
            features: 输入特征 [B, C, H, W]
            use_learnable_temp: 是否使用可学习温度
            return_full_weights: 是否返回包含共享专家的完整权重
            
        Returns:
            router_weights: 路由权重
                - 如果 return_full_weights=True 且 use_shared_expert=True:
                  [B, num_experts+1, H, W]，最后一个通道是共享专家权重
                - 否则: [B, num_experts, H, W]
            expert_info: 包含专家信息的字典
        """
        logits = self.router(features)
        
        # 使用温度缩放的softmax
        temp = self.learnable_temperature if use_learnable_temp else self.temperature
        router_weights = F.softmax(logits / temp, dim=1)  # [B, num_experts, H, W]
        
        expert_info = {
            'num_routed_experts': self.num_experts,
            'use_shared_expert': self.use_shared_expert,
            'total_experts': self.total_experts,
        }
        
        if return_full_weights and self.use_shared_expert:
            # 添加共享专家权重通道
            B, _, H, W = router_weights.shape
            shared_weight = self.learnable_shared_weight.expand(B, 1, H, W)
            
            # 归一化：路由专家权重 + 共享专家权重 = 1
            # 路由专家分配 (1 - shared_weight) 的总权重
            scaled_router_weights = router_weights * (1 - shared_weight)
            
            # 拼接：[路由专家权重..., 共享专家权重]
            full_weights = torch.cat([scaled_router_weights, shared_weight], dim=1)
            
            expert_info['shared_expert_idx'] = self.num_experts  # 共享专家的索引
            
            return full_weights, expert_info
        
        return router_weights, expert_info
    
    def get_routing_weights_only(self, features):
        """
        只获取路由专家的权重（不包括共享专家）
        
        Args:
            features: 输入特征 [B, C, H, W]
            
        Returns:
            router_weights: [B, num_experts, H, W] 路由专家权重
        """
        logits = self.router(features)
        router_weights = F.softmax(logits / self.learnable_temperature, dim=1)
        return router_weights
    
    def get_hard_assignment(self, features, threshold=0.5):
        """
        获取硬分配（用于推理和可视化）
        
        Args:
            features: 输入特征 [B, C, H, W]
            threshold: 阈值
            
        Returns:
            hard_assignment: [B, num_experts, H, W] 每个位置的专家分配
        """
        router_weights, _ = self.forward(features, use_learnable_temp=False, return_full_weights=False)
        # 取最大权重的专家
        hard_assignment = torch.argmax(router_weights, dim=1, keepdim=True)
        return hard_assignment


class MultiScaleMoERouter(nn.Module):
    """
    多尺度MoE路由器
    
    融合多尺度特征进行路由决策，提高分离精度。
    
    Args:
        feat_dims: 各尺度特征维度列表，如 [32, 48, 96]
        num_experts: 路由专家数量
        hidden_channels: 隐藏层通道数
        use_shared_expert: 是否使用共享专家
        shared_expert_weight: 共享专家权重
    """
    
    def __init__(self, feat_dims, num_experts=2, hidden_channels=64,
                 use_shared_expert=True, shared_expert_weight=0.5):
        super().__init__()
        self.num_experts = num_experts
        self.use_shared_expert = use_shared_expert
        self.shared_expert_weight = shared_expert_weight
        self.total_experts = num_experts + 1 if use_shared_expert else num_experts
        
        # 各尺度特征的投影层
        self.projectors = nn.ModuleList([
            nn.Conv2d(dim, hidden_channels, kernel_size=1)
            for dim in feat_dims
        ])
        
        # 融合后的路由网络
        self.router = nn.Sequential(
            nn.Conv2d(hidden_channels * len(feat_dims), hidden_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels * 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_experts, kernel_size=1)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
        if use_shared_expert:
            self.learnable_shared_weight = nn.Parameter(torch.ones(1) * shared_expert_weight)
        
    def forward(self, multi_scale_features, return_full_weights=True):
        """
        前向传播
        
        Args:
            multi_scale_features: 多尺度特征列表 [(B,C1,H1,W1), (B,C2,H2,W2), ...]
            return_full_weights: 是否返回包含共享专家的完整权重
            
        Returns:
            router_weights: 路由权重
            expert_info: 专家信息字典
        """
        # 获取目标尺寸（最高分辨率）
        target_size = multi_scale_features[0].shape[2:]
        
        # 投影并上采样各尺度特征
        projected_feats = []
        for i, feat in enumerate(multi_scale_features):
            proj = self.projectors[i](feat)
            if proj.shape[2:] != target_size:
                proj = F.interpolate(proj, size=target_size, mode='bilinear', align_corners=False)
            projected_feats.append(proj)
        
        # 融合多尺度特征
        fused_feat = torch.cat(projected_feats, dim=1)
        
        # 路由决策
        logits = self.router(fused_feat)
        router_weights = F.softmax(logits / self.temperature, dim=1)
        
        expert_info = {
            'num_routed_experts': self.num_experts,
            'use_shared_expert': self.use_shared_expert,
            'total_experts': self.total_experts,
        }
        
        if return_full_weights and self.use_shared_expert:
            B, _, H, W = router_weights.shape
            shared_weight = self.learnable_shared_weight.expand(B, 1, H, W)
            scaled_router_weights = router_weights * (1 - shared_weight)
            full_weights = torch.cat([scaled_router_weights, shared_weight], dim=1)
            expert_info['shared_expert_idx'] = self.num_experts
            return full_weights, expert_info
        
        return router_weights, expert_info


class DepthAwareMoERouter(nn.Module):
    """
    深度感知MoE路由器
    
    结合深度信息进行路由决策，利用深度不连续性辅助分离。
    
    Args:
        img_channels: 图像特征通道数
        depth_channels: 深度特征通道数（默认1）
        num_experts: 路由专家数量
        hidden_channels: 隐藏层通道数
        use_shared_expert: 是否使用共享专家
        shared_expert_weight: 共享专家权重
    """
    
    def __init__(self, img_channels, depth_channels=1, num_experts=2, hidden_channels=64,
                 use_shared_expert=True, shared_expert_weight=0.5):
        super().__init__()
        self.num_experts = num_experts
        self.use_shared_expert = use_shared_expert
        self.shared_expert_weight = shared_expert_weight
        self.total_experts = num_experts + 1 if use_shared_expert else num_experts
        
        # 图像特征编码
        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 深度特征编码（包括深度梯度）
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels + 2, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True)
        )
        
        # 融合路由网络
        self.router = nn.Sequential(
            nn.Conv2d(hidden_channels + hidden_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels // 2, num_experts, kernel_size=1)
        )
        
        self.temperature = nn.Parameter(torch.ones(1))
        
        if use_shared_expert:
            self.learnable_shared_weight = nn.Parameter(torch.ones(1) * shared_expert_weight)
        
        # Sobel算子用于计算深度梯度
        self.register_buffer('sobel_x', torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]).float().view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]).float().view(1, 1, 3, 3))
        
    def compute_depth_gradient(self, depth):
        """计算深度梯度"""
        grad_x = F.conv2d(depth, self.sobel_x, padding=1)
        grad_y = F.conv2d(depth, self.sobel_y, padding=1)
        return grad_x, grad_y
        
    def forward(self, img_features, depth, return_full_weights=True):
        """
        前向传播
        
        Args:
            img_features: 图像特征 [B, C, H, W]
            depth: 深度图 [B, 1, H, W]
            return_full_weights: 是否返回包含共享专家的完整权重
            
        Returns:
            router_weights: 路由权重
            expert_info: 专家信息字典
        """
        # 编码图像特征
        img_feat = self.img_encoder(img_features)
        
        # 计算深度梯度并编码
        grad_x, grad_y = self.compute_depth_gradient(depth)
        depth_with_grad = torch.cat([depth, grad_x, grad_y], dim=1)
        depth_feat = self.depth_encoder(depth_with_grad)
        
        # 融合并路由
        fused_feat = torch.cat([img_feat, depth_feat], dim=1)
        logits = self.router(fused_feat)
        router_weights = F.softmax(logits / self.temperature, dim=1)
        
        expert_info = {
            'num_routed_experts': self.num_experts,
            'use_shared_expert': self.use_shared_expert,
            'total_experts': self.total_experts,
        }
        
        if return_full_weights and self.use_shared_expert:
            B, _, H, W = router_weights.shape
            shared_weight = self.learnable_shared_weight.expand(B, 1, H, W)
            scaled_router_weights = router_weights * (1 - shared_weight)
            full_weights = torch.cat([scaled_router_weights, shared_weight], dim=1)
            expert_info['shared_expert_idx'] = self.num_experts
            return full_weights, expert_info
        
        return router_weights, expert_info


def create_moe_router(cfg, in_channels):
    """
    创建MoE路由器
    
    Args:
        cfg: 配置对象
        in_channels: 输入通道数
        
    Returns:
        router: MoE路由器实例
    """
    moe_cfg = getattr(cfg, 'moe', None)
    if moe_cfg is None:
        # 默认配置
        return MoERouter(
            in_channels=in_channels,
            num_experts=2,
            hidden_channels=64,
            temperature=1.0,
            use_shared_expert=True,
            shared_expert_weight=0.5
        )
    
    router_type = getattr(moe_cfg, 'router_type', 'basic')
    num_experts = getattr(moe_cfg, 'num_experts', 2)
    hidden_channels = getattr(moe_cfg, 'router_channels', 64)
    use_shared_expert = getattr(moe_cfg, 'use_shared_expert', True)
    shared_expert_weight = getattr(moe_cfg, 'shared_expert_weight', 0.5)
    
    if router_type == 'basic':
        return MoERouter(
            in_channels=in_channels,
            num_experts=num_experts,
            hidden_channels=hidden_channels,
            use_shared_expert=use_shared_expert,
            shared_expert_weight=shared_expert_weight
        )
    elif router_type == 'multiscale':
        feat_dims = getattr(cfg.raft, 'encoder_dims', [32, 48, 96])
        return MultiScaleMoERouter(
            feat_dims=feat_dims,
            num_experts=num_experts,
            hidden_channels=hidden_channels,
            use_shared_expert=use_shared_expert,
            shared_expert_weight=shared_expert_weight
        )
    elif router_type == 'depth_aware':
        return DepthAwareMoERouter(
            img_channels=in_channels,
            num_experts=num_experts,
            hidden_channels=hidden_channels,
            use_shared_expert=use_shared_expert,
            shared_expert_weight=shared_expert_weight
        )
    else:
        raise ValueError(f"未知的路由器类型: {router_type}")
