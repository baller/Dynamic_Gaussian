"""
MoE (Mixture of Experts) 路由器模块

实现动态的背景/人体特征分离，通过可学习的软路由网络
输出像素级的路由权重，用于指导双流高斯参数预测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoERouter(nn.Module):
    """
    MoE路由器 - 输出像素级的背景/人体路由权重
    
    使用软路由（soft routing）而非硬分割，允许梯度反传。
    路由权重可作为"伪mask"用于分离特征。
    
    Args:
        in_channels: 输入特征通道数
        num_experts: 专家数量（默认2：背景+人体）
        hidden_channels: 隐藏层通道数
        temperature: softmax温度参数，控制分布锐度
    """
    
    def __init__(self, in_channels, num_experts=2, hidden_channels=64, temperature=1.0):
        super().__init__()
        self.num_experts = num_experts
        self.temperature = temperature
        
        # 多尺度特征融合路由网络
        self.router = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, num_experts, kernel_size=1)
        )
        
        # 可学习的温度参数
        self.learnable_temperature = nn.Parameter(torch.ones(1) * temperature)
        
    def forward(self, features, use_learnable_temp=True):
        """
        前向传播
        
        Args:
            features: 输入特征 [B, C, H, W]
            use_learnable_temp: 是否使用可学习温度
            
        Returns:
            router_weights: 路由权重 [B, num_experts, H, W]
                           channel 0: 背景权重
                           channel 1: 人体权重
        """
        logits = self.router(features)
        
        # 使用温度缩放的softmax
        temp = self.learnable_temperature if use_learnable_temp else self.temperature
        router_weights = F.softmax(logits / temp, dim=1)
        
        return router_weights
    
    def get_hard_assignment(self, features, threshold=0.5):
        """
        获取硬分配（用于推理和可视化）
        
        Args:
            features: 输入特征 [B, C, H, W]
            threshold: 阈值
            
        Returns:
            hard_assignment: 硬分配 [B, 1, H, W]，1表示人体，0表示背景
        """
        router_weights = self.forward(features, use_learnable_temp=False)
        # 人体权重大于阈值则为1
        hard_assignment = (router_weights[:, 1:2] > threshold).float()
        return hard_assignment


class MultiScaleMoERouter(nn.Module):
    """
    多尺度MoE路由器
    
    融合多尺度特征进行路由决策，提高分离精度。
    
    Args:
        feat_dims: 各尺度特征维度列表，如 [32, 48, 96]
        num_experts: 专家数量
        hidden_channels: 隐藏层通道数
    """
    
    def __init__(self, feat_dims, num_experts=2, hidden_channels=64):
        super().__init__()
        self.num_experts = num_experts
        
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
        
    def forward(self, multi_scale_features):
        """
        前向传播
        
        Args:
            multi_scale_features: 多尺度特征列表 [(B,C1,H1,W1), (B,C2,H2,W2), ...]
            
        Returns:
            router_weights: 路由权重 [B, num_experts, H, W] (最高分辨率)
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
        
        return router_weights


class DepthAwareMoERouter(nn.Module):
    """
    深度感知MoE路由器
    
    结合深度信息进行路由决策，利用深度不连续性辅助分离。
    
    Args:
        img_channels: 图像特征通道数
        depth_channels: 深度特征通道数（默认1）
        num_experts: 专家数量
        hidden_channels: 隐藏层通道数
    """
    
    def __init__(self, img_channels, depth_channels=1, num_experts=2, hidden_channels=64):
        super().__init__()
        self.num_experts = num_experts
        
        # 图像特征编码
        self.img_encoder = nn.Sequential(
            nn.Conv2d(img_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 深度特征编码（包括深度梯度）
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_channels + 2, hidden_channels // 2, kernel_size=3, padding=1),  # +2 for gradients
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
        
    def forward(self, img_features, depth):
        """
        前向传播
        
        Args:
            img_features: 图像特征 [B, C, H, W]
            depth: 深度图 [B, 1, H, W]
            
        Returns:
            router_weights: 路由权重 [B, num_experts, H, W]
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
        
        return router_weights


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
            temperature=1.0
        )
    
    router_type = getattr(moe_cfg, 'router_type', 'basic')
    num_experts = getattr(moe_cfg, 'num_experts', 2)
    hidden_channels = getattr(moe_cfg, 'router_channels', 64)
    
    if router_type == 'basic':
        return MoERouter(
            in_channels=in_channels,
            num_experts=num_experts,
            hidden_channels=hidden_channels
        )
    elif router_type == 'multiscale':
        feat_dims = getattr(cfg.raft, 'encoder_dims', [32, 48, 96])
        return MultiScaleMoERouter(
            feat_dims=feat_dims,
            num_experts=num_experts,
            hidden_channels=hidden_channels
        )
    elif router_type == 'depth_aware':
        return DepthAwareMoERouter(
            img_channels=in_channels,
            num_experts=num_experts,
            hidden_channels=hidden_channels
        )
    else:
        raise ValueError(f"未知的路由器类型: {router_type}")
