"""
可学习的高斯数量分配网络

根据场景复杂度动态分配高斯数量给背景和人体区域。
借鉴ml-sharp的MultiLayerInitializer思路。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianAllocationNet(nn.Module):
    """
    可学习的高斯数量分配网络
    
    根据输入特征和路由权重，动态预测背景和人体的高斯分配比例。
    
    Args:
        feat_dim: 输入特征维度
        total_gaussians: 总高斯数量
        min_ratio: 最小分配比例（防止退化）
        num_experts: 专家数量
    """
    
    def __init__(self, feat_dim, total_gaussians=1024*1024, min_ratio=0.1, num_experts=2):
        super().__init__()
        self.total = total_gaussians
        self.min_ratio = min_ratio
        self.num_experts = num_experts
        
        # 全局特征提取
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # 分配预测网络
        self.allocator = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_experts)
        )
        
        # 路由统计分支（利用路由权重的统计信息）
        self.router_stats = nn.Sequential(
            nn.Linear(num_experts * 4, 32),  # mean, std, max, area_ratio per expert
            nn.ReLU(inplace=True),
            nn.Linear(32, num_experts)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(num_experts * 2, num_experts),
        )
        
    def compute_router_stats(self, router_weights):
        """
        计算路由权重的统计信息
        
        Args:
            router_weights: [B, num_experts, H, W]
            
        Returns:
            stats: [B, num_experts * 4]
        """
        B = router_weights.shape[0]
        stats_list = []
        
        for i in range(self.num_experts):
            weights_i = router_weights[:, i]  # [B, H, W]
            mean_i = weights_i.mean(dim=[1, 2])  # [B]
            std_i = weights_i.std(dim=[1, 2])  # [B]
            max_i = weights_i.amax(dim=[1, 2])  # [B]
            # 面积比例（权重>0.5的像素比例）
            area_i = (weights_i > 0.5).float().mean(dim=[1, 2])  # [B]
            stats_list.extend([mean_i, std_i, max_i, area_i])
        
        stats = torch.stack(stats_list, dim=1)  # [B, num_experts * 4]
        return stats
        
    def forward(self, features, router_weights=None):
        """
        前向传播
        
        Args:
            features: 输入特征 [B, C, H, W]
            router_weights: 路由权重 [B, num_experts, H, W]，可选
            
        Returns:
            allocation_ratio: 分配比例 [B, num_experts]
            bg_count: 背景高斯数量 [B]
            human_count: 人体高斯数量 [B]
        """
        B = features.shape[0]
        
        # 全局特征
        global_feat = self.global_pool(features).view(B, -1)
        feat_pred = self.allocator(global_feat)  # [B, num_experts]
        
        if router_weights is not None:
            # 利用路由统计信息
            router_stats = self.compute_router_stats(router_weights)
            router_pred = self.router_stats(router_stats)  # [B, num_experts]
            
            # 融合两种预测
            combined = torch.cat([feat_pred, router_pred], dim=1)
            ratio_logits = self.fusion(combined)
        else:
            ratio_logits = feat_pred
        
        # Softmax得到分配比例
        ratio = F.softmax(ratio_logits, dim=1)
        
        # 确保最小分配比例
        # ratio = ratio * (1 - num_experts * min_ratio) + min_ratio
        ratio = ratio * (1 - self.num_experts * self.min_ratio) + self.min_ratio
        
        # 计算实际高斯数量
        bg_count = (self.total * ratio[:, 0]).int()
        human_count = (self.total * ratio[:, 1]).int()
        
        return ratio, bg_count, human_count


class AdaptiveGaussianSampler(nn.Module):
    """
    自适应高斯采样器
    
    根据分配比例从高斯参数图中采样指定数量的高斯。
    支持基于重要性的采样策略。
    
    Args:
        sampling_mode: 采样模式 ('uniform', 'importance', 'hybrid')
    """
    
    def __init__(self, sampling_mode='importance'):
        super().__init__()
        self.sampling_mode = sampling_mode
        
    def uniform_sample(self, params, num_samples, valid_mask=None):
        """
        均匀采样
        
        Args:
            params: 高斯参数 [B, C, H, W]
            num_samples: 采样数量
            valid_mask: 有效区域mask [B, 1, H, W]
            
        Returns:
            sampled_params: [B, num_samples, C]
            sample_indices: [B, num_samples]
        """
        B, C, H, W = params.shape
        params_flat = params.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        
        if valid_mask is not None:
            valid_flat = valid_mask.view(B, -1)  # [B, H*W]
        else:
            valid_flat = torch.ones(B, H * W, device=params.device)
        
        sampled_params_list = []
        sample_indices_list = []
        
        for b in range(B):
            valid_indices = valid_flat[b].nonzero(as_tuple=True)[0]
            n_valid = len(valid_indices)
            
            if n_valid >= num_samples:
                # 随机选择
                perm = torch.randperm(n_valid, device=params.device)[:num_samples]
                indices = valid_indices[perm]
            else:
                # 重复填充
                repeat_times = (num_samples // n_valid) + 1
                indices = valid_indices.repeat(repeat_times)[:num_samples]
            
            sampled_params_list.append(params_flat[b, indices])
            sample_indices_list.append(indices)
        
        sampled_params = torch.stack(sampled_params_list, dim=0)
        sample_indices = torch.stack(sample_indices_list, dim=0)
        
        return sampled_params, sample_indices
    
    def importance_sample(self, params, num_samples, importance_weights, valid_mask=None):
        """
        基于重要性的采样
        
        Args:
            params: 高斯参数 [B, C, H, W]
            num_samples: 采样数量
            importance_weights: 重要性权重 [B, 1, H, W]
            valid_mask: 有效区域mask [B, 1, H, W]
            
        Returns:
            sampled_params: [B, num_samples, C]
            sample_indices: [B, num_samples]
        """
        B, C, H, W = params.shape
        params_flat = params.view(B, C, -1).permute(0, 2, 1)  # [B, H*W, C]
        weights_flat = importance_weights.view(B, -1)  # [B, H*W]
        
        if valid_mask is not None:
            valid_flat = valid_mask.view(B, -1)
            weights_flat = weights_flat * valid_flat
        
        # 归一化权重为概率
        weights_sum = weights_flat.sum(dim=1, keepdim=True).clamp(min=1e-8)
        probs = weights_flat / weights_sum
        
        sampled_params_list = []
        sample_indices_list = []
        
        for b in range(B):
            # 使用多项式分布采样
            indices = torch.multinomial(probs[b], num_samples, replacement=True)
            sampled_params_list.append(params_flat[b, indices])
            sample_indices_list.append(indices)
        
        sampled_params = torch.stack(sampled_params_list, dim=0)
        sample_indices = torch.stack(sample_indices_list, dim=0)
        
        return sampled_params, sample_indices
    
    def forward(self, params, num_samples, importance_weights=None, valid_mask=None):
        """
        前向传播
        
        Args:
            params: 高斯参数 [B, C, H, W]
            num_samples: 采样数量（标量或[B]）
            importance_weights: 重要性权重 [B, 1, H, W]，可选
            valid_mask: 有效区域mask [B, 1, H, W]，可选
            
        Returns:
            sampled_params: 采样的高斯参数
            sample_indices: 采样索引
        """
        if self.sampling_mode == 'uniform' or importance_weights is None:
            return self.uniform_sample(params, num_samples, valid_mask)
        elif self.sampling_mode == 'importance':
            return self.importance_sample(params, num_samples, importance_weights, valid_mask)
        elif self.sampling_mode == 'hybrid':
            # 混合采样：一半均匀，一半重要性
            n_uniform = num_samples // 2
            n_importance = num_samples - n_uniform
            
            uniform_params, uniform_idx = self.uniform_sample(params, n_uniform, valid_mask)
            importance_params, importance_idx = self.importance_sample(
                params, n_importance, importance_weights, valid_mask
            )
            
            sampled_params = torch.cat([uniform_params, importance_params], dim=1)
            sample_indices = torch.cat([uniform_idx, importance_idx], dim=1)
            
            return sampled_params, sample_indices
        else:
            raise ValueError(f"未知的采样模式: {self.sampling_mode}")


def create_gaussian_allocator(cfg):
    """
    创建高斯分配器
    
    Args:
        cfg: 配置对象
        
    Returns:
        allocator: GaussianAllocationNet实例
    """
    moe_cfg = getattr(cfg, 'moe', None)
    
    # 使用encoder的第一层特征维度（与lr_img_feat[0]匹配）
    # encoder_dims: [32, 48, 96] -> 第一层是32
    feat_dim = cfg.raft.encoder_dims[0]  # 32
    
    if moe_cfg is None:
        # 默认配置
        return GaussianAllocationNet(
            feat_dim=feat_dim,
            total_gaussians=1024 * 1024,
            min_ratio=0.1,
            num_experts=2
        )
    
    alloc_cfg = getattr(moe_cfg, 'allocation', None)
    if alloc_cfg is None:
        return GaussianAllocationNet(
            feat_dim=feat_dim,
            total_gaussians=1024 * 1024,
            min_ratio=0.1,
            num_experts=getattr(moe_cfg, 'num_experts', 2)
        )
    
    return GaussianAllocationNet(
        feat_dim=feat_dim,
        total_gaussians=getattr(alloc_cfg, 'total_gaussians', 1024 * 1024),
        min_ratio=getattr(alloc_cfg, 'min_ratio', 0.1),
        num_experts=getattr(moe_cfg, 'num_experts', 2)
    )
