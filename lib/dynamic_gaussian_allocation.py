"""
动态高斯分配模块

实现基于纹理复杂度的动态高斯数量分配：
- 纹理复杂区域: 分配更多高斯，保留多层
- 平滑区域: 减少高斯数量，仅保留第一层

核心设计：
1. 每像素预测 L 层高斯 (L=3 或 4)
2. 通过 Opacity 阈值裁剪无效高斯
3. 结合纹理复杂度调整保留策略
4. 展平输出为点云格式供渲染

优势：
- 自适应高斯数量，计算更高效
- 复杂区域细节更丰富
- 支持可微训练
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class DynamicGaussianAllocation(nn.Module):
    """
    动态高斯分配模块
    
    Args:
        cfg: 配置对象
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 从配置读取参数
        dgs_cfg = getattr(cfg, 'dynamic_gs', None)
        if dgs_cfg is not None:
            self.num_layers = getattr(dgs_cfg, 'num_layers', 3)
            self.opacity_threshold = getattr(dgs_cfg, 'opacity_threshold', 0.01)
            self.complexity_threshold = getattr(dgs_cfg, 'complexity_threshold', 0.3)
            self.use_complexity_guidance = getattr(dgs_cfg, 'use_complexity_guidance', True)
            self.gumbel_temperature = getattr(dgs_cfg, 'gumbel_temperature', 1.0)
            self.gumbel_opacity_scale = getattr(dgs_cfg, 'gumbel_opacity_scale', 10.0)
            self.soft_pruning_alpha = getattr(dgs_cfg, 'soft_pruning_alpha', 10.0)
        else:
            self.num_layers = 3
            self.opacity_threshold = 0.01
            self.complexity_threshold = 0.3
            self.use_complexity_guidance = True
            self.gumbel_temperature = 1.0
            self.gumbel_opacity_scale = 10.0
            self.soft_pruning_alpha = 10.0
        
        # 复杂度到保留层数的映射
        if self.use_complexity_guidance:
            self.complexity_to_layers = nn.Sequential(
                nn.Conv2d(1, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, self.num_layers, 1),
                nn.Softmax(dim=1),  # 每层的保留概率
            )
        
        logger.info(f"[DynamicGaussianAllocation] 初始化完成: "
                   f"num_layers={self.num_layers}, opacity_threshold={self.opacity_threshold}, "
                   f"use_complexity_guidance={self.use_complexity_guidance}")
    
    def forward(
        self, 
        gaussian_params: Dict[str, torch.Tensor], 
        texture_complexity: Optional[torch.Tensor] = None,
        step: Optional[int] = None
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            gaussian_params: dict - 高斯参数
                - rotation: [B, 4, L, H, W]
                - scale: [B, 3, L, H, W]
                - opacity: [B, 1, L, H, W]
                - depth_offset: [B, 1, L, H, W]
            texture_complexity: [B, 1, H, W] - 纹理复杂度 (可选)
            
        Returns:
            filtered_params: dict - 过滤后的高斯参数
            valid_mask: [B, L, H, W] - 有效高斯掩码
            stats: dict - 统计信息
        """
        opacity = gaussian_params['opacity']  # [B, 1, L, H, W]
        B, _, L, H, W = opacity.shape
        opacity_squeezed = opacity.squeeze(1)  # [B, L, H, W]
        
        # 方法1: 基于 Opacity 阈值裁剪
        warmup_iters = getattr(self.cfg.dynamic_gs, 'warmup_iters', 0) if hasattr(self.cfg, 'dynamic_gs') else 0
        if step is not None and warmup_iters > 0:
            warmup_ratio = min(1.0, float(step) / float(warmup_iters))
            opacity_threshold = self.opacity_threshold * warmup_ratio
        else:
            opacity_threshold = self.opacity_threshold
        valid_mask = opacity_squeezed > opacity_threshold  # [B, L, H, W]
        
        # 方法2: 结合纹理复杂度指导
        if self.use_complexity_guidance and texture_complexity is not None:
            # 计算每层的保留概率
            layer_probs = self.complexity_to_layers(texture_complexity)  # [B, L, H, W]
            
            if self.training:
                # 训练时：使用 Gumbel-Softmax 实现可微采样
                keep_decision = self._gumbel_softmax_sample(
                    layer_probs,
                    opacity_squeezed,
                    temperature=self.gumbel_temperature
                )
                valid_mask = valid_mask & (keep_decision > 0.5)
            else:
                # 推理时：根据复杂度决定保留层数
                complexity_expanded = texture_complexity.expand(-1, L, -1, -1)  # [B, L, H, W]
                
                # 复杂区域 (complexity > threshold): 保留所有层
                # 简单区域: 只保留第一层
                simple_mask = complexity_expanded[:, 0:1] < self.complexity_threshold
                simple_mask = simple_mask.expand(-1, L, -1, -1)  # [B, L, H, W]
                
                # 对于简单区域，只保留第一层
                layer_mask = torch.ones(L, device=opacity.device)
                layer_mask[1:] = 0  # [L]
                layer_mask = layer_mask.view(1, L, 1, 1).expand(B, -1, H, W)
                
                # 简单区域只保留第一层
                valid_mask = torch.where(simple_mask, valid_mask & (layer_mask > 0), valid_mask)
        
        # 应用掩码到所有参数
        filtered_params = {}
        for key, value in gaussian_params.items():
            if key == 'texture_complexity':
                filtered_params[key] = value
                continue
            
            if value.dim() == 5:  # [B, C, L, H, W]
                if key == 'opacity':
                    # 将无效高斯的 opacity 设为 0
                    filtered_params[key] = value * valid_mask.unsqueeze(1).float()
                else:
                    filtered_params[key] = value
            else:
                filtered_params[key] = value
        
        # 计算统计信息
        total_gaussians = B * L * H * W
        valid_count = valid_mask.sum().item()
        stats = {
            'total_gaussians': total_gaussians,
            'valid_gaussians': valid_count,
            'pruning_ratio': 1.0 - valid_count / total_gaussians,
            'avg_layers_per_pixel': valid_mask.float().sum(dim=1).mean().item(),
            'opacity_threshold': float(opacity_threshold),
        }
        
        return filtered_params, valid_mask, stats
    
    def _gumbel_softmax_sample(
        self, 
        layer_probs: torch.Tensor, 
        opacity: torch.Tensor, 
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        可微的 Gumbel-Softmax 采样
        
        Args:
            layer_probs: [B, L, H, W] - 每层保留概率
            opacity: [B, L, H, W] - 不透明度
            temperature: Gumbel-Softmax 温度
            
        Returns:
            keep_decision: [B, L, H, W] - 保留决策 (软掩码)
        """
        # 结合 opacity 和 layer_probs
        combined = layer_probs * torch.sigmoid(opacity * self.gumbel_opacity_scale)
        
        # Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(combined) + 1e-8) + 1e-8)
        
        # Softmax with temperature
        y = F.softmax((torch.log(combined + 1e-8) + gumbel_noise) / temperature, dim=1)
        
        # 累积和作为阈值 (层0始终保留，层1-2根据概率)
        cumsum = torch.cumsum(y, dim=1)  # [B, L, H, W]
        
        return cumsum
    
    def flatten_gaussians(
        self, 
        gaussian_params: Dict[str, torch.Tensor], 
        valid_mask: torch.Tensor, 
        depth: torch.Tensor, 
        intrinsics: torch.Tensor, 
        extrinsics: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        将多层高斯展平为点云格式
        
        Args:
            gaussian_params: dict - 高斯参数
                - rotation: [B, 4, L, H, W]
                - scale: [B, 3, L, H, W]
                - opacity: [B, 1, L, H, W]
                - depth_offset: [B, 1, L, H, W]
            valid_mask: [B, L, H, W] - 有效掩码
            depth: [B, 1, H, W] - 基础深度图
            intrinsics: [B, 3, 3] - 相机内参
            extrinsics: [B, 3, 4] 或 [B, 4, 4] - 相机外参
            
        Returns:
            flattened_params: dict - 展平后的参数
                - xyz: [B, N, 3] - 3D坐标 (N = L * H * W)
                - rotation: [B, N, 4] - 旋转
                - scale: [B, N, 3] - 缩放
                - opacity: [B, N] - 不透明度
                - valid_mask: [B, N] - 有效掩码
        """
        rotation = gaussian_params['rotation']  # [B, 4, L, H, W]
        scale = gaussian_params['scale']        # [B, 3, L, H, W]
        opacity = gaussian_params['opacity']    # [B, 1, L, H, W]
        depth_offset = gaussian_params['depth_offset']  # [B, 1, L, H, W]
        
        B, _, L, H, W = rotation.shape
        device = rotation.device
        
        # 计算每层的深度
        # 基础深度 + 每层偏移
        layer_depths = depth.unsqueeze(2) + depth_offset  # [B, 1, L, H, W]
        layer_depths = F.relu(layer_depths) + 1e-6  # 确保深度为正
        
        # 生成像素坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  # [B, L, H, W]
        y_coords = y_coords.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        
        # 从内参获取焦距和主点
        fx = intrinsics[:, 0, 0].view(B, 1, 1, 1)  # [B, 1, 1, 1]
        fy = intrinsics[:, 1, 1].view(B, 1, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1, 1)
        
        # 反投影到相机坐标系
        z = layer_depths.squeeze(1)  # [B, L, H, W]
        x = (x_coords - cx) * z / fx
        y = (y_coords - cy) * z / fy
        
        xyz_camera = torch.stack([x, y, z], dim=-1)  # [B, L, H, W, 3]
        
        # 转换到世界坐标系
        # 外参: world_to_camera，需要求逆
        if extrinsics.shape[1] == 4:
            R = extrinsics[:, :3, :3]  # [B, 3, 3]
            t = extrinsics[:, :3, 3]   # [B, 3]
        else:
            R = extrinsics[:, :, :3]  # [B, 3, 3]
            t = extrinsics[:, :, 3]   # [B, 3]
        
        # camera_to_world = inverse(world_to_camera)
        # P_world = R^T @ (P_camera - t)
        # 或者如果外参是 camera_to_world: P_world = R @ P_camera + t
        # 这里假设外参是 world_to_camera
        R_inv = R.transpose(1, 2)  # [B, 3, 3]
        
        # 重塑 xyz_camera: [B, L, H, W, 3] -> [B, L*H*W, 3]
        xyz_camera_flat = xyz_camera.view(B, L * H * W, 3)
        
        # 转换: P_world = R^T @ P_camera - R^T @ t
        xyz_world = torch.bmm(xyz_camera_flat, R_inv.transpose(1, 2))  # [B, N, 3]
        xyz_world = xyz_world - torch.bmm(t.unsqueeze(1), R_inv.transpose(1, 2))
        
        # 展平其他参数
        # [B, C, L, H, W] -> [B, L, H, W, C] -> [B, N, C]
        rotation_flat = rotation.permute(0, 2, 3, 4, 1).reshape(B, L * H * W, 4)
        scale_flat = scale.permute(0, 2, 3, 4, 1).reshape(B, L * H * W, 3)
        opacity_flat = opacity.squeeze(1).reshape(B, L * H * W)  # [B, N]
        valid_flat = valid_mask.reshape(B, L * H * W)  # [B, N]
        
        flattened_params = {
            'xyz': xyz_world,         # [B, N, 3]
            'rotation': rotation_flat, # [B, N, 4]
            'scale': scale_flat,       # [B, N, 3]
            'opacity': opacity_flat,   # [B, N]
            'valid_mask': valid_flat,  # [B, N]
        }
        
        return flattened_params
    
    def filter_by_mask(
        self, 
        flattened_params: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        根据有效掩码过滤高斯点
        
        注意: 这会改变每个 batch 的点数，不适合批量操作
        仅用于推理或可视化
        
        Args:
            flattened_params: dict - 展平后的参数
            
        Returns:
            filtered_params: dict - 过滤后的参数 (list of tensors)
        """
        B = flattened_params['xyz'].shape[0]
        valid_mask = flattened_params['valid_mask']  # [B, N]
        
        filtered = {
            'xyz': [],
            'rotation': [],
            'scale': [],
            'opacity': [],
        }
        
        for b in range(B):
            mask = valid_mask[b]  # [N]
            filtered['xyz'].append(flattened_params['xyz'][b, mask])
            filtered['rotation'].append(flattened_params['rotation'][b, mask])
            filtered['scale'].append(flattened_params['scale'][b, mask])
            filtered['opacity'].append(flattened_params['opacity'][b, mask])
        
        return filtered
    
    def compute_allocation_entropy_loss(
        self, 
        valid_mask: torch.Tensor, 
        texture_complexity: torch.Tensor
    ) -> torch.Tensor:
        """
        计算分配熵损失
        
        目标: 避免分配退化（全平或全尖）
        
        Args:
            valid_mask: [B, L, H, W] - 有效掩码
            texture_complexity: [B, 1, H, W] - 纹理复杂度
            
        Returns:
            loss: 分配熵损失
        """
        B, L, H, W = valid_mask.shape
        
        # 每像素保留的层数
        layers_per_pixel = valid_mask.float().sum(dim=1)  # [B, H, W]
        
        # 归一化为概率分布
        layers_probs = layers_per_pixel / (L + 1e-8)  # [B, H, W]
        
        # 计算熵
        eps = 1e-8
        entropy = -(layers_probs * torch.log(layers_probs + eps) + 
                   (1 - layers_probs) * torch.log(1 - layers_probs + eps))
        
        # 我们希望熵较高（不要退化为全选或全不选）
        # 但也不能太高（需要有区分度）
        # 目标: 让分配与复杂度相关
        
        # 复杂度高的地方应该有更多层
        complexity = texture_complexity.squeeze(1)  # [B, H, W]
        target_layers = complexity * L  # [B, H, W]
        
        # 损失: 实际层数与期望层数的差异
        allocation_loss = F.mse_loss(layers_per_pixel, target_layers)
        
        return allocation_loss


class DynamicGaussianAllocationSimple(nn.Module):
    """
    简化版动态高斯分配
    
    用于与原有渲染流程兼容，直接基于 opacity 进行软裁剪
    不改变张量形状，仅将低 opacity 高斯的贡献降低
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        dgs_cfg = getattr(cfg, 'dynamic_gs', None)
        if dgs_cfg is not None:
            self.opacity_threshold = getattr(dgs_cfg, 'opacity_threshold', 0.01)
            self.soft_pruning = getattr(dgs_cfg, 'soft_pruning', True)
            self.soft_pruning_alpha = getattr(dgs_cfg, 'soft_pruning_alpha', 10.0)
            self.complexity_weight_min = getattr(dgs_cfg, 'complexity_weight_min', 0.5)
            self.complexity_weight_max = getattr(dgs_cfg, 'complexity_weight_max', 1.0)
        else:
            self.opacity_threshold = 0.01
            self.soft_pruning = True
            self.soft_pruning_alpha = 10.0
            self.complexity_weight_min = 0.5
            self.complexity_weight_max = 1.0
    
    def forward(
        self, 
        opacity: torch.Tensor, 
        texture_complexity: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            opacity: [B, 1, H, W] - 不透明度图
            texture_complexity: [B, 1, H, W] - 纹理复杂度 (可选)
            
        Returns:
            adjusted_opacity: [B, 1, H, W] - 调整后的不透明度
            stats: dict - 统计信息
        """
        if self.soft_pruning:
            # 软裁剪: 使用 sigmoid 平滑过渡
            # opacity < threshold 时接近 0
            # opacity > threshold 时接近原值
            mask = torch.sigmoid(self.soft_pruning_alpha * (opacity - self.opacity_threshold))
            adjusted_opacity = opacity * mask
        else:
            # 硬裁剪
            mask = (opacity > self.opacity_threshold).float()
            adjusted_opacity = opacity * mask
        
        # 结合纹理复杂度调整 (可选)
        if texture_complexity is not None:
            # 复杂区域: 保持或增强 opacity
            # 简单区域: 降低 opacity (更激进的裁剪)
            complexity_weight = self.complexity_weight_min + (
                self.complexity_weight_max - self.complexity_weight_min
            ) * texture_complexity
            adjusted_opacity = adjusted_opacity * complexity_weight
        
        # 统计
        stats = {
            'mean_opacity_before': opacity.mean().item(),
            'mean_opacity_after': adjusted_opacity.mean().item(),
            'pruning_ratio': (adjusted_opacity < self.opacity_threshold).float().mean().item(),
        }
        
        return adjusted_opacity, stats


def create_dynamic_gaussian_allocation(cfg) -> DynamicGaussianAllocation:
    """
    创建动态高斯分配模块
    
    Args:
        cfg: 配置对象
        
    Returns:
        DynamicGaussianAllocation 实例
    """
    return DynamicGaussianAllocation(cfg)


def create_dynamic_gaussian_allocation_simple(cfg) -> DynamicGaussianAllocationSimple:
    """
    创建简化版动态高斯分配模块
    
    Args:
        cfg: 配置对象
        
    Returns:
        DynamicGaussianAllocationSimple 实例
    """
    return DynamicGaussianAllocationSimple(cfg)
