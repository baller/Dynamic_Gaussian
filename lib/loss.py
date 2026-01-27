"""
GPS_plus 损失函数模块

包含:
1. 原始光流序列损失 (RAFT)
2. 深度一致性损失 (DA3 深度融合)
3. MoE 负载均衡损失
4. 高斯分配熵正则
5. 边缘感知深度平滑损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    flow_loss = 0.0

    valid = (valid >= 0.5)
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        # assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()  # 
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()  # 
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'train_epe': epe.mean().item(),
        'train_1px': (epe < 1).float().mean().item(),
        'train_3px': (epe < 3).float().mean().item()
    }

    return flow_loss, metrics


class DepthConsistencyLoss(nn.Module):
    """
    深度一致性损失
    
    确保左右视图的深度在几何上一致
    使用立体几何约束: disp = baseline * fx / depth
    """
    
    def __init__(self, robust: bool = True, eps: float = 1e-6):
        super().__init__()
        self.robust = robust
        self.eps = eps
    
    def forward(
        self, 
        depth_l: torch.Tensor, 
        depth_r: torch.Tensor,
        intrinsics: torch.Tensor,
        baseline: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算深度一致性损失
        
        Args:
            depth_l: [B, 1, H, W] - 左视图深度
            depth_r: [B, 1, H, W] - 右视图深度
            intrinsics: [B, 3, 3] - 相机内参
            baseline: [B] - 基线距离
            mask: [B, 1, H, W] - 有效区域掩码 (可选)
            
        Returns:
            loss: 深度一致性损失
        """
        B, _, H, W = depth_l.shape
        
        # 获取焦距
        fx = intrinsics[:, 0, 0].view(B, 1, 1, 1)
        
        # 计算视差
        disp_l = baseline.view(B, 1, 1, 1) * fx / (depth_l + self.eps)
        disp_r = baseline.view(B, 1, 1, 1) * fx / (depth_r + self.eps)
        
        # 一致性误差
        diff = torch.abs(disp_l - disp_r)
        
        if self.robust:
            # 使用 Huber loss (对异常值更鲁棒)
            delta = 1.0
            loss = torch.where(
                diff < delta,
                0.5 * diff ** 2,
                delta * (diff - 0.5 * delta)
            )
        else:
            loss = diff
        
        # 应用掩码
        if mask is not None:
            loss = loss * mask
            loss = loss.sum() / (mask.sum() + self.eps)
        else:
            loss = loss.mean()
        
        return loss


class MoEBalanceLoss(nn.Module):
    """
    MoE 负载均衡损失
    
    确保所有专家被均匀使用，避免路由坍塌
    """
    
    def __init__(self, num_experts: int = 8, auxiliary_weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.auxiliary_weight = auxiliary_weight
    
    def forward(
        self, 
        router_weights: torch.Tensor,
        top_k_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算负载均衡损失
        
        Args:
            router_weights: [B, N, num_experts] - 路由权重
            top_k_indices: [B, N, top_k] - top-k 专家索引 (可选)
            
        Returns:
            loss: 负载均衡损失
        """
        B, N, E = router_weights.shape
        
        # 方法1: 简单的均匀性损失
        # 计算每个专家的平均使用率
        mean_usage = router_weights.mean(dim=(0, 1))  # [E]
        
        # 目标: 每个专家使用率 = 1/E
        target = 1.0 / E
        
        # 使用 KL 散度
        mean_usage = mean_usage.clamp(min=1e-8)
        loss = (mean_usage * (torch.log(mean_usage) - torch.log(torch.tensor(target, device=router_weights.device)))).sum()
        
        return loss * self.auxiliary_weight


class AllocationEntropyLoss(nn.Module):
    """
    分配熵损失
    
    确保高斯分配具有合理的分布：
    - 不要退化为全选或全不选
    - 分配与纹理复杂度正相关
    """
    
    def __init__(self, target_entropy: float = 0.5):
        super().__init__()
        self.target_entropy = target_entropy
    
    def forward(
        self, 
        valid_mask: torch.Tensor,
        texture_complexity: Optional[torch.Tensor] = None,
        num_layers: int = 3
    ) -> torch.Tensor:
        """
        计算分配熵损失
        
        Args:
            valid_mask: [B, L, H, W] - 有效高斯掩码
            texture_complexity: [B, 1, H, W] - 纹理复杂度 (可选)
            num_layers: 最大层数
            
        Returns:
            loss: 分配熵损失
        """
        B, L, H, W = valid_mask.shape
        eps = 1e-8
        
        # 每像素保留的层数
        layers_per_pixel = valid_mask.float().sum(dim=1)  # [B, H, W]
        
        # 归一化到 [0, 1]
        p = layers_per_pixel / (L + eps)  # [B, H, W]
        
        # 计算二元熵
        entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
        
        # 如果有纹理复杂度，让分配与之相关
        if texture_complexity is not None:
            complexity = texture_complexity.squeeze(1)  # [B, H, W]
            # 期望: 复杂度高 -> 层数多
            target_layers = complexity * L
            correlation_loss = F.mse_loss(layers_per_pixel, target_layers)
            
            # 总损失 = 熵正则 + 相关性损失
            loss = -entropy.mean() + correlation_loss
        else:
            # 仅熵正则: 鼓励适中的熵
            loss = (entropy - self.target_entropy).abs().mean()
        
        return loss


class EdgeAwareDepthSmoothLoss(nn.Module):
    """
    边缘感知深度平滑损失
    
    在低纹理区域强制深度平滑，在边缘区域允许深度不连续
    """
    
    def __init__(self, lambda_smooth: float = 1.0):
        super().__init__()
        self.lambda_smooth = lambda_smooth
    
    def forward(
        self, 
        depth: torch.Tensor, 
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算边缘感知平滑损失
        
        Args:
            depth: [B, 1, H, W] - 深度图
            image: [B, 3, H, W] - RGB 图像
            mask: [B, 1, H, W] - 有效区域掩码 (可选)
            
        Returns:
            loss: 平滑损失
        """
        # 计算深度梯度
        depth_dx = depth[:, :, :, 1:] - depth[:, :, :, :-1]  # [B, 1, H, W-1]
        depth_dy = depth[:, :, 1:, :] - depth[:, :, :-1, :]  # [B, 1, H-1, W]
        
        # 计算图像梯度 (用于边缘检测)
        image_gray = image.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        image_dx = image_gray[:, :, :, 1:] - image_gray[:, :, :, :-1]
        image_dy = image_gray[:, :, 1:, :] - image_gray[:, :, :-1, :]
        
        # 边缘权重: 图像梯度大的地方权重小
        weight_x = torch.exp(-self.lambda_smooth * torch.abs(image_dx))
        weight_y = torch.exp(-self.lambda_smooth * torch.abs(image_dy))
        
        # 加权深度梯度
        smooth_x = torch.abs(depth_dx) * weight_x
        smooth_y = torch.abs(depth_dy) * weight_y
        
        # 应用掩码
        if mask is not None:
            mask_x = mask[:, :, :, 1:] * mask[:, :, :, :-1]
            mask_y = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            
            smooth_x = smooth_x * mask_x
            smooth_y = smooth_y * mask_y
            
            loss = (smooth_x.sum() + smooth_y.sum()) / (mask_x.sum() + mask_y.sum() + 1e-8)
        else:
            loss = smooth_x.mean() + smooth_y.mean()
        
        return loss


class CombinedLoss(nn.Module):
    """
    组合损失函数
    
    整合所有损失项，便于训练使用
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 获取损失权重
        loss_cfg = getattr(cfg, 'loss', None)
        if loss_cfg is not None:
            self.l1_weight = getattr(loss_cfg, 'l1_weight', 0.8)
            self.ssim_weight = getattr(loss_cfg, 'ssim_weight', 0.2)
            self.chamfer_weight = getattr(loss_cfg, 'chamfer_weight', 0.5)
            self.depth_consistency_weight = getattr(loss_cfg, 'depth_consistency_weight', 0.1)
            self.moe_balance_weight = getattr(loss_cfg, 'moe_balance_weight', 0.01)
            self.allocation_entropy_weight = getattr(loss_cfg, 'allocation_entropy_weight', 0.001)
        else:
            self.l1_weight = 0.8
            self.ssim_weight = 0.2
            self.chamfer_weight = 0.5
            self.depth_consistency_weight = 0.1
            self.moe_balance_weight = 0.01
            self.allocation_entropy_weight = 0.001
        
        # 初始化损失模块
        self.depth_consistency_loss = DepthConsistencyLoss()
        self.moe_balance_loss = MoEBalanceLoss()
        self.allocation_entropy_loss = AllocationEntropyLoss()
        self.edge_smooth_loss = EdgeAwareDepthSmoothLoss()
    
    def forward(
        self, 
        data: Dict,
        l1_loss: torch.Tensor,
        ssim_loss: torch.Tensor,
        chamfer_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        计算总损失
        
        Args:
            data: 模型输出数据
            l1_loss: L1 渲染损失
            ssim_loss: SSIM 损失
            chamfer_loss: Chamfer 距离损失 (可选)
            
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {
            'l1': l1_loss.item(),
            'ssim': ssim_loss.item(),
        }
        
        # 主损失
        total_loss = self.l1_weight * l1_loss + self.ssim_weight * ssim_loss
        
        # Chamfer 损失
        if chamfer_loss is not None:
            total_loss = total_loss + self.chamfer_weight * chamfer_loss
            loss_dict['chamfer'] = chamfer_loss.item()
        
        # 深度一致性损失 (如果有深度融合输出)
        if 'fusion_aux' in data and self.depth_consistency_weight > 0:
            aux = data['fusion_aux']
            depth_l = aux.get('depth_l_metric')
            depth_r = aux.get('depth_r_metric')
            
            if depth_l is not None and depth_r is not None:
                intr = data['lmain']['intr']
                baseline = self._compute_baseline(data['lmain']['extr'], data['rmain']['extr'])
                
                dc_loss = self.depth_consistency_loss(depth_l, depth_r, intr, baseline)
                total_loss = total_loss + self.depth_consistency_weight * dc_loss
                loss_dict['depth_consistency'] = dc_loss.item()
        
        # MoE 负载均衡损失 (如果使用 Transformer+MoE)
        if 'moe_router_weights' in data and self.moe_balance_weight > 0:
            router_weights = data['moe_router_weights']
            if isinstance(router_weights, list) and len(router_weights) > 0:
                moe_loss = sum(self.moe_balance_loss(rw) for rw in router_weights)
                moe_loss = moe_loss / len(router_weights)
                total_loss = total_loss + moe_loss
                loss_dict['moe_balance'] = moe_loss.item()
        
        return total_loss, loss_dict
    
    def _compute_baseline(self, extr_l: torch.Tensor, extr_r: torch.Tensor) -> torch.Tensor:
        """计算基线距离"""
        if extr_l.shape[1] == 4:
            t_l = extr_l[:, :3, 3]
            t_r = extr_r[:, :3, 3]
        else:
            t_l = extr_l[:, :, 3]
            t_r = extr_r[:, :, 3]
        return torch.norm(t_l - t_r, dim=1)


def create_combined_loss(cfg) -> CombinedLoss:
    """
    创建组合损失函数
    
    Args:
        cfg: 配置对象
        
    Returns:
        CombinedLoss 实例
    """
    return CombinedLoss(cfg)
