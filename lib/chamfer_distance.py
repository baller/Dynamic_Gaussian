"""
Chamfer Distance 实现

替代 pytorch3d.loss.chamfer_distance，不依赖 pytorch3d 库
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional


def chamfer_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    x_lengths: Optional[torch.Tensor] = None,
    y_lengths: Optional[torch.Tensor] = None,
    x_normals: Optional[torch.Tensor] = None,
    y_normals: Optional[torch.Tensor] = None,
    weights: Optional[torch.Tensor] = None,
    batch_reduction: str = "mean",
    point_reduction: str = "mean",
    norm: int = 2,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    计算两个点云之间的 Chamfer Distance
    
    Args:
        x: 第一个点云 [B, N, 3] 或 [N, 3]
        y: 第二个点云 [B, M, 3] 或 [M, 3]
        x_lengths: 每个batch中x的实际点数 (未使用，保持接口兼容)
        y_lengths: 每个batch中y的实际点数 (未使用，保持接口兼容)
        x_normals: x的法向量 (未使用，保持接口兼容)
        y_normals: y的法向量 (未使用，保持接口兼容)
        weights: 权重 (未使用，保持接口兼容)
        batch_reduction: batch维度的聚合方式 ('mean', 'sum', 'none')
        point_reduction: 点维度的聚合方式 ('mean', 'sum')
        norm: 距离范数 (1 或 2)
        
    Returns:
        loss: Chamfer distance
        loss_normals: 法向量损失 (始终为None，保持接口兼容)
    """
    # 确保输入是3D张量 [B, N, 3]
    if x.dim() == 2:
        x = x.unsqueeze(0)
    if y.dim() == 2:
        y = y.unsqueeze(0)
    
    B, N, D = x.shape
    _, M, _ = y.shape
    
    # 计算距离矩阵
    # x: [B, N, 1, D], y: [B, 1, M, D]
    # diff: [B, N, M, D]
    diff = x.unsqueeze(2) - y.unsqueeze(1)
    
    if norm == 2:
        # L2 距离
        dist = (diff ** 2).sum(dim=-1)  # [B, N, M]
    elif norm == 1:
        # L1 距离
        dist = diff.abs().sum(dim=-1)  # [B, N, M]
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    
    # x -> y 的最近距离
    min_dist_x_to_y, _ = dist.min(dim=2)  # [B, N]
    
    # y -> x 的最近距离
    min_dist_y_to_x, _ = dist.min(dim=1)  # [B, M]
    
    # 点维度聚合
    if point_reduction == "mean":
        loss_x = min_dist_x_to_y.mean(dim=1)  # [B]
        loss_y = min_dist_y_to_x.mean(dim=1)  # [B]
    elif point_reduction == "sum":
        loss_x = min_dist_x_to_y.sum(dim=1)  # [B]
        loss_y = min_dist_y_to_x.sum(dim=1)  # [B]
    else:
        raise ValueError(f"Unsupported point_reduction: {point_reduction}")
    
    # 双向 Chamfer Distance
    loss = loss_x + loss_y  # [B]
    
    # Batch 维度聚合
    if batch_reduction == "mean":
        loss = loss.mean()
    elif batch_reduction == "sum":
        loss = loss.sum()
    elif batch_reduction == "none":
        pass
    else:
        raise ValueError(f"Unsupported batch_reduction: {batch_reduction}")
    
    return loss, None


def chamfer_distance_naive(
    pc1: torch.Tensor,
    pc2: torch.Tensor,
) -> torch.Tensor:
    """
    朴素的 Chamfer Distance 计算
    
    更简单的接口，直接返回损失值
    
    Args:
        pc1: [B, N, 3] 或 [N, 3]
        pc2: [B, M, 3] 或 [M, 3]
        
    Returns:
        loss: Chamfer distance (标量)
    """
    loss, _ = chamfer_distance(pc1, pc2)
    return loss
