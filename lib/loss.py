"""
损失函数模块

包含:
1. 原始的光流序列损失
2. MoE相关损失函数（稀疏性、时序一致性、分配平衡）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class MoELoss(nn.Module):
    """
    MoE相关损失函数
    
    包含:
    - 路由稀疏性损失: 鼓励清晰的背景/人体分离
    - 时序一致性损失: 背景路由权重应在时间上稳定
    - 分配平衡损失: 防止高斯分配退化
    - 分离一致性损失: 鼓励分离的高斯参数差异化
    
    Args:
        sparsity_weight: 稀疏性损失权重
        temporal_weight: 时序一致性损失权重
        balance_weight: 分配平衡损失权重
        separation_weight: 分离一致性损失权重
        target_bg_ratio: 目标背景比例
    """
    
    def __init__(self, sparsity_weight=0.1, temporal_weight=0.05, 
                 balance_weight=0.01, separation_weight=0.01,
                 target_bg_ratio=0.3):
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.temporal_weight = temporal_weight
        self.balance_weight = balance_weight
        self.separation_weight = separation_weight
        self.target_bg_ratio = target_bg_ratio
        
        self.l1 = nn.L1Loss()
        
    def sparsity_loss(self, router_weights):
        """
        路由稀疏性损失
        
        使用熵作为稀疏性度量，鼓励路由权重接近0或1。
        
        Args:
            router_weights: [B, num_experts, H, W]
            
        Returns:
            loss: 稀疏性损失
        """
        # 确保数值稳定
        router_weights = router_weights.clamp(min=1e-8, max=1.0 - 1e-8)
        
        # 计算熵: -p*log(p)
        entropy = -router_weights * torch.log(router_weights)
        entropy = entropy.sum(dim=1)  # [B, H, W]
        
        # 熵越低越稀疏（更接近0或1）
        loss = entropy.mean()
        
        # 防止 NaN
        if torch.isnan(loss) or torch.isinf(loss):
            return torch.tensor(0.0, device=router_weights.device)
        
        return loss
    
    def temporal_consistency_loss(self, router_weights, prev_router_weights):
        """
        时序一致性损失
        
        鼓励背景区域的路由权重在时间上保持稳定。
        
        Args:
            router_weights: 当前帧路由权重 [B, 2, H, W]
            prev_router_weights: 上一帧路由权重 [B, 2, H, W]
            
        Returns:
            loss: 时序一致性损失
        """
        if prev_router_weights is None:
            return torch.tensor(0.0, device=router_weights.device)
        
        # 只在背景概率较高的区域计算一致性
        bg_mask = (router_weights[:, 0:1] > 0.5).float()
        prev_bg_mask = (prev_router_weights[:, 0:1] > 0.5).float()
        
        # 两帧都是背景的区域
        common_bg_mask = bg_mask * prev_bg_mask
        
        if common_bg_mask.sum() < 1:
            return torch.tensor(0.0, device=router_weights.device)
        
        # 计算背景区域的L1差异
        diff = (router_weights[:, 0:1] - prev_router_weights[:, 0:1]).abs()
        loss = (diff * common_bg_mask).sum() / (common_bg_mask.sum() + 1e-8)
        
        return loss
    
    def balance_loss(self, allocation_ratio):
        """
        分配平衡损失
        
        防止高斯分配过于极端（全部分配给背景或人体）。
        
        Args:
            allocation_ratio: 分配比例 [B, 2]
            
        Returns:
            loss: 分配平衡损失
        """
        if allocation_ratio is None:
            return torch.tensor(0.0)
        
        # 鼓励背景比例接近目标值
        bg_ratio = allocation_ratio[:, 0]
        loss = (bg_ratio - self.target_bg_ratio).abs().mean()
        
        return loss
    
    def separation_loss(self, bg_params, human_params, router_weights):
        """
        分离一致性损失
        
        鼓励背景和人体专家预测不同的高斯参数。
        使用数值稳定的计算方式。
        
        Args:
            bg_params: 背景高斯参数字典
            human_params: 人体高斯参数字典
            router_weights: 路由权重 [B, num_experts, H, W]
            
        Returns:
            loss: 分离一致性损失（Tensor）
        """
        device = router_weights.device
        
        if bg_params is None or human_params is None:
            return torch.tensor(0.0, device=device)
        
        # 获取专家数量
        num_experts = router_weights.shape[1]
        if num_experts < 2:
            return torch.tensor(0.0, device=device)
        
        total_loss = torch.tensor(0.0, device=device)
        count = 0
        
        # 1. 参数差异性损失：鼓励两个专家预测不同的参数
        # 使用负L1距离（差异越大，损失越小）
        if 'scale_maps' in bg_params and 'scale_maps' in human_params:
            bg_scale = bg_params['scale_maps']
            human_scale = human_params['scale_maps']
            
            # 计算尺度差异，使用 clamp 防止 NaN
            scale_diff = torch.abs(bg_scale - human_scale).clamp(min=1e-8).mean()
            # 转换为损失：1 / (1 + diff * scale_factor)，差异越大损失越小
            separation_scale = 1.0 / (1.0 + scale_diff * 1000)
            total_loss = total_loss + separation_scale
            count += 1
        
        if 'opacity_maps' in bg_params and 'opacity_maps' in human_params:
            bg_opacity = bg_params['opacity_maps']
            human_opacity = human_params['opacity_maps']
            
            # 计算不透明度差异
            opacity_diff = torch.abs(bg_opacity - human_opacity).clamp(min=1e-8).mean()
            separation_opacity = 1.0 / (1.0 + opacity_diff * 10)
            total_loss = total_loss + separation_opacity
            count += 1
        
        # 2. 专家置信度损失：鼓励专家预测高不透明度
        if 'opacity_maps' in bg_params and 'opacity_maps' in human_params:
            bg_opacity = bg_params['opacity_maps'].clamp(0, 1)
            human_opacity = human_params['opacity_maps'].clamp(0, 1)
            
            # 鼓励整体高不透明度
            avg_opacity = (bg_opacity.mean() + human_opacity.mean()) / 2
            conf_loss = 1.0 - avg_opacity
            total_loss = total_loss + conf_loss.clamp(0, 1)
            count += 1
        
        if count > 0:
            total_loss = total_loss / count
        
        # 最终检查，防止 NaN
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            return torch.tensor(0.0, device=device)
        
        return total_loss
    
    def forward(self, data, prev_data=None):
        """
        计算MoE总损失
        
        Args:
            data: 当前数据字典，包含:
                - lmain/rmain['router_weights']: 路由权重
                - lmain/rmain['bg_params']: 背景参数
                - lmain/rmain['human_params']: 人体参数
                - allocation_ratio: 分配比例（可选）
            prev_data: 上一帧数据（用于时序一致性）
            
        Returns:
            total_loss: 总MoE损失
            loss_dict: 各项损失的字典
        """
        total_loss = 0.0
        loss_dict = {}
        
        # 收集路由权重
        router_weights_list = []
        for view in ['lmain', 'rmain']:
            if 'router_weights' in data[view]:
                router_weights_list.append(data[view]['router_weights'])
        
        if len(router_weights_list) == 0:
            return total_loss, loss_dict
        
        router_weights = torch.cat(router_weights_list, dim=0)
        
        # 1. 稀疏性损失
        sparsity = self.sparsity_loss(router_weights)
        total_loss += self.sparsity_weight * sparsity
        loss_dict['sparsity'] = sparsity.item()
        
        # 2. 时序一致性损失
        if prev_data is not None:
            prev_router_list = []
            for view in ['lmain', 'rmain']:
                if 'router_weights' in prev_data[view]:
                    prev_router_list.append(prev_data[view]['router_weights'])
            
            if len(prev_router_list) > 0:
                prev_router = torch.cat(prev_router_list, dim=0)
                temporal = self.temporal_consistency_loss(router_weights, prev_router)
                total_loss += self.temporal_weight * temporal
                loss_dict['temporal'] = temporal.item()
        
        # 3. 分配平衡损失
        allocation_ratio = data.get('allocation_ratio', None)
        if allocation_ratio is not None:
            balance = self.balance_loss(allocation_ratio)
            total_loss += self.balance_weight * balance
            loss_dict['balance'] = balance.item()
        
        # 4. 分离一致性损失
        bg_params = data['lmain'].get('bg_params', None)
        human_params = data['lmain'].get('human_params', None)
        if bg_params is not None and human_params is not None:
            separation = self.separation_loss(
                bg_params, human_params, data['lmain']['router_weights']
            )
            total_loss += self.separation_weight * separation
            loss_dict['separation'] = separation.item() if isinstance(separation, torch.Tensor) else separation
        
        # 检查并处理 NaN
        if isinstance(total_loss, torch.Tensor):
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                total_loss = torch.tensor(0.0, device=router_weights.device)
            loss_dict['moe_total'] = total_loss.item()
        else:
            loss_dict['moe_total'] = total_loss
        
        return total_loss, loss_dict


def create_moe_loss(cfg):
    """
    创建MoE损失函数
    
    Args:
        cfg: 配置对象
        
    Returns:
        loss_fn: MoELoss实例
    """
    moe_cfg = getattr(cfg, 'moe', None)
    
    if moe_cfg is None:
        return MoELoss()
    
    loss_cfg = getattr(moe_cfg, 'loss', None)
    if loss_cfg is None:
        return MoELoss()
    
    return MoELoss(
        sparsity_weight=getattr(loss_cfg, 'sparsity_weight', 0.1),
        temporal_weight=getattr(loss_cfg, 'temporal_weight', 0.05),
        balance_weight=getattr(loss_cfg, 'balance_weight', 0.01),
        separation_weight=getattr(loss_cfg, 'separation_weight', 0.01),
        target_bg_ratio=getattr(loss_cfg, 'target_bg_ratio', 0.3)
    )
