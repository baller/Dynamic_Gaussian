"""
动态高斯分配模块

根据背景更新信号动态分配高斯点到背景和人体区域。
支持两种模式:
1. 正常模式 (bg_update_signal=True): 按路由权重分配高斯
2. 人体专注模式 (bg_update_signal=False): 所有高斯分配给人体，背景使用缓存

架构设计:
- 固定总高斯数量
- TopK 选择策略
- 背景缓存机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class GaussianParams:
    """高斯参数数据类"""
    xyz: torch.Tensor          # [N, 3] 位置
    rgb: torch.Tensor          # [N, 3] 颜色
    rotation: torch.Tensor     # [N, 4] 旋转四元数
    scale: torch.Tensor        # [N, 3] 尺度
    opacity: torch.Tensor      # [N, 1] 不透明度
    
    def to(self, device):
        """移动到指定设备"""
        return GaussianParams(
            xyz=self.xyz.to(device),
            rgb=self.rgb.to(device),
            rotation=self.rotation.to(device),
            scale=self.scale.to(device),
            opacity=self.opacity.to(device)
        )
    
    def detach(self):
        """分离梯度"""
        return GaussianParams(
            xyz=self.xyz.detach(),
            rgb=self.rgb.detach(),
            rotation=self.rotation.detach(),
            scale=self.scale.detach(),
            opacity=self.opacity.detach()
        )
    
    @staticmethod
    def from_dict(d: Dict[str, torch.Tensor]) -> 'GaussianParams':
        """从字典创建"""
        return GaussianParams(
            xyz=d.get('xyz'),
            rgb=d.get('rgb'),
            rotation=d.get('rotation'),
            scale=d.get('scale'),
            opacity=d.get('opacity')
        )
    
    def to_dict(self) -> Dict[str, torch.Tensor]:
        """转换为字典"""
        return {
            'xyz': self.xyz,
            'rgb': self.rgb,
            'rotation': self.rotation,
            'scale': self.scale,
            'opacity': self.opacity
        }


class BackgroundCache:
    """
    背景高斯缓存
    
    存储上一次计算的背景高斯参数，用于人体专注模式。
    """
    
    def __init__(self):
        self._cache: Optional[GaussianParams] = None
        self._is_valid: bool = False
        self._last_update_step: int = -1
    
    def update(self, params: GaussianParams, step: int = 0):
        """更新缓存"""
        self._cache = params.detach()
        self._is_valid = True
        self._last_update_step = step
    
    def get(self) -> Optional[GaussianParams]:
        """获取缓存"""
        return self._cache if self._is_valid else None
    
    def is_valid(self) -> bool:
        """检查缓存是否有效"""
        return self._is_valid
    
    def invalidate(self):
        """使缓存失效"""
        self._is_valid = False
    
    def get_last_update_step(self) -> int:
        """获取上次更新步数"""
        return self._last_update_step


class DynamicGaussianAllocator(nn.Module):
    """
    动态高斯分配器
    
    根据路由权重和背景更新信号，动态分配高斯点。
    
    Args:
        total_gaussians: 总高斯数量
        min_bg_ratio: 最小背景比例（防止完全没有背景）
        min_human_ratio: 最小人体比例
    """
    
    def __init__(
        self,
        total_gaussians: int = 1048576,
        min_bg_ratio: float = 0.1,
        min_human_ratio: float = 0.3
    ):
        super().__init__()
        
        self.total_gaussians = total_gaussians
        self.min_bg_ratio = min_bg_ratio
        self.min_human_ratio = min_human_ratio
        
        # 背景缓存
        self.bg_cache = BackgroundCache()
        
        # 统计信息
        self._last_bg_count = 0
        self._last_human_count = 0
    
    def forward(
        self,
        expert_params: List[Dict[str, torch.Tensor]],
        router_weights: torch.Tensor,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        bg_update_signal: bool = True,
        step: int = 0
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        动态分配高斯
        
        Args:
            expert_params: 各专家预测的参数列表 [bg_params, human_params, ...]
            router_weights: [B, N, num_experts] 路由权重
            xyz: [B, N, 3] 3D 位置
            rgb: [B, N, 3] 颜色
            bg_update_signal: 是否更新背景
            step: 当前训练步数
            
        Returns:
            selected_params: 选中的高斯参数
            allocation_info: 分配信息
        """
        B, N, num_experts = router_weights.shape
        device = router_weights.device
        
        # 假设专家0是背景，专家1是人体
        bg_params = expert_params[0] if len(expert_params) > 0 else None
        human_params = expert_params[1] if len(expert_params) > 1 else expert_params[0]
        
        # 获取路由权重
        bg_weight = router_weights[:, :, 0]      # [B, N]
        human_weight = router_weights[:, :, 1] if num_experts > 1 else router_weights[:, :, 0]
        
        if bg_update_signal:
            # 正常模式: 按路由权重分配
            selected, allocation_info = self._allocate_normal(
                bg_params, human_params,
                bg_weight, human_weight,
                xyz, rgb,
                device
            )
            
            # 更新背景缓存
            if allocation_info.get('bg_gaussians') is not None:
                bg_gaussians = allocation_info['bg_gaussians']
                self.bg_cache.update(GaussianParams.from_dict(bg_gaussians), step)
        else:
            # 人体专注模式: 所有高斯给人体
            selected, allocation_info = self._allocate_human_focus(
                human_params,
                human_weight,
                xyz, rgb,
                device
            )
        
        # 更新统计
        self._last_bg_count = allocation_info.get('bg_count', 0)
        self._last_human_count = allocation_info.get('human_count', 0)
        
        return selected, allocation_info
    
    def _allocate_normal(
        self,
        bg_params: Dict[str, torch.Tensor],
        human_params: Dict[str, torch.Tensor],
        bg_weight: torch.Tensor,
        human_weight: torch.Tensor,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        正常模式分配
        
        按路由权重比例分配高斯数量到背景和人体。
        """
        B, N = bg_weight.shape
        
        # 计算分配比例
        bg_weight_sum = bg_weight.sum()
        human_weight_sum = human_weight.sum()
        total_weight = bg_weight_sum + human_weight_sum + 1e-8
        
        bg_ratio = (bg_weight_sum / total_weight).item()
        
        # 应用最小比例约束
        bg_ratio = max(self.min_bg_ratio, min(1 - self.min_human_ratio, bg_ratio))
        
        # 计算分配数量
        bg_count = int(self.total_gaussians * bg_ratio)
        human_count = self.total_gaussians - bg_count
        
        # 选择 Top-K
        # 对于批次中的每个样本，选择权重最高的 K 个点
        selected_params = {}
        bg_gaussians = {}
        human_gaussians = {}
        
        for b in range(B):
            # 背景 Top-K
            bg_w = bg_weight[b]  # [N]
            _, bg_indices = torch.topk(bg_w, min(bg_count, N), largest=True)
            
            # 人体 Top-K
            human_w = human_weight[b]  # [N]
            _, human_indices = torch.topk(human_w, min(human_count, N), largest=True)
            
            # 提取参数
            if b == 0:  # 只处理第一个批次样本
                # 背景高斯
                bg_gaussians = {
                    'xyz': xyz[b, bg_indices],
                    'rgb': rgb[b, bg_indices],
                    'rotation': bg_params['rotation'][b, bg_indices],
                    'scale': bg_params['scale'][b, bg_indices],
                    'opacity': bg_params['opacity'][b, bg_indices]
                }
                
                # 人体高斯
                human_gaussians = {
                    'xyz': xyz[b, human_indices],
                    'rgb': rgb[b, human_indices],
                    'rotation': human_params['rotation'][b, human_indices],
                    'scale': human_params['scale'][b, human_indices],
                    'opacity': human_params['opacity'][b, human_indices]
                }
                
                # 合并
                selected_params = self._concat_params(bg_gaussians, human_gaussians)
        
        allocation_info = {
            'mode': 'normal',
            'bg_count': bg_count,
            'human_count': human_count,
            'bg_ratio': bg_ratio,
            'bg_gaussians': bg_gaussians,
            'human_gaussians': human_gaussians
        }
        
        return selected_params, allocation_info
    
    def _allocate_human_focus(
        self,
        human_params: Dict[str, torch.Tensor],
        human_weight: torch.Tensor,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        device: torch.device
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        人体专注模式分配
        
        所有高斯分配给人体，背景使用缓存。
        """
        B, N = human_weight.shape
        
        # 全部分配给人体
        human_count = self.total_gaussians
        
        selected_params = {}
        human_gaussians = {}
        
        for b in range(B):
            # 人体 Top-K (全部)
            human_w = human_weight[b]
            _, human_indices = torch.topk(human_w, min(human_count, N), largest=True)
            
            if b == 0:
                human_gaussians = {
                    'xyz': xyz[b, human_indices],
                    'rgb': rgb[b, human_indices],
                    'rotation': human_params['rotation'][b, human_indices],
                    'scale': human_params['scale'][b, human_indices],
                    'opacity': human_params['opacity'][b, human_indices]
                }
                selected_params = human_gaussians
        
        # 获取背景缓存
        cached_bg = self.bg_cache.get()
        
        allocation_info = {
            'mode': 'human_focus',
            'bg_count': 0,
            'human_count': human_count,
            'bg_ratio': 0.0,
            'human_gaussians': human_gaussians,
            'bg_cached': cached_bg is not None,
            'cached_bg_params': cached_bg.to_dict() if cached_bg else None
        }
        
        return selected_params, allocation_info
    
    def _concat_params(
        self,
        params1: Dict[str, torch.Tensor],
        params2: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """合并两个参数字典"""
        result = {}
        for key in params1.keys():
            if key in params2:
                result[key] = torch.cat([params1[key], params2[key]], dim=0)
            else:
                result[key] = params1[key]
        return result
    
    def get_allocation_stats(self) -> Dict[str, int]:
        """获取分配统计"""
        return {
            'bg_count': self._last_bg_count,
            'human_count': self._last_human_count,
            'total': self.total_gaussians,
            'cache_valid': self.bg_cache.is_valid()
        }


class CurriculumScheduler:
    """
    课程学习调度器
    
    控制训练过程中 bg_update_signal 的采样概率。
    
    训练阶段:
    1. Warmup (0-warmup_steps): 100% 有信号
    2. Decay (warmup-warmup+decay): 线性衰减到 min_prob
    3. Final: 维持 min_prob
    
    Args:
        warmup_steps: 预热步数
        decay_steps: 衰减步数
        min_bg_prob: 最小背景更新概率
    """
    
    def __init__(
        self,
        warmup_steps: int = 10000,
        decay_steps: int = 40000,
        min_bg_prob: float = 0.3
    ):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.min_bg_prob = min_bg_prob
    
    def get_bg_signal_prob(self, step: int) -> float:
        """
        获取当前步的 bg_update_signal=True 概率
        
        Args:
            step: 当前训练步数
            
        Returns:
            概率值 [0, 1]
        """
        if step < self.warmup_steps:
            # 阶段1: 全部有信号
            return 1.0
        elif step < self.warmup_steps + self.decay_steps:
            # 阶段2: 线性衰减
            progress = (step - self.warmup_steps) / self.decay_steps
            return 1.0 - (1.0 - self.min_bg_prob) * progress
        else:
            # 阶段3: 维持最小概率
            return self.min_bg_prob
    
    def sample_bg_signal(self, step: int) -> bool:
        """
        采样 bg_update_signal
        
        Args:
            step: 当前训练步数
            
        Returns:
            bg_update_signal
        """
        prob = self.get_bg_signal_prob(step)
        return torch.rand(1).item() < prob
    
    def get_phase(self, step: int) -> str:
        """获取当前训练阶段"""
        if step < self.warmup_steps:
            return 'warmup'
        elif step < self.warmup_steps + self.decay_steps:
            return 'decay'
        else:
            return 'final'


def create_dynamic_allocator(cfg) -> DynamicGaussianAllocator:
    """
    创建动态高斯分配器
    
    Args:
        cfg: 配置对象
        
    Returns:
        DynamicGaussianAllocator 实例
    """
    moe_cfg = getattr(cfg, 'moe', None)
    
    if moe_cfg is not None:
        total_gaussians = getattr(
            getattr(moe_cfg, 'allocation', None), 
            'total_gaussians', 
            1048576
        )
        min_ratio = getattr(
            getattr(moe_cfg, 'allocation', None),
            'min_ratio',
            0.1
        )
    else:
        total_gaussians = 1048576
        min_ratio = 0.1
    
    return DynamicGaussianAllocator(
        total_gaussians=total_gaussians,
        min_bg_ratio=min_ratio,
        min_human_ratio=min_ratio
    )


def create_curriculum_scheduler(cfg) -> CurriculumScheduler:
    """
    创建课程学习调度器
    
    Args:
        cfg: 配置对象
        
    Returns:
        CurriculumScheduler 实例
    """
    moe_cfg = getattr(cfg, 'moe', None)
    
    if moe_cfg is not None:
        training_cfg = getattr(moe_cfg, 'training', None)
        if training_cfg is not None:
            curriculum_cfg = getattr(training_cfg, 'curriculum', None)
            if curriculum_cfg is not None:
                return CurriculumScheduler(
                    warmup_steps=getattr(curriculum_cfg, 'warmup_steps', 10000),
                    decay_steps=getattr(curriculum_cfg, 'decay_steps', 40000),
                    min_bg_prob=getattr(curriculum_cfg, 'min_bg_prob', 0.3)
                )
    
    # 默认配置
    return CurriculumScheduler()
