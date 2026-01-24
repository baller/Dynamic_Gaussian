"""
自适应背景高斯缓存模块

实现背景高斯的缓存和自适应更新机制，
根据渲染质量决定是否需要更新背景高斯。
"""

import torch
import torch.nn.functional as F
from lib.gs_utils.loss_utils import ssim as compute_ssim


def compute_psnr(pred, gt, mask=None):
    """
    计算PSNR
    
    Args:
        pred: 预测图像 [B, C, H, W]
        gt: 真实图像 [B, C, H, W]
        mask: 可选的mask [B, 1, H, W]
        
    Returns:
        psnr: PSNR值
    """
    if mask is not None:
        pred = pred * mask
        gt = gt * mask
        mse = ((pred - gt) ** 2).sum() / (mask.sum() * pred.shape[1] + 1e-8)
    else:
        mse = ((pred - gt) ** 2).mean()
    
    if mse < 1e-10:
        return 100.0
    
    psnr = 10 * torch.log10(1.0 / mse)
    return psnr.item()


class AdaptiveBackgroundCache:
    """
    自适应背景高斯缓存
    
    缓存背景高斯参数，根据渲染质量决定是否更新。
    静态背景只需生成一次，运动较少的背景可周期性更新。
    
    Args:
        update_threshold: 质量阈值（PSNR），低于此值时触发更新
        min_update_interval: 最小更新间隔（帧数）
        quality_metric: 质量评估指标 ('psnr', 'ssim', 'combined')
        momentum: 缓存更新动量（用于渐进更新）
    """
    
    def __init__(self, update_threshold=25.0, min_update_interval=1, 
                 quality_metric='psnr', momentum=0.0):
        self.update_threshold = update_threshold
        self.min_update_interval = min_update_interval
        self.quality_metric = quality_metric
        self.momentum = momentum
        
        # 缓存状态
        self.cached_gaussians = None
        self.cached_frame_idx = -1
        self.cached_xyz = None
        self.cached_rot = None
        self.cached_scale = None
        self.cached_opacity = None
        self.cached_rgb = None
        
        # 统计信息
        self.update_count = 0
        self.total_frames = 0
        self.quality_history = []
        
    def reset(self):
        """重置缓存"""
        self.cached_gaussians = None
        self.cached_frame_idx = -1
        self.cached_xyz = None
        self.cached_rot = None
        self.cached_scale = None
        self.cached_opacity = None
        self.cached_rgb = None
        self.update_count = 0
        self.total_frames = 0
        self.quality_history = []
        
    def _compute_quality(self, current_render, gt_image, bg_mask):
        """
        计算背景区域的渲染质量
        
        Args:
            current_render: 当前渲染结果 [B, C, H, W]
            gt_image: 真实图像 [B, C, H, W]
            bg_mask: 背景区域mask [B, 1, H, W]
            
        Returns:
            quality: 质量分数
        """
        if self.quality_metric == 'psnr':
            return compute_psnr(current_render, gt_image, bg_mask)
        elif self.quality_metric == 'ssim':
            # 只在背景区域计算SSIM
            masked_render = current_render * bg_mask
            masked_gt = gt_image * bg_mask
            ssim_val = compute_ssim(masked_render, masked_gt)
            return ssim_val.item() * 100  # 转换到类似PSNR的尺度
        elif self.quality_metric == 'combined':
            psnr = compute_psnr(current_render, gt_image, bg_mask)
            masked_render = current_render * bg_mask
            masked_gt = gt_image * bg_mask
            ssim_val = compute_ssim(masked_render, masked_gt).item()
            return psnr * 0.5 + ssim_val * 50  # 加权组合
        else:
            return compute_psnr(current_render, gt_image, bg_mask)
    
    def should_update(self, current_render=None, gt_image=None, router_weights=None, 
                      frame_idx=None, force=False):
        """
        判断是否需要更新背景缓存
        
        Args:
            current_render: 当前渲染结果 [B, C, H, W]
            gt_image: 真实图像 [B, C, H, W]
            router_weights: 路由权重 [B, 2, H, W]
            frame_idx: 当前帧索引
            force: 强制更新标志
            
        Returns:
            should_update: 是否需要更新
            reason: 更新原因字符串
        """
        self.total_frames += 1
        
        # 首次运行，必须更新
        if self.cached_gaussians is None:
            return True, "first_frame"
        
        # 强制更新
        if force:
            return True, "forced"
        
        # 检查最小更新间隔
        if frame_idx is not None:
            frames_since_update = frame_idx - self.cached_frame_idx
            if frames_since_update < self.min_update_interval:
                return False, "min_interval"
        
        # 如果没有提供渲染结果，不更新
        if current_render is None or gt_image is None:
            return False, "no_render"
        
        # 计算背景区域的渲染质量
        if router_weights is not None:
            bg_mask = (router_weights[:, 0:1] > 0.5).float()
        else:
            # 如果没有路由权重，假设整个图像都是背景
            bg_mask = torch.ones_like(current_render[:, :1])
        
        quality = self._compute_quality(current_render, gt_image, bg_mask)
        self.quality_history.append(quality)
        
        # 质量低于阈值时更新
        if quality < self.update_threshold:
            return True, f"low_quality_{quality:.2f}"
        
        return False, f"quality_ok_{quality:.2f}"
    
    def update(self, bg_gaussians, frame_idx=None):
        """
        更新背景缓存
        
        Args:
            bg_gaussians: 背景高斯参数字典，包含:
                - xyz: 3D位置 [B, N, 3]
                - rot_maps: 旋转参数
                - scale_maps: 尺度参数
                - opacity_maps: 不透明度
                - rgb: 颜色（可选）
            frame_idx: 当前帧索引
        """
        if self.momentum > 0 and self.cached_gaussians is not None:
            # 渐进更新（使用动量）
            for key in bg_gaussians:
                if key in self.cached_gaussians and bg_gaussians[key] is not None:
                    self.cached_gaussians[key] = (
                        self.momentum * self.cached_gaussians[key] +
                        (1 - self.momentum) * bg_gaussians[key].clone().detach()
                    )
        else:
            # 直接替换
            self.cached_gaussians = {}
            for key, value in bg_gaussians.items():
                if value is not None:
                    self.cached_gaussians[key] = value.clone().detach()
        
        if frame_idx is not None:
            self.cached_frame_idx = frame_idx
        
        self.update_count += 1
    
    def get(self):
        """
        获取缓存的背景高斯
        
        Returns:
            cached_gaussians: 缓存的高斯参数字典，如果没有缓存则返回None
        """
        return self.cached_gaussians
    
    def get_update_ratio(self):
        """获取更新率（更新次数/总帧数）"""
        if self.total_frames == 0:
            return 0.0
        return self.update_count / self.total_frames
    
    def get_stats(self):
        """获取统计信息"""
        return {
            'update_count': self.update_count,
            'total_frames': self.total_frames,
            'update_ratio': self.get_update_ratio(),
            'cached_frame_idx': self.cached_frame_idx,
            'avg_quality': sum(self.quality_history) / len(self.quality_history) if self.quality_history else 0,
            'has_cache': self.cached_gaussians is not None
        }


class TemporalBackgroundTracker:
    """
    时序背景跟踪器
    
    跟踪多帧的背景信息，用于检测背景变化（如相机移动）。
    
    Args:
        window_size: 时序窗口大小
        change_threshold: 变化检测阈值
    """
    
    def __init__(self, window_size=5, change_threshold=0.1):
        self.window_size = window_size
        self.change_threshold = change_threshold
        
        self.router_history = []
        self.depth_history = []
        
    def update(self, router_weights, depth=None):
        """
        更新跟踪历史
        
        Args:
            router_weights: 路由权重 [B, 2, H, W]
            depth: 深度图 [B, 1, H, W]，可选
        """
        # 保存下采样的路由权重（节省内存）
        downsampled = F.interpolate(router_weights, size=(64, 64), mode='bilinear', align_corners=False)
        self.router_history.append(downsampled.detach().cpu())
        
        if depth is not None:
            depth_down = F.interpolate(depth, size=(64, 64), mode='bilinear', align_corners=False)
            self.depth_history.append(depth_down.detach().cpu())
        
        # 保持窗口大小
        if len(self.router_history) > self.window_size:
            self.router_history.pop(0)
        if len(self.depth_history) > self.window_size:
            self.depth_history.pop(0)
    
    def detect_background_change(self):
        """
        检测背景是否发生变化
        
        Returns:
            changed: 是否检测到变化
            change_score: 变化分数
        """
        if len(self.router_history) < 2:
            return False, 0.0
        
        # 比较最近两帧的背景路由权重
        current = self.router_history[-1][:, 0]  # 背景权重
        previous = self.router_history[-2][:, 0]
        
        # 计算变化量
        change = (current - previous).abs().mean().item()
        
        # 如果有深度信息，也考虑深度变化
        if len(self.depth_history) >= 2:
            depth_current = self.depth_history[-1]
            depth_previous = self.depth_history[-2]
            depth_change = (depth_current - depth_previous).abs().mean().item()
            change = change * 0.7 + depth_change * 0.3
        
        return change > self.change_threshold, change
    
    def get_temporal_consistency_loss(self, current_router_weights):
        """
        计算时序一致性损失
        
        Args:
            current_router_weights: 当前路由权重 [B, 2, H, W]
            
        Returns:
            loss: 时序一致性损失
        """
        if len(self.router_history) < 1:
            return torch.tensor(0.0, device=current_router_weights.device)
        
        # 获取上一帧的路由权重
        previous = self.router_history[-1].to(current_router_weights.device)
        
        # 上采样到当前分辨率
        previous_up = F.interpolate(previous, size=current_router_weights.shape[2:], 
                                    mode='bilinear', align_corners=False)
        
        # 只在背景区域计算一致性损失
        bg_mask = (previous_up[:, 0:1] > 0.5).float()
        
        loss = F.l1_loss(
            current_router_weights[:, 0:1] * bg_mask,
            previous_up[:, 0:1] * bg_mask
        )
        
        return loss


def create_bg_cache(cfg):
    """
    创建背景缓存
    
    Args:
        cfg: 配置对象
        
    Returns:
        cache: AdaptiveBackgroundCache实例
    """
    moe_cfg = getattr(cfg, 'moe', None)
    
    if moe_cfg is None:
        return AdaptiveBackgroundCache()
    
    cache_cfg = getattr(moe_cfg, 'bg_cache', None)
    if cache_cfg is None:
        return AdaptiveBackgroundCache()
    
    return AdaptiveBackgroundCache(
        update_threshold=getattr(cache_cfg, 'update_threshold', 25.0),
        min_update_interval=getattr(cache_cfg, 'min_update_interval', 1),
        quality_metric=getattr(cache_cfg, 'quality_metric', 'psnr'),
        momentum=getattr(cache_cfg, 'momentum', 0.0)
    )
