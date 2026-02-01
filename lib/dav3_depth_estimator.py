"""
Depth Anything V3 (DAV3) 深度估计器 - 简化版

正确的 DA3 使用方式：
1. 使用 DA3 的高层 API 或正确归一化后的 forward()
2. 不使用 Cost Volume 进行特征匹配（DA3 的高层特征是语义特征，不适合立体匹配）
3. 使用简单的基于基线的尺度对齐

Key improvements over the previous version:
- Removed SparseStereoCostVolume (semantic features don't work for stereo matching)
- Removed complex Scale/Shift networks (unreliable with bad stereo matches)
- Use DA3's multi-view inference or simple baseline-based scale alignment
- Proper ImageNet normalization for DA3 input
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from torch.amp import autocast
import logging

logger = logging.getLogger(__name__)

# Add Depth-Anything-3 to path
sys.path.insert(0, '/home/user_3/3DGS/Depth-Anything-3/src')

# ImageNet normalization constants for DA3
# DA3 expects ImageNet-normalized inputs
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

# DA3/DINOv2 patch size
PATCH_SIZE = 14

try:
    from depth_anything_3.api import DepthAnything3
    DA3_AVAILABLE = True
except ImportError:
    logger.warning("Depth-Anything-3 not found. Please install it first.")
    DepthAnything3 = None
    DA3_AVAILABLE = False


@dataclass
class DAV3Config:
    """Configuration for DAV3 depth estimator."""
    # Model configuration
    model_name: str = "depth-anything/DA3-LARGE"
    export_feat_layers: Tuple[int, ...] = (11, 15, 19, 23)
    
    # Input preprocessing
    freeze_backbone: bool = True
    mixed_precision: bool = False
    
    # Depth alignment configuration
    use_baseline_alignment: bool = True
    depth_min: float = 0.1
    depth_max: float = 100.0
    
    # Typical depth range estimation (for baseline alignment)
    typical_disp_min_ratio: float = 0.01  # Minimum disparity as fraction of image width
    typical_disp_max_ratio: float = 0.15  # Maximum disparity as fraction of image width


class DAV3DepthEstimator(nn.Module):
    """
    DAV3 深度估计器 - 简化版
    
    使用 Depth Anything 3 进行单目深度估计，并通过立体基线进行尺度对齐。
    
    Key features:
    1. 正确的 ImageNet 归一化
    2. 使用 DA3 的高层 API
    3. 基于基线的尺度对齐（而不是有问题的 Cost Volume）
    
    Args:
        cfg: 配置对象
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 从 cfg 获取 dav3 配置
        dav3_cfg = getattr(cfg, 'dav3', None)
        if dav3_cfg is None:
            self.config = DAV3Config()
        else:
            self.config = DAV3Config(
                model_name=getattr(dav3_cfg, 'model_name', 'depth-anything/DA3-LARGE'),
                export_feat_layers=tuple(getattr(dav3_cfg, 'export_feat_layers', [11, 15, 19, 23])),
                freeze_backbone=getattr(dav3_cfg, 'freeze_backbone', True),
                mixed_precision=getattr(dav3_cfg, 'mixed_precision', False),
                use_baseline_alignment=getattr(dav3_cfg, 'use_baseline_alignment', True),
                depth_min=getattr(dav3_cfg, 'depth_min', 0.1),
                depth_max=getattr(dav3_cfg, 'depth_max', 100.0),
            )
        
        # 深度对齐配置
        depth_align_cfg = getattr(cfg, 'depth_align', None)
        if depth_align_cfg is not None:
            self.config.typical_disp_min_ratio = getattr(depth_align_cfg, 'typical_disp_min_ratio', 0.01)
            self.config.typical_disp_max_ratio = getattr(depth_align_cfg, 'typical_disp_max_ratio', 0.15)
        
        # 模型延迟加载
        self.model = None
        self.is_loaded = False
        
        # 推断特征维度
        model_name = self.config.model_name.upper()
        if 'LARGE' in model_name or 'GIANT' in model_name:
            self.feat_dim = 1024
        elif 'BASE' in model_name:
            self.feat_dim = 768
        else:
            self.feat_dim = 384
    
    def _lazy_load_model(self):
        """延迟加载模型，避免初始化时占用显存"""
        if self.is_loaded:
            return
        
        if not DA3_AVAILABLE:
            raise ImportError("Depth-Anything-3 not found. Please install it first.")
        
        logger.info(f"[DAV3] 加载模型: {self.config.model_name}")
        
        # 从 HuggingFace 加载预训练模型
        self.model = DepthAnything3.from_pretrained(self.config.model_name)
        
        # 获取当前设备
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else 'cuda'
        self.model = self.model.to(device)
        self.model.eval()
        
        # 冻结参数
        if self.config.freeze_backbone:
            for param in self.model.parameters():
                param.requires_grad = False
            logger.info("[DAV3] 模型参数已冻结")
        
        self.is_loaded = True
        logger.info("[DAV3] 模型加载完成")
    
    def _normalize_for_da3(self, images: torch.Tensor) -> torch.Tensor:
        """
        将输入图像归一化到 DA3 期望的格式
        
        输入可能是:
        - [-1, 1] 范围 (GPS_plus 的 human_loader)
        - [0, 1] 范围
        - [0, 255] 范围
        
        输出: [0, 1] 范围 (DA3 的 inference() 会自动进行 ImageNet 归一化)
        
        Args:
            images: [B, 3, H, W] 输入图像
            
        Returns:
            images_01: [B, 3, H, W] 归一化到 [0, 1] 的图像
        """
        # 检查输入范围
        img_min = images.min().item()
        img_max = images.max().item()
        
        # 从配置获取输入范围
        img_range = getattr(getattr(self.cfg, 'dataset', None), 'img_range', None)
        if img_range is not None and len(img_range) == 2:
            range_min, range_max = img_range
            images = (images - range_min) / (range_max - range_min)
        elif img_min >= -1.1 and img_max <= 1.1 and img_min < 0:
            # [-1, 1] 范围
            images = (images + 1.0) / 2.0
        elif img_min >= 0 and img_max > 1.1:
            # [0, 255] 范围
            images = images / 255.0
        # 否则假设已经是 [0, 1]
        
        return images.clamp(0, 1)
    
    def _apply_imagenet_normalization(self, images: torch.Tensor) -> torch.Tensor:
        """
        应用 ImageNet 归一化
        
        Args:
            images: [B, 3, H, W] 或 [B, N, 3, H, W] 图像，范围 [0, 1]
            
        Returns:
            normalized: ImageNet 归一化后的图像
        """
        device = images.device
        dtype = images.dtype
        
        if images.dim() == 4:
            # [B, 3, H, W]
            mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(device=device, dtype=dtype)
            std = IMAGENET_STD.view(1, 3, 1, 1).to(device=device, dtype=dtype)
        else:
            # [B, N, 3, H, W]
            mean = IMAGENET_MEAN.view(1, 1, 3, 1, 1).to(device=device, dtype=dtype)
            std = IMAGENET_STD.view(1, 1, 3, 1, 1).to(device=device, dtype=dtype)
        
        return (images - mean) / std
    
    def _pad_to_patch_size(self, images: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        填充图像使其尺寸是 PATCH_SIZE 的倍数
        
        Args:
            images: [B, 3, H, W] 或 [B, N, 3, H, W]
            
        Returns:
            padded: 填充后的图像
            original_size: (H, W) 原始尺寸
        """
        if images.dim() == 4:
            B, C, H, W = images.shape
        else:
            B, N, C, H, W = images.shape
        
        pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
        
        if pad_h > 0 or pad_w > 0:
            if images.dim() == 4:
                images = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
            else:
                # [B, N, 3, H, W] -> 需要重塑后填充
                images = images.reshape(B * N, C, H, W)
                images = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
                images = images.reshape(B, N, C, H + pad_h, W + pad_w)
        
        return images, (H, W)
    
    def forward(self, data: Dict, is_train: bool = True) -> Dict:
        """
        前向传播
        
        使用 DA3 进行深度估计，并通过基线进行尺度对齐。
        
        Args:
            data: 包含以下键的字典:
                - 'lmain': {'img': [B, 3, H, W], 'intr': [B, 3, 3], 'extr': [B, 3, 4], 'mask': [B, C, H, W]}
                - 'rmain': 同上
            is_train: 是否为训练模式
            
        Returns:
            data: 更新后的字典，添加:
                - 'depth': [B, 1, H, W] 对齐后的深度
                - 'depth_relative': [B, 1, H, W] 原始相对深度
                - 'scale': [B, 1] 尺度因子
                - 'shift': [B, 1] 偏移量
        """
        self._lazy_load_model()
        
        device = data['lmain']['img'].device
        bs = data['lmain']['img'].shape[0]
        _, _, H, W = data['lmain']['img'].shape
        
        # 合并左右视图
        l_img = data['lmain']['img']
        r_img = data['rmain']['img']
        
        # 归一化到 [0, 1]
        l_img_01 = self._normalize_for_da3(l_img)
        r_img_01 = self._normalize_for_da3(r_img)
        
        # 应用 ImageNet 归一化
        l_img_norm = self._apply_imagenet_normalization(l_img_01)
        r_img_norm = self._apply_imagenet_normalization(r_img_01)
        
        # 填充到 patch size 的倍数
        l_img_padded, (orig_H, orig_W) = self._pad_to_patch_size(l_img_norm)
        r_img_padded, _ = self._pad_to_patch_size(r_img_norm)
        
        # 合并为 batch: [2B, 3, H', W']
        lr_img = torch.cat([l_img_padded, r_img_padded], dim=0)
        
        # 转换为 DA3 期望的格式: [2B, 1, 3, H', W']
        lr_img_da3 = lr_img.unsqueeze(1)
        
        # DA3 推理
        with autocast('cuda', enabled=self.config.mixed_precision):
            with torch.no_grad():
                output = self.model(
                    lr_img_da3,
                    export_feat_layers=list(self.config.export_feat_layers)
                )
        
        # 提取深度图
        depth = output.depth.clone()  # [2B, 1, H', W'] 或 [2B, H', W']
        if depth.dim() == 4:
            depth = depth[:, 0]  # [2B, H', W']
        depth = depth.unsqueeze(1)  # [2B, 1, H', W']
        
        # 移除 padding
        depth = depth[:, :, :orig_H, :orig_W]
        
        # 确保尺寸匹配
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
        
        # 分离左右视图
        l_depth_rel, r_depth_rel = torch.split(depth, [bs, bs])
        
        # 存储相对深度
        data['lmain']['depth_relative'] = l_depth_rel
        data['rmain']['depth_relative'] = r_depth_rel
        
        # 尺度对齐
        if self.config.use_baseline_alignment:
            l_depth, r_depth, scale, shift = self._align_depth_with_baseline(
                l_depth_rel, r_depth_rel,
                data['lmain']['intr'], data['rmain']['intr'],
                data['lmain']['extr'], data['rmain']['extr'],
                data['lmain']['mask'], data['rmain']['mask'],
                W
            )
        else:
            # 不对齐，直接使用相对深度（注意：这通常不适合渲染）
            l_depth = l_depth_rel
            r_depth = r_depth_rel
            scale = torch.ones(bs, 1, device=device)
            shift = torch.zeros(bs, 1, device=device)
        
        # 存储对齐后的深度
        data['lmain']['depth'] = l_depth
        data['rmain']['depth'] = r_depth
        data['scale'] = scale
        data['shift'] = shift
        
        # 创建伪 flow_pred (用于兼容性)
        data['lmain']['flow_pred'] = torch.zeros(bs, 2, H, W, device=device)
        data['rmain']['flow_pred'] = torch.zeros(bs, 2, H, W, device=device)
        
        # 存储稀疏立体匹配的占位符 (为了兼容原有代码)
        data['sparse_stereo'] = {
            'num_anchors': torch.zeros(bs, device=device, dtype=torch.long)
        }
        
        return data
    
    def _align_depth_with_baseline(
        self,
        l_depth: torch.Tensor,
        r_depth: torch.Tensor,
        l_intr: torch.Tensor,
        r_intr: torch.Tensor,
        l_extr: torch.Tensor,
        r_extr: torch.Tensor,
        l_mask: torch.Tensor,
        r_mask: torch.Tensor,
        img_width: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        使用基线信息对齐深度
        
        DA3 输出的是相对深度/逆深度，需要转换为绝对深度。
        使用立体基线和焦距估计合理的深度范围。
        
        Args:
            l_depth, r_depth: [B, 1, H, W] 相对深度
            l_intr, r_intr: [B, 3, 3] 内参
            l_extr, r_extr: [B, 3, 4] 或 [B, 4, 4] 外参
            l_mask, r_mask: [B, C, H, W] 前景掩码
            img_width: 图像宽度
            
        Returns:
            l_depth_aligned, r_depth_aligned: [B, 1, H, W] 对齐后的深度
            scale: [B, 1] 尺度因子
            shift: [B, 1] 偏移量
        """
        B = l_depth.shape[0]
        device = l_depth.device
        eps = 1e-6
        
        # 计算基线
        if l_extr.shape[1] == 4:
            t_l = l_extr[:, :3, 3]
            t_r = r_extr[:, :3, 3]
        else:
            t_l = l_extr[:, :, 3]
            t_r = r_extr[:, :, 3]
        
        baseline = torch.norm(t_l - t_r, dim=1)  # [B]
        
        # 获取焦距
        fx_l = l_intr[:, 0, 0]  # [B]
        
        # 估计典型深度范围
        # disp = baseline * fx / depth  =>  depth = baseline * fx / disp
        typical_disp_min = img_width * self.config.typical_disp_min_ratio
        typical_disp_max = img_width * self.config.typical_disp_max_ratio
        
        depth_far = baseline * fx_l / (typical_disp_min + eps)  # [B]
        depth_near = baseline * fx_l / (typical_disp_max + eps)  # [B]
        
        target_median = (depth_near + depth_far) / 2  # [B]
        target_range = (depth_far - depth_near) / 2   # [B]
        
        # 计算每个样本的相对深度统计量
        l_mask_valid = (l_mask[:, :1] > 0.5).float()
        r_mask_valid = (r_mask[:, :1] > 0.5).float()
        
        scales = []
        shifts = []
        l_depths_aligned = []
        r_depths_aligned = []
        
        for b in range(B):
            # 左视图
            l_valid = l_depth[b] * l_mask_valid[b]
            l_valid_flat = l_valid[l_mask_valid[b] > 0.5]
            
            if l_valid_flat.numel() > 0:
                l_median = l_valid_flat.median()
                l_mad = (l_valid_flat - l_median).abs().median() + eps
            else:
                l_median = torch.tensor(0.5, device=device)
                l_mad = torch.tensor(0.25, device=device)
            
            # 右视图
            r_valid = r_depth[b] * r_mask_valid[b]
            r_valid_flat = r_valid[r_mask_valid[b] > 0.5]
            
            if r_valid_flat.numel() > 0:
                r_median = r_valid_flat.median()
                r_mad = (r_valid_flat - r_median).abs().median() + eps
            else:
                r_median = l_median
                r_mad = l_mad
            
            # 计算尺度和偏移
            # normalized = (depth - median) / mad
            # aligned = normalized * target_range + target_median
            # => aligned = (depth - median) / mad * target_range + target_median
            # => aligned = depth * (target_range / mad) + (target_median - median * target_range / mad)
            # => scale = target_range / mad, shift = target_median - median * scale
            
            scale_b = target_range[b] * 0.5 / l_mad  # 使用较保守的范围
            shift_b = target_median[b] - l_median * scale_b
            
            scales.append(scale_b)
            shifts.append(shift_b)
            
            # 对齐深度
            l_aligned = l_depth[b] * scale_b + shift_b
            r_aligned = r_depth[b] * scale_b + shift_b  # 使用相同的尺度和偏移保证一致性
            
            l_depths_aligned.append(l_aligned)
            r_depths_aligned.append(r_aligned)
        
        l_depth_aligned = torch.stack(l_depths_aligned, dim=0)
        r_depth_aligned = torch.stack(r_depths_aligned, dim=0)
        scale = torch.stack(scales, dim=0).view(B, 1)
        shift = torch.stack(shifts, dim=0).view(B, 1)
        
        # 限制深度范围
        l_depth_aligned = l_depth_aligned.clamp(self.config.depth_min, self.config.depth_max)
        r_depth_aligned = r_depth_aligned.clamp(self.config.depth_min, self.config.depth_max)
        
        return l_depth_aligned, r_depth_aligned, scale, shift
    
    def to(self, device):
        """移动模型到指定设备"""
        if self.model is not None:
            self.model = self.model.to(device)
        return super().to(device)


def create_dav3_depth_estimator(cfg) -> DAV3DepthEstimator:
    """
    创建 DAV3 深度估计器的工厂函数
    
    Args:
        cfg: 配置对象
        
    Returns:
        DAV3DepthEstimator 实例
    """
    return DAV3DepthEstimator(cfg)
