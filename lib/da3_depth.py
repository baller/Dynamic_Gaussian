"""
DA3 深度估计模块
用于封装 Depth Anything 3 模型，提供深度估计和特征提取功能

Features:
- 深度估计: 输出相对深度图
- 多尺度特征提取: 从 DINOv2 backbone 提取多尺度特征
- 注意力熵计算: 作为纹理复杂度信号
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
import logging

logger = logging.getLogger(__name__)


class DA3DepthEstimator(nn.Module):
    """
    DA3 深度估计器
    
    封装 Depth Anything 3 模型，提供:
    1. 深度估计
    2. 多尺度 DINOv2 特征
    3. 注意力熵（纹理复杂度信号）
    
    Args:
        cfg: 配置对象，包含 da3 相关配置
        device: 模型运行设备
    """
    
    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device = device
        
        # DA3 配置
        self.da3_cfg = getattr(cfg, 'da3', None)
        if self.da3_cfg is None:
            # 默认配置
            self.da3_cfg = {
                'model_name': 'depth-anything/DA3-LARGE',
                'export_feat_layers': [11, 15, 19, 23],
                'freeze_backbone': True,
                'mixed_precision': False,
            }
        else:
            # 转换为字典
            self.da3_cfg = dict(self.da3_cfg) if hasattr(self.da3_cfg, 'items') else {
                'model_name': getattr(self.da3_cfg, 'model_name', 'depth-anything/DA3-LARGE'),
                'export_feat_layers': getattr(self.da3_cfg, 'export_feat_layers', [11, 15, 19, 23]),
                'freeze_backbone': getattr(self.da3_cfg, 'freeze_backbone', True),
                'mixed_precision': getattr(self.da3_cfg, 'mixed_precision', False),
            }
        
        # 根据模型名推断特征维度
        model_name = self.da3_cfg.get('model_name', 'depth-anything/DA3-LARGE')
        if 'LARGE' in model_name.upper() or 'GIANT' in model_name.upper():
            self.da3_feat_dim = 1024
        elif 'BASE' in model_name.upper():
            self.da3_feat_dim = 768
        else:
            self.da3_feat_dim = 384
        
        self.model = None
        self.is_loaded = False
        
    def _lazy_load_model(self):
        """延迟加载模型，避免初始化时占用显存"""
        if self.is_loaded:
            return
            
        try:
            from depth_anything_3.api import DepthAnything3
            import os
            
            model_name = self.da3_cfg.get('model_name', 'depth-anything/DA3-LARGE')
            local_files_only = self.da3_cfg.get('local_files_only', True)
            
            # 智能处理 model_name：支持本地路径和 HuggingFace repo id
            model_path = self._resolve_model_path(model_name)
            logger.info(f"[DA3] 加载模型: {model_path} (local_files_only={local_files_only})")
            
            # 从 HuggingFace 或本地路径加载预训练模型
            self.model = DepthAnything3.from_pretrained(model_path, local_files_only=local_files_only)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # 冻结参数
            if self.da3_cfg.get('freeze_backbone', True):
                for param in self.model.parameters():
                    param.requires_grad = False
                logger.info("[DA3] 模型参数已冻结")
            
            self.is_loaded = True
            logger.info("[DA3] 模型加载完成")
            
        except ImportError as e:
            logger.error(f"[DA3] 无法导入 depth_anything_3: {e}")
            logger.error("[DA3] 请安装: pip install -e /home/user_3/3DGS/Depth-Anything-3")
            raise
        except Exception as e:
            logger.error(f"[DA3] 模型加载失败: {e}")
            raise
    
    def _resolve_model_path(self, model_name: str) -> str:
        """
        解析模型路径，支持多种格式：
        1. HuggingFace repo id: 'depth-anything/DA3-LARGE'
        2. 本地目录路径: '/path/to/model_dir/' (包含 config.json 和 model.safetensors)
        3. 本地文件路径: '/path/to/model.safetensors' -> 转换为其父目录
        
        Args:
            model_name: 模型名称或路径
            
        Returns:
            可用于 from_pretrained 的路径
        """
        import os
        
        # 检查是否是本地路径（以 / 开头或包含路径分隔符）
        if os.path.isabs(model_name) or os.path.exists(model_name):
            # 如果是 .safetensors 文件，使用其父目录
            if model_name.endswith('.safetensors'):
                parent_dir = os.path.dirname(model_name)
                logger.info(f"[DA3] 检测到 safetensors 文件路径，使用父目录: {parent_dir}")
                return parent_dir
            # 如果是目录，直接返回
            elif os.path.isdir(model_name):
                return model_name
            # 如果是其他文件，使用其父目录
            elif os.path.isfile(model_name):
                parent_dir = os.path.dirname(model_name)
                logger.info(f"[DA3] 检测到文件路径，使用父目录: {parent_dir}")
                return parent_dir
        
        # 否则认为是 HuggingFace repo id
        return model_name
    
    def forward(self, images, intrinsics=None, return_features=True, return_entropy=True):
        """
        前向传播
        
        Args:
            images: 输入图像 [B, 3, H, W]，值范围 [0, 1]
            intrinsics: 相机内参 [B, 3, 3]，用于 metric 模型
            return_features: 是否返回多尺度特征
            return_entropy: 是否返回注意力熵
            
        Returns:
            dict: {
                'depth': [B, 1, H, W] 深度图 (metric 模型返回绝对深度),
                'features': list of [B, C, H', W'] 多尺度特征 (可选),
                'entropy': [B, 1, H, W] 注意力熵 (可选),
                'is_metric': bool 是否为 metric 深度
            }
        """
        self._lazy_load_model()
        
        B, C, H, W = images.shape
        device = images.device
        
        # DA3/DINOv2 要求输入尺寸是 14 的倍数
        PATCH_SIZE = 14
        pad_h = (PATCH_SIZE - H % PATCH_SIZE) % PATCH_SIZE
        pad_w = (PATCH_SIZE - W % PATCH_SIZE) % PATCH_SIZE
        
        if pad_h > 0 or pad_w > 0:
            # 使用反射填充
            images = F.pad(images, (0, pad_w, 0, pad_h), mode='reflect')
        
        H_padded, W_padded = images.shape[-2:]
        
        # DA3 期望输入格式: [B, N, 3, H, W]，N 是视图数
        # 对于单视图输入，N=1
        images_da3 = images.unsqueeze(1)  # [B, 1, 3, H, W]
        
        # 准备内参 (用于 metric 模型)
        intrinsics_da3 = None
        if intrinsics is not None:
            # 调整内参以适应 padding 后的尺寸
            intrinsics_da3 = intrinsics.clone()
            # 不需要调整，因为深度会被裁剪回原始尺寸
            intrinsics_da3 = intrinsics_da3.unsqueeze(1)  # [B, 1, 3, 3]
        
        # 获取要导出的特征层
        export_feat_layers = self.da3_cfg.get('export_feat_layers', [11, 15, 19, 23])
        
        with autocast('cuda', enabled=self.da3_cfg.get('mixed_precision', False)):
            # DA3 前向传播
            output = self.model(
                images_da3,
                intrinsics=intrinsics_da3,
                export_feat_layers=export_feat_layers if return_features else []
            )
        
        result = {}
        
        # 检查是否为 metric 模型
        is_metric_model = 'metric' in self.da3_cfg.get('model_name', '').lower()
        result['is_metric'] = is_metric_model
        
        # 提取深度图
        # DA3 输出深度形状: [B, N, H, W]
        # 注意: DA3 使用 inference_mode，需要 clone() 使张量可用于训练
        depth = output.depth.clone()  # [B, 1, H, W] -> [B, H, W]
        if depth.dim() == 4:
            depth = depth[:, 0]  # [B, H, W]
        depth = depth.unsqueeze(1)  # [B, 1, H, W]
        
        # 移除 padding，恢复到原始尺寸
        if pad_h > 0 or pad_w > 0:
            depth = depth[:, :, :H, :W].contiguous()
        
        # 确保深度图尺寸与输入一致
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
        
        # 对于 metric 模型，应用 focal scaling: metric_depth = focal * raw_depth / 300
        # 参考官方代码: https://github.com/DepthAnything/Depth-Anything-3
        if is_metric_model and intrinsics is not None:
            # intrinsics: [B, 3, 3]
            # 提取焦距 (fx, fy 的平均值)
            focal_length = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2.0  # [B]
            
            # 应用 metric scaling (从配置读取 scale factor)
            scale_factor = getattr(self.da3_cfg, 'metric_scale_factor', 300.0) if self.da3_cfg else 300.0
            depth = depth * (focal_length[:, None, None, None] / scale_factor)
            
            logger.debug(f"[DA3] Applied metric scaling with focal={focal_length.mean().item():.2f}, "
                        f"depth range: [{depth.min().item():.3f}, {depth.max().item():.3f}]")
        
        result['depth'] = depth
        result['depth_raw'] = output.depth.clone()  # 保存原始深度
        
        # 提取多尺度特征
        if return_features and hasattr(output, 'aux') and output.aux is not None:
            features = []
            for layer_idx in export_feat_layers:
                key = f'feat_layer_{layer_idx}'
                if key in output.aux:
                    feat = output.aux[key].clone()  # clone 使张量可用于训练
                    # 特征形状: [B, N, C, H', W'] -> [B, C, H', W']
                    if feat.dim() == 5:
                        feat = feat[:, 0]
                    # DA3 输出可能是 [B, H', W', C] 格式，需要转换为 [B, C, H', W']
                    if feat.dim() == 4 and feat.shape[-1] == self.da3_feat_dim:
                        feat = feat.permute(0, 3, 1, 2).contiguous()  # [B, H', W', C] -> [B, C, H', W']
                    features.append(feat)
            result['features'] = features
        
        # 计算注意力熵（纹理复杂度信号）
        if return_entropy:
            entropy = self._compute_attention_entropy(output, H, W)
            result['entropy'] = entropy.clone() if entropy is not None else None
        
        return result
    
    def _compute_attention_entropy(self, output, H, W):
        """
        从注意力权重计算熵，作为纹理复杂度信号
        
        高熵区域 = 注意力分散 = 纹理复杂区域
        低熵区域 = 注意力集中 = 平滑区域
        
        Args:
            output: DA3 模型输出
            H, W: 目标尺寸
            
        Returns:
            entropy: [B, 1, H, W] 注意力熵图
        """
        # 尝试从 aux 中获取注意力权重
        if hasattr(output, 'aux') and output.aux is not None:
            # 查找注意力相关的输出
            for key in output.aux:
                if 'attn' in key.lower():
                    attn = output.aux[key]
                    # 计算熵
                    eps = 1e-8
                    attn = attn.clamp(min=eps)
                    entropy = -(attn * torch.log(attn)).sum(dim=-1)
                    # 调整形状
                    if entropy.dim() == 4:  # [B, heads, H', W']
                        entropy = entropy.mean(dim=1, keepdim=True)
                    elif entropy.dim() == 3:  # [B, H', W']
                        entropy = entropy.unsqueeze(1)
                    
                    # 上采样到目标尺寸
                    entropy = F.interpolate(entropy, size=(H, W), mode='bilinear', align_corners=False)
                    return entropy
        
        # 如果没有注意力权重，使用深度图梯度作为替代
        depth = output.depth
        if depth.dim() == 4:
            depth = depth[:, 0]
        depth = depth.unsqueeze(1)
        
        # 计算深度图梯度
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(depth, sobel_x, padding=1)
        grad_y = F.conv2d(depth, sobel_y, padding=1)
        gradient_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        
        # 归一化
        gradient_magnitude = gradient_magnitude / (gradient_magnitude.max() + 1e-8)
        
        # 上采样到目标尺寸
        if gradient_magnitude.shape[-2:] != (H, W):
            gradient_magnitude = F.interpolate(gradient_magnitude, size=(H, W), 
                                               mode='bilinear', align_corners=False)
        
        return gradient_magnitude
    
    def get_feature_dims(self):
        """
        获取各层特征的通道数
        
        Returns:
            list: 各层特征的通道数
        """
        # DINOv2-L 的特征维度
        # Layer 11, 15, 19, 23 都是 1024 维
        return [1024, 1024, 1024, 1024]
    
    def to(self, device):
        """移动模型到指定设备"""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return super().to(device)


class DA3FeatureAdapter(nn.Module):
    """
    DA3 特征适配器
    
    将 DA3 的多尺度特征转换为与原 RAFT encoder 兼容的格式
    用于与现有 GSRegresser 等模块兼容
    
    Args:
        in_dims: DA3 特征维度列表，如 [1024, 1024, 1024, 1024]
        out_dims: 目标特征维度列表，如 [32, 48, 96] (与 RAFT encoder 一致)
    """
    
    def __init__(self, in_dims=[1024, 1024, 1024, 1024], out_dims=[32, 48, 96]):
        super().__init__()
        
        self.in_dims = in_dims
        self.out_dims = out_dims
        self.num_in_layers = len(in_dims)
        
        # 创建适配层
        # 支持 3 层或 4 层输入，映射到 3 个输出层
        
        # Level 1: 第一层 -> out_dims[0]
        self.adapter1 = nn.Sequential(
            nn.Conv2d(in_dims[0], out_dims[0], 1),
            nn.BatchNorm2d(out_dims[0]),
            nn.ReLU(inplace=True)
        )
        
        # Level 2: 第二层 -> out_dims[1]
        self.adapter2 = nn.Sequential(
            nn.Conv2d(in_dims[1], out_dims[1], 1),
            nn.BatchNorm2d(out_dims[1]),
            nn.ReLU(inplace=True)
        )
        
        # Level 3: 融合剩余层 -> out_dims[2]
        if self.num_in_layers >= 4:
            # 4 层输入: 融合 layer 3, 4
            in_dim_3 = in_dims[2] + in_dims[3]
        else:
            # 3 层输入: 只使用 layer 3
            in_dim_3 = in_dims[2]
        
        self.adapter3 = nn.Sequential(
            nn.Conv2d(in_dim_3, out_dims[2], 1),
            nn.BatchNorm2d(out_dims[2]),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, features, target_sizes=None):
        """
        适配特征
        
        Args:
            features: DA3 多尺度特征列表，3 或 4 个张量
            target_sizes: 目标尺寸列表 [(H1, W1), (H2, W2), (H3, W3)]
            
        Returns:
            adapted_features: 适配后的特征列表，3 个张量
        """
        if len(features) < 3:
            raise ValueError(f"Expected at least 3 feature levels, got {len(features)}")
        
        f1, f2, f3 = features[0], features[1], features[2]
        
        # DA3 输出格式是 [B, H, W, C]，需要转换为 [B, C, H, W]
        if f1.dim() == 4 and f1.shape[-1] == self.in_dims[0]:
            f1 = f1.permute(0, 3, 1, 2).contiguous()
        if f2.dim() == 4 and f2.shape[-1] == self.in_dims[1]:
            f2 = f2.permute(0, 3, 1, 2).contiguous()
        if f3.dim() == 4 and f3.shape[-1] == self.in_dims[2]:
            f3 = f3.permute(0, 3, 1, 2).contiguous()
        
        # Level 1
        out1 = self.adapter1(f1)
        
        # Level 2
        out2 = self.adapter2(f2)
        
        # Level 3: 根据输入层数处理
        if self.num_in_layers >= 4 and len(features) >= 4:
            f4 = features[3]
            # DA3 输出格式转换
            if f4.dim() == 4 and f4.shape[-1] == self.in_dims[3]:
                f4 = f4.permute(0, 3, 1, 2).contiguous()
            # 首先确保尺寸一致
            if f3.shape[-2:] != f4.shape[-2:]:
                f4 = F.interpolate(f4, size=f3.shape[-2:], mode='bilinear', align_corners=False)
            f34 = torch.cat([f3, f4], dim=1)
            out3 = self.adapter3(f34)
        else:
            # 只使用 f3
            out3 = self.adapter3(f3)
        
        # 调整到目标尺寸
        if target_sizes is not None:
            if out1.shape[-2:] != target_sizes[0]:
                out1 = F.interpolate(out1, size=target_sizes[0], mode='bilinear', align_corners=False)
            if out2.shape[-2:] != target_sizes[1]:
                out2 = F.interpolate(out2, size=target_sizes[1], mode='bilinear', align_corners=False)
            if out3.shape[-2:] != target_sizes[2]:
                out3 = F.interpolate(out3, size=target_sizes[2], mode='bilinear', align_corners=False)
        
        return [out1, out2, out3]


def create_da3_estimator(cfg, device='cuda'):
    """
    创建 DA3 深度估计器的工厂函数
    
    Args:
        cfg: 配置对象
        device: 设备
        
    Returns:
        DA3DepthEstimator 实例
    """
    return DA3DepthEstimator(cfg, device)
