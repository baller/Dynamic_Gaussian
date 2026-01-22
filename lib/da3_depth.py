"""
DA3 (Depth-Anything-3) 深度估计模块
用于替换GPS_plus中的RAFT-Stereo深度估计

该模块封装DA3模型，提供与GPS_plus兼容的深度估计接口。
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

# 添加DA3路径 - DA3模块位于 src 子目录下
DA3_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Depth-Anything-3')
DA3_SRC_PATH = os.path.join(DA3_ROOT, 'src')
if DA3_SRC_PATH not in sys.path:
    sys.path.insert(0, DA3_SRC_PATH)


class DA3DepthEstimator(nn.Module):
    """
    DA3深度估计器
    
    封装Depth-Anything-3模型，提供与GPS_plus兼容的接口。
    支持单目和双目深度估计模式。
    """
    
    def __init__(self, cfg):
        """
        初始化DA3深度估计器
        
        Args:
            cfg: 配置对象，包含DA3相关配置
        """
        super().__init__()
        self.cfg = cfg
        self.da3_cfg = cfg.da3
        self.model = None
        self.device = None
        self._initialized = False
        
        # DA3模型配置
        self.model_name = self.da3_cfg.model_name  # 例如 "depth-anything/DA3-LARGE"
        self.use_metric = self.da3_cfg.get('use_metric', False)
        self.scale_factor = self.da3_cfg.get('scale_factor', 1.0)
        
    def _lazy_init(self, device):
        """延迟初始化DA3模型（首次forward时调用）"""
        if self._initialized:
            return
            
        self.device = device
        
        try:
            from depth_anything_3.api import DepthAnything3
            
            print(f"[DA3] 加载模型: {self.model_name}")
            self.model = DepthAnything3.from_pretrained(self.model_name)
            self.model = self.model.to(device=device)
            self.model.eval()
            
            # 冻结DA3参数（可选，根据配置决定是否微调）
            if not self.da3_cfg.get('finetune', False):
                for param in self.model.parameters():
                    param.requires_grad = False
                print("[DA3] 模型参数已冻结")
            else:
                print("[DA3] 模型参数可训练")
                
            self._initialized = True
            print("[DA3] 模型加载完成")
            
        except ImportError as e:
            raise ImportError(
                f"无法导入DA3模块，请确保Depth-Anything-3已正确安装。\n"
                f"DA3根目录: {DA3_ROOT}\n"
                f"DA3源码路径: {DA3_SRC_PATH}\n"
                f"错误: {e}"
            )
    
    def freeze_bn(self):
        """冻结BatchNorm层"""
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm)):
                m.eval()
    
    def preprocess_images(self, img_tensor):
        """
        预处理图像张量为DA3输入格式
        
        Args:
            img_tensor: [B, C, H, W] 范围 [-1, 1] 或 [0, 1]
            
        Returns:
            处理后的图像张量 [B, 1, C, H, W] 范围 [0, 1]
        """
        # 如果输入范围是[-1, 1]，转换到[0, 1]
        if img_tensor.min() < 0:
            img_tensor = img_tensor * 0.5 + 0.5
        
        # DA3期望输入形状为 [B, N, C, H, W]，其中N是视图数
        # 对于单目输入，N=1
        if img_tensor.dim() == 4:
            img_tensor = img_tensor.unsqueeze(1)  # [B, 1, C, H, W]
            
        return img_tensor
    
    def forward(self, data, is_train=True):
        """
        前向传播，估计深度
        
        Args:
            data: 包含 'lmain' 和 'rmain' 的数据字典
                  每个视图包含 'img' 键，形状为 [B, C, H, W]
            is_train: 是否训练模式
            
        Returns:
            data: 更新后的数据字典，添加 'depth' 键
            depth_loss: 深度损失（如果有GT）
            metrics: 指标字典
        """
        device = data['lmain']['img'].device
        self._lazy_init(device)
        
        bs = data['lmain']['img'].shape[0]
        metrics = {}
        depth_loss = None
        
        # 获取左右视图图像
        img_l = data['lmain']['img']  # [B, C, H, W], 范围 [-1, 1]
        img_r = data['rmain']['img']  # [B, C, H, W], 范围 [-1, 1]
        
        # 分别对左右视图进行深度估计
        with torch.set_grad_enabled(is_train and self.da3_cfg.get('finetune', False)):
            depth_l = self._estimate_depth_single(img_l)
            depth_r = self._estimate_depth_single(img_r)
        
        # 深度尺度对齐
        # DA3输出的是相对深度，需要根据场景尺度进行调整
        # 使用配置中的逆深度初始值进行尺度对齐
        inverse_depth_init = self.cfg.dataset.inverse_depth_init
        
        # 将深度转换为逆深度格式（与原始GPS_plus一致）
        # DA3输出的深度值越大表示越远
        depth_l = self._align_depth_scale(depth_l, inverse_depth_init)
        depth_r = self._align_depth_scale(depth_r, inverse_depth_init)
        
        # 存储深度结果
        data['lmain']['depth'] = depth_l
        data['rmain']['depth'] = depth_r
        
        # 为了兼容原有流程，也生成伪flow_pred
        # 注意：这里的flow_pred仅用于兼容性，不参与实际计算
        data['lmain']['flow_pred'] = self._depth_to_pseudo_flow(depth_l, data['lmain'])
        data['rmain']['flow_pred'] = self._depth_to_pseudo_flow(depth_r, data['rmain'])
        
        return data, depth_loss, metrics
    
    def _pad_to_patch_size(self, img, patch_size=14):
        """
        将图像填充到patch_size的倍数
        
        Args:
            img: [B, C, H, W] 输入图像
            patch_size: patch大小，默认14
            
        Returns:
            padded_img: 填充后的图像
            (H, W): 原始尺寸
        """
        B, C, H, W = img.shape
        
        # 计算需要填充到的尺寸（向上取整到patch_size的倍数）
        new_H = ((H + patch_size - 1) // patch_size) * patch_size
        new_W = ((W + patch_size - 1) // patch_size) * patch_size
        
        if new_H == H and new_W == W:
            return img, (H, W)
        
        # 使用反射填充
        pad_h = new_H - H
        pad_w = new_W - W
        # F.pad格式: (left, right, top, bottom)
        padded_img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
        
        return padded_img, (H, W)
    
    def _estimate_depth_single(self, img):
        """
        对单张图像进行深度估计
        
        Args:
            img: [B, C, H, W] 范围 [-1, 1]
            
        Returns:
            depth: [B, 1, H, W] 深度图
        """
        B, C, H, W = img.shape
        original_size = (H, W)
        
        # 转换到[0, 1]范围，然后进行ImageNet标准化
        img_normalized = img * 0.5 + 0.5  # [0, 1]
        
        # ImageNet标准化
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img_normalized = (img_normalized - mean) / std
        
        # DA3使用patch_size=14，需要输入尺寸是14的倍数
        img_padded, _ = self._pad_to_patch_size(img_normalized, patch_size=14)
        
        # DA3期望输入 [B, N, C, H, W]
        img_input = img_padded.unsqueeze(1)  # [B, 1, C, H_padded, W_padded]
        
        # 调用DA3模型 - 使用self.model.model访问底层网络
        with autocast(enabled=self.da3_cfg.get('mixed_precision', False)):
            # DepthAnything3.model 是底层的 DepthAnything3Net
            prediction = self.model.model(
                img_input,
                extrinsics=None,
                intrinsics=None,
                export_feat_layers=[],
                infer_gs=False,
                use_ray_pose=False,
            )
        
        # 获取深度输出
        # DA3输出格式: prediction.depth [B, N, H, W]
        depth = prediction.depth  # [B, 1, H_out, W_out]
        
        # 确保输出形状为 [B, 1, H, W]
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        elif depth.dim() == 4 and depth.shape[1] != 1:
            depth = depth[:, :1]
        
        # 裁剪回原始尺寸并进行插值（如果需要）
        if depth.shape[-2:] != original_size:
            # 首先裁剪到原始尺寸（如果深度图更大）
            depth_h, depth_w = depth.shape[-2:]
            if depth_h >= H and depth_w >= W:
                depth = depth[:, :, :H, :W]
            
            # 如果尺寸仍不匹配，进行插值
            if depth.shape[-2:] != original_size:
                depth = F.interpolate(
                    depth,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )
            
        return depth
    
    def _align_depth_scale(self, depth, inverse_depth_init):
        """
        对齐深度尺度
        
        DA3输出的深度是相对深度，需要根据场景尺度进行调整。
        使用median normalization方法。
        
        Args:
            depth: [B, 1, H, W] DA3输出的深度
            inverse_depth_init: 场景的逆深度初始值
            
        Returns:
            aligned_depth: [B, 1, H, W] 对齐后的逆深度
        """
        # DA3输出的是深度值（距离），转换为逆深度
        # 首先对深度进行归一化
        B = depth.shape[0]
        
        aligned_depths = []
        for b in range(B):
            d = depth[b]  # [1, H, W]
            
            # 计算有效深度的中位数
            valid_mask = d > 1e-6
            if valid_mask.sum() > 0:
                median_depth = d[valid_mask].median()
            else:
                median_depth = d.mean()
            
            # 根据逆深度初始值对齐尺度
            # 假设中位数深度对应的逆深度为inverse_depth_init
            target_median_inv_depth = inverse_depth_init
            
            # 计算尺度因子
            # inv_depth = 1 / depth
            # 目标: median(1/depth) ≈ inverse_depth_init
            # 所以: depth_scaled = depth * (1 / (inverse_depth_init * median_depth))
            if median_depth > 1e-6:
                scale = 1.0 / (target_median_inv_depth * median_depth)
                d_scaled = d * scale
            else:
                d_scaled = d
            
            # 转换为逆深度
            inv_depth = 1.0 / (d_scaled + 1e-6)
            
            # 裁剪到合理范围
            inv_depth = torch.clamp(inv_depth, min=0.01, max=10.0)
            
            aligned_depths.append(inv_depth)
        
        return torch.stack(aligned_depths, dim=0)
    
    def _depth_to_pseudo_flow(self, depth, view_data):
        """
        将深度转换为伪光流（用于兼容原有流程）
        
        GPS_plus原本使用光流来计算深度:
        disparity = offset - flow
        depth = -disparity / Tf_x
        
        反过来:
        disparity = -depth * Tf_x
        flow = offset - disparity = offset + depth * Tf_x
        
        Args:
            depth: [B, 1, H, W] 逆深度
            view_data: 包含相机参数的视图数据
            
        Returns:
            pseudo_flow: [B, 1, H, W] 伪光流
        """
        Tf_x = view_data['Tf_x']  # [B]
        offset = view_data['ref_intr'][:, 0, 2] - view_data['intr'][:, 0, 2]  # [B]
        
        # 扩展维度以匹配depth
        Tf_x = Tf_x[:, None, None, None]  # [B, 1, 1, 1]
        offset = offset[:, None, None, None]  # [B, 1, 1, 1]
        
        # 计算视差
        disparity = -depth * Tf_x
        
        # 计算伪光流
        pseudo_flow = offset - disparity
        
        return pseudo_flow


class DA3DepthEstimatorDual(DA3DepthEstimator):
    """
    DA3双目深度估计器
    
    利用双目信息进行更准确的深度估计。
    将左右视图一起输入DA3，利用多视图一致性。
    """
    
    def forward(self, data, is_train=True):
        """
        双目深度估计
        
        Args:
            data: 包含 'lmain' 和 'rmain' 的数据字典
            is_train: 是否训练模式
            
        Returns:
            data: 更新后的数据字典
            depth_loss: 深度损失
            metrics: 指标字典
        """
        device = data['lmain']['img'].device
        self._lazy_init(device)
        
        bs = data['lmain']['img'].shape[0]
        metrics = {}
        depth_loss = None
        
        # 获取左右视图图像
        img_l = data['lmain']['img']  # [B, C, H, W]
        img_r = data['rmain']['img']  # [B, C, H, W]
        
        B, C, H, W = img_l.shape
        original_size = (H, W)
        
        # 将左右视图组合为多视图输入
        # DA3支持多视图输入 [B, N, C, H, W]
        # 首先转换到[0, 1]范围
        img_normalized_l = img_l * 0.5 + 0.5
        img_normalized_r = img_r * 0.5 + 0.5
        
        # ImageNet标准化
        mean = torch.tensor([0.485, 0.456, 0.406], device=img_l.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img_l.device).view(1, 3, 1, 1)
        img_normalized_l = (img_normalized_l - mean) / std
        img_normalized_r = (img_normalized_r - mean) / std
        
        # DA3使用patch_size=14，需要输入尺寸是14的倍数
        img_padded_l, _ = self._pad_to_patch_size(img_normalized_l, patch_size=14)
        img_padded_r, _ = self._pad_to_patch_size(img_normalized_r, patch_size=14)
        
        # 组合为双视图输入
        img_stereo = torch.stack([img_padded_l, img_padded_r], dim=1)  # [B, 2, C, H_padded, W_padded]
        
        with torch.set_grad_enabled(is_train and self.da3_cfg.get('finetune', False)):
            with autocast(enabled=self.da3_cfg.get('mixed_precision', False)):
                # DepthAnything3.model 是底层的 DepthAnything3Net
                prediction = self.model.model(
                    img_stereo,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[],
                    infer_gs=False,
                    use_ray_pose=False,
                )
        
        # 获取双视图深度
        depth = prediction.depth  # [B, 2, H_out, W_out]
        
        # 裁剪回原始尺寸并进行插值（如果需要）
        if depth.shape[-2:] != original_size:
            depth_h, depth_w = depth.shape[-2:]
            if depth_h >= H and depth_w >= W:
                depth = depth[:, :, :H, :W]
            
            if depth.shape[-2:] != original_size:
                depth = F.interpolate(
                    depth,
                    size=original_size,
                    mode='bilinear',
                    align_corners=False
                )
        
        # 分离左右视图深度
        depth_l = depth[:, 0:1]  # [B, 1, H, W]
        depth_r = depth[:, 1:2]  # [B, 1, H, W]
        
        # 深度尺度对齐
        inverse_depth_init = self.cfg.dataset.inverse_depth_init
        depth_l = self._align_depth_scale(depth_l, inverse_depth_init)
        depth_r = self._align_depth_scale(depth_r, inverse_depth_init)
        
        # 存储结果
        data['lmain']['depth'] = depth_l
        data['rmain']['depth'] = depth_r
        
        # 生成伪flow_pred用于兼容性
        data['lmain']['flow_pred'] = self._depth_to_pseudo_flow(depth_l, data['lmain'])
        data['rmain']['flow_pred'] = self._depth_to_pseudo_flow(depth_r, data['rmain'])
        
        return data, depth_loss, metrics


def create_da3_depth_estimator(cfg, mode='single'):
    """
    创建DA3深度估计器
    
    Args:
        cfg: 配置对象
        mode: 'single' 单目模式，'dual' 双目模式
        
    Returns:
        DA3深度估计器实例
    """
    if mode == 'single':
        return DA3DepthEstimator(cfg)
    elif mode == 'dual':
        return DA3DepthEstimatorDual(cfg)
    else:
        raise ValueError(f"未知的DA3模式: {mode}")
