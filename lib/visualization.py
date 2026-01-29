"""
训练过程中间结果可视化工具

支持可视化：
1. 深度图 (左/右视图，DA3 预测)
2. 高斯分布 (多层不透明度，裁剪前后对比)
3. Novel View 预测 vs GT 对比
4. 不透明度/缩放图
5. 特征图 (可选)
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端


class TrainingVisualizer:
    """训练过程可视化器"""
    
    def __init__(self, cfg, save_dir: str):
        """
        初始化可视化器
        
        Args:
            cfg: 配置对象
            save_dir: 保存目录
        """
        self.cfg = cfg
        self.save_dir = Path(save_dir)
        self.img_range = getattr(getattr(cfg, 'dataset', None), 'img_range', None)
        
        # 从配置读取参数
        vis_cfg = getattr(cfg, 'visualization', None)
        if vis_cfg is not None:
            self.enabled = getattr(vis_cfg, 'enabled', True)
            self.vis_freq = getattr(vis_cfg, 'vis_freq', 100)
            self.save_depth = getattr(vis_cfg, 'save_depth', True)
            self.save_gaussian = getattr(vis_cfg, 'save_gaussian', True)
            self.save_novel_view = getattr(vis_cfg, 'save_novel_view', True)
            self.save_opacity = getattr(vis_cfg, 'save_opacity', True)
            self.save_features = getattr(vis_cfg, 'save_features', False)
            self.colormap = getattr(vis_cfg, 'colormap', 'turbo')
            self.max_images = getattr(vis_cfg, 'max_images', 4)
            self.layer_patch_size = getattr(vis_cfg, 'layer_patch_size', 16)
        else:
            self.enabled = True
            self.vis_freq = 100
            self.save_depth = True
            self.save_gaussian = True
            self.save_novel_view = True
            self.save_opacity = True
            self.save_features = False
            self.colormap = 'turbo'
            self.max_images = 4
            self.layer_patch_size = 16
        
        # 创建子目录
        self.depth_dir = self.save_dir / 'depth'
        self.gaussian_dir = self.save_dir / 'gaussian'
        self.novel_view_dir = self.save_dir / 'novel_view'
        self.opacity_dir = self.save_dir / 'opacity'
        self.features_dir = self.save_dir / 'features'
        
        for d in [self.depth_dir, self.gaussian_dir, self.novel_view_dir, 
                  self.opacity_dir, self.features_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # 颜色映射
        self.cmap = plt.get_cmap(self.colormap)
    
    def should_visualize(self, step: int) -> bool:
        """检查是否应该在当前步骤进行可视化"""
        return self.enabled and step > 0 and step % self.vis_freq == 0
    
    def visualize(self, data: Dict, step: int, phase: str = 'train'):
        """
        执行可视化
        
        Args:
            data: 训练数据字典
            step: 当前训练步数
            phase: 'train' 或 'val'
        """
        if not self.should_visualize(step):
            return
        
        prefix = f"{phase}_{step:06d}"
        
        try:
            # 1. 深度图可视化
            if self.save_depth:
                self._visualize_depth(data, prefix)
            
            # 2. 高斯分布可视化
            if self.save_gaussian:
                self._visualize_gaussian(data, prefix)
            
            # 3. Novel View 对比可视化
            if self.save_novel_view:
                self._visualize_novel_view(data, prefix)
            
            # 4. 不透明度可视化
            if self.save_opacity:
                self._visualize_opacity(data, prefix)
            
            # 5. 特征图可视化 (可选)
            if self.save_features:
                self._visualize_features(data, prefix)
                
        except Exception as e:
            print(f"[Visualizer] Warning: visualization failed at step {step}: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_depth(self, data: Dict, prefix: str):
        """可视化深度图"""
        has_mono = 'depth_mono' in data.get('lmain', {})
        ncols = 5 if has_mono else 4
        fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 5))
        if ncols == 1:
            axes = [axes]
        
        batch_idx = 0  # 只可视化第一个样本
        
        # 左视图深度
        if 'lmain' in data and 'depth' in data['lmain']:
            depth_l = data['lmain']['depth'][batch_idx]
            if depth_l.dim() == 3:
                depth_l = depth_l[0]
            depth_l_np = self._depth_to_color(depth_l)
            axes[0].imshow(depth_l_np)
            axes[0].set_title('Left Absolute Depth')
            axes[0].axis('off')
        
        # 右视图深度
        if 'rmain' in data and 'depth' in data['rmain']:
            depth_r = data['rmain']['depth'][batch_idx]
            if depth_r.dim() == 3:
                depth_r = depth_r[0]
            depth_r_np = self._depth_to_color(depth_r)
            axes[1].imshow(depth_r_np)
            axes[1].set_title('Right Absolute Depth')
            axes[1].axis('off')

        col_offset = 2
        if has_mono:
            depth_mono = data['lmain']['depth_mono'][batch_idx]
            if depth_mono.dim() == 3:
                depth_mono = depth_mono[0]
            depth_mono_np = self._depth_to_color(depth_mono)
            axes[2].imshow(depth_mono_np)
            axes[2].set_title('DA3 Mono Depth')
            axes[2].axis('off')
            col_offset = 3
        
        # 左视图原始图像
        if 'lmain' in data and 'img' in data['lmain']:
            img_l = data['lmain']['img'][batch_idx]
            img_l_np = self._tensor_to_image(img_l)
            axes[col_offset].imshow(img_l_np)
            axes[col_offset].set_title('Left Image')
            axes[col_offset].axis('off')
        
        # 右视图原始图像
        if 'rmain' in data and 'img' in data['rmain']:
            img_r = data['rmain']['img'][batch_idx]
            img_r_np = self._tensor_to_image(img_r)
            axes[col_offset + 1].imshow(img_r_np)
            axes[col_offset + 1].set_title('Right Image')
            axes[col_offset + 1].axis('off')
        
        # 添加 scale/shift 信息
        if 'scale' in data and 'shift' in data:
            scale_val = data['scale'][batch_idx].item() if torch.is_tensor(data['scale']) else data['scale']
            shift_val = data['shift'][batch_idx].item() if torch.is_tensor(data['shift']) else data['shift']
            fig.suptitle(f'Scale: {scale_val:.4f}, Shift: {shift_val:.4f}', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.depth_dir / f'{prefix}_depth.jpg', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _visualize_gaussian(self, data: Dict, prefix: str):
        """可视化高斯分布 (多层 + 裁剪)"""
        batch_idx = 0
        
        # 检查是否有高斯参数
        views_to_check = ['lmain', 'rmain']
        for view in views_to_check:
            if view not in data:
                continue
            
            view_data = data[view]
            
            # 创建子图
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            has_content = False
            
            # 检查是否有多层参数格式
            multilayer_params = view_data.get('multilayer_params')
            
            # 第一行: 不透明度、缩放、旋转
            if multilayer_params is not None:
                # 新的多层格式: [B, C, L, H, W] 或 [B, 1, L, H, W]
                if 'opacity' in multilayer_params:
                    opacity = multilayer_params['opacity'][batch_idx]  # [1, L, H, W]
                    # 可视化所有层的平均不透明度
                    opacity_mean = opacity.squeeze(0).mean(dim=0)  # [H, W]
                    opacity_np = opacity_mean.detach().cpu().numpy()
                    im = axes[0, 0].imshow(opacity_np, cmap='viridis', vmin=0, vmax=1)
                    axes[0, 0].set_title(f'{view} Opacity (mean over {opacity.shape[1]} layers)')
                    axes[0, 0].axis('off')
                    plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
                    has_content = True
                
                if 'scale' in multilayer_params:
                    scales = multilayer_params['scale'][batch_idx]  # [3, L, H, W]
                    # 计算所有层的平均缩放模
                    scale_mag = scales.norm(dim=0).mean(dim=0)  # [H, W]
                    scale_np = scale_mag.detach().cpu().numpy()
                    im = axes[0, 1].imshow(scale_np, cmap='plasma')
                    axes[0, 1].set_title(f'{view} Scale Magnitude (mean)')
                    axes[0, 1].axis('off')
                    plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
                    has_content = True
                
                if 'rotation' in multilayer_params:
                    rotations = multilayer_params['rotation'][batch_idx]  # [4, L, H, W]
                    # 可视化第一层的 w 分量
                    rot_w = rotations[0, 0]  # [H, W]
                    rot_np = rot_w.detach().cpu().numpy()
                    im = axes[0, 2].imshow(rot_np, cmap='coolwarm', vmin=-1, vmax=1)
                    axes[0, 2].set_title(f'{view} Rotation w (layer 0)')
                    axes[0, 2].axis('off')
                    plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
                    has_content = True
            else:
                # 旧格式兼容
                if 'opacity_maps' in view_data:
                    opacity = view_data['opacity_maps'][batch_idx]
                    if opacity.dim() == 3:
                        opacity = opacity[0]
                    elif opacity.dim() == 4:
                        opacity = opacity[0, 0]
                    
                    if opacity.dim() == 2:
                        opacity_np = opacity.detach().cpu().numpy()
                        im = axes[0, 0].imshow(opacity_np, cmap='viridis', vmin=0, vmax=1)
                        axes[0, 0].set_title(f'{view} Opacity')
                        axes[0, 0].axis('off')
                        plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
                        has_content = True
                
                if 'scale_maps' in view_data:
                    scales = view_data['scale_maps'][batch_idx]
                    if scales.dim() == 4:
                        scales = scales[0]
                    if scales.dim() == 3:
                        scale_mag = scales.norm(dim=0)
                    else:
                        scale_mag = scales
                    
                    if scale_mag.dim() == 2:
                        scale_np = scale_mag.detach().cpu().numpy()
                        im = axes[0, 1].imshow(scale_np, cmap='plasma')
                        axes[0, 1].set_title(f'{view} Scale Magnitude')
                        axes[0, 1].axis('off')
                        plt.colorbar(im, ax=axes[0, 1], fraction=0.046)
                        has_content = True
                
                if 'rot_maps' in view_data:
                    rotations = view_data['rot_maps'][batch_idx]
                    if rotations.dim() == 4:
                        rotations = rotations[0]
                    if rotations.dim() == 3:
                        rot_w = rotations[0]
                    else:
                        rot_w = rotations
                    
                    if rot_w.dim() == 2:
                        rot_np = rot_w.detach().cpu().numpy()
                        im = axes[0, 2].imshow(rot_np, cmap='coolwarm', vmin=-1, vmax=1)
                        axes[0, 2].set_title(f'{view} Rotation w')
                        axes[0, 2].axis('off')
                        plt.colorbar(im, ax=axes[0, 2], fraction=0.046)
                        has_content = True
            
            # 第二行: 深度、点云统计、稀疏立体匹配
            if 'depth' in view_data:
                depth = view_data['depth'][batch_idx]
                if depth.dim() == 3:
                    depth = depth[0]
                depth_colored = self._depth_to_color(depth)
                axes[1, 0].imshow(depth_colored)
                axes[1, 0].set_title(f'{view} Depth (colored)')
                axes[1, 0].axis('off')
                has_content = True
            
            # 点云数量统计
            if 'pts_valid' in view_data:
                pts_valid = view_data['pts_valid'][batch_idx]
                valid_count = pts_valid.sum().item()
                total_count = pts_valid.numel()
                ratio = valid_count / total_count * 100
                axes[1, 1].text(0.5, 0.5, 
                               f"Valid Points:\n{valid_count:,.0f} / {total_count:,}\n({ratio:.1f}%)",
                               ha='center', va='center', fontsize=14,
                               transform=axes[1, 1].transAxes)
                axes[1, 1].set_title(f'{view} Point Statistics')
                axes[1, 1].axis('off')
                has_content = True
            
            # 稀疏立体匹配可视化
            if 'sparse_stereo' in data and view == 'lmain':
                sparse = data['sparse_stereo']
                if 'valid_mask' in sparse:
                    valid_mask = sparse['valid_mask'][batch_idx].detach().cpu().numpy()
                    num_anchors = sparse['num_anchors'][batch_idx].item()
                    axes[1, 2].imshow(valid_mask, cmap='Greens')
                    axes[1, 2].set_title(f'Anchor Points: {int(num_anchors)}')
                    axes[1, 2].axis('off')
                    has_content = True
            
            if has_content:
                plt.tight_layout()
                plt.savefig(self.gaussian_dir / f'{prefix}_{view}_gaussian.jpg', dpi=150, bbox_inches='tight')
            plt.close(fig)
    
    def _visualize_novel_view(self, data: Dict, prefix: str):
        """可视化 Novel View 预测 vs GT"""
        if 'novel_view' not in data:
            return
        
        novel_data = data['novel_view']
        batch_idx = 0
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 预测图像 - 渲染输出已经在 [0, 1] 范围
        if 'img_pred' in novel_data:
            img_pred = novel_data['img_pred'][batch_idx]
            img_pred_np = self._tensor_to_image(img_pred, use_img_range=False)
            axes[0].imshow(img_pred_np)
            axes[0].set_title('Predicted Novel View')
            axes[0].axis('off')
        
        # GT 图像 - 来自数据集
        if 'img' in novel_data:
            img_gt = novel_data['img']
            if torch.is_tensor(img_gt):
                img_gt = img_gt[batch_idx]
            img_gt_np = self._tensor_to_image(img_gt, use_img_range=True)
            axes[1].imshow(img_gt_np)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
        
        # 差异图
        if 'img_pred' in novel_data and 'img' in novel_data:
            img_pred = novel_data['img_pred'][batch_idx]
            img_gt = novel_data['img']
            if torch.is_tensor(img_gt):
                img_gt = img_gt[batch_idx]
            
            # 确保在同一设备
            if img_pred.device != img_gt.device:
                img_gt = img_gt.to(img_pred.device)
            
            # 将 GT 归一化到 [0, 1]
            if self.img_range is not None and len(self.img_range) == 2:
                img_min, img_max = self.img_range
                img_gt_normalized = (img_gt - img_min) / (img_max - img_min)
            else:
                img_gt_normalized = img_gt
            
            diff = (img_pred - img_gt_normalized).abs()
            diff_np = diff.mean(dim=0).detach().cpu().numpy()  # 平均通道
            
            # 归一化差异
            diff_np = diff_np / (diff_np.max() + 1e-8)
            diff_colored = self.cmap(diff_np)[:, :, :3]
            
            axes[2].imshow(diff_colored)
            axes[2].set_title('Absolute Difference')
            axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.novel_view_dir / f'{prefix}_novel_view.jpg', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _visualize_opacity(self, data: Dict, prefix: str):
        """可视化不透明度分布"""
        batch_idx = 0
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        has_content = False
        
        for idx, view in enumerate(['lmain', 'rmain']):
            if view not in data:
                continue
            
            view_data = data[view]
            opacity = None
            
            # 优先使用多层格式
            if 'multilayer_params' in view_data and 'opacity' in view_data['multilayer_params']:
                opacity = view_data['multilayer_params']['opacity'][batch_idx]  # [1, L, H, W]
                opacity = opacity.squeeze(0).mean(dim=0)  # [H, W]
            elif 'opacity_maps' in view_data:
                opacity = view_data['opacity_maps'][batch_idx]
                if opacity.dim() == 3:
                    opacity = opacity[0]
                elif opacity.dim() == 4:
                    opacity = opacity[0, 0]
            
            if opacity is None or opacity.dim() != 2:
                continue
            
            opacity_np = opacity.detach().cpu().numpy()
            
            # 不透明度热力图
            im = axes[idx].imshow(opacity_np, cmap='hot', vmin=0, vmax=1)
            axes[idx].set_title(f'{view} Opacity')
            axes[idx].axis('off')
            plt.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)
            has_content = True
        
        if has_content:
            plt.tight_layout()
            plt.savefig(self.opacity_dir / f'{prefix}_opacity.jpg', dpi=150, bbox_inches='tight')
        plt.close(fig)
    
    def _visualize_features(self, data: Dict, prefix: str):
        """可视化特征图 (PCA 降维)"""
        batch_idx = 0
        
        for view in ['lmain', 'rmain']:
            if view not in data:
                continue
            
            # 检查是否有特征数据
            feat_key = 'dav3_features' if 'dav3_features' in data[view] else 'features'
            if feat_key not in data[view]:
                continue
            
            features = data[view][feat_key]
            if isinstance(features, (list, tuple)):
                features = features[0]  # 取第一层特征
            
            feat = features[batch_idx]  # [C, H, W]
            
            # PCA 降维到 3 维用于 RGB 可视化
            feat_flat = feat.view(feat.shape[0], -1).T  # [H*W, C]
            feat_np = feat_flat.detach().cpu().numpy()
            
            # 简单 PCA
            feat_centered = feat_np - feat_np.mean(axis=0)
            try:
                U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
                feat_pca = U[:, :3] * S[:3]  # 取前 3 个主成分
            except:
                continue
            
            # 归一化到 [0, 1]
            feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)
            
            # 重塑为图像
            H, W = feat.shape[1], feat.shape[2]
            feat_rgb = feat_pca.reshape(H, W, 3)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(feat_rgb)
            plt.title(f'{view} Feature (PCA)')
            plt.axis('off')
            plt.savefig(self.features_dir / f'{prefix}_{view}_features.jpg', dpi=150, bbox_inches='tight')
            plt.close()
    
    def _depth_to_color(self, depth: torch.Tensor) -> np.ndarray:
        """将深度图转换为彩色图像"""
        depth_np = depth.detach().cpu().numpy()
        
        # 归一化
        valid_mask = depth_np > 0
        if valid_mask.sum() > 0:
            min_val = np.percentile(depth_np[valid_mask], 1)
            max_val = np.percentile(depth_np[valid_mask], 99)
            depth_norm = (depth_np - min_val) / (max_val - min_val + 1e-8)
        else:
            depth_norm = depth_np
        
        depth_norm = np.clip(depth_norm, 0, 1)
        
        # 应用颜色映射
        depth_colored = self.cmap(depth_norm)[:, :, :3]
        
        return depth_colored
    
    def _tensor_to_image(self, tensor: torch.Tensor, use_img_range: bool = True) -> np.ndarray:
        """
        将张量转换为 numpy 图像
        
        Args:
            tensor: 输入张量
            use_img_range: 是否使用 img_range 进行归一化
        """
        img = tensor.detach().cpu()
        
        # 处理维度
        if img.dim() == 4:
            img = img[0]
        if img.dim() == 3:
            if img.shape[0] in [1, 3, 4]:  # CHW
                img = img.permute(1, 2, 0)
        
        img_np = img.numpy()
        
        # 归一化到 [0, 1]
        if use_img_range and self.img_range is not None and len(self.img_range) == 2:
            img_min, img_max = self.img_range
            img_np = (img_np - img_min) / (img_max - img_min)
        elif img_np.max() > 1.0:
            img_np = img_np / 255.0
        
        img_np = np.clip(img_np, 0, 1)
        
        # 灰度图转 RGB
        if img_np.ndim == 2:
            img_np = np.stack([img_np] * 3, axis=-1)
        elif img_np.shape[-1] == 1:
            img_np = np.concatenate([img_np] * 3, axis=-1)
        
        return img_np
    
    def save_comparison_grid(self, images: List[Tuple[np.ndarray, str]], 
                            save_path: str, cols: int = 4):
        """保存图像对比网格"""
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes = np.atleast_2d(axes)
        
        for idx, (img, title) in enumerate(images):
            row, col = idx // cols, idx % cols
            axes[row, col].imshow(img)
            axes[row, col].set_title(title)
            axes[row, col].axis('off')
        
        # 隐藏空白子图
        for idx in range(n_images, rows * cols):
            row, col = idx // cols, idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)


def create_visualizer(cfg, save_dir: str) -> TrainingVisualizer:
    """
    创建可视化器工厂函数
    
    Args:
        cfg: 配置对象
        save_dir: 保存目录
        
    Returns:
        TrainingVisualizer 实例
    """
    return TrainingVisualizer(cfg, save_dir)


# 兼容旧接口的函数
def create_visualization_grid(data: Dict, step: int, save_dir: str, prefix: str = "") -> Dict[str, str]:
    """
    兼容旧接口的可视化函数
    
    Args:
        data: 数据字典
        step: 当前步数
        save_dir: 保存目录
        prefix: 文件名前缀
        
    Returns:
        paths: 保存的文件路径字典
    """
    save_dir = Path(save_dir)
    
    # 创建分类子目录
    depth_dir = save_dir / 'depth'
    sparse_dir = save_dir / 'sparse_stereo'
    novel_dir = save_dir / 'novel_view'
    pointcloud_dir = save_dir / 'pointcloud'
    
    for d in [depth_dir, sparse_dir, novel_dir, pointcloud_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    paths = {}
    batch_idx = 0
    
    # 使用默认 colormap
    cmap = plt.get_cmap('turbo')
    
    def depth_to_color(depth):
        """将深度图转换为彩色图，处理无效值"""
        depth_np = depth.detach().cpu().numpy().astype(np.float32)
        
        # 处理无效值
        depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 使用有效掩码
        valid_mask = (depth_np > 0.01) & (depth_np < 100)
        
        if valid_mask.sum() > 10:
            min_val = np.percentile(depth_np[valid_mask], 2)
            max_val = np.percentile(depth_np[valid_mask], 98)
            depth_norm = (depth_np - min_val) / (max_val - min_val + 1e-8)
        else:
            depth_norm = np.zeros_like(depth_np)
        
        depth_norm = np.clip(depth_norm, 0, 1)
        depth_colored = cmap(depth_norm)[:, :, :3]
        
        # 无效区域设为灰色
        depth_colored[~valid_mask] = 0.5
        
        return depth_colored
    
    # 1. 深度对比图
    if 'lmain' in data and 'depth' in data['lmain']:
        has_mono = 'depth_mono' in data['lmain']
        ncols = 2 if has_mono else 1
        
        fig, axes = plt.subplots(1, ncols, figsize=(8 * ncols, 8))
        if ncols == 1:
            axes = [axes]
        
        # 绝对深度
        depth = data['lmain']['depth'][batch_idx]
        if depth.dim() == 3:
            depth = depth[0]
        depth_colored = depth_to_color(depth)
        axes[0].imshow(depth_colored)
        axes[0].set_title('Absolute Depth', fontsize=14)
        axes[0].axis('off')
        
        # Mono 深度
        if has_mono:
            depth_mono = data['lmain']['depth_mono'][batch_idx]
            if depth_mono.dim() == 3:
                depth_mono = depth_mono[0]
            mono_colored = depth_to_color(depth_mono)
            axes[1].imshow(mono_colored)
            axes[1].set_title('DA3 Mono Depth', fontsize=14)
            axes[1].axis('off')
        
        # 添加 scale/shift 信息
        if 'scale' in data and 'shift' in data:
            scale_val = data['scale'][batch_idx].item() if torch.is_tensor(data['scale']) else data['scale']
            shift_val = data['shift'][batch_idx].item() if torch.is_tensor(data['shift']) else data['shift']
            fig.suptitle(f'Scale: {scale_val:.4f}, Shift: {shift_val:.4f}', fontsize=12)
        
        plt.tight_layout()
        depth_path = str(depth_dir / f'{prefix}{step:06d}.jpg')
        plt.savefig(depth_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths['depth_comparison'] = depth_path
    
    # 2. 稀疏立体匹配
    if 'sparse_stereo' in data and 'lmain' in data and 'img' in data['lmain']:
        sparse = data['sparse_stereo']
        img = data['lmain']['img'][batch_idx].permute(1, 2, 0).detach().cpu().numpy()
        
        # 图像归一化
        img_min, img_max = img.min(), img.max()
        if img_max - img_min > 0.01:
            img = (img - img_min) / (img_max - img_min)
        img = np.clip(img, 0, 1)
        
        valid_mask = sparse['valid_mask'][batch_idx].detach().cpu().numpy()
        confidence = sparse['confidence'][batch_idx].detach().cpu().numpy()
        disparity = sparse['disparity'][batch_idx].detach().cpu().numpy()
        
        # 上采样 mask 到图像大小
        H_img, W_img = img.shape[:2]
        H_ds, W_ds = valid_mask.shape
        scale_h, scale_w = H_img / H_ds, W_img / W_ds
        
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        
        # 绘制锚点
        ys, xs = np.where(valid_mask)
        for y, x in zip(ys, xs):
            y_img = int(y * scale_h + scale_h / 2)
            x_img = int(x * scale_w + scale_w / 2)
            c = confidence[y, x]
            d = disparity[y, x]
            color = plt.cm.RdYlGn(c)
            circle = plt.Circle((x_img, y_img), radius=max(3, d/2), 
                               color=color, fill=False, linewidth=2)
            ax.add_patch(circle)
        
        num_anchors = valid_mask.sum()
        ax.set_title(f'Sparse Stereo Anchors: {num_anchors}', fontsize=14)
        ax.axis('off')
        
        sparse_path = str(sparse_dir / f'{prefix}{step:06d}.jpg')
        plt.savefig(sparse_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths['sparse_stereo'] = sparse_path
    
    # 3. Novel View
    if 'novel_view' in data and 'img_pred' in data['novel_view']:
        novel = data['novel_view']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 预测图像 - 渲染输出应该在 [0, 1] 范围
        pred = novel['img_pred'][batch_idx]
        if pred.dim() == 3:
            pred = pred.permute(1, 2, 0)
        pred_np = pred.detach().cpu().numpy()
        
        # 检查值范围并归一化
        pred_min, pred_max = pred_np.min(), pred_np.max()
        if pred_max > 1.5 or pred_min < -0.5:
            # 可能在其他范围，尝试归一化
            pred_np = (pred_np - pred_min) / (pred_max - pred_min + 1e-8)
        pred_np = np.clip(pred_np, 0, 1)
        
        axes[0].imshow(pred_np)
        axes[0].set_title('Predicted', fontsize=14)
        axes[0].axis('off')
        
        # GT 图像
        if 'img' in novel:
            gt = novel['img']
            if torch.is_tensor(gt):
                gt = gt[batch_idx]
                if gt.dim() == 3:
                    gt = gt.permute(1, 2, 0)
                gt_np = gt.detach().cpu().numpy()
            else:
                gt_np = gt
            
            # 检查值范围并归一化
            gt_min, gt_max = gt_np.min(), gt_np.max()
            if gt_max > 1.5 or gt_min < -0.5:
                gt_np = (gt_np - gt_min) / (gt_max - gt_min + 1e-8)
            gt_np = np.clip(gt_np, 0, 1)
            
            axes[1].imshow(gt_np)
            axes[1].set_title('Ground Truth', fontsize=14)
            axes[1].axis('off')
            
            # 差异图
            diff = np.abs(pred_np - gt_np).mean(axis=2)
            diff = diff / (diff.max() + 1e-8)
            axes[2].imshow(cmap(diff)[:, :, :3])
            axes[2].set_title('Difference', fontsize=14)
            axes[2].axis('off')
        
        plt.tight_layout()
        novel_path = str(novel_dir / f'{prefix}{step:06d}.jpg')
        plt.savefig(novel_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths['novel_view'] = novel_path
    
    # 4. 点云投影
    if 'lmain' in data and 'xyz' in data['lmain'] and 'pts_valid' in data['lmain']:
        xyz = data['lmain']['xyz'][batch_idx].detach().cpu().numpy()
        valid = data['lmain']['pts_valid'][batch_idx].detach().cpu().numpy().astype(bool)
        
        if 'depth' in data['lmain']:
            depth = data['lmain']['depth'][batch_idx]
            H, W = depth.shape[-2:]
        else:
            H, W = 1024, 1024
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # 过滤有效点
        xyz_valid = xyz[valid]
        
        # 过滤异常值
        if len(xyz_valid) > 0:
            # 移除 NaN 和 Inf
            finite_mask = np.all(np.isfinite(xyz_valid), axis=1)
            xyz_valid = xyz_valid[finite_mask]
        
        if len(xyz_valid) > 100:
            # 使用 z 坐标作为深度着色
            z_vals = xyz_valid[:, 2]
            z_min, z_max = np.percentile(z_vals, [2, 98])
            z_norm = (z_vals - z_min) / (z_max - z_min + 1e-8)
            z_norm = np.clip(z_norm, 0, 1)
            
            colors = cmap(z_norm)[:, :3]
            
            # 随机采样显示
            n_show = min(100000, len(xyz_valid))
            indices = np.random.choice(len(xyz_valid), n_show, replace=False)
            
            ax.scatter(xyz_valid[indices, 0], xyz_valid[indices, 1], 
                      c=colors[indices], s=0.3, alpha=0.6)
        
        ax.set_title(f'Point Cloud (valid: {valid.sum():,})', fontsize=14)
        ax.set_aspect('equal')
        ax.axis('off')
        
        pc_path = str(pointcloud_dir / f'{prefix}{step:06d}.jpg')
        plt.savefig(pc_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        paths['pointcloud'] = pc_path
    
    # 不再生成 Summary
    
    return paths
