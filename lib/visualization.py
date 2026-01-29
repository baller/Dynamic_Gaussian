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
        has_wide = 'depth_wide' in data
        ncols = 5 if has_wide else 4
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
            axes[0].set_title('Left Depth')
            axes[0].axis('off')
        
        # 右视图深度
        if 'rmain' in data and 'depth' in data['rmain']:
            depth_r = data['rmain']['depth'][batch_idx]
            if depth_r.dim() == 3:
                depth_r = depth_r[0]
            depth_r_np = self._depth_to_color(depth_r)
            axes[1].imshow(depth_r_np)
            axes[1].set_title('Right Depth')
            axes[1].axis('off')

        col_offset = 2
        if has_wide:
            depth_wide = data['depth_wide'][batch_idx]
            if depth_wide.dim() == 3:
                depth_wide = depth_wide[0]
            depth_wide_np = self._depth_to_color(depth_wide)
            axes[2].imshow(depth_wide_np)
            axes[2].set_title('Wide FOV Depth')
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
        
        plt.tight_layout()
        plt.savefig(self.depth_dir / f'{prefix}_depth.jpg', dpi=150, bbox_inches='tight', pil_kwargs={'quality': 95})
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
                    if opacity.dim() == 1:  # [N] 展平格式
                        # 尝试重塑为图像
                        if 'depth' in view_data:
                            depth = view_data['depth'][batch_idx]
                            if depth.dim() >= 2:
                                H, W = depth.shape[-2:]
                                N = opacity.numel()
                                L = N // (H * W) if H * W > 0 else 1
                                if N == L * H * W:
                                    opacity = opacity.view(L, H, W).mean(dim=0)
                    elif opacity.dim() == 2:  # [N, 1]
                        if 'depth' in view_data:
                            depth = view_data['depth'][batch_idx]
                            if depth.dim() >= 2:
                                H, W = depth.shape[-2:]
                                opacity = opacity.squeeze(-1)
                                N = opacity.numel()
                                L = N // (H * W) if H * W > 0 else 1
                                if N == L * H * W:
                                    opacity = opacity.view(L, H, W).mean(dim=0)
                    elif opacity.dim() == 3:
                        opacity = opacity[0]
                    
                    if opacity.dim() == 2:
                        opacity_np = opacity.detach().cpu().numpy()
                        im = axes[0, 0].imshow(opacity_np, cmap='viridis', vmin=0, vmax=1)
                        axes[0, 0].set_title(f'{view} Opacity')
                        axes[0, 0].axis('off')
                        plt.colorbar(im, ax=axes[0, 0], fraction=0.046)
                        has_content = True
            
            # 第二行: 有效点分布、纹理复杂度、点云统计
            valid_mask = view_data.get('valid_mask')
            if valid_mask is not None:
                # [B, L, H, W] 格式
                vm = valid_mask[batch_idx]  # [L, H, W]
                if vm.dim() == 3:
                    # 可视化每像素的有效层数
                    layers_per_pixel = vm.float().sum(dim=0)  # [H, W]
                    L = vm.shape[0]
                    patch_size = int(self.layer_patch_size)
                    patch_size = max(1, min(patch_size, layers_per_pixel.shape[-2], layers_per_pixel.shape[-1]))
                    if patch_size > 1:
                        pooled = F.avg_pool2d(
                            layers_per_pixel.unsqueeze(0).unsqueeze(0),
                            kernel_size=patch_size,
                            stride=patch_size
                        )
                        patch_map = F.interpolate(
                            pooled, size=layers_per_pixel.shape[-2:],
                            mode='nearest'
                        ).squeeze(0).squeeze(0)
                    else:
                        patch_map = layers_per_pixel
                    im = axes[1, 0].imshow(patch_map.detach().cpu().numpy(), cmap='Greens', vmin=0, vmax=L)
                    axes[1, 0].set_title(f'{view} Layer Count Patches (patch={patch_size})')
                    axes[1, 0].axis('off')
                    plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
                    has_content = True
            elif 'pts_valid' in view_data:
                pts_valid = view_data['pts_valid'][batch_idx]
                if 'depth' in view_data:
                    depth = view_data['depth'][batch_idx]
                    if depth.dim() >= 2:
                        H, W = depth.shape[-2:]
                        N = pts_valid.numel()
                        L = N // (H * W) if H * W > 0 else 1
                        if N == L * H * W:
                            pts_valid_layers = pts_valid.float().view(L, H, W)
                            layers_per_pixel = pts_valid_layers.sum(dim=0)
                            patch_size = int(self.layer_patch_size)
                            patch_size = max(1, min(patch_size, H, W))
                            if patch_size > 1:
                                pooled = F.avg_pool2d(
                                    layers_per_pixel.unsqueeze(0).unsqueeze(0),
                                    kernel_size=patch_size,
                                    stride=patch_size
                                )
                                patch_map = F.interpolate(
                                    pooled, size=(H, W),
                                    mode='nearest'
                                ).squeeze(0).squeeze(0)
                            else:
                                patch_map = layers_per_pixel
                            im = axes[1, 0].imshow(patch_map.detach().cpu().numpy(), cmap='Greens', vmin=0, vmax=L)
                            axes[1, 0].set_title(f'{view} Layer Count Patches (patch={patch_size})')
                            axes[1, 0].axis('off')
                            plt.colorbar(im, ax=axes[1, 0], fraction=0.046)
                            has_content = True
            
            # 纹理复杂度
            texture = view_data.get('texture_complexity')
            if texture is not None:
                tex = texture[batch_idx]
                if tex.dim() == 3:
                    tex = tex[0]
                axes[1, 1].imshow(tex.detach().cpu().numpy(), cmap='hot', vmin=0, vmax=1)
                axes[1, 1].set_title(f'{view} Texture Complexity')
                axes[1, 1].axis('off')
                has_content = True
            elif 'depth' in view_data:
                depth = view_data['depth'][batch_idx]
                if depth.dim() == 3:
                    depth = depth[0]
                depth_colored = self._depth_to_color(depth)
                axes[1, 1].imshow(depth_colored)
                axes[1, 1].set_title(f'{view} Depth (colored)')
                axes[1, 1].axis('off')
                has_content = True
            
            # 点云数量统计
            if 'pts_valid' in view_data:
                pts_valid = view_data['pts_valid'][batch_idx]
                valid_count = pts_valid.sum().item()
                total_count = pts_valid.numel()
                ratio = valid_count / total_count * 100
                axes[1, 2].text(0.5, 0.5, 
                               f"Valid Points:\n{valid_count:,.0f} / {total_count:,}\n({ratio:.1f}%)",
                               ha='center', va='center', fontsize=14,
                               transform=axes[1, 2].transAxes)
                axes[1, 2].set_title(f'{view} Point Statistics')
                axes[1, 2].axis('off')
                has_content = True
            
            if has_content:
                plt.tight_layout()
                plt.savefig(self.gaussian_dir / f'{prefix}_{view}_gaussian.jpg', dpi=150, bbox_inches='tight', pil_kwargs={'quality': 95})
            plt.close(fig)
    
    def _visualize_novel_view(self, data: Dict, prefix: str):
        """可视化 Novel View 预测 vs GT"""
        if 'novel_view' not in data:
            return
        
        novel_data = data['novel_view']
        batch_idx = 0
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 预测图像 - 渲染输出已经在 [0, 1] 范围，不需要 img_range 转换
        if 'img_pred' in novel_data:
            img_pred = novel_data['img_pred'][batch_idx]
            img_pred_np = self._tensor_to_image(img_pred, use_img_range=False)
            axes[0].imshow(img_pred_np)
            axes[0].set_title('Predicted Novel View')
            axes[0].axis('off')
        
        # GT 图像 - 来自数据集，使用 img_range 转换
        if 'img' in novel_data:
            img_gt = novel_data['img'][batch_idx]
            img_gt_np = self._tensor_to_image(img_gt, use_img_range=True)
            axes[1].imshow(img_gt_np)
            axes[1].set_title('Ground Truth')
            axes[1].axis('off')
        
        # 差异图 - 需要将两者都转换到 [0, 1] 范围再比较
        if 'img_pred' in novel_data and 'img' in novel_data:
            img_pred = novel_data['img_pred'][batch_idx]  # 已经在 [0, 1]
            img_gt = novel_data['img'][batch_idx]  # 在 img_range (如 [-1, 1])
            
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
        plt.savefig(self.novel_view_dir / f'{prefix}_novel_view.jpg', dpi=150, bbox_inches='tight', pil_kwargs={'quality': 95})
        plt.close(fig)
    
    def _visualize_opacity(self, data: Dict, prefix: str):
        """可视化不透明度分布"""
        batch_idx = 0
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        has_content = False
        
        for idx, view in enumerate(['lmain', 'rmain']):
            if view not in data:
                continue
            
            # 注意: 键名是 'opacity_maps'
            if 'opacity_maps' not in data[view]:
                continue
            
            # 优先使用多层格式
            if 'multilayer_params' in data[view] and 'opacity' in data[view]['multilayer_params']:
                opacity = data[view]['multilayer_params']['opacity'][batch_idx]  # [1, L, H, W]
                opacity = opacity.squeeze(0).mean(dim=0)  # [H, W]
            else:
                opacity = data[view]['opacity_maps'][batch_idx]
                if opacity.dim() == 3:
                    opacity = opacity[0]
                elif opacity.dim() == 2:
                    opacity = opacity.squeeze(-1)
                elif opacity.dim() == 1 and 'depth' in data[view]:
                    depth = data[view]['depth'][batch_idx]
                    if depth.dim() >= 2:
                        H, W = depth.shape[-2:]
                        N = opacity.numel()
                        if H * W > 0 and N % (H * W) == 0:
                            L = N // (H * W)
                            opacity = opacity.view(L, H, W).mean(dim=0)
            
            if opacity.dim() != 2:
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
            plt.savefig(self.opacity_dir / f'{prefix}_opacity.jpg', dpi=150, bbox_inches='tight', pil_kwargs={'quality': 95})
        plt.close(fig)
    
    def _visualize_features(self, data: Dict, prefix: str):
        """可视化特征图 (PCA 降维)"""
        batch_idx = 0
        
        for view in ['lmain', 'rmain']:
            if view not in data:
                continue
            
            # 检查是否有特征数据
            if 'features' not in data[view]:
                continue
            
            features = data[view]['features']
            if isinstance(features, (list, tuple)):
                features = features[0]  # 取第一层特征
            
            feat = features[batch_idx]  # [C, H, W]
            
            # PCA 降维到 3 维用于 RGB 可视化
            feat_flat = feat.view(feat.shape[0], -1).T  # [H*W, C]
            feat_np = feat_flat.detach().cpu().numpy()
            
            # 简单 PCA
            feat_centered = feat_np - feat_np.mean(axis=0)
            U, S, Vt = np.linalg.svd(feat_centered, full_matrices=False)
            feat_pca = U[:, :3] * S[:3]  # 取前 3 个主成分
            
            # 归一化到 [0, 1]
            feat_pca = (feat_pca - feat_pca.min()) / (feat_pca.max() - feat_pca.min() + 1e-8)
            
            # 重塑为图像
            H, W = feat.shape[1], feat.shape[2]
            feat_rgb = feat_pca.reshape(H, W, 3)
            
            plt.figure(figsize=(8, 8))
            plt.imshow(feat_rgb)
            plt.title(f'{view} Feature (PCA)')
            plt.axis('off')
            plt.savefig(self.features_dir / f'{prefix}_{view}_features.jpg', dpi=150, bbox_inches='tight', pil_kwargs={'quality': 95})
            plt.close()
    
    def _depth_to_color(self, depth: torch.Tensor) -> np.ndarray:
        """将深度图转换为彩色图像"""
        depth_np = depth.detach().cpu().numpy()
        
        # 归一化
        valid_mask = depth_np > 0
        if valid_mask.sum() > 0:
            min_val = depth_np[valid_mask].min()
            max_val = depth_np[valid_mask].max()
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
                - True: 用于原始图像数据 (来自数据集，如 [-1, 1])
                - False: 用于渲染输出 (已经在 [0, 1] 范围)
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight', pil_kwargs={'quality': 95})
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
