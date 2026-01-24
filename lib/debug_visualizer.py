"""
调试可视化工具 - 用于在调试控制台中显示训练中间变量的图片数据

使用方法:
    1. 在调试断点处导入: from lib.debug_visualizer import DebugVisualizer, dbg_vis
    2. 使用全局实例: dbg_vis.show_tensor(tensor, "name")
    3. 或创建新实例: vis = DebugVisualizer(save_dir="debug_output")

支持的数据类型:
    - torch.Tensor (自动处理不同维度和数据范围)
    - numpy.ndarray
    - PIL.Image
    - 训练中的data字典
"""

import os
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Union, Optional, Dict, List, Any
import cv2

try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class DebugVisualizer:
    """调试可视化器 - 用于保存和显示训练中间变量"""
    
    def __init__(self, save_dir: str = "debug_images", auto_open: bool = False):
        """
        初始化调试可视化器
        
        Args:
            save_dir: 保存图片的目录
            auto_open: 是否自动打开图片（仅在有显示器时有效）
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.auto_open = auto_open
        self.counter = 0
        self.session_id = datetime.now().strftime("%H%M%S")
        
        print(f"[DebugVisualizer] 初始化完成，图片保存目录: {self.save_dir.absolute()}")
    
    def _to_numpy(self, data: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """将tensor转换为numpy数组"""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        return np.array(data)
    
    def _normalize_image(self, img: np.ndarray, normalize: bool = True) -> np.ndarray:
        """
        将图像归一化到 [0, 255] 范围
        
        Args:
            img: 输入图像
            normalize: 是否自动归一化
        """
        if img.dtype == np.uint8:
            return img
            
        if normalize:
            # 自动归一化到 [0, 1]
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = (img - img_min) / (img_max - img_min)
            else:
                img = np.zeros_like(img)
        else:
            # 假设已经在 [0, 1] 范围
            img = np.clip(img, 0, 1)
        
        return (img * 255).astype(np.uint8)
    
    def _process_tensor(self, tensor: Union[torch.Tensor, np.ndarray], 
                        normalize: bool = True) -> np.ndarray:
        """
        处理不同维度的tensor为可显示的图像
        
        支持的维度:
            - [H, W]: 灰度图
            - [H, W, C]: HWC格式图像
            - [C, H, W]: CHW格式图像 (PyTorch格式)
            - [B, C, H, W]: 批次图像 (取第一个)
            - [B, H, W]: 批次灰度图 (取第一个)
        """
        img = self._to_numpy(tensor)
        
        # 处理不同维度
        if img.ndim == 4:
            # [B, C, H, W] -> 取第一个样本
            img = img[0]
        
        if img.ndim == 3:
            if img.shape[0] in [1, 3, 4]:
                # [C, H, W] -> [H, W, C]
                img = np.transpose(img, (1, 2, 0))
            # 否则假设是 [H, W, C]
        
        if img.ndim == 2:
            # [H, W] 灰度图 -> [H, W, 1]
            img = img[:, :, np.newaxis]
        
        # 处理单通道
        if img.shape[-1] == 1:
            img = np.repeat(img, 3, axis=-1)
        
        # 处理4通道 (RGBA -> RGB)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        
        return self._normalize_image(img, normalize)
    
    def _generate_filename(self, name: str, ext: str = "png") -> Path:
        """生成唯一的文件名"""
        self.counter += 1
        filename = f"{self.session_id}_{self.counter:04d}_{name}.{ext}"
        return self.save_dir / filename
    
    def show_tensor(self, tensor: Union[torch.Tensor, np.ndarray], 
                    name: str = "tensor",
                    normalize: bool = True,
                    colormap: Optional[str] = None) -> str:
        """
        显示单个tensor为图像
        
        Args:
            tensor: 要显示的tensor
            name: 图像名称
            normalize: 是否自动归一化
            colormap: 使用的colormap (如 'jet', 'viridis' 等，用于深度图)
            
        Returns:
            保存的文件路径
        """
        img = self._process_tensor(tensor, normalize)
        
        if colormap and HAS_MATPLOTLIB:
            # 使用colormap (适用于深度图等)
            if img.ndim == 3:
                img = img.mean(axis=-1)  # 转为灰度
            plt.figure(figsize=(10, 8))
            plt.imshow(img, cmap=colormap)
            plt.colorbar()
            plt.title(name)
            filepath = self._generate_filename(name)
            plt.savefig(filepath, bbox_inches='tight', dpi=100)
            plt.close()
        else:
            filepath = self._generate_filename(name)
            cv2.imwrite(str(filepath), img[:, :, ::-1])  # RGB -> BGR
        
        print(f"[DebugVisualizer] 保存: {filepath}")
        return str(filepath)
    
    def show_image_grid(self, images: List[Union[torch.Tensor, np.ndarray]], 
                        names: List[str] = None,
                        grid_name: str = "grid",
                        cols: int = 4,
                        normalize: bool = True) -> str:
        """
        显示多个图像为网格
        
        Args:
            images: 图像列表
            names: 图像名称列表
            grid_name: 网格图像名称
            cols: 列数
            normalize: 是否自动归一化
            
        Returns:
            保存的文件路径
        """
        if not HAS_MATPLOTLIB:
            # 回退到保存单独图像
            paths = []
            for i, img in enumerate(images):
                name = names[i] if names and i < len(names) else f"{grid_name}_{i}"
                paths.append(self.show_tensor(img, name, normalize))
            return paths[0] if paths else ""
        
        n_images = len(images)
        rows = (n_images + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if rows == 1:
            axes = [axes] if cols == 1 else list(axes)
        else:
            axes = [ax for row in axes for ax in row]
        
        for i, img in enumerate(images):
            processed = self._process_tensor(img, normalize)
            axes[i].imshow(processed)
            if names and i < len(names):
                axes[i].set_title(names[i])
            axes[i].axis('off')
        
        # 隐藏多余的子图
        for i in range(n_images, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        filepath = self._generate_filename(grid_name)
        plt.savefig(filepath, bbox_inches='tight', dpi=100)
        plt.close()
        
        print(f"[DebugVisualizer] 保存网格图: {filepath}")
        return str(filepath)
    
    def show_training_data(self, data: Dict[str, Any], 
                           step: Optional[int] = None) -> Dict[str, str]:
        """
        显示训练数据中的所有图像
        
        Args:
            data: 训练数据字典 (包含 'lmain', 'rmain', 'novel_view' 等)
            step: 训练步数 (用于命名)
            
        Returns:
            保存的文件路径字典
        """
        saved_paths = {}
        step_str = f"step{step}" if step is not None else ""
        
        # 收集所有要显示的图像
        images = []
        names = []
        
        # 处理左视图
        if 'lmain' in data:
            if 'img' in data['lmain']:
                images.append(data['lmain']['img'])
                names.append(f"L_input_{step_str}")
            if 'depth' in data['lmain']:
                # 深度图单独保存，使用colormap
                path = self.show_tensor(data['lmain']['depth'], 
                                       f"L_depth_{step_str}", 
                                       colormap='jet')
                saved_paths['L_depth'] = path
        
        # 处理右视图
        if 'rmain' in data:
            if 'img' in data['rmain']:
                images.append(data['rmain']['img'])
                names.append(f"R_input_{step_str}")
            if 'depth' in data['rmain']:
                path = self.show_tensor(data['rmain']['depth'], 
                                       f"R_depth_{step_str}", 
                                       colormap='jet')
                saved_paths['R_depth'] = path
        
        # 处理新视角
        if 'novel_view' in data:
            if 'img_pred' in data['novel_view']:
                images.append(data['novel_view']['img_pred'])
                names.append(f"novel_pred_{step_str}")
            if 'img' in data['novel_view']:
                images.append(data['novel_view']['img'])
                names.append(f"novel_gt_{step_str}")
        
        # 保存图像网格
        if images:
            grid_path = self.show_image_grid(images, names, 
                                            f"training_data_{step_str}")
            saved_paths['grid'] = grid_path
        
        return saved_paths
    
    def show_comparison(self, pred: Union[torch.Tensor, np.ndarray],
                        gt: Union[torch.Tensor, np.ndarray],
                        name: str = "comparison",
                        show_diff: bool = True) -> str:
        """
        显示预测和GT的对比
        
        Args:
            pred: 预测图像
            gt: GT图像
            name: 名称
            show_diff: 是否显示差异图
            
        Returns:
            保存的文件路径
        """
        pred_np = self._process_tensor(pred, normalize=False)
        gt_np = self._process_tensor(gt, normalize=False)
        
        images = [pred_np, gt_np]
        names = ["Prediction", "Ground Truth"]
        
        if show_diff:
            diff = np.abs(pred_np.astype(float) - gt_np.astype(float))
            diff = (diff / diff.max() * 255).astype(np.uint8) if diff.max() > 0 else diff.astype(np.uint8)
            images.append(diff)
            names.append("Difference")
        
        return self.show_image_grid(images, names, name, cols=3, normalize=False)
    
    def show_depth(self, depth: Union[torch.Tensor, np.ndarray],
                   name: str = "depth",
                   colormap: str = 'jet') -> str:
        """
        显示深度图
        
        Args:
            depth: 深度图
            name: 名称
            colormap: 使用的colormap
            
        Returns:
            保存的文件路径
        """
        return self.show_tensor(depth, name, normalize=True, colormap=colormap)
    
    def show_gaussians_info(self, data: Dict[str, Any], 
                            view: str = 'lmain',
                            step: Optional[int] = None) -> Dict[str, str]:
        """
        显示高斯点相关信息
        
        Args:
            data: 训练数据字典
            view: 视图名称 ('lmain' 或 'rmain')
            step: 训练步数
            
        Returns:
            保存的文件路径字典
        """
        saved_paths = {}
        step_str = f"step{step}" if step is not None else ""
        
        if view not in data:
            print(f"[DebugVisualizer] 警告: 视图 '{view}' 不存在于数据中")
            return saved_paths
        
        view_data = data[view]
        
        # 显示xyz位置的统计信息
        if 'xyz' in view_data:
            xyz = self._to_numpy(view_data['xyz'])
            print(f"[DebugVisualizer] {view} xyz统计:")
            print(f"  形状: {xyz.shape}")
            print(f"  X范围: [{xyz[..., 0].min():.3f}, {xyz[..., 0].max():.3f}]")
            print(f"  Y范围: [{xyz[..., 1].min():.3f}, {xyz[..., 1].max():.3f}]")
            print(f"  Z范围: [{xyz[..., 2].min():.3f}, {xyz[..., 2].max():.3f}]")
        
        # 显示颜色/特征
        if 'rgb' in view_data:
            path = self.show_tensor(view_data['rgb'], 
                                   f"{view}_rgb_{step_str}")
            saved_paths['rgb'] = path
        
        # 显示不透明度
        if 'opacity' in view_data:
            path = self.show_tensor(view_data['opacity'], 
                                   f"{view}_opacity_{step_str}",
                                   colormap='viridis')
            saved_paths['opacity'] = path
        
        # 显示有效点掩码
        if 'pts_valid' in view_data:
            valid = self._to_numpy(view_data['pts_valid'])
            print(f"[DebugVisualizer] {view} 有效点比例: {valid.sum() / valid.size:.2%}")
        
        # MoE路由器权重
        if 'router_weights' in view_data:
            weights = self._to_numpy(view_data['router_weights'])
            print(f"[DebugVisualizer] {view} MoE路由器权重:")
            print(f"  形状: {weights.shape}")
            print(f"  范围: [{weights.min():.3f}, {weights.max():.3f}]")
        
        return saved_paths
    
    def clear(self):
        """清空保存目录中的所有文件"""
        import shutil
        if self.save_dir.exists():
            shutil.rmtree(self.save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        self.counter = 0
        print(f"[DebugVisualizer] 已清空目录: {self.save_dir}")


# 全局实例，方便在调试时快速使用
dbg_vis = DebugVisualizer(save_dir="debug_images")


# ============ 便捷函数 ============

def show(tensor: Union[torch.Tensor, np.ndarray], 
         name: str = "debug",
         normalize: bool = True) -> str:
    """快速显示tensor的便捷函数"""
    return dbg_vis.show_tensor(tensor, name, normalize)


def show_depth(depth: Union[torch.Tensor, np.ndarray], 
               name: str = "depth") -> str:
    """快速显示深度图的便捷函数"""
    return dbg_vis.show_depth(depth, name)


def show_data(data: Dict[str, Any], step: int = None) -> Dict[str, str]:
    """快速显示训练数据的便捷函数"""
    return dbg_vis.show_training_data(data, step)


def compare(pred: Union[torch.Tensor, np.ndarray],
            gt: Union[torch.Tensor, np.ndarray],
            name: str = "compare") -> str:
    """快速显示预测和GT对比的便捷函数"""
    return dbg_vis.show_comparison(pred, gt, name)


# ============ 调试时直接在控制台使用的示例 ============
"""
使用示例 (在调试断点处的控制台中输入):

# 1. 显示单个tensor
from lib.debug_visualizer import show, show_depth, show_data, compare

# 显示渲染结果
show(data['novel_view']['img_pred'], 'render')

# 显示深度图
show_depth(data['lmain']['depth'], 'left_depth')

# 显示训练数据
show_data(data, step=1000)

# 对比预测和GT
compare(render_novel, gt_novel, 'novel_comparison')

# 2. 使用完整的DebugVisualizer实例
from lib.debug_visualizer import DebugVisualizer
vis = DebugVisualizer(save_dir='my_debug')
vis.show_training_data(data, step=self.total_steps)
vis.show_gaussians_info(data, view='lmain')
"""
