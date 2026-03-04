"""
PAG-Splat: Prior-Aware 2D Gaussian Splatting
一种无代价体积的前馈式 3D 高斯泼溅框架

架构三模块：
  Module 1 - MonoPriorExtractor: 冻结 DA3 提取单目特征 + 相对深度，ScaleAlignmentMLP 赋予物理尺度
  Module 2 - SingleSurfaceWarping: 基于度量深度的单表面特征扭曲
  Module 3 - GaussianDecoder: 不确定性感知 U-Net 解码 2D 高斯参数图
"""

from .model import PAGSplat
from .prior_extractor import MonoPriorExtractor
from .scale_align import ScaleAlignmentMLP
from .warping import SingleSurfaceWarping
from .gaussian_decoder import GaussianDecoder

__all__ = [
    "PAGSplat",
    "MonoPriorExtractor",
    "ScaleAlignmentMLP",
    "SingleSurfaceWarping",
    "GaussianDecoder",
]
