"""
PAG-Splat 简化版模型入口。

当前训练链路保留 DA3 单目先验与尺度对齐，
并回退为 GPS+ 风格的 GS 参数回归与 RGB 渲染。
"""

from .model import PAGSplat
from .prior_extractor import MonoPriorExtractor
from .scale_align import ScaleAlignmentMLP

__all__ = [
    "PAGSplat",
    "MonoPriorExtractor",
    "ScaleAlignmentMLP",
]
