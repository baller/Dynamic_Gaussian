"""
PAG-Splat 渲染兼容层。

当前简化模型已回退到 GPS+ 风格字段，渲染直接复用 lib.GaussianRender.pts2render。
"""

from __future__ import annotations

import torch

from lib.GaussianRender import pts2render


def move_data_to_cuda(data: dict) -> dict:
    """
    将数据字典中的所有 Tensor 移至 CUDA。

    覆盖 lmain / rmain / novel_view 中的所有 Tensor。
    非 Tensor 类型保持原样。
    """
    def _to_cuda(value):
        if isinstance(value, torch.Tensor):
            return value.cuda()
        if isinstance(value, dict):
            return {k: _to_cuda(v) for k, v in value.items()}
        return value

    for key in ["lmain", "rmain", "novel_view"]:
        if key in data:
            data[key] = {k: _to_cuda(v) for k, v in data[key].items()}
    return data


__all__ = ["pts2render", "move_data_to_cuda"]
