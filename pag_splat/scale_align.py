"""
Module 1 (续) — ScaleAlignmentMLP (相机感知的尺度对齐网络)

将 DA3 输出的无尺度相对深度 D_rel 对齐到物理米制深度：
    D_metric = exp(log_s) * D_rel + t

其中 log_s 和 t 由 MLP 从相机参数中回归，per-image 独立预测。

输入特征编码：
  - 相机内参 K: (fx, fy, cx, cy) → 归一化后拼接
  - 相对位姿 [R12|t12]: 视图1 → 视图2 的变换
      R12 以 6D 旋转表示 (连续性优于四元数)
      t12 直接使用 3D 向量
  - 合计输入维度: 4 + 6 + 3 = 13 → hidden_dim → 2 (log_scale, shift)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def rotation_to_6d(R: torch.Tensor) -> torch.Tensor:
    """
    将 3×3 旋转矩阵转为 6D 连续表示 (Zhou et al. 2019)。
    取矩阵前两列展平: (B, 3, 3) → (B, 6)
    """
    return R[..., :2].reshape(*R.shape[:-2], 6)  # (..., 6)


def encode_intrinsics(
    intr: torch.Tensor, img_h: int, img_w: int
) -> torch.Tensor:
    """
    将内参矩阵归一化为图像尺寸无关的 4D 向量。

    Args:
        intr: (B, 3, 3)
        img_h, img_w: 图像高宽 (用于归一化)

    Returns:
        (B, 4): [fx/W, fy/H, cx/W, cy/H]
    """
    fx = intr[:, 0, 0] / img_w
    fy = intr[:, 1, 1] / img_h
    cx = intr[:, 0, 2] / img_w
    cy = intr[:, 1, 2] / img_h
    return torch.stack([fx, fy, cx, cy], dim=-1)  # (B, 4)


class ScaleAlignmentMLP(nn.Module):
    """
    从相机参数中预测尺度-偏移对，将 DA3 相对深度转为度量深度。

    D_metric = exp(log_s) * D_rel + t_shift

    设计选择：
      - 使用 exp(log_s) 保证尺度恒正
      - t_shift 无约束，允许相对深度存在系统偏移
      - 对视图1 和视图2 分别预测（共享权重，但输入不同）
      - 输入: 视图1 内参 + 视图2 内参 + 相对位姿 = 4 + 4 + 9 = 17 维

    Args:
        hidden_dim:   MLP 隐藏层维度
        num_layers:   MLP 层数
        use_view2_intr: 是否把视图2 内参也作为输入
    """

    INPUT_DIM: int = 4 + 6 + 3  # K1(4) + R12_6d(6) + t12(3) = 13
    # 也可选择加入 K2: 4 + 4 + 6 + 3 = 17

    def __init__(
        self,
        hidden_dim: int = 256,
        num_layers: int = 4,
        use_view2_intr: bool = True,
    ) -> None:
        super().__init__()
        self.use_view2_intr = use_view2_intr
        in_dim = (self.INPUT_DIM + 4) if use_view2_intr else self.INPUT_DIM

        layers = []
        prev = in_dim
        for i in range(num_layers - 1):
            layers += [nn.Linear(prev, hidden_dim), nn.SiLU()]
            prev = hidden_dim
        # 最后一层输出 log_scale + shift
        layers.append(nn.Linear(prev, 2))
        self.mlp = nn.Sequential(*layers)

        # 初始化最后一层接近 0，使得初始预测接近 scale=1, shift=0
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def _build_input(
        self,
        intr1: torch.Tensor,
        intr2: torch.Tensor,
        extr1: torch.Tensor,
        extr2: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """
        构建 MLP 的输入特征向量。

        Args:
            intr1: (B, 3, 3) 视图1 内参
            intr2: (B, 3, 3) 视图2 内参
            extr1: (B, 3, 4) 视图1 外参 [R1|t1]  (world→camera)
            extr2: (B, 3, 4) 视图2 外参 [R2|t2]
            img_h, img_w: 图像尺寸

        Returns:
            feat: (B, input_dim)
        """
        B = intr1.shape[0]
        device = intr1.device

        # --- 内参归一化 ---
        k1_enc = encode_intrinsics(intr1, img_h, img_w)  # (B, 4)
        k2_enc = encode_intrinsics(intr2, img_h, img_w)  # (B, 4)

        # --- 相对旋转 R12 = R2 @ R1^T ---
        R1 = extr1[:, :3, :3]  # (B, 3, 3)
        t1 = extr1[:, :3, 3]   # (B, 3)
        R2 = extr2[:, :3, :3]
        t2 = extr2[:, :3, 3]

        R12 = R2 @ R1.transpose(-1, -2)   # (B, 3, 3)
        r12_6d = rotation_to_6d(R12)       # (B, 6)

        # --- 相对平移 t12 = t2 - R2 @ R1^T @ t1 = t2 - R12 @ t1 ---
        # 即视图1 原点在视图2 坐标系中的表示
        t12 = t2 - (R12 @ t1.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        # 归一化平移：用 t12 的 L2 范数除，防止量纲影响
        t12_norm = t12 / (t12.norm(dim=-1, keepdim=True).clamp(min=1e-6))

        parts = [k1_enc, r12_6d, t12_norm]
        if self.use_view2_intr:
            parts.insert(1, k2_enc)

        return torch.cat(parts, dim=-1)  # (B, input_dim)

    def forward(
        self,
        intr1: torch.Tensor,
        intr2: torch.Tensor,
        extr1: torch.Tensor,
        extr2: torch.Tensor,
        d_rel: torch.Tensor,
        img_h: int,
        img_w: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        预测尺度/偏移，并应用到相对深度上。

        Args:
            intr1:  (B, 3, 3)
            intr2:  (B, 3, 3)
            extr1:  (B, 3, 4)
            extr2:  (B, 3, 4)
            d_rel:  (B, 1, H, W)  DA3 输出的相对深度
            img_h, img_w: 图像尺寸 (与 d_rel 一致)

        Returns:
            d_metric:  (B, 1, H, W)  度量深度 (米)
            log_scale: (B,)          预测的对数尺度
            t_shift:   (B,)          预测的深度偏移
        """
        feat = self._build_input(intr1, intr2, extr1, extr2, img_h, img_w)
        out = self.mlp(feat)  # (B, 2)

        log_scale = out[:, 0]   # (B,)
        t_shift = out[:, 1]     # (B,)

        scale = torch.exp(log_scale).view(-1, 1, 1, 1)   # (B, 1, 1, 1)
        shift = t_shift.view(-1, 1, 1, 1)

        d_metric = scale * d_rel + shift

        # 保证深度为正 (避免负深度导致反投影错误)
        d_metric = F.softplus(d_metric) + 1e-3

        return d_metric, log_scale, t_shift
