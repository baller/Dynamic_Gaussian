"""
Module 2 — Single-Surface Warping (单表面特征扭曲)

核心思路：
  1. 给定视图1 的度量深度 D_metric1，将每个像素反投影到 3D 世界坐标
  2. 将 3D 点投影到视图2，查询视图2 的特征 F_mono2，得到扭曲特征 F_{2→1}
  3. 构造误差/交互特征图 ΔF = Concat(F1, F_{2→1}, |F1 - F_{2→1}|)

投影公式：
  X_cam1 = K1_feat^{-1} @ [u, v, 1] * z1        (特征坐标 → 相机1坐标)
  X_world = R1^T @ X_cam1 - R1^T @ t1             (相机1 → 世界坐标)
  X_cam2  = R2 @ X_world + t2                     (世界 → 相机2坐标)
  [u2, v2] = K2_feat @ X_cam2 / z_cam2            (相机2坐标 → 特征坐标)
  采样 F_mono2 at [u2, v2]

所有投影在**特征分辨率** (H/feat_stride, W/feat_stride) 下进行，
内参相应缩放: K_feat = K / feat_stride。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


def scale_intrinsics(intr: torch.Tensor, feat_stride: int) -> torch.Tensor:
    """
    将内参缩放到特征分辨率。

    Args:
        intr: (B, 3, 3)
        feat_stride: 特征图相对于原图的下采样倍数

    Returns:
        (B, 3, 3) 缩放后的内参
    """
    intr_feat = intr.clone()
    intr_feat[:, :2, :] = intr[:, :2, :] / feat_stride  # 缩放 fx, fy, cx, cy
    return intr_feat


def build_pixel_grid(
    batch: int, height: int, width: int, device: torch.device
) -> torch.Tensor:
    """
    构建像素坐标网格。

    Returns:
        (B, H, W, 2)  每个位置的 (u, v) 坐标 (列优先: u=x, v=y)
    """
    v_coords = torch.arange(height, device=device, dtype=torch.float32)
    u_coords = torch.arange(width,  device=device, dtype=torch.float32)
    grid_v, grid_u = torch.meshgrid(v_coords, u_coords, indexing="ij")
    # (H, W, 2): [u, v]
    grid = torch.stack([grid_u, grid_v], dim=-1)
    return grid.unsqueeze(0).expand(batch, -1, -1, -1)  # (B, H, W, 2)


def unproject_depth(
    depth: torch.Tensor,
    intr: torch.Tensor,
    feat_stride: int = 1,
) -> torch.Tensor:
    """
    将深度图反投影为相机坐标系下的 3D 点云。

    Args:
        depth:      (B, 1, H, W)  度量深度 (z 轴，单位米)
        intr:       (B, 3, 3)     对应分辨率下的内参
        feat_stride: 若 intr 是全分辨率内参且 depth 是特征分辨率，传入 feat_stride 自动缩放

    Returns:
        pts_cam: (B, H, W, 3)  相机坐标系下的 3D 点 (X, Y, Z)
    """
    B, _, H, W = depth.shape
    intr_feat = scale_intrinsics(intr, feat_stride)

    fx = intr_feat[:, 0, 0].view(B, 1, 1)  # (B, 1, 1)
    fy = intr_feat[:, 1, 1].view(B, 1, 1)
    cx = intr_feat[:, 0, 2].view(B, 1, 1)
    cy = intr_feat[:, 1, 2].view(B, 1, 1)

    grid = build_pixel_grid(B, H, W, depth.device)  # (B, H, W, 2)
    u = grid[..., 0]  # (B, H, W)
    v = grid[..., 1]

    z = depth[:, 0]  # (B, H, W)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    return torch.stack([x, y, z], dim=-1)  # (B, H, W, 3)


def project_to_view(
    pts_world: torch.Tensor,
    intr: torch.Tensor,
    extr: torch.Tensor,
    feat_stride: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将世界坐标系 3D 点投影到某视图的特征坐标。

    Args:
        pts_world: (B, H, W, 3)  世界坐标
        intr:      (B, 3, 3)     目标视图全分辨率内参
        extr:      (B, 3, 4)     目标视图外参 [R|t] (world→camera)
        feat_stride: 特征图下采样倍数

    Returns:
        uv_feat:   (B, H, W, 2)  特征坐标 (u, v)，可能超出 [0, W/H] 范围
        z_cam:     (B, H, W)     投影后的深度
    """
    B, H, W, _ = pts_world.shape
    intr_feat = scale_intrinsics(intr, feat_stride)

    R = extr[:, :3, :3]  # (B, 3, 3)
    t = extr[:, :3, 3]   # (B, 3)

    # (B, H, W, 3) → (B, H*W, 3) → R @ pts + t
    pts_flat = pts_world.reshape(B, H * W, 3)
    # pts_cam = R @ pts + t  →  (B, H*W, 3)
    pts_cam_flat = (pts_flat @ R.transpose(-1, -2)) + t.unsqueeze(1)

    z_cam = pts_cam_flat[..., 2].clamp(min=1e-3)  # (B, H*W)

    fx = intr_feat[:, 0, 0].view(B, 1)
    fy = intr_feat[:, 1, 1].view(B, 1)
    cx = intr_feat[:, 0, 2].view(B, 1)
    cy = intr_feat[:, 1, 2].view(B, 1)

    u = pts_cam_flat[..., 0] / z_cam * fx + cx  # (B, H*W)
    v = pts_cam_flat[..., 1] / z_cam * fy + cy

    uv_feat = torch.stack([u, v], dim=-1).reshape(B, H, W, 2)
    z_cam = z_cam.reshape(B, H, W)

    return uv_feat, z_cam


def normalize_grid_coords(
    uv: torch.Tensor, feat_h: int, feat_w: int
) -> torch.Tensor:
    """
    将像素坐标归一化到 [-1, 1] 用于 F.grid_sample。

    grid_sample 约定: (-1,-1)=左上角, (1,1)=右下角, (u对应W维, v对应H维)

    Args:
        uv:     (B, H, W, 2)  [u, v] 像素坐标
        feat_h, feat_w: 特征图尺寸

    Returns:
        (B, H, W, 2)  归一化坐标 [x_norm, y_norm]
    """
    u_norm = (uv[..., 0] / (feat_w - 1)) * 2 - 1  # u → x
    v_norm = (uv[..., 1] / (feat_h - 1)) * 2 - 1  # v → y
    return torch.stack([u_norm, v_norm], dim=-1)   # (B, H, W, 2)


class SingleSurfaceWarping(nn.Module):
    """
    基于单视图度量深度的单表面特征扭曲模块。

    该模块不包含可学习参数（纯几何计算），完全可微。

    计算流程：
      1. D_metric1 (特征分辨率) 反投影 → 3D 点云 (相机1坐标)
      2. 3D 点云变换到世界坐标
      3. 投影到视图2 特征坐标 → UV grid
      4. grid_sample 采样 F_mono2 → F_{2→1}
      5. 组合 ΔF = [F1 ; F_{2→1} ; |F1 - F_{2→1}|]

    Args:
        feat_stride: 特征图下采样倍数，用于缩放内参
        padding_mode: grid_sample 的边界处理方式
                      "zeros"  — 视野外置 0
                      "border" — 最近边界值重复
    """

    def __init__(self, feat_stride: int = 4, padding_mode: str = "zeros") -> None:
        super().__init__()
        self.feat_stride = feat_stride
        self.padding_mode = padding_mode

    @staticmethod
    def cam2world(pts_cam: torch.Tensor, extr: torch.Tensor) -> torch.Tensor:
        """
        相机坐标 → 世界坐标。

        Args:
            pts_cam: (B, H, W, 3)
            extr:    (B, 3, 4) [R|t]  world→camera

        Returns:
            pts_world: (B, H, W, 3)
        """
        R = extr[:, :3, :3]  # (B, 3, 3)
        t = extr[:, :3, 3]   # (B, 3)

        B, H, W, _ = pts_cam.shape
        pts_flat = pts_cam.reshape(B, H * W, 3)

        # 行向量约定: X_cam = X_world @ R^T + t → X_world = (X_cam - t) @ R
        # (因 R^{-1} = R^T，行向量下 (R^T)^{-1} = R)
        pts_world_flat = (pts_flat - t.unsqueeze(1)) @ R
        return pts_world_flat.reshape(B, H, W, 3)

    def forward(
        self,
        f_mono1: torch.Tensor,
        f_mono2: torch.Tensor,
        d_metric1: torch.Tensor,
        intr1: torch.Tensor,
        intr2: torch.Tensor,
        extr1: torch.Tensor,
        extr2: torch.Tensor,
        img2: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        执行特征扭曲，计算交互特征 ΔF；可选同步扭曲 img2 颜色图。

        Args:
            f_mono1:   (B, C, Hf, Wf)  视图1 特征
            f_mono2:   (B, C, Hf, Wf)  视图2 特征
            d_metric1: (B, 1, H,  W )  视图1 全分辨率度量深度
            intr1:     (B, 3, 3)  视图1 全分辨率内参
            intr2:     (B, 3, 3)  视图2 全分辨率内参
            extr1:     (B, 3, 4)  视图1 外参
            extr2:     (B, 3, 4)  视图2 外参
            img2:      (B, 3, H,  W )  视图2 原始 RGB [-1,1]（可为 None）

        Returns:
            delta_f:       (B, 3*C, Hf, Wf)  交互特征图 ΔF
            valid_mask:    (B, 1, Hf, Wf)     有效扭曲区域掩码 (1=有效)
            img2_warped:   (B, 3, Hf, Wf)     扭曲到视图1 的 img2 颜色图（若 img2=None 则为 None）
        """
        B, C, Hf, Wf = f_mono1.shape

        # --- 将深度下采样到特征分辨率 ---
        d_feat = F.interpolate(
            d_metric1, size=(Hf, Wf), mode="bilinear", align_corners=False
        )  # (B, 1, Hf, Wf)

        # --- Step 1: 反投影 D_feat1 → 相机1 3D 点云 ---
        pts_cam1 = unproject_depth(d_feat, intr1, feat_stride=self.feat_stride)
        # (B, Hf, Wf, 3)

        # --- Step 2: 相机1 → 世界坐标 ---
        R1 = extr1[:, :3, :3]  # (B, 3, 3)
        t1 = extr1[:, :3, 3]   # (B, 3)
        B_, H_, W_, _ = pts_cam1.shape
        pts_flat = pts_cam1.reshape(B_, H_ * W_, 3)
        pts_world_flat = (pts_flat - t1.unsqueeze(1)) @ R1
        pts_world = pts_world_flat.reshape(B_, H_, W_, 3)

        # --- Step 3: 世界坐标 → 相机2 坐标 → 特征坐标 UV ---
        uv2_feat, z_cam2 = project_to_view(pts_world, intr2, extr2, self.feat_stride)
        # uv2_feat: (B, Hf, Wf, 2)  z_cam2: (B, Hf, Wf)

        # --- Step 4: 有效掩码 ---
        u_valid = (uv2_feat[..., 0] >= 0) & (uv2_feat[..., 0] <= Wf - 1)
        v_valid = (uv2_feat[..., 1] >= 0) & (uv2_feat[..., 1] <= Hf - 1)
        depth_valid = z_cam2 > 1e-3
        valid_mask = (u_valid & v_valid & depth_valid).unsqueeze(1).float()
        # (B, 1, Hf, Wf)

        # --- Step 5: 归一化 UV → grid_sample 坐标 ---
        grid = normalize_grid_coords(uv2_feat, Hf, Wf)  # (B, Hf, Wf, 2)

        # --- Step 6: 双线性采样 F_mono2 ---
        f_2to1 = F.grid_sample(
            f_mono2,
            grid,
            mode="bilinear",
            padding_mode=self.padding_mode,
            align_corners=True,
        )  # (B, C, Hf, Wf)
        f_2to1 = f_2to1 * valid_mask

        # --- Step 7: 构造交互特征 ΔF ---
        diff = torch.abs(f_mono1 - f_2to1)
        delta_f = torch.cat([f_mono1, f_2to1, diff], dim=1)  # (B, 3C, Hf, Wf)

        # --- Step 8 (可选): 同步扭曲 img2 颜色图 ---
        img2_warped = None
        if img2 is not None:
            # 将 img2 下采样到特征分辨率后用同一 grid 采样
            img2_down = F.interpolate(
                img2, size=(Hf, Wf), mode="bilinear", align_corners=False
            )  # (B, 3, Hf, Wf)
            # padding_mode="zeros"：视野外区域置 0，配合 valid_mask 使 decoder 显式感知无效区
            # 不用 "border" 是因为 border 重复边缘色素，在重叠小时污染非重叠区颜色
            img2_warped = F.grid_sample(
                img2_down,
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=True,
            )  # (B, 3, Hf, Wf)
            # 用 valid_mask 进一步清零非重叠像素（防止 grid 边界的插值残留）
            img2_warped = img2_warped * valid_mask  # (B, 3, Hf, Wf)

        return delta_f, valid_mask, img2_warped
