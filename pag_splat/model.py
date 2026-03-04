"""
PAGSplat — Prior-Aware 2D Gaussian Splatting (主模型)

前向传播完整数据流：
  输入: 两张稀疏视角图像 {img1, img2} + 相机参数 {K1, K2, E1, E2}
  ↓
  [Module 1] MonoPriorExtractor  → F_mono1, F_mono2, D_rel1, D_rel2
  [Module 1] ScaleAlignmentMLP   → D_metric1 = S·D_rel1 + T
  ↓
  [Module 2] SingleSurfaceWarping → ΔF = [F1; F_{2→1}; |F1-F_{2→1}|]
  ↓
  [Module 3] GaussianDecoder      → rot(4), scale(3), opacity(1),
                                     delta_depth(1), uncertainty(1)
  ↓
  深度细化: D_final = D_metric1 + delta_depth
  depth2pc: D_final → 3D 世界坐标 xyz
  ↓
  pts2render (复用 GPS+ diff_gaussian_rasterization)
  ↓
  输出: rendered novel-view image

兼容性：
  - 数据字典格式与 GPS+ 的 'lmain'/'rmain' 兼容
  - 渲染接口与 GPS+ 的 pts2render 兼容
  - 颜色直接取自输入像素 (不预测颜色，与 GPS+ 一致)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .prior_extractor import MonoPriorExtractor
from .scale_align import ScaleAlignmentMLP
from .warping import SingleSurfaceWarping
from .gaussian_decoder import GaussianDecoder


# ──────────────────────────────────────────────
#  辅助：深度 → 3D 点云
# ──────────────────────────────────────────────

def depth_to_pointcloud(
    depth: torch.Tensor,
    intr: torch.Tensor,
    extr: torch.Tensor,
) -> torch.Tensor:
    """
    将度量深度图 (z 轴) 转换为世界坐标系 3D 点云。

    Args:
        depth: (B, 1, H, W)  度量深度 (正值，米)
        intr:  (B, 3, 3)     相机内参
        extr:  (B, 3, 4)     相机外参 [R|t] (world→camera)

    Returns:
        pts_world: (B, H*W, 3)
    """
    B, _, H, W = depth.shape

    fx = intr[:, 0, 0].view(B, 1, 1)
    fy = intr[:, 1, 1].view(B, 1, 1)
    cx = intr[:, 0, 2].view(B, 1, 1)
    cy = intr[:, 1, 2].view(B, 1, 1)

    # 像素坐标网格
    v_c = torch.arange(H, device=depth.device, dtype=torch.float32).view(1, H, 1)
    u_c = torch.arange(W, device=depth.device, dtype=torch.float32).view(1, 1, W)
    u_c = u_c.expand(B, H, W)
    v_c = v_c.expand(B, H, W)

    z = depth[:, 0]
    x_cam = (u_c - cx) * z / fx
    y_cam = (v_c - cy) * z / fy
    pts_cam = torch.stack([x_cam, y_cam, z], dim=-1)  # (B, H, W, 3)

    # 相机 → 世界坐标
    R = extr[:, :3, :3]   # (B, 3, 3)
    t = extr[:, :3, 3]    # (B, 3)
    pts_flat = pts_cam.reshape(B, H * W, 3)
    # X_world = R^T @ (X_cam - t)
    pts_world = (pts_flat - t.unsqueeze(1)) @ R   # (B, H*W, 3)

    return pts_world


# ──────────────────────────────────────────────
#  主模型
# ──────────────────────────────────────────────

class PAGSplat(nn.Module):
    """
    PAG-Splat: Prior-Aware 2D Gaussian Splatting

    Args:
        da3_net:        DA3 核心网络实例 (DepthAnything3Net / NestedDepthAnything3Net)
        embed_dim:      DA3 backbone 的 patch 特征维度 (ViT-B=768, ViT-L=1024)
        feat_channels:  内部统一特征通道数
        feat_stride:    特征图下采样倍数 (推荐 4)
        feat_layer:     DA3 DINOv2 特征提取层索引
        mlp_hidden:     ScaleAlignmentMLP 隐藏层维度
        enc_dims:       GaussianDecoder 编码器通道配置
        dec_dims:       GaussianDecoder 解码器通道配置
        head_ch:        预测头共享通道数
        scale_max:      高斯缩放上限
    """

    def __init__(
        self,
        da3_net: nn.Module,
        embed_dim: int = 768,
        feat_channels: int = 256,
        feat_stride: int = 4,
        feat_layer: int = 8,
        mlp_hidden: int = 256,
        enc_dims: list[int] | None = None,
        dec_dims: list[int] | None = None,
        head_ch: int = 32,
        scale_max: float = 0.002,
    ) -> None:
        super().__init__()

        enc_dims = enc_dims or [128, 256, 512]
        dec_dims = dec_dims or [128, 256, 512]

        # ── Module 1a: 冻结 DA3 先验提取 ──
        self.prior_extractor = MonoPriorExtractor(
            da3_net=da3_net,
            embed_dim=embed_dim,
            feat_channels=feat_channels,
            feat_stride=feat_stride,
            feat_layer=feat_layer,
        )

        # ── Module 1b: 尺度对齐 MLP ──
        self.scale_align = ScaleAlignmentMLP(hidden_dim=mlp_hidden)

        # ── Module 2: 单表面特征扭曲 ──
        self.warping = SingleSurfaceWarping(feat_stride=feat_stride)

        # ── Module 3: 不确定性感知高斯图解码 ──
        self.gaussian_decoder = GaussianDecoder(
            delta_f_channels=feat_channels * 3,
            feat_stride=feat_stride,
            enc_dims=enc_dims,
            dec_dims=dec_dims,
            head_ch=head_ch,
            scale_max=scale_max,
        )

    # ──────────────────────────────────────────
    #  完整前向传播
    # ──────────────────────────────────────────

    def forward(
        self,
        data: Dict,
        is_train: bool = True,
    ) -> Dict:
        """
        主前向传播，兼容 GPS+ 数据字典格式。

        Args:
            data: 包含以下键的字典:
              'lmain' / 'rmain':
                'img':   (B, 3, H, W)  归一化图像
                'intr':  (B, 3, 3)     内参
                'extr':  (B, 3, 4)     外参
                'mask':  (B, 3, H, W)  前景掩码 (可选)
              data['novel_view']: 目标视角相机参数 (渲染用)

        Returns:
            data (原地更新):
              'lmain'/'rmain' 新增:
                'xyz':        (B, H*W, 3)  世界坐标
                'rot':        (B, H*W, 4)  四元数
                'scale':      (B, H*W, 3)  缩放
                'opacity':    (B, H*W, 1)  不透明度
                'uncertainty':(B, H*W, 1)  不确定性
              'metric_depth_l': (B, 1, H, W)  视图1 度量深度
              'warp_valid_mask': (B, 1, Hf, Wf) 有效扭曲区域
        """
        lm = data["lmain"]
        rm = data["rmain"]

        img1  = lm["img"]    # (B, 3, H, W)
        img2  = rm["img"]
        intr1 = lm["intr"]   # (B, 3, 3)
        intr2 = rm["intr"]
        extr1 = lm["extr"]   # (B, 3, 4)
        extr2 = rm["extr"]

        B, _, H, W = img1.shape

        # ═══════════════════════════════════════
        # MODULE 1: 单目先验提取 + 尺度对齐
        # ═══════════════════════════════════════

        f_mono1, f_mono2, d_rel1, d_rel2 = self.prior_extractor(img1, img2)
        # f_mono1/2: (B, feat_ch, H/4, W/4)
        # d_rel1/2:  (B, 1, H, W)

        # 视图1 → 度量深度
        d_metric1, log_s1, t1 = self.scale_align(
            intr1, intr2, extr1, extr2, d_rel1, H, W
        )
        # 视图2 → 度量深度 (交换视角关系)
        d_metric2, log_s2, t2 = self.scale_align(
            intr2, intr1, extr2, extr1, d_rel2, H, W
        )

        # ═══════════════════════════════════════
        # MODULE 2: 单表面特征扭曲
        # ═══════════════════════════════════════

        delta_f, valid_mask = self.warping(
            f_mono1, f_mono2, d_metric1,
            intr1, intr2, extr1, extr2,
        )
        # delta_f: (B, 3*feat_ch, H/4, W/4)
        # valid_mask: (B, 1, H/4, W/4)

        # ═══════════════════════════════════════
        # MODULE 3: 高斯参数图解码
        # ═══════════════════════════════════════

        gs_params = self.gaussian_decoder(delta_f, img1)
        # gs_params: dict 含 rot/scale/opacity/delta_depth/uncertainty，均 (B, ch, H, W)

        # ═══════════════════════════════════════
        # 深度细化 + 反投影
        # ═══════════════════════════════════════

        # D_final = D_metric1 + Δdepth
        d_final = d_metric1 + gs_params["delta_depth"]
        d_final = F.softplus(d_final) + 1e-3   # 再次确保正值

        # 将高斯参数展平为点云格式 (B, H*W, ch)
        def flatten_map(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        xyz1 = depth_to_pointcloud(d_final, intr1, extr1)   # (B, H*W, 3)

        # 写回 data['lmain']
        lm["xyz"]         = xyz1
        lm["rot"]         = flatten_map(gs_params["rot"])         # (B, H*W, 4)
        lm["scale"]       = flatten_map(gs_params["scale"])       # (B, H*W, 3)
        lm["opacity"]     = flatten_map(gs_params["opacity"])     # (B, H*W, 1)
        lm["uncertainty"] = flatten_map(gs_params["uncertainty"]) # (B, H*W, 1)

        # ── 视图2 的高斯参数 (对称地用视图1特征扭曲到视图2) ──
        # 为简化: 对视图2 做同样的处理 (swap 输入顺序)
        delta_f2, valid_mask2 = self.warping(
            f_mono2, f_mono1, d_metric2,
            intr2, intr1, extr2, extr1,
        )
        gs_params2 = self.gaussian_decoder(delta_f2, img2)

        d_final2 = d_metric2 + gs_params2["delta_depth"]
        d_final2 = F.softplus(d_final2) + 1e-3

        xyz2 = depth_to_pointcloud(d_final2, intr2, extr2)

        rm["xyz"]         = xyz2
        rm["rot"]         = flatten_map(gs_params2["rot"])
        rm["scale"]       = flatten_map(gs_params2["scale"])
        rm["opacity"]     = flatten_map(gs_params2["opacity"])
        rm["uncertainty"] = flatten_map(gs_params2["uncertainty"])

        # 保存中间结果供 loss 计算
        data["metric_depth_l"]   = d_metric1          # (B, 1, H, W)
        data["metric_depth_r"]   = d_metric2
        data["warp_valid_mask"]  = valid_mask          # (B, 1, Hf, Wf)
        data["warp_valid_mask2"] = valid_mask2
        data["log_scale_l"]      = log_s1              # (B,)
        data["log_scale_r"]      = log_s2

        return data

    # ──────────────────────────────────────────
    #  便捷方法
    # ──────────────────────────────────────────

    def count_parameters(self) -> dict[str, int]:
        """统计各子模块参数量。"""
        def count(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)

        return {
            "prior_extractor (trainable proj)": count(self.prior_extractor),
            "scale_align_mlp":   count(self.scale_align),
            "warping":           count(self.warping),      # 0 (纯几何)
            "gaussian_decoder":  count(self.gaussian_decoder),
            "total_trainable":   count(self),
        }


# ──────────────────────────────────────────────
#  pts2render 兼容层 (复用 GPS+ 渲染管线)
# ──────────────────────────────────────────────

def pag_pts2render(data: Dict, bg_color: list[float] = [0, 0, 0]) -> Dict:
    """
    将 PAGSplat 预测的高斯参数送入 GPS+ 的 diff_gaussian_rasterization 渲染。

    与 GPS+ pts2render 的差异：
      - 不透明度额外乘以 (1 - uncertainty) 降低不可靠高斯的贡献
      - 颜色仍直接来自输入像素 (与 GPS+ 一致)

    Args:
        data: PAGSplat.forward() 返回的数据字典
        bg_color: 背景颜色 [R, G, B]

    Returns:
        data (新增 'novel_view_img' 键)
    """
    # 延迟导入以避免在无 CUDA 环境下崩溃
    try:
        from lib.GaussianRender import pts2render as _gps_pts2render
    except ImportError:
        raise ImportError(
            "需要 GPS+ 的 lib.GaussianRender 模块，"
            "请确保在 GPS_plus 目录下运行，或单独安装 diff_gaussian_rasterization。"
        )

    # 将 uncertainty 整合进 opacity
    for view_key in ["lmain", "rmain"]:
        view = data[view_key]
        if "uncertainty" in view and "opacity" in view:
            # 高不确定性 → 降低不透明度
            view["opacity"] = view["opacity"] * (1.0 - view["uncertainty"])

    return _gps_pts2render(data, bg_color=bg_color)


# ──────────────────────────────────────────────
#  工厂函数
# ──────────────────────────────────────────────

def build_pag_splat(
    da3_checkpoint: str | None = None,
    da3_model_name: str = "da3-large",
    feat_channels: int = 256,
    feat_stride: int = 4,
    feat_layer: int = 8,
    mlp_hidden: int = 256,
    enc_dims: list[int] | None = None,
    dec_dims: list[int] | None = None,
    head_ch: int = 32,
    scale_max: float = 0.002,
    device: str = "cuda",
) -> PAGSplat:
    """
    构建 PAGSplat 模型，可选从 HuggingFace Hub 或本地路径加载 DA3。

    Args:
        da3_checkpoint:  DA3 预训练权重路径或 HF Hub repo id
                         None 时从 Hub 自动下载
        da3_model_name:  DA3 preset 名称 ("da3-large" / "da3-base" 等)
        feat_channels:   PAGSplat 内部特征维度
        feat_stride:     特征下采样倍数
        feat_layer:      从 DINOv2 第几层提取特征
        mlp_hidden:      ScaleAlignmentMLP 隐藏层维度
        enc_dims:        GaussianDecoder 编码器通道配置
        dec_dims:        GaussianDecoder 解码器通道配置
        head_ch:         预测头共享通道数
        scale_max:       高斯缩放上限
        device:          目标设备

    Returns:
        PAGSplat 实例 (DA3 已冻结)
    """
    enc_dims = enc_dims or [128, 256, 512]
    dec_dims = dec_dims or [128, 256, 512]
    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError:
        raise ImportError(
            "未找到 depth_anything_3 包，"
            "请将 /data/sifang/Depth-Anything-3/src 加入 PYTHONPATH。"
        )

    if da3_checkpoint is not None:
        da3_api = DepthAnything3.from_pretrained(da3_checkpoint)
    else:
        da3_api = DepthAnything3(model_name=da3_model_name)

    da3_api = da3_api.to(device).eval()

    # 从 patch_embed 投影权重反推 DINOv2 embed_dim
    # 权重形状: (embed_dim, 3, patch_size, patch_size)
    try:
        for name, param in da3_api.model.backbone.named_parameters():
            if "patch_embed" in name and "weight" in name and param.ndim == 4:
                embed_dim = param.shape[0]
                break
        else:
            embed_dim = 768
    except Exception:
        embed_dim = 768
    print(f"[PAGSplat] embed_dim = {embed_dim}")

    model = PAGSplat(
        da3_net=da3_api.model,
        embed_dim=embed_dim,
        feat_channels=feat_channels,
        feat_stride=feat_stride,
        feat_layer=feat_layer,
        mlp_hidden=mlp_hidden,
        enc_dims=enc_dims,
        dec_dims=dec_dims,
        head_ch=head_ch,
        scale_max=scale_max,
    ).to(device)

    return model
