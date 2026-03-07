"""
PAGSplat — Prior-Aware 2D Gaussian Splatting (主模型)

前向传播完整数据流：
  输入: 两张稀疏视角图像 {img1, img2} + 相机参数 {K1, K2, E1, E2}
  ↓
  [Module 1] MonoPriorExtractor  → F_mono1, F_mono2, D_rel1, D_rel2
  [Module 1] ScaleAlignmentMLP   → D_metric1 = S·D_rel1 + T  (含双视图全局特征)
  ↓
  [迭代深度精化] DepthGRUCell × N_iters:
    (re-warp F_mono2 with current depth) → ΔF_i → GRU → Δd_feat → D_{i+1}
  ↓
  [Module 2] SingleSurfaceWarping (最后一次，同时输出 img2_warped_feat)
  ↓
  [Module 3] GaussianDecoder      → rot(4), scale(3), opacity(1),
                                     delta_depth(1), color_map(3)
  ↓
  深度细化: D_final = D_gru + delta_depth
  depth2pc: D_final → 3D 世界坐标 xyz
  ↓
  pts2render (复用 GPS+ diff_gaussian_rasterization)，颜色使用 color_map
  ↓
  输出: rendered novel-view image

兼容性：
  - 数据字典格式与 GPS+ 的 'lmain'/'rmain' 兼容
  - 渲染接口与 GPS+ 的 pts2render 兼容
"""

from __future__ import annotations

import math
import os
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
#  迭代深度精化 ConvGRU 单元
# ──────────────────────────────────────────────

class DepthGRUCell(nn.Module):
    """
    轻量 ConvGRU 单元，在特征分辨率 (H/4) 下迭代精化像素级深度偏移。

    参考: Splat-SAP Translation module (Eq.5), RAFT-Stereo ConvGRU。

    每轮迭代:
      1. 将 f_mono2 用当前深度扭曲至视图1 → f_2to1_i
      2. compress([f_mono1, f_2to1_i, d_feat]) → x (B, hidden_ch, Hf, Wf)
      3. GRU 门控: h_new = GRU(x, h_prev)
      4. Δd_feat = depth_head(h_new)  →  上采样后加入当前深度 d_i

    Args:
        feat_channels: DA3 特征通道数 (f_mono1/f_2to1 各占此数)
        hidden_ch:     GRU 隐藏状态通道数
    """

    def __init__(self, feat_channels: int = 256, hidden_ch: int = 64) -> None:
        super().__init__()
        self.hidden_ch = hidden_ch

        # 输入压缩: [f_mono1(C), f_2to1(C), d_feat(1)] → hidden_ch
        self.compress = nn.Sequential(
            nn.Conv2d(feat_channels * 2 + 1, hidden_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
        )

        # GRU 门控 (输入维度 = hidden_ch + hidden_ch)
        gate_in = hidden_ch * 2
        self.update_gate = nn.Conv2d(gate_in, hidden_ch, 3, 1, 1)
        self.reset_gate  = nn.Conv2d(gate_in, hidden_ch, 3, 1, 1)
        self.new_gate    = nn.Conv2d(gate_in, hidden_ch, 3, 1, 1)

        # 深度增量预测（最后一层零初始化，确保训练初期从零增量开始）
        _last = nn.Conv2d(32, 1, 3, 1, 1)
        nn.init.zeros_(_last.weight)
        nn.init.zeros_(_last.bias)
        self.depth_head = nn.Sequential(
            nn.Conv2d(hidden_ch, 32, 3, 1, 1),
            nn.GELU(),
            _last,
        )

    def forward(
        self,
        f_mono1: torch.Tensor,   # (B, C, Hf, Wf)
        f_2to1:  torch.Tensor,   # (B, C, Hf, Wf)  用当前深度扭曲的视图2特征
        d_feat:  torch.Tensor,   # (B, 1, Hf, Wf)  当前深度（特征分辨率）
        h:       torch.Tensor,   # (B, hidden_ch, Hf, Wf)  GRU 隐藏状态
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_new:        (B, hidden_ch, Hf, Wf)
            delta_d_feat: (B, 1, Hf, Wf)  特征分辨率深度增量
        """
        x = self.compress(torch.cat([f_mono1, f_2to1, d_feat], dim=1))
        xh = torch.cat([x, h], dim=1)

        z = torch.sigmoid(self.update_gate(xh))
        r = torch.sigmoid(self.reset_gate(xh))
        n = torch.tanh(self.new_gate(torch.cat([x, r * h], dim=1)))
        h_new = (1.0 - z) * h + z * n

        delta_d_feat = torch.tanh(self.depth_head(h_new)) * 1.0   # ±1m at feat resolution
        return h_new, delta_d_feat


# ──────────────────────────────────────────────
#  主模型
# ──────────────────────────────────────────────

class PAGSplat(nn.Module):
    """
    PAG-Splat: Prior-Aware 2D Gaussian Splatting

    Args:
        da3_net:        DA3 核心网络实例
        embed_dim:      DA3 backbone 的 patch 特征维度 (ViT-B=768, ViT-L=1024)
        feat_channels:  内部统一特征通道数
        feat_stride:    特征图下采样倍数 (推荐 4)
        feat_layer:     DA3 DINOv2 特征提取层索引
        mlp_hidden:     ScaleAlignmentMLP 隐藏层维度
        enc_dims:       GaussianDecoder 编码器通道配置
        dec_dims:       GaussianDecoder 解码器通道配置
        head_ch:        预测头共享通道数
        scale_max:      高斯缩放上限
        gru_iters:      迭代深度精化次数（类 Splat-SAP）
        gru_hidden_ch:  DepthGRUCell 隐藏通道数
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
        t12_mode: str = "log_len",
        gru_iters: int = 3,
        gru_hidden_ch: int = 64,
    ) -> None:
        super().__init__()

        enc_dims = enc_dims or [128, 256, 512]
        dec_dims = dec_dims or [128, 256, 512]
        self.feat_channels = feat_channels
        self.feat_stride   = feat_stride
        self.gru_iters     = gru_iters

        # ── Module 1a: 冻结 DA3 先验提取 ──
        self.prior_extractor = MonoPriorExtractor(
            da3_net=da3_net,
            embed_dim=embed_dim,
            feat_channels=feat_channels,
            feat_stride=feat_stride,
            feat_layer=feat_layer,
        )

        # ── Module 1b: 尺度对齐 MLP (含双视图全局图像特征) ──
        self.scale_align = ScaleAlignmentMLP(
            hidden_dim=mlp_hidden,
            t12_mode=t12_mode,
            use_img_feat=True,
            feat_channels=feat_channels,
        )

        # ── 迭代深度精化 GRU ──
        self.depth_gru = DepthGRUCell(feat_channels=feat_channels, hidden_ch=gru_hidden_ch)

        # ── Module 2: 单表面特征扭曲 ──
        self.warping = SingleSurfaceWarping(feat_stride=feat_stride)

        # ── Module 3: 不确定性感知高斯图解码 (含颜色融合头) ──
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

    def _run_one_view(
        self,
        img_self:  torch.Tensor,   # (B, 3, H, W)
        img_other: torch.Tensor,   # (B, 3, H, W)
        intr_self: torch.Tensor,
        intr_other: torch.Tensor,
        extr_self:  torch.Tensor,
        extr_other: torch.Tensor,
        f_mono_self:  torch.Tensor,  # (B, C, Hf, Wf)
        f_mono_other: torch.Tensor,
        d_rel_self:   torch.Tensor,  # (B, 1, H, W)
    ) -> Tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        对单个视图执行完整的深度估计 + 高斯解码。

        Returns:
            gs_params:   {rot, scale, opacity, delta_depth, color_map}
            d_final:     (B, 1, H, W)
            d_metric:    (B, 1, H, W)
            valid_mask:  (B, 1, Hf, Wf)
            log_s:       (B,)
        """
        B, _, H, W = img_self.shape
        _, _, Hf, Wf = f_mono_self.shape

        # ── 1. 尺度对齐（含双视图全局图像特征）──
        d_metric, log_s, _ = self.scale_align(
            intr_self, intr_other, extr_self, extr_other,
            d_rel_self, H, W,
            f_self=f_mono_self, f_other=f_mono_other,
        )

        # ── 2. 迭代深度精化（DepthGRUCell） ──
        # 最后一次迭代时同步输出 delta_f 和 img2_warped，避免额外一次冗余 warp
        d_curr = d_metric                                 # (B, 1, H, W)
        h = torch.zeros(
            B, self.depth_gru.hidden_ch, Hf, Wf,
            device=img_self.device, dtype=img_self.dtype
        )
        delta_f = valid_mask = img_other_warped = None
        for i in range(self.gru_iters):
            is_last = (i == self.gru_iters - 1)
            d_feat = F.interpolate(
                d_curr, size=(Hf, Wf), mode="bilinear", align_corners=False
            )
            # 最后一次迭代同步扭曲 img_other，复用同一次 warp 输出
            delta_f_i, valid_mask, img_other_warped = self.warping(
                f_mono_self, f_mono_other, d_curr,
                intr_self, intr_other, extr_self, extr_other,
                img2=(img_other if is_last else None),
            )
            f_2to1_i = delta_f_i[:, self.feat_channels: 2 * self.feat_channels]
            h, delta_d_feat = self.depth_gru(f_mono_self, f_2to1_i, d_feat, h)
            delta_d = F.interpolate(
                delta_d_feat, size=(H, W), mode="bilinear", align_corners=False
            )
            d_curr = d_curr + delta_d
            d_curr = F.softplus(d_curr) + 1e-3
            if is_last:
                delta_f = delta_f_i   # 保存最后一次 delta_f 给解码器用

        d_gru = d_curr

        # ── 3. 高斯解码（含颜色融合头，delta_f 来自最后一次 GRU 迭代）──
        gs_params = self.gaussian_decoder(
            delta_f, img_self,
            img2_warped_feat=img_other_warped,
        )

        # ── 4. 最终深度细化（pixel-level delta_depth on top of GRU result）──
        d_final = d_gru + gs_params["delta_depth"]
        d_final = F.softplus(d_final) + 1e-3

        return gs_params, d_final, d_metric, valid_mask, log_s

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
                'color_map':  (B, 3, H, W) 预测融合颜色图 [0,1]
        """
        lm = data["lmain"]
        rm = data["rmain"]

        img1  = lm["img"];  img2  = rm["img"]
        intr1 = lm["intr"]; intr2 = rm["intr"]
        extr1 = lm["extr"]; extr2 = rm["extr"]

        B, _, H, W = img1.shape

        # ═══════════════════════════════════════
        # 单目先验提取
        # ═══════════════════════════════════════
        f_mono1, f_mono2, d_rel1, d_rel2 = self.prior_extractor(img1, img2)

        # ═══════════════════════════════════════
        # 视图 1: 深度精化 + 高斯解码
        # ═══════════════════════════════════════
        gs_params, d_final, d_metric1, valid_mask, log_s1 = self._run_one_view(
            img1, img2, intr1, intr2, extr1, extr2,
            f_mono1, f_mono2, d_rel1,
        )

        def flatten_map(t: torch.Tensor) -> torch.Tensor:
            return t.permute(0, 2, 3, 1).reshape(B, H * W, -1)

        xyz1 = depth_to_pointcloud(d_final, intr1, extr1)
        lm["xyz"]     = xyz1
        lm["rot"]     = flatten_map(gs_params["rot"])
        lm["scale"]   = flatten_map(gs_params["scale"])
        lm["opacity"] = flatten_map(gs_params["opacity"])
        if "color_map" in gs_params:
            lm["color_map"] = gs_params["color_map"]   # (B, 3, H, W) [0,1]

        # ═══════════════════════════════════════
        # 视图 2: 深度精化 + 高斯解码 (对称)
        # ═══════════════════════════════════════
        gs_params2, d_final2, d_metric2, valid_mask2, log_s2 = self._run_one_view(
            img2, img1, intr2, intr1, extr2, extr1,
            f_mono2, f_mono1, d_rel2,
        )

        xyz2 = depth_to_pointcloud(d_final2, intr2, extr2)
        rm["xyz"]     = xyz2
        rm["rot"]     = flatten_map(gs_params2["rot"])
        rm["scale"]   = flatten_map(gs_params2["scale"])
        rm["opacity"] = flatten_map(gs_params2["opacity"])
        if "color_map" in gs_params2:
            rm["color_map"] = gs_params2["color_map"]

        # ── 保存中间结果供 loss 计算 ──
        data["metric_depth_l"]   = d_metric1
        data["metric_depth_r"]   = d_metric2
        data["warp_valid_mask"]  = valid_mask
        data["warp_valid_mask2"] = valid_mask2
        data["log_scale_l"]      = log_s1
        data["log_scale_r"]      = log_s2
        data["final_depth_l"]    = d_final
        data["final_depth_r"]    = d_final2
        data["rot_map_l"]        = gs_params["rot"]
        data["scale_map_l"]      = gs_params["scale"]
        data["rot_map_r"]        = gs_params2["rot"]
        data["scale_map_r"]      = gs_params2["scale"]
        data["xyz_map_l"]        = xyz1.reshape(B, H, W, 3)
        data["xyz_map_r"]        = xyz2.reshape(B, H, W, 3)

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
            "depth_gru":         count(self.depth_gru),
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
      - 不透明度由 opacity_head 和 valid_mask 直接控制
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

    # uncertainty_head 已移除，opacity 直接由网络学习

    return _gps_pts2render(data, bg_color=bg_color)


# ──────────────────────────────────────────────
#  工厂函数
# ──────────────────────────────────────────────

def detect_t12_mode(ckpt_path: str | None) -> str:
    """
    从 checkpoint 文件中自动检测 ScaleAlignmentMLP 的 t12 编码模式。

    通过检查 scale_align.mlp.0.weight 的输入维度推断：
      - 输入维度 17 (use_view2_intr=True) → "norm" 旧版 (K1+K2+R12_6d+t12_dir3 = 4+4+6+3)
      - 输入维度 18 (use_view2_intr=True) → "log_len" 新版 (K1+K2+R12_6d+t12_dir3+loglen1 = 4+4+6+4)

    Args:
        ckpt_path: checkpoint 文件路径，None 时默认返回 "log_len"

    Returns:
        "log_len" 或 "norm"
    """
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return "log_len"
    try:
        import torch as _torch
        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("network", ckpt)
        w = state.get("scale_align.mlp.0.weight")
        if w is not None:
            in_dim = w.shape[1]
            mode = "norm" if in_dim in (13, 17) else "log_len"
            print(f"[PAGSplat] 检测到 scale_align 输入维度={in_dim}, t12_mode='{mode}'")
            return mode
    except Exception as e:
        print(f"[PAGSplat] checkpoint 检测失败 ({e})，使用默认 t12_mode='log_len'")
    return "log_len"


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
    ckpt_path: str | None = None,
    t12_mode: str | None = None,
    gru_iters: int = 3,
    gru_hidden_ch: int = 64,
) -> PAGSplat:
    """
    构建 PAGSplat 模型，可选从 HuggingFace Hub 或本地路径加载 DA3。

    Args:
        da3_checkpoint:  DA3 预训练权重路径或 HF Hub repo id，None 时从 Hub 自动下载
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
        ckpt_path:       待加载的 PAGSplat checkpoint 路径（用于自动检测 t12_mode）
        t12_mode:        显式指定 t12 编码模式，覆盖自动检测

    Returns:
        PAGSplat 实例 (DA3 已冻结)
    """
    import os as _os
    enc_dims = enc_dims or [128, 256, 512]
    dec_dims = dec_dims or [128, 256, 512]

    # 自动检测旧/新 checkpoint 的 t12 编码维度
    if t12_mode is None:
        t12_mode = detect_t12_mode(ckpt_path)

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
    try:
        for name, param in da3_api.model.backbone.named_parameters():
            if "patch_embed" in name and "weight" in name and param.ndim == 4:
                embed_dim = param.shape[0]
                break
        else:
            embed_dim = 768
    except Exception:
        embed_dim = 768
    print(f"[PAGSplat] embed_dim={embed_dim}, t12_mode='{t12_mode}'")

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
        t12_mode=t12_mode,
        gru_iters=gru_iters,
        gru_hidden_ch=gru_hidden_ch,
    ).to(device)

    return model
