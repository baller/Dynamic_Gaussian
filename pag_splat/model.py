"""
PAGSplat 简化主模型

目标：
  - 保留 DA3 单目先验提取
  - 保留相机感知的尺度对齐
  - 回退到原 GPS+ 风格的 GSRegresser
  - 输出与 lib.GaussianRender.pts2render 兼容的字段
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser

from .prior_extractor import MonoPriorExtractor
from .scale_align import ScaleAlignmentMLP


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
    bsz, _, height, width = depth.shape

    fx = intr[:, 0, 0].view(bsz, 1, 1)
    fy = intr[:, 1, 1].view(bsz, 1, 1)
    cx = intr[:, 0, 2].view(bsz, 1, 1)
    cy = intr[:, 1, 2].view(bsz, 1, 1)

    v_c = torch.arange(height, device=depth.device, dtype=torch.float32).view(1, height, 1)
    u_c = torch.arange(width, device=depth.device, dtype=torch.float32).view(1, 1, width)
    u_c = u_c.expand(bsz, height, width)
    v_c = v_c.expand(bsz, height, width)

    z = depth[:, 0]
    x_cam = (u_c - cx) * z / fx
    y_cam = (v_c - cy) * z / fy
    pts_cam = torch.stack([x_cam, y_cam, z], dim=-1)

    rot = extr[:, :3, :3]
    trans = extr[:, :3, 3]
    pts_flat = pts_cam.reshape(bsz, height * width, 3)
    pts_world = (pts_flat - trans.unsqueeze(1)) @ rot
    return pts_world


def _build_gs_cfg(
    raft_encoder_dims: list[int],
    gs_encoder_dims: list[int],
    gs_decoder_dims: list[int],
    head_dim: int,
) -> SimpleNamespace:
    """构造 GSRegresser 所需的最小配置对象。"""
    return SimpleNamespace(
        raft=SimpleNamespace(encoder_dims=list(raft_encoder_dims)),
        gsnet=SimpleNamespace(
            encoder_dims=list(gs_encoder_dims),
            decoder_dims=list(gs_decoder_dims),
            parm_head_dim=head_dim,
        ),
    )


class PAGSplat(nn.Module):
    """
    简化版 PAG-Splat：
      DA3 prior -> scale align -> GPS+ GSRegresser -> RGB Gaussian render
    """

    def __init__(
        self,
        da3_net: nn.Module,
        embed_dim: int = 768,
        feat_channels: int = 256,
        feat_stride: int = 4,
        feat_layer: int = 8,
        mlp_hidden: int = 256,
        gs_cfg: SimpleNamespace | None = None,
        t12_mode: str = "log_len",
    ) -> None:
        super().__init__()

        if gs_cfg is None:
            gs_cfg = _build_gs_cfg(
                raft_encoder_dims=[32, 48, 96],
                gs_encoder_dims=[32, 48, 96],
                gs_decoder_dims=[48, 64, 96],
                head_dim=32,
            )

        self.prior_extractor = MonoPriorExtractor(
            da3_net=da3_net,
            embed_dim=embed_dim,
            feat_channels=feat_channels,
            feat_stride=feat_stride,
            feat_layer=feat_layer,
        )
        self.scale_align = ScaleAlignmentMLP(
            hidden_dim=mlp_hidden,
            t12_mode=t12_mode,
            use_img_feat=True,
            feat_channels=feat_channels,
        )
        self.img_encoder = UnetExtractor(
            in_channel=3,
            encoder_dim=list(gs_cfg.raft.encoder_dims),
        )
        self.gs_parm_regresser = GSRegresser(gs_cfg, rgb_dim=3, depth_dim=1)

    def _run_scale_align(
        self,
        img_self: torch.Tensor,
        img_other: torch.Tensor,
        intr_self: torch.Tensor,
        intr_other: torch.Tensor,
        extr_self: torch.Tensor,
        extr_other: torch.Tensor,
        f_mono_self: torch.Tensor,
        f_mono_other: torch.Tensor,
        d_rel_self: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """单视图尺度对齐。"""
        _, _, height, width = img_self.shape
        d_metric, log_s, _ = self.scale_align(
            intr_self,
            intr_other,
            extr_self,
            extr_other,
            d_rel_self,
            height,
            width,
            f_self=f_mono_self,
            f_other=f_mono_other,
        )
        return d_metric, log_s

    def forward(
        self,
        data: Dict,
        is_train: bool = True,
    ) -> Dict:
        del is_train

        lm = data["lmain"]
        rm = data["rmain"]

        img1 = lm["img"]
        img2 = rm["img"]
        intr1 = lm["intr"]
        intr2 = rm["intr"]
        extr1 = lm["extr"]
        extr2 = rm["extr"]

        bsz, _, height, width = img1.shape

        f_mono1, f_mono2, d_rel1, d_rel2 = self.prior_extractor(img1, img2)

        d_metric1, log_s1 = self._run_scale_align(
            img1, img2, intr1, intr2, extr1, extr2, f_mono1, f_mono2, d_rel1
        )
        d_metric2, log_s2 = self._run_scale_align(
            img2, img1, intr2, intr1, extr2, extr1, f_mono2, f_mono1, d_rel2
        )

        lr_img = torch.cat([img1, img2], dim=0)
        lr_depth = torch.cat([d_metric1, d_metric2], dim=0)
        lr_img_feat = self.img_encoder(lr_img)

        rot_maps, scale_maps, opacity_maps, depth_res = self.gs_parm_regresser(
            lr_img, lr_depth, lr_img_feat
        )
        l_resdepth, r_resdepth = torch.split(depth_res, [bsz, bsz], dim=0)

        d_final1 = F.softplus(d_metric1 + l_resdepth) + 1e-3
        d_final2 = F.softplus(d_metric2 + r_resdepth) + 1e-3

        xyz1 = depth_to_pointcloud(d_final1, intr1, extr1)
        xyz2 = depth_to_pointcloud(d_final2, intr2, extr2)

        rot_l, rot_r = torch.split(rot_maps, [bsz, bsz], dim=0)
        scale_l, scale_r = torch.split(scale_maps, [bsz, bsz], dim=0)
        opacity_l, opacity_r = torch.split(opacity_maps, [bsz, bsz], dim=0)

        for view_key, rot_map, scale_map, opacity_map, xyz in [
            ("lmain", rot_l, scale_l, opacity_l, xyz1),
            ("rmain", rot_r, scale_r, opacity_r, xyz2),
        ]:
            view = data[view_key]
            view["xyz"] = xyz
            view["rot_maps"] = rot_map
            view["scale_maps"] = scale_map
            view["opacity_maps"] = opacity_map

            if "mask" in view:
                valid = view["mask"][:, :1] > 0.5
            else:
                valid = torch.ones(
                    bsz, 1, height, width, dtype=torch.bool, device=xyz.device
                )
            view["pts_valid"] = valid.view(bsz, -1)

        data["metric_depth_l"] = d_metric1
        data["metric_depth_r"] = d_metric2
        data["final_depth_l"] = d_final1
        data["final_depth_r"] = d_final2
        data["log_scale_l"] = log_s1
        data["log_scale_r"] = log_s2
        data["novel_view"]["scale_regular"] = torch.mean(scale_maps)

        return data

    def count_parameters(self) -> dict[str, int]:
        """统计各子模块参数量。"""
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "prior_extractor (trainable proj)": count(self.prior_extractor),
            "scale_align_mlp": count(self.scale_align),
            "img_encoder": count(self.img_encoder),
            "gs_parm_regresser": count(self.gs_parm_regresser),
            "total_trainable": count(self),
        }


def detect_t12_mode(ckpt_path: str | None) -> str:
    """
    从 checkpoint 文件中自动检测 ScaleAlignmentMLP 的 t12 编码模式。
    仅用于同系列简化模型的 checkpoint。
    """
    if ckpt_path is None or not os.path.exists(ckpt_path):
        return "log_len"
    try:
        import torch as _torch

        ckpt = _torch.load(ckpt_path, map_location="cpu", weights_only=True)
        state = ckpt.get("network", ckpt)
        weight = state.get("scale_align.mlp.0.weight")
        if weight is not None:
            in_dim = weight.shape[1]
            mode = "norm" if in_dim in (13, 17) else "log_len"
            print(f"[PAGSplat] 检测到 scale_align 输入维度={in_dim}, t12_mode='{mode}'")
            return mode
    except Exception as exc:
        print(f"[PAGSplat] checkpoint 检测失败 ({exc})，使用默认 t12_mode='log_len'")
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
    raft_encoder_dims: list[int] | None = None,
) -> PAGSplat:
    """
    构建简化版 PAGSplat 模型。

    保留原工厂函数入口，内部回退到 GPS+ 风格 GS 参数回归。
    """
    del scale_max, gru_iters, gru_hidden_ch

    enc_dims = enc_dims or [32, 48, 96]
    dec_dims = dec_dims or [48, 64, 96]
    raft_encoder_dims = raft_encoder_dims or [32, 48, 96]

    if t12_mode is None:
        t12_mode = detect_t12_mode(ckpt_path)

    try:
        from depth_anything_3.api import DepthAnything3
    except ImportError as exc:
        raise ImportError(
            "未找到 depth_anything_3 包，请将 /data/sifang/Depth-Anything-3/src 加入 PYTHONPATH。"
        ) from exc

    if da3_checkpoint is not None:
        da3_api = DepthAnything3.from_pretrained(da3_checkpoint)
    else:
        da3_api = DepthAnything3(model_name=da3_model_name)

    da3_api = da3_api.to(device).eval()

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

    gs_cfg = _build_gs_cfg(
        raft_encoder_dims=raft_encoder_dims,
        gs_encoder_dims=enc_dims,
        gs_decoder_dims=dec_dims,
        head_dim=head_ch,
    )

    model = PAGSplat(
        da3_net=da3_api.model,
        embed_dim=embed_dim,
        feat_channels=feat_channels,
        feat_stride=feat_stride,
        feat_layer=feat_layer,
        mlp_hidden=mlp_hidden,
        gs_cfg=gs_cfg,
        t12_mode=t12_mode,
    ).to(device)
    return model
