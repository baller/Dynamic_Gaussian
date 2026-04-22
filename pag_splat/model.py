"""
PAGSplat 主模型：DA3 + metric depth + coarse GS + wavelet-guided child splitting.
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
    uv_offsets: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    将 metric depth 转换为世界坐标点云。

    Args:
        depth:      (B, C, H, W)
        intr:       (B, 3, 3)
        extr:       (B, 3, 4)
        uv_offsets: (B, C, 2, H, W)，单位为像素；可选

    Returns:
        pts_world:  (B, C*H*W, 3)
    """
    bsz, channels, height, width = depth.shape
    device = depth.device

    fx = intr[:, 0, 0].view(bsz, 1, 1, 1)
    fy = intr[:, 1, 1].view(bsz, 1, 1, 1)
    cx = intr[:, 0, 2].view(bsz, 1, 1, 1)
    cy = intr[:, 1, 2].view(bsz, 1, 1, 1)

    v_base = torch.arange(height, device=device, dtype=torch.float32).view(1, 1, height, 1)
    u_base = torch.arange(width, device=device, dtype=torch.float32).view(1, 1, 1, width)
    u = u_base.expand(bsz, channels, height, width)
    v = v_base.expand(bsz, channels, height, width)

    if uv_offsets is not None:
        u = u + uv_offsets[:, :, 0]
        v = v + uv_offsets[:, :, 1]

    z = depth
    x_cam = (u - cx) * z / fx
    y_cam = (v - cy) * z / fy
    pts_cam = torch.stack([x_cam, y_cam, z], dim=-1)  # (B, C, H, W, 3)

    rot = extr[:, :3, :3]
    trans = extr[:, :3, 3]
    pts_flat = pts_cam.reshape(bsz, channels * height * width, 3)
    pts_world = (pts_flat - trans.unsqueeze(1)) @ rot
    return pts_world


def _build_gs_cfg(
    raft_encoder_dims: list[int],
    gs_encoder_dims: list[int],
    gs_decoder_dims: list[int],
    head_dim: int,
) -> SimpleNamespace:
    return SimpleNamespace(
        raft=SimpleNamespace(encoder_dims=list(raft_encoder_dims)),
        gsnet=SimpleNamespace(
            encoder_dims=list(gs_encoder_dims),
            decoder_dims=list(gs_decoder_dims),
            parm_head_dim=head_dim,
        ),
    )


class ConvGNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        groups = max(1, out_ch // 8)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class WaveletFeatureDecomposer(nn.Module):
    """1-level Haar DWT on DA3 mono features."""

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.shape[-2] % 2 != 0:
            x = F.pad(x, (0, 0, 0, 1), mode="replicate")
        if x.shape[-1] % 2 != 0:
            x = F.pad(x, (0, 1, 0, 0), mode="replicate")

        x00 = x[:, :, 0::2, 0::2]
        x01 = x[:, :, 0::2, 1::2]
        x10 = x[:, :, 1::2, 0::2]
        x11 = x[:, :, 1::2, 1::2]

        ll = 0.5 * (x00 + x01 + x10 + x11)
        lh = 0.5 * (x00 - x01 + x10 - x11)
        hl = 0.5 * (x00 + x01 - x10 - x11)
        hh = 0.5 * (x00 - x01 - x10 + x11)
        return ll, torch.cat([lh, hl, hh], dim=1)


class SplitProposalHead(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, split_k_max: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvGNReLU(in_ch, hidden_ch),
            ConvGNReLU(hidden_ch, hidden_ch),
        )
        self.score_head = nn.Conv2d(hidden_ch, 1, kernel_size=1)
        self.weight_head = nn.Conv2d(hidden_ch, split_k_max, kernel_size=1)
        self.orientation_head = nn.Conv2d(hidden_ch, 3, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.net(x)
        split_score = torch.sigmoid(self.score_head(feat))
        child_weight_logits = self.weight_head(feat)
        orientation_hint = torch.softmax(self.orientation_head(feat), dim=1)
        return split_score, child_weight_logits, orientation_hint


class HierarchicalRefinementHead(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, split_k_max: int) -> None:
        super().__init__()
        self.split_k_max = split_k_max
        self.net = nn.Sequential(
            ConvGNReLU(in_ch, hidden_ch),
            ConvGNReLU(hidden_ch, hidden_ch),
        )
        self.delta_uv = nn.Conv2d(hidden_ch, 2 * split_k_max, kernel_size=1)
        self.delta_depth = nn.Conv2d(hidden_ch, split_k_max, kernel_size=1)
        self.delta_scale = nn.Conv2d(hidden_ch, 3 * split_k_max, kernel_size=1)
        self.delta_rot = nn.Conv2d(hidden_ch, 4 * split_k_max, kernel_size=1)
        self.delta_opacity = nn.Conv2d(hidden_ch, split_k_max, kernel_size=1)
        self.delta_color = nn.Conv2d(hidden_ch, 3 * split_k_max, kernel_size=1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.net(x)
        return {
            "delta_uv": torch.tanh(self.delta_uv(feat)) * 0.5,
            "delta_depth": torch.tanh(self.delta_depth(feat)),
            "delta_scale": self.delta_scale(feat),
            "delta_rot": self.delta_rot(feat),
            "delta_opacity": self.delta_opacity(feat),
            "delta_color": torch.tanh(self.delta_color(feat)) * 0.15,
        }


class PAGSplat(nn.Module):
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
        split_k_max: int = 4,
        split_score_thresh: float = 0.20,
        child_weight_thresh: float = 0.10,
        split_topk_ratio: float = 0.01,
    ) -> None:
        super().__init__()

        if gs_cfg is None:
            gs_cfg = _build_gs_cfg(
                raft_encoder_dims=[32, 48, 96],
                gs_encoder_dims=[32, 48, 96],
                gs_decoder_dims=[48, 64, 96],
                head_dim=32,
            )

        self.split_k_max = split_k_max
        self.split_score_thresh = split_score_thresh
        self.child_weight_thresh = child_weight_thresh
        self.split_topk_ratio = split_topk_ratio
        self._runtime_flags: dict[str, bool] = {}

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

        rgb_dim = int(gs_cfg.raft.encoder_dims[0])
        self.wavelet = WaveletFeatureDecomposer()
        self.low_reduce = nn.Conv2d(feat_channels, 64, kernel_size=1)
        self.high_reduce = nn.Conv2d(feat_channels * 3, 64, kernel_size=1)
        self.rgb_reduce = nn.Conv2d(rgb_dim, 32, kernel_size=1)
        proposal_in_ch = 64 + 64 + 32 + 2
        refine_in_ch = proposal_in_ch + 4
        self.split_proposal = SplitProposalHead(proposal_in_ch, 128, split_k_max)
        self.refine_head = HierarchicalRefinementHead(refine_in_ch, 128, split_k_max)

    def _run_scale_align(
        self,
        intr_self: torch.Tensor,
        intr_other: torch.Tensor,
        extr_self: torch.Tensor,
        extr_other: torch.Tensor,
        f_mono_self: torch.Tensor,
        f_mono_other: torch.Tensor,
        d_rel_self: torch.Tensor,
        height: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

    @staticmethod
    def _select_split_mask(
        split_score: torch.Tensor,
        valid_map: torch.Tensor,
        topk_ratio: float,
        thresh: float,
        training: bool,
        force_topk: bool = False,
    ) -> torch.Tensor:
        """在可用区域内选择少量父像素进入 child 分裂。"""
        bsz, _, height, width = split_score.shape
        score_flat = split_score.view(bsz, -1)
        valid_flat = valid_map.view(bsz, -1).bool()
        out = []
        for batch_idx in range(bsz):
            mask = torch.zeros_like(valid_flat[batch_idx])
            valid_idx = torch.nonzero(valid_flat[batch_idx], as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                out.append(mask)
                continue
            scores = score_flat[batch_idx, valid_idx]
            if training or force_topk:
                topk = max(1, int(valid_idx.numel() * topk_ratio))
                topk = min(topk, valid_idx.numel())
                chosen = valid_idx[scores.topk(topk).indices]
                mask[chosen] = True
            else:
                keep = valid_idx[scores > thresh]
                if keep.numel() == 0:
                    keep = valid_idx[scores.topk(1).indices]
                mask[keep] = True
            out.append(mask)
        return torch.stack(out, dim=0).view(bsz, 1, height, width)

    def _refine_one_view(
        self,
        img: torch.Tensor,
        intr: torch.Tensor,
        extr: torch.Tensor,
        rgb_feat: torch.Tensor,
        f_mono: torch.Tensor,
        d_final: torch.Tensor,
        rot_map: torch.Tensor,
        scale_map: torch.Tensor,
        opacity_map: torch.Tensor,
        valid_map: torch.Tensor,
        warmup_alpha: float,
        is_train: bool,
    ) -> dict[str, torch.Tensor]:
        bsz, _, height, width = img.shape
        parent_rgb = img * 0.5 + 0.5
        parent_valid = valid_map.bool()
        force_topk_eval = bool(not is_train and self._runtime_flags.get("force_topk_split_eval", False))

        f_low, f_high = self.wavelet(f_mono)
        low_up = F.interpolate(self.low_reduce(f_low), size=(height, width), mode="bilinear", align_corners=False)
        high_up = F.interpolate(self.high_reduce(f_high), size=(height, width), mode="bilinear", align_corners=False)
        rgb_up = F.interpolate(self.rgb_reduce(rgb_feat), size=(height, width), mode="bilinear", align_corners=False)
        prior_map = F.interpolate(
            f_high.abs().mean(dim=1, keepdim=True),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        prior_map = prior_map / (prior_map.amax(dim=(-2, -1), keepdim=True) + 1e-6)

        proposal_input = torch.cat([low_up, high_up, rgb_up, d_final, opacity_map], dim=1)
        split_score, child_weight_logits, orientation_hint = self.split_proposal(proposal_input)
        child_weight_maps = torch.softmax(child_weight_logits, dim=1)
        split_gate = split_score * float(warmup_alpha)

        refine_input = torch.cat([proposal_input, split_gate, orientation_hint], dim=1)
        refine_out = self.refine_head(refine_input)

        k_max = self.split_k_max
        delta_uv = refine_out["delta_uv"].view(bsz, k_max, 2, height, width)
        delta_depth = refine_out["delta_depth"].view(bsz, k_max, height, width)
        delta_scale = refine_out["delta_scale"].view(bsz, k_max, 3, height, width)
        delta_rot = refine_out["delta_rot"].view(bsz, k_max, 4, height, width)
        delta_opacity = refine_out["delta_opacity"].view(bsz, k_max, height, width)
        delta_color = refine_out["delta_color"].view(bsz, k_max, 3, height, width)

        split_parent_mask = self._select_split_mask(
            split_score,
            parent_valid,
            self.split_topk_ratio,
            self.split_score_thresh,
            training=is_train,
            force_topk=force_topk_eval,
        )
        split_parent_mask_k = split_parent_mask.expand(-1, k_max, -1, -1)

        parent_depth_k = d_final.expand(-1, k_max, -1, -1)
        parent_scale_k = scale_map.unsqueeze(1).expand(-1, k_max, -1, -1, -1)
        parent_rot_k = rot_map.unsqueeze(1).expand(-1, k_max, -1, -1, -1)
        parent_rgb_k = parent_rgb.unsqueeze(1).expand(-1, k_max, -1, -1, -1)
        parent_opacity_k = opacity_map.expand(-1, k_max, -1, -1)

        child_depth = torch.clamp_min(parent_depth_k * (1.0 + 0.15 * delta_depth), 1e-3)
        child_scale_ratio = 0.25 + 0.75 * torch.sigmoid(delta_scale)
        child_scale = parent_scale_k * child_scale_ratio
        child_rot = F.normalize(parent_rot_k + delta_rot, dim=2)
        child_rgb = torch.clamp(parent_rgb_k + delta_color, 0.0, 1.0)
        child_opacity = parent_opacity_k * split_gate.expand_as(parent_opacity_k) * child_weight_maps
        child_opacity = child_opacity * torch.sigmoid(delta_opacity)

        child_weight_mask = child_weight_maps > self.child_weight_thresh
        if is_train:
            child_valid = split_parent_mask_k
        else:
            child_valid = split_parent_mask_k & child_weight_mask

        child_xyz = depth_to_pointcloud(child_depth, intr, extr, uv_offsets=delta_uv)
        child_rgb_flat = child_rgb.permute(0, 1, 3, 4, 2).reshape(bsz, -1, 3)
        child_rot_flat = child_rot.permute(0, 1, 3, 4, 2).reshape(bsz, -1, 4)
        child_scale_flat = child_scale.permute(0, 1, 3, 4, 2).reshape(bsz, -1, 3)
        child_opacity_flat = child_opacity.reshape(bsz, -1, 1)
        child_valid_flat = child_valid.reshape(bsz, -1)

        return {
            "split_score_maps": split_score,
            "split_prior_maps": prior_map,
            "child_weight_maps": child_weight_maps,
            "orientation_hint_maps": orientation_hint,
            "child_delta_uv_maps": refine_out["delta_uv"],
            "child_depth_res_maps": refine_out["delta_depth"],
            "child_scale_ratio_maps": child_scale_ratio.view(bsz, -1, height, width),
            "child_rot_delta_maps": refine_out["delta_rot"],
            "child_color_res_maps": refine_out["delta_color"],
            "child_xyz": child_xyz,
            "child_rgb": child_rgb_flat,
            "child_rot": child_rot_flat,
            "child_scale": child_scale_flat,
            "child_opacity": child_opacity_flat,
            "child_valid": child_valid_flat,
            "child_uv_offset_maps": refine_out["delta_uv"],
            "child_parent_mask": split_parent_mask,
            "child_opacity_maps": child_opacity,
        }

    def forward(
        self,
        data: Dict,
        is_train: bool = True,
    ) -> Dict:
        lm = data["lmain"]
        rm = data["rmain"]

        img1 = lm["img"]
        img2 = rm["img"]
        intr1 = lm["intr"]
        intr2 = rm["intr"]
        extr1 = lm["extr"]
        extr2 = rm["extr"]

        bsz, _, height, width = img1.shape
        warmup_alpha = float(data.get("_refine_warmup_alpha", 1.0))
        self._runtime_flags = {
            "force_topk_split_eval": bool(data.get("_force_topk_split_eval", False)),
        }

        f_mono1, f_mono2, d_rel1, d_rel2 = self.prior_extractor(img1, img2)

        d_metric1, log_s1 = self._run_scale_align(
            intr1, intr2, extr1, extr2, f_mono1, f_mono2, d_rel1, height, width
        )
        d_metric2, log_s2 = self._run_scale_align(
            intr2, intr1, extr2, extr1, f_mono2, f_mono1, d_rel2, height, width
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

        img_feat1, img_feat2, _ = lr_img_feat
        feat_l, feat_r = torch.split(img_feat1, [bsz, bsz], dim=0)
        rot_l, rot_r = torch.split(rot_maps, [bsz, bsz], dim=0)
        scale_l, scale_r = torch.split(scale_maps, [bsz, bsz], dim=0)
        opacity_l, opacity_r = torch.split(opacity_maps, [bsz, bsz], dim=0)

        for view_key, img, intr, extr, rgb_feat, f_mono, d_metric, d_final, log_s, rot_map, scale_map, opacity_map, xyz in [
            ("lmain", img1, intr1, extr1, feat_l, f_mono1, d_metric1, d_final1, log_s1, rot_l, scale_l, opacity_l, xyz1),
            ("rmain", img2, intr2, extr2, feat_r, f_mono2, d_metric2, d_final2, log_s2, rot_r, scale_r, opacity_r, xyz2),
        ]:
            view = data[view_key]
            if "mask" in view:
                valid_map = view["mask"][:, :1] > 0.5
            else:
                valid_map = torch.ones(
                    bsz, 1, height, width, dtype=torch.bool, device=img.device
                )

            view["xyz"] = xyz
            view["rot_maps"] = rot_map
            view["scale_maps"] = scale_map
            view["opacity_maps"] = opacity_map
            view["pts_valid"] = valid_map.view(bsz, -1)

            refine = self._refine_one_view(
                img=img,
                intr=intr,
                extr=extr,
                rgb_feat=rgb_feat,
                f_mono=f_mono,
                d_final=d_final,
                rot_map=rot_map,
                scale_map=scale_map,
                opacity_map=opacity_map,
                valid_map=valid_map,
                warmup_alpha=warmup_alpha,
                is_train=is_train,
            )
            view.update(refine)
            view["metric_depth"] = d_metric
            view["final_depth"] = d_final
            view["log_scale"] = log_s

        data["metric_depth_l"] = d_metric1
        data["metric_depth_r"] = d_metric2
        data["final_depth_l"] = d_final1
        data["final_depth_r"] = d_final2
        data["log_scale_l"] = log_s1
        data["log_scale_r"] = log_s2
        data["novel_view"]["scale_regular"] = torch.mean(scale_maps)
        return data

    def count_parameters(self) -> dict[str, int]:
        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "prior_extractor (trainable proj)": count(self.prior_extractor),
            "scale_align_mlp": count(self.scale_align),
            "img_encoder": count(self.img_encoder),
            "gs_parm_regresser": count(self.gs_parm_regresser),
            "split_proposal": count(self.split_proposal),
            "refine_head": count(self.refine_head),
            "total_trainable": count(self),
        }


def detect_t12_mode(ckpt_path: str | None) -> str:
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
    split_k_max: int = 4,
    split_score_thresh: float = 0.20,
    child_weight_thresh: float = 0.10,
    split_topk_ratio: float = 0.01,
) -> PAGSplat:
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
        split_k_max=split_k_max,
        split_score_thresh=split_score_thresh,
        child_weight_thresh=child_weight_thresh,
        split_topk_ratio=split_topk_ratio,
    ).to(device)
    return model
