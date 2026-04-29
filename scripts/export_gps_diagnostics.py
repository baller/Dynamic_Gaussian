#!/usr/bin/env python3
"""Export comparable GPS/Gaussian diagnostics for a single validation sample.

Run this script once per method in a clean Python process. It imports modules
from the requested repository root, restores the checkpoint, renders one novel
view, and writes intermediate tensors/visualizations for hole analysis.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--sample-name", required=True)
    parser.add_argument("--novel-id", type=int, default=2)
    parser.add_argument("--max-ply-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=1314)
    parser.add_argument("--scale-mult", type=float, default=1.0)
    parser.add_argument("--opacity-mult", type=float, default=1.0)
    parser.add_argument("--sub-opacity-mult", type=float, default=1.0)
    parser.add_argument("--drop-sub-gaussians", action="store_true")
    parser.add_argument(
        "--cvct-identity",
        choices=["auto", "true", "false"],
        default="auto",
        help="Only applies to StereoGS checkpoints with CVCT.",
    )
    parser.add_argument(
        "--skip-full-npz",
        action="store_true",
        help="Only save sampled PLY files, not full compressed tensor dumps.",
    )
    return parser.parse_args()


def setup_repo(repo_root: str) -> None:
    root = str(Path(repo_root).resolve())
    os.chdir(root)
    sys.path.insert(0, root)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def to_cuda_tree(obj: Any) -> Any:
    if torch.is_tensor(obj):
        return obj.cuda(non_blocking=True)
    if isinstance(obj, dict):
        return {k: to_cuda_tree(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cuda_tree(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cuda_tree(v) for v in obj)
    return obj


def tensor_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().float().cpu().numpy()


def image_01_to_u8(t: torch.Tensor) -> np.ndarray:
    arr = tensor_to_numpy(t[0].clamp(0, 1).permute(1, 2, 0))
    return (arr * 255.0).round().astype(np.uint8)


def image_m11_to_u8(t: torch.Tensor) -> np.ndarray:
    return image_01_to_u8(t * 0.5 + 0.5)


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    cv2.imwrite(str(path), rgb[:, :, ::-1])


def colorize_scalar(
    arr: np.ndarray,
    label: str,
    out_path: Path,
    cmap: int = cv2.COLORMAP_INFERNO,
    robust: bool = True,
) -> dict[str, float]:
    values = arr.astype(np.float32)
    finite = np.isfinite(values)
    if not finite.any():
        norm = np.zeros(values.shape, dtype=np.uint8)
        lo = hi = 0.0
    else:
        valid = values[finite]
        if robust:
            lo = float(np.percentile(valid, 1))
            hi = float(np.percentile(valid, 99))
        else:
            lo = float(valid.min())
            hi = float(valid.max())
        norm = np.clip((values - lo) / (hi - lo + 1e-8), 0, 1)
        norm = (norm * 255.0).astype(np.uint8)
    img = cv2.applyColorMap(norm, cmap)
    text = f"{label} [{lo:.4g}, {hi:.4g}]"
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
    return {"min": float(np.nanmin(values)), "max": float(np.nanmax(values)), "p01": lo, "p99": hi}


def quantiles(arr: np.ndarray) -> dict[str, float]:
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {}
    qs = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
    return {f"p{q:02d}": float(np.percentile(arr, q)) for q in qs}


def sample_indices(n: int, max_points: int, seed: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, max_points, replace=False))


def write_point_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray, max_points: int, seed: int) -> int:
    idx = sample_indices(xyz.shape[0], max_points, seed)
    pts = xyz[idx].astype(np.float32)
    colors = np.clip(rgb[idx] * 255.0, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(pts, colors):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")
    return int(pts.shape[0])


def write_gaussian_ply(
    path: Path,
    xyz: np.ndarray,
    rgb: np.ndarray,
    rot: np.ndarray,
    scale: np.ndarray,
    opacity: np.ndarray,
    max_points: int,
    seed: int,
) -> int:
    idx = sample_indices(xyz.shape[0], max_points, seed)
    pts = xyz[idx].astype(np.float32)
    colors = np.clip(rgb[idx] * 255.0, 0, 255).astype(np.uint8)
    rr = rot[idx].astype(np.float32)
    ss = scale[idx].astype(np.float32)
    oo = opacity[idx].reshape(-1).astype(np.float32)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("property float opacity\n")
        f.write("property float scale_0\nproperty float scale_1\nproperty float scale_2\n")
        f.write("property float rot_0\nproperty float rot_1\nproperty float rot_2\nproperty float rot_3\n")
        f.write("end_header\n")
        for p, c, o, s, r in zip(pts, colors, oo, ss, rr):
            f.write(
                f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                f"{int(c[0])} {int(c[1])} {int(c[2])} {o:.6f} "
                f"{s[0]:.8f} {s[1]:.8f} {s[2]:.8f} "
                f"{r[0]:.8f} {r[1]:.8f} {r[2]:.8f} {r[3]:.8f}\n"
            )
    return int(pts.shape[0])


def project_to_novel(
    xyz: np.ndarray,
    extr: np.ndarray,
    fovx: float,
    fovy: float,
    height: int,
    width: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fx = width / (2.0 * math.tan(fovx * 0.5))
    fy = height / (2.0 * math.tan(fovy * 0.5))
    cx = width / 2.0
    cy = height / 2.0
    r = extr[:3, :3].astype(np.float32)
    t = extr[:3, 3].astype(np.float32)
    pts_cam = xyz.astype(np.float32) @ r.T + t[None, :]
    z = pts_cam[:, 2]
    z_safe = np.clip(z, 1e-6, None)
    u = (fx * pts_cam[:, 0] / z_safe + cx).astype(np.int64)
    v = (fy * pts_cam[:, 1] / z_safe + cy).astype(np.int64)
    valid = (z > 0.1) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return u, v, valid


def save_projection_density(
    path_prefix: Path,
    xyz: np.ndarray,
    extr: np.ndarray,
    fovx: float,
    fovy: float,
    height: int,
    width: int,
) -> dict[str, float]:
    u, v, valid = project_to_novel(xyz, extr, fovx, fovy, height, width)
    density = np.zeros((height, width), dtype=np.float32)
    np.add.at(density, (v[valid], u[valid]), 1.0)
    coverage = density > 0
    den_log = np.log1p(density)
    den_norm = (den_log / (den_log.max() + 1e-8) * 255.0).astype(np.uint8)
    cv2.imwrite(str(path_prefix.with_name(path_prefix.name + "_density.png")), cv2.applyColorMap(den_norm, cv2.COLORMAP_PLASMA))
    cv2.imwrite(str(path_prefix.with_name(path_prefix.name + "_coverage.png")), (coverage.astype(np.uint8) * 255))
    np.save(path_prefix.with_name(path_prefix.name + "_density.npy"), density)
    return {
        "projected_points": int(valid.sum()),
        "total_points": int(xyz.shape[0]),
        "projection_valid_ratio": float(valid.mean()) if xyz.shape[0] else 0.0,
        "covered_pixels": int(coverage.sum()),
        "coverage_ratio": float(coverage.mean()),
        "density_mean": float(density.mean()),
        "density_p99": float(np.percentile(density, 99)),
        "density_max": float(density.max()),
    }


def load_cfg(config_path: str):
    from config.stereo_human_config import ConfigStereoHuman

    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(config_path)
    cfg = cfg_obj.get_cfg()
    cfg.defrost()
    cfg.freeze()
    return cfg


def load_model(cfg, ckpt_path: str):
    from lib.network import RtStereoHumanModel

    model = RtStereoHumanModel(cfg, with_gs_render=True).cuda()
    ckpt = torch.load(ckpt_path, map_location="cuda", weights_only=False)
    payload = ckpt["network"] if isinstance(ckpt, dict) and "network" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(payload, strict=False)
    step = int(ckpt.get("total_steps", -1)) if isinstance(ckpt, dict) else -1
    return model.eval(), step, missing, unexpected


def configure_cvct(model, cfg, step: int, mode: str) -> dict[str, Any]:
    out: dict[str, Any] = {"requested": mode, "applied": None, "blend": None}
    stereo_model = getattr(model, "stereo_gs_model", None)
    if stereo_model is None or not hasattr(stereo_model, "set_cvct_identity"):
        return out
    if mode == "true":
        identity = True
        blend = 1.0
    elif mode == "false":
        identity = False
        blend = 0.0
    else:
        wcvct = getattr(cfg, "wcvct", None)
        phase2_end = getattr(getattr(wcvct, "schedule", None), "phase2_end", 10**18)
        warmup = getattr(getattr(wcvct, "schedule", None), "cvct_warmup_steps", 2000)
        identity = step < phase2_end
        blend = 1.0 if identity else max(0.0, 1.0 - (step - phase2_end) / max(1, warmup))
    stereo_model.set_cvct_identity(identity)
    if hasattr(stereo_model, "set_cvct_blend"):
        stereo_model.set_cvct_blend(blend)
    out.update({"applied": bool(identity), "blend": float(blend)})
    return out


def load_sample(cfg, sample_name: str, novel_id: int):
    from lib.human_loader import StereoHumanDataset

    dataset = StereoHumanDataset(cfg.dataset, phase="val")
    if sample_name not in dataset.sample_list:
        similar = [s for s in dataset.sample_list if sample_name[:4] in s][:10]
        raise ValueError(f"sample {sample_name!r} not found. Similar: {similar}")
    idx = dataset.sample_list.index(sample_name)
    sample = dataset.get_item(idx, novel_id=[novel_id])
    batch = default_collate([sample])
    return to_cuda_tree(batch)


def apply_inference_perturbations(data, args: argparse.Namespace) -> dict[str, Any]:
    changes = {
        "scale_mult": float(args.scale_mult),
        "opacity_mult": float(args.opacity_mult),
        "sub_opacity_mult": float(args.sub_opacity_mult),
        "drop_sub_gaussians": bool(args.drop_sub_gaussians),
    }
    for view in ("lmain", "rmain"):
        if args.scale_mult != 1.0 and "scale_maps" in data[view]:
            data[view]["scale_maps"] = data[view]["scale_maps"] * args.scale_mult
        if args.opacity_mult != 1.0 and "opacity_maps" in data[view]:
            data[view]["opacity_maps"] = (data[view]["opacity_maps"] * args.opacity_mult).clamp(0, 1)
        if "sub_opacity" in data[view]:
            if args.sub_opacity_mult != 1.0:
                data[view]["sub_opacity"] = (data[view]["sub_opacity"] * args.sub_opacity_mult).clamp(0, 1)
            if args.drop_sub_gaussians:
                data[view]["sub_opacity"] = torch.zeros_like(data[view]["sub_opacity"])
                data[view]["sub_valid"] = torch.zeros_like(data[view]["sub_valid"]).bool()
    return changes


def render_data(model, cfg, data, args: argparse.Namespace):
    from lib.GaussianRender import pts2render

    with torch.no_grad():
        data, _, metrics = model(data, is_train=False)
        perturbations = apply_inference_perturbations(data, args)
        use_cags = bool(getattr(getattr(cfg, "stereo_gs", None), "use_cags", False))
        if use_cags:
            from lib.GaussianRender import pts2render_cags

            data = pts2render_cags(data, bg_color=cfg.dataset.bg_color)
        else:
            data = pts2render(data, bg_color=cfg.dataset.bg_color)
        stereo_model = getattr(model, "stereo_gs_model", None)
        if bool(getattr(getattr(cfg, "stereo_gs", None), "use_post_refine", False)) and stereo_model is not None:
            data = stereo_model.refine_rendered(data)
    return data, metrics or {}, perturbations


def get_view_arrays(data: dict[str, Any], view: str) -> dict[str, np.ndarray]:
    v = data[view]
    valid = v["pts_valid"][0].detach().bool()
    img = v.get("img_orig", v["img"])
    rgb = (img[0] * 0.5 + 0.5).clamp(0, 1).permute(1, 2, 0).reshape(-1, 3)
    rot = v["rot_maps"][0].permute(1, 2, 0).reshape(-1, 4)
    scale = v["scale_maps"][0].permute(1, 2, 0).reshape(-1, 3)
    opacity = v["opacity_maps"][0].permute(1, 2, 0).reshape(-1, 1)
    return {
        "xyz": tensor_to_numpy(v["xyz"][0][valid]),
        "rgb": tensor_to_numpy(rgb[valid]),
        "rot": tensor_to_numpy(rot[valid]),
        "scale": tensor_to_numpy(scale[valid]),
        "opacity": tensor_to_numpy(opacity[valid]),
        "valid_mask": tensor_to_numpy(valid.float()).astype(np.uint8),
    }


def get_sub_arrays(data: dict[str, Any], view: str) -> dict[str, np.ndarray] | None:
    v = data[view]
    if "sub_xyz" not in v:
        return None
    valid = v["sub_valid"][0].detach().bool()
    return {
        "xyz": tensor_to_numpy(v["sub_xyz"][0][valid]),
        "rgb": tensor_to_numpy((v["sub_rgb"][0][valid] * 0.5 + 0.5).clamp(0, 1)),
        "rot": tensor_to_numpy(v["sub_rot"][0][valid]),
        "scale": tensor_to_numpy(v["sub_scale"][0][valid]),
        "opacity": tensor_to_numpy(v["sub_opacity"][0][valid]),
        "valid_mask": tensor_to_numpy(valid.float()).astype(np.uint8),
    }


def save_method_outputs(args: argparse.Namespace, cfg, data, stats: dict[str, Any]) -> None:
    out = ensure_dir(args.out)
    for sub in ["images", "maps", "npz", "ply", "projection"]:
        ensure_dir(out / sub)

    lm = data["lmain"]
    rm = data["rmain"]
    nv = data["novel_view"]
    render = nv["img_pred"]
    gt = nv["img"].cuda() if not nv["img"].is_cuda else nv["img"]
    h, w = render.shape[-2:]

    input_l = image_m11_to_u8(lm.get("img_orig", lm["img"]))
    input_r = image_m11_to_u8(rm.get("img_orig", rm["img"]))
    render_u8 = image_01_to_u8(render)
    gt_u8 = image_01_to_u8(gt)
    diff = np.clip(np.abs(render_u8.astype(np.float32) - gt_u8.astype(np.float32)) * 3.0, 0, 255).astype(np.uint8)
    hole = ((render_u8.mean(axis=2) < 18) & (gt_u8.mean(axis=2) > 35)).astype(np.uint8)

    save_rgb(out / "images" / "input_l.png", input_l)
    save_rgb(out / "images" / "input_r.png", input_r)
    save_rgb(out / "images" / "render.png", render_u8)
    save_rgb(out / "images" / "gt.png", gt_u8)
    save_rgb(out / "images" / "diff_x3.png", diff)
    cv2.imwrite(str(out / "images" / "hole_candidates.png"), hole * 255)

    overview_top = np.concatenate([input_l, input_r], axis=1)
    overview_bottom = np.concatenate([render_u8, gt_u8, diff], axis=1)
    pad = overview_bottom.shape[1] - overview_top.shape[1]
    if pad > 0:
        overview_top = np.pad(overview_top, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), constant_values=255)
    save_rgb(out / "images" / "overview.png", np.concatenate([overview_top, overview_bottom], axis=0))

    stats["image"] = {
        "height": int(h),
        "width": int(w),
        "mae_rgb_0_255": float(np.abs(render_u8.astype(np.float32) - gt_u8.astype(np.float32)).mean()),
        "hole_candidate_pixels": int(hole.sum()),
        "hole_candidate_ratio": float(hole.mean()),
    }

    extras = data.get("_stereo_gs_extras", {})
    view_stats: dict[str, Any] = {}
    all_xyz = []
    for view_key, label in [("lmain", "left"), ("rmain", "right")]:
        view_dir = out / "npz" / label
        ensure_dir(view_dir)
        arrays = get_view_arrays(data, view_key)
        sub_arrays = get_sub_arrays(data, view_key)

        if not args.skip_full_npz:
            np.savez_compressed(view_dir / "base_gaussians_full.npz", **arrays)
            if sub_arrays is not None:
                np.savez_compressed(view_dir / "sub_gaussians_full.npz", **sub_arrays)

        write_point_ply(out / "ply" / f"{label}_base_points_sampled.ply", arrays["xyz"], arrays["rgb"], args.max_ply_points, args.seed)
        write_gaussian_ply(
            out / "ply" / f"{label}_base_gaussians_sampled.ply",
            arrays["xyz"],
            arrays["rgb"],
            arrays["rot"],
            arrays["scale"],
            arrays["opacity"],
            args.max_ply_points,
            args.seed,
        )
        all_xyz.append(arrays["xyz"])

        if sub_arrays is not None and sub_arrays["xyz"].shape[0] > 0:
            write_point_ply(out / "ply" / f"{label}_sub_points_sampled.ply", sub_arrays["xyz"], sub_arrays["rgb"], args.max_ply_points, args.seed + 1)
            write_gaussian_ply(
                out / "ply" / f"{label}_sub_gaussians_sampled.ply",
                sub_arrays["xyz"],
                sub_arrays["rgb"],
                sub_arrays["rot"],
                sub_arrays["scale"],
                sub_arrays["opacity"],
                args.max_ply_points,
                args.seed + 1,
            )
            all_xyz.append(sub_arrays["xyz"])

        depth = tensor_to_numpy(data[view_key]["depth"][0, 0])
        opacity_map = tensor_to_numpy(data[view_key]["opacity_maps"][0, 0])
        scale_map = tensor_to_numpy(data[view_key]["scale_maps"][0].mean(dim=0))
        np.save(view_dir / "depth.npy", depth)
        np.save(view_dir / "opacity.npy", opacity_map)
        np.save(view_dir / "scale_mean.npy", scale_map)
        colorize_scalar(depth, f"{label} depth", out / "maps" / f"{label}_depth.png")
        colorize_scalar(opacity_map, f"{label} opacity", out / "maps" / f"{label}_opacity.png", cv2.COLORMAP_PLASMA, robust=False)
        colorize_scalar(scale_map, f"{label} scale_mean", out / "maps" / f"{label}_scale_mean.png", cv2.COLORMAP_VIRIDIS, robust=False)
        cv2.imwrite(str(out / "maps" / f"{label}_valid_mask.png"), arrays["valid_mask"].reshape(h, w) * 255)

        view_stats[label] = {
            "base_points": int(arrays["xyz"].shape[0]),
            "base_opacity": quantiles(arrays["opacity"].reshape(-1)),
            "base_scale_mean": quantiles(arrays["scale"].mean(axis=1)),
            "depth": quantiles(depth.reshape(-1)),
        }
        if sub_arrays is not None:
            view_stats[label]["sub_points"] = int(sub_arrays["xyz"].shape[0])
            view_stats[label]["sub_opacity"] = quantiles(sub_arrays["opacity"].reshape(-1))
            view_stats[label]["sub_scale_mean"] = quantiles(sub_arrays["scale"].mean(axis=1))

        conf_key = "confidence_left" if view_key == "lmain" else "confidence_right"
        if conf_key in extras:
            conf = tensor_to_numpy(extras[conf_key][0, 0])
            np.save(view_dir / "confidence.npy", conf)
            colorize_scalar(conf, f"{label} confidence", out / "maps" / f"{label}_confidence.png", cv2.COLORMAP_VIRIDIS, robust=False)
            view_stats[label]["confidence"] = quantiles(conf.reshape(-1))

        sw_key = "split_weights_left" if view_key == "lmain" else "split_weights_right"
        if sw_key in extras:
            sw = tensor_to_numpy(extras[sw_key][0])
            sw_sum = sw.sum(axis=0)
            np.save(view_dir / "split_weights.npy", sw)
            colorize_scalar(sw_sum, f"{label} split_sum", out / "maps" / f"{label}_split_sum.png", cv2.COLORMAP_HOT, robust=False)
            view_stats[label]["split_sum"] = quantiles(sw_sum.reshape(-1))

    merged_xyz = np.concatenate(all_xyz, axis=0) if all_xyz else np.zeros((0, 3), np.float32)
    nv_extr = tensor_to_numpy(nv["extr"][0])
    fovx = float(nv["FovX"][0])
    fovy = float(nv["FovY"][0])
    proj_stats = save_projection_density(out / "projection" / "all_gaussians", merged_xyz, nv_extr, fovx, fovy, h, w)
    density = np.load(out / "projection" / "all_gaussians_density.npy")
    if hole.sum() > 0:
        proj_stats["density_on_hole_candidates_mean"] = float(density[hole.astype(bool)].mean())
        proj_stats["density_on_hole_candidates_p50"] = float(np.percentile(density[hole.astype(bool)], 50))
        proj_stats["density_on_hole_candidates_p90"] = float(np.percentile(density[hole.astype(bool)], 90))
    stats["views"] = view_stats
    stats["projection"] = proj_stats


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    setup_repo(args.repo_root)

    cfg = load_cfg(args.config)
    model, step, missing, unexpected = load_model(cfg, args.ckpt)
    cvct_state = configure_cvct(model, cfg, step, args.cvct_identity)
    data = load_sample(cfg, args.sample_name, args.novel_id)
    data, metrics, perturbations = render_data(model, cfg, data, args)

    stats: dict[str, Any] = {
        "method": args.method_name,
        "repo_root": str(Path(args.repo_root).resolve()),
        "config": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.ckpt).resolve()),
        "checkpoint_step": step,
        "sample_name": args.sample_name,
        "novel_id": args.novel_id,
        "cvct": cvct_state,
        "inference_perturbations": perturbations,
        "load_state": {
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "missing_first": [str(x) for x in missing[:20]],
            "unexpected_first": [str(x) for x in unexpected[:20]],
        },
        "metrics": {k: float(v) for k, v in metrics.items()} if isinstance(metrics, dict) else {},
    }
    save_method_outputs(args, cfg, data, stats)
    out = ensure_dir(args.out)
    with open(out / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps({"out": str(out), "method": args.method_name, "sample": args.sample_name}, indent=2))


if __name__ == "__main__":
    main()
