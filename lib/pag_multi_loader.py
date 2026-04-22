"""
PAGMultiDataset — 支持两种数据格式、多数据集混合训练的 Dataset。

支持格式：
  1. mini 格式：parameter/{sample}/cameras.json
     - N 个相机，每个含 intrinsic / extrinsic
     - 每次随机从 N 个相机中选 3 个：2 输入视图 + 1 新视角 GT

  2. legacy 格式：parameter/{sample}/0_1.json + {id}_intrinsic/extrinsic.npy
     - source 固定为 0、1，novel 从配置的 novel_ids 中随机选 1 个

两种格式输出相同的 dict：
  {
    "name":       str
    "lmain":      {"img": (3,H,W) [-1,1], "intr": (3,3), "extr": (3,4)}
    "rmain":      {"img": (3,H,W) [-1,1], "intr": (3,3), "extr": (3,4)}
    "novel_view": {"img": (3,H,W) [0,1], "intr": (3,3), "extr": (3,4),
                   "FovX", "FovY", "width", "height",
                   "world_view_transform", "full_proj_transform",
                   "camera_center"}
  }
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import ConcatDataset, Dataset

from lib.gs_utils.graphics_utils import (
    focal2fov,
    getProjectionMatrix,
    getWorld2View2,
)

_DATA_PROCESS_DIR = Path(__file__).resolve().parent.parent / "data_process"
if str(_DATA_PROCESS_DIR) not in sys.path:
    sys.path.append(str(_DATA_PROCESS_DIR))

from colmap_read_model import read_cameras_binary, read_images_binary


# ──────────────────────────────────────────────────────────────────────────────
#  Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _read_img(path: str) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return img


def _resize_img(img: np.ndarray, target_hw: tuple[int, int] | None) -> np.ndarray:
    if target_hw is None:
        return img
    h, w = target_hw
    if img.shape[0] == h and img.shape[1] == w:
        return img
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)


def _scale_intr(intr: np.ndarray, src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> np.ndarray:
    """按分辨率缩放内参。src_hw/dst_hw = (H, W)。"""
    if src_hw == dst_hw:
        return intr.copy()
    src_h, src_w = src_hw
    dst_h, dst_w = dst_hw
    out = intr.copy()
    out[0] *= dst_w / max(src_w, 1)
    out[1] *= dst_h / max(src_h, 1)
    return out


def _center_crop_square(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int]]:
    """中心裁剪为正方形，返回裁剪图与 (x0, y0, size)。"""
    h, w = img.shape[:2]
    size = min(h, w)
    x0 = max((w - size) // 2, 0)
    y0 = max((h - size) // 2, 0)
    return img[y0:y0 + size, x0:x0 + size], (x0, y0, size)


def _crop_resize_with_intr(
    img: np.ndarray,
    intr: np.ndarray,
    crop_box: tuple[int, int, int] | None,
    target_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    对图像执行方形裁剪 + resize，并同步调整内参。
    crop_box = (x0, y0, size)。若为 None 则默认中心裁剪。
    """
    if crop_box is None:
        img, crop_box = _center_crop_square(img)
    else:
        x0, y0, size = crop_box
        img = img[y0:y0 + size, x0:x0 + size]
    x0, y0, size = crop_box

    intr_adj = intr.copy()
    intr_adj[0, 2] -= x0
    intr_adj[1, 2] -= y0
    dst_h, dst_w = target_hw
    intr_adj[0] *= dst_w / max(size, 1)
    intr_adj[1] *= dst_h / max(size, 1)
    img = _resize_img(img, target_hw)
    return img, intr_adj


def _sample_frame_idx(sample_name: str) -> int | None:
    parts = sample_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return int(parts[1])
    return None


def _colmap_camera_to_intr_dist(camera) -> tuple[np.ndarray, np.ndarray]:
    model = str(camera.model).upper()
    params = np.asarray(camera.params, dtype=np.float32)
    intr = np.eye(3, dtype=np.float32)
    dist = np.zeros(5, dtype=np.float32)

    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params[:3]
        intr[0, 0] = f
        intr[1, 1] = f
        intr[0, 2] = cx
        intr[1, 2] = cy
    elif model == "PINHOLE":
        fx, fy, cx, cy = params[:4]
        intr[0, 0] = fx
        intr[1, 1] = fy
        intr[0, 2] = cx
        intr[1, 2] = cy
    elif model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params[:4]
        intr[0, 0] = f
        intr[1, 1] = f
        intr[0, 2] = cx
        intr[1, 2] = cy
        dist[0] = k1
    elif model == "RADIAL":
        f, cx, cy, k1, k2 = params[:5]
        intr[0, 0] = f
        intr[1, 1] = f
        intr[0, 2] = cx
        intr[1, 2] = cy
        dist[0] = k1
        dist[1] = k2
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
        intr[0, 0] = fx
        intr[1, 1] = fy
        intr[0, 2] = cx
        intr[1, 2] = cy
        dist[:] = np.array([k1, k2, p1, p2, 0.0], dtype=np.float32)
    else:
        raise ValueError(f"暂不支持的 COLMAP 相机模型: {camera.model}")

    return intr, dist


def _extr_from_colmap_image(image) -> np.ndarray:
    extr = np.zeros((3, 4), dtype=np.float32)
    extr[:3, :3] = image.qvec2rotmat().astype(np.float32)
    extr[:3, 3] = np.asarray(image.tvec, dtype=np.float32)
    return extr


def _crop_intr(intr: np.ndarray, crop_box: tuple[int, int, int], target_hw: tuple[int, int]) -> np.ndarray:
    x0, y0, size = crop_box
    intr_out = intr.copy()
    intr_out[0, 2] -= x0
    intr_out[1, 2] -= y0
    dst_h, dst_w = target_hw
    intr_out[0] *= dst_w / max(size, 1)
    intr_out[1] *= dst_h / max(size, 1)
    return intr_out


def _rectify_raw_stereo_pair(
    img0: np.ndarray,
    intr0: np.ndarray,
    dist0: np.ndarray,
    extr0: np.ndarray,
    img1: np.ndarray,
    intr1: np.ndarray,
    dist1: np.ndarray,
    extr1: np.ndarray,
    crop_box: tuple[int, int, int] | None,
    target_hw: tuple[int, int],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    height, width = img0.shape[:2]
    r0, t0 = extr0[:3, :3], extr0[:3, 3:4]
    r1, t1 = extr1[:3, :3], extr1[:3, 3:4]
    rel_r = r1 @ r0.T
    rel_t = t1 - rel_r @ t0

    r_rect0, r_rect1, proj0, proj1, _, _, _ = cv2.stereoRectify(
        intr0,
        dist0,
        intr1,
        dist1,
        (width, height),
        rel_r,
        rel_t.reshape(3),
        flags=0,
    )

    map0_x, map0_y = cv2.initUndistortRectifyMap(
        intr0, dist0, r_rect0, proj0, (width, height), cv2.CV_32FC1
    )
    map1_x, map1_y = cv2.initUndistortRectifyMap(
        intr1, dist1, r_rect1, proj1, (width, height), cv2.CV_32FC1
    )

    img0_rect = cv2.remap(img0, map0_x, map0_y, cv2.INTER_LINEAR)
    img1_rect = cv2.remap(img1, map1_x, map1_y, cv2.INTER_LINEAR)

    extr0_rect = np.concatenate([r_rect0 @ extr0[:3, :3], r_rect0 @ extr0[:3, 3:4]], axis=1)
    extr1_rect = np.concatenate([r_rect1 @ extr1[:3, :3], r_rect1 @ extr1[:3, 3:4]], axis=1)
    intr0_rect = proj0[:3, :3].astype(np.float32)
    intr1_rect = proj1[:3, :3].astype(np.float32)

    if crop_box is None:
        img0_rect, crop_box = _center_crop_square(img0_rect)
        img1_rect = img1_rect[crop_box[1]:crop_box[1] + crop_box[2], crop_box[0]:crop_box[0] + crop_box[2]]
    else:
        x0, y0, size = crop_box
        img0_rect = img0_rect[y0:y0 + size, x0:x0 + size]
        img1_rect = img1_rect[y0:y0 + size, x0:x0 + size]

    img0_rect = _resize_img(img0_rect, target_hw)
    img1_rect = _resize_img(img1_rect, target_hw)
    intr0_rect = _crop_intr(intr0_rect, crop_box, target_hw)
    intr1_rect = _crop_intr(intr1_rect, crop_box, target_hw)

    return {
        "left": (img0_rect, intr0_rect, extr0_rect.astype(np.float32)),
        "right": (img1_rect, intr1_rect, extr1_rect.astype(np.float32)),
    }


def _img_to_tensor(img: np.ndarray, normalize: bool = True) -> torch.Tensor:
    """HWC uint8  →  CHW float  (normalize: [-1,1] if True, else [0,1])"""
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    if normalize:
        t = 2.0 * t - 1.0
    return t


def _build_novel_view_tensors(
    intr: np.ndarray,         # (3,3)
    extr: np.ndarray,         # (3,4)
    img:  np.ndarray,         # HxWx3  uint8
    cfg_dataset,
    target_hw: tuple[int, int] | None,
) -> dict:
    """
    将 (intr, extr, img) 打包成 diff_gaussian_rasterization 需要的全套相机矩阵。
    坐标约定与 StereoHumanDataset.get_novel_view_tensor 保持完全一致。
    """
    img = _resize_img(img, target_hw)
    height, width = img.shape[:2]

    img_t = _img_to_tensor(img, normalize=False)          # [0,1]

    R = np.array(extr[:3, :3], np.float32).T              # camera←world  →  R 为 world←camera
    T = np.array(extr[:3, 3],  np.float32)

    FovX = focal2fov(float(intr[0, 0]), width)
    FovY = focal2fov(float(intr[1, 1]), height)

    proj_mat = getProjectionMatrix(
        znear=cfg_dataset.znear, zfar=cfg_dataset.zfar,
        fovX=FovX, fovY=FovY, K=intr, h=height, w=width,
    ).transpose(0, 1)

    world_view = torch.tensor(
        getWorld2View2(R, T, np.array(cfg_dataset.trans), cfg_dataset.scale)
    ).transpose(0, 1)

    full_proj = world_view.unsqueeze(0).bmm(proj_mat.unsqueeze(0)).squeeze(0)
    cam_center = world_view.inverse()[3, :3]

    return {
        "img":                    img_t,
        "intr":                   torch.FloatTensor(intr),
        "extr":                   torch.FloatTensor(extr),
        "FovX":                   FovX,
        "FovY":                   FovY,
        "width":                  width,
        "height":                 height,
        "world_view_transform":   world_view,
        "full_proj_transform":    full_proj,
        "camera_center":          cam_center,
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Mini-format Dataset  (cameras.json)
# ──────────────────────────────────────────────────────────────────────────────

class PAGMiniSceneDataset(Dataset):
    """
    单个 mini 场景的 Dataset（cameras.json 格式）。

    Args:
        scene_root:   e.g. /…/processed_mini/enerf_outdoor/actor1_4/train
        cfg_dataset:  YACS dataset config node (znear/zfar/trans/scale/target_hw 等)
        phase:        "train" | "val" | "test"
        min_cam_gap:  选两个输入视角时，至少间隔这么多个相机索引（防止太相近）
        max_cam_gap:  选两个输入视角时，最多间隔这么多个相机索引（防止视野完全不重叠）
    """

    def __init__(
        self,
        scene_root: str,
        cfg_dataset,
        phase: str = "train",
        min_cam_gap: int = 2,
        max_cam_gap: int = 12,
    ) -> None:
        super().__init__()
        self.scene_root  = scene_root
        self.cfg         = cfg_dataset
        self.phase       = phase
        self.min_cam_gap = min_cam_gap
        self.max_cam_gap = max_cam_gap

        self.target_hw: tuple[int, int] | None = getattr(cfg_dataset, "target_hw", None)
        self.render_hw: tuple[int, int] | None = getattr(cfg_dataset, "render_hw", None) or self.target_hw
        self.raw_data_root: str = str(getattr(cfg_dataset, "raw_data_root", "") or "")

        img_dir = os.path.join(scene_root, "img")
        if not os.path.isdir(img_dir):
            raise FileNotFoundError(f"img 目录不存在: {img_dir}")
        self.sample_list: list[str] = sorted(os.listdir(img_dir))

        self.train_boost = getattr(cfg_dataset, "train_boost", 50)
        self.val_boost   = getattr(cfg_dataset, "val_boost",   200)

    # ── 内部辅助 ──

    def _load_cameras(self, sample_name: str) -> dict:
        path = os.path.join(
            self.scene_root, "parameter", sample_name, "cameras.json"
        )
        with open(path) as f:
            raw = json.load(f)
        cams = {}
        for k, v in raw.items():
            try:
                cam_id = int(k)
            except (ValueError, TypeError):
                continue   # 跳过非整数 key（如 processed_data 中的 'Tf_x'）
            if not isinstance(v, dict) or "intrinsic" not in v or "extrinsic" not in v:
                continue   # 跳过不含 intrinsic/extrinsic 的条目
            cams[cam_id] = {
                "intrinsic": np.array(v["intrinsic"], dtype=np.float32).reshape(3, 3),
                "extrinsic": np.array(v["extrinsic"], dtype=np.float32).reshape(3, 4),
                "original_cam_id": str(v.get("original_cam_id", cam_id)),
            }
        return cams   # {int_id: {"intrinsic": (3,3), "extrinsic": (3,4)}}

    def _pick_cam_triple(self, n_cams: int) -> tuple[int, int, int]:
        """
        找所有满足相邻间隔约束的有序三元组 (a, b, c)（a<b<c，b-a 和 c-b 均在
        [min_cam_gap, max_cam_gap] 内），再随机打乱分配为 (id_l, id_r, id_novel)。
        """
        ids = list(range(n_cams))
        triples = [
            (a, b, c)
            for a in ids for b in ids for c in ids
            if a < b < c
            and self.min_cam_gap <= b - a <= self.max_cam_gap
            and self.min_cam_gap <= c - b <= self.max_cam_gap
        ]
        if not triples:
            # 退化：宽松选3个（间隔约束放开）
            triples = [
                (a, b, c)
                for a in ids for b in ids for c in ids
                if a < b < c
            ]
        if not triples:
            chosen = list(range(min(3, n_cams)))
            while len(chosen) < 3:
                chosen.append(chosen[-1])
            triples = [tuple(chosen)]

        a, b, c = random.choice(triples)
        perm = random.sample([a, b, c], 3)   # 随机分配哪两个是输入、哪个是 novel
        return perm[0], perm[1], perm[2]

    def _load_one(self, sample_name: str, cam_id: int, cams: dict) -> tuple:
        img_path = os.path.join(
            self.scene_root, "img", sample_name, f"{cam_id}.jpg"
        )
        pil_img = Image.open(img_path)
        orig_w, orig_h = pil_img.size   # PIL: (W, H)
        img = np.array(pil_img)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        intr = cams[cam_id]["intrinsic"].copy()
        extr = cams[cam_id]["extrinsic"].copy()

        if self.target_hw is not None:
            tgt_h, tgt_w = self.target_hw
            if orig_h != tgt_h or orig_w != tgt_w:
                img  = cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
                intr = intr.copy()
                intr[0] *= tgt_w / orig_w   # fx, cx
                intr[1] *= tgt_h / orig_h   # fy, cy

        return img, intr, extr

    def _load_raw_hr_view(
        self,
        sample_name: str,
        cam_id: int,
        cams: dict,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.raw_data_root or self.render_hw is None:
            return None
        frame_idx = _sample_frame_idx(sample_name)
        if frame_idx is None:
            return None
        cam_name = str(cams[cam_id].get("original_cam_id", cam_id))
        raw_path = os.path.join(self.raw_data_root, cam_name, f"frame_{frame_idx:06d}.jpg")
        if not os.path.isfile(raw_path):
            return None
        raw_img = _read_img(raw_path)
        raw_img, _ = _center_crop_square(raw_img)
        raw_img = _resize_img(raw_img, self.render_hw)

        proc_hw = self.target_hw or raw_img.shape[:2]
        intr_hr = _scale_intr(cams[cam_id]["intrinsic"], proc_hw, self.render_hw)
        return raw_img, intr_hr

    # ── Dataset 接口 ──

    def __len__(self) -> int:
        boost = self.train_boost if self.phase == "train" else self.val_boost
        return len(self.sample_list) * boost

    def __getitem__(self, index: int) -> dict:
        sample_name = self.sample_list[index % len(self.sample_list)]
        cams = self._load_cameras(sample_name)
        n_cams = len(cams)

        id_l, id_r, id_novel = self._pick_cam_triple(n_cams)

        img_l, intr_l, extr_l = self._load_one(sample_name, id_l,     cams)
        img_r, intr_r, extr_r = self._load_one(sample_name, id_r,     cams)
        img_n, intr_n, extr_n = self._load_one(sample_name, id_novel, cams)
        proc_hw = img_l.shape[:2]

        hr_l = self._load_raw_hr_view(sample_name, id_l, cams)
        hr_r = self._load_raw_hr_view(sample_name, id_r, cams)
        hr_n = self._load_raw_hr_view(sample_name, id_novel, cams)
        img_l_hr, intr_l_hr = hr_l if hr_l is not None else (
            _resize_img(img_l, self.render_hw),
            _scale_intr(intr_l, proc_hw, self.render_hw or proc_hw),
        )
        img_r_hr, intr_r_hr = hr_r if hr_r is not None else (
            _resize_img(img_r, self.render_hw),
            _scale_intr(intr_r, img_r.shape[:2], self.render_hw or img_r.shape[:2]),
        )
        img_n_hr, intr_n_hr = hr_n if hr_n is not None else (
            _resize_img(img_n, self.render_hw),
            _scale_intr(intr_n, img_n.shape[:2], self.render_hw or img_n.shape[:2]),
        )

        lmain = {
            "img":  _img_to_tensor(img_l, normalize=True),
            "img_hr": _img_to_tensor(img_l_hr, normalize=True),
            "intr": torch.FloatTensor(intr_l),
            "intr_hr": torch.FloatTensor(intr_l_hr),
            "extr": torch.FloatTensor(extr_l),
        }
        rmain = {
            "img":  _img_to_tensor(img_r, normalize=True),
            "img_hr": _img_to_tensor(img_r_hr, normalize=True),
            "intr": torch.FloatTensor(intr_r),
            "intr_hr": torch.FloatTensor(intr_r_hr),
            "extr": torch.FloatTensor(extr_r),
        }
        novel_view = _build_novel_view_tensors(
            intr_n_hr, extr_n, img_n_hr, self.cfg, None
        )
        novel_view["sample_name"] = sample_name

        return {
            "name":       sample_name,
            "lmain":      lmain,
            "rmain":      rmain,
            "novel_view": novel_view,
            "cam_id_l":   id_l,
            "cam_id_r":   id_r,
            "cam_id_n":   id_novel,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Legacy-format Dataset  (0_1.json + npy)
# ──────────────────────────────────────────────────────────────────────────────

class PAGLegacyDataset(Dataset):
    """
    processed_data 格式的 Dataset（0_1.json + {id}_intrinsic/extrinsic.npy）。

    保持原始 GPS+ 风格行为：
    - 输入视角固定为 0 和 1
    - novel view 从 legacy_novel_ids 中随机选 1 个

    视角 0,1 的参数来自 0_1.json；视角 2~5 的参数来自 {id}_intrinsic/extrinsic.npy。

    Args:
        data_root:  e.g. /…/processed_data/train
        cfg_dataset: YACS dataset config node
        phase:      "train" | "val"
    """

    def __init__(
        self,
        data_root:  str,
        cfg_dataset,
        phase:      str = "train",
        # novel_ids 参数保留但不再使用，避免破坏已有调用
        novel_ids:  list[int] = None,
    ) -> None:
        super().__init__()
        self.data_root = data_root
        self.cfg       = cfg_dataset
        self.phase     = phase
        self.target_hw: tuple[int, int] | None = getattr(cfg_dataset, "target_hw", None)
        self.render_hw: tuple[int, int] | None = getattr(cfg_dataset, "render_hw", None) or self.target_hw
        self.raw_data_root: str = str(getattr(cfg_dataset, "raw_data_root", "") or "")
        self.legacy_novel_ids: list[int] = list(getattr(cfg_dataset, "legacy_novel_ids", [2, 3, 4, 5]))
        self.report = self._load_export_report()
        self._raw_sparse_cache = self._load_raw_sparse_model()

        img_dir = os.path.join(data_root, "img")
        self.sample_list: list[str] = sorted(os.listdir(img_dir))

        self.train_boost = getattr(cfg_dataset, "train_boost", 50)
        self.val_boost   = getattr(cfg_dataset, "val_boost",   200)

    def _load_export_report(self) -> dict | None:
        report_path = os.path.join(os.path.dirname(self.data_root), "export_report.json")
        if not os.path.isfile(report_path):
            return None
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logging.warning(f"[PAGLegacyDataset] 读取 export_report 失败: {exc}")
            return None

    def _legacy_scene_name(self, sample_name: str) -> str:
        parts = sample_name.rsplit("_", 1)
        return parts[0] if len(parts) == 2 and parts[1].isdigit() else sample_name

    def _load_raw_sparse_model(self) -> dict | None:
        if not self.raw_data_root:
            return None
        cam_path = os.path.join(self.raw_data_root, "sparse", "0", "cameras.bin")
        img_path = os.path.join(self.raw_data_root, "sparse", "0", "images.bin")
        if not (os.path.isfile(cam_path) and os.path.isfile(img_path)):
            return None

        try:
            cams_raw = read_cameras_binary(cam_path)
            imgs_raw = read_images_binary(img_path)
        except Exception as exc:
            logging.warning(f"[PAGLegacyDataset] 读取 raw sparse COLMAP 失败: {exc}")
            return None

        cameras = {}
        for cam_id, cam in cams_raw.items():
            intr, dist = _colmap_camera_to_intr_dist(cam)
            cameras[cam_id] = {
                "intr": intr,
                "dist": dist,
                "height": int(cam.height),
                "width": int(cam.width),
            }

        images = {}
        for img in imgs_raw.values():
            images[str(img.name)] = {
                "camera_id": int(img.camera_id),
                "extr": _extr_from_colmap_image(img),
            }

        return {"cameras": cameras, "images": images}

    def _legacy_raw_cam_name(self, sample_name: str, view_id: int) -> str | None:
        if self.report is None:
            return None
        scene = self._legacy_scene_name(sample_name)
        for workset in self.report.get("worksets", []):
            if workset.get("scene") != scene:
                continue
            source_pair = list(workset.get("source_pair", []))
            novel_order = list(workset.get("novel_order", []))
            if view_id == 0 and len(source_pair) >= 1:
                return source_pair[0]
            if view_id == 1 and len(source_pair) >= 2:
                return source_pair[1]
            novel_idx = view_id - 2
            if 0 <= novel_idx < len(novel_order):
                return novel_order[novel_idx]
        return None

    def _legacy_crop_box(self) -> tuple[int, int, int] | None:
        if self.report is None:
            return None
        crop = self.report.get("crop_resize", {})
        try:
            return int(crop["crop_x0"]), int(crop["crop_y0"]), int(crop["crop_size"])
        except Exception:
            return None

    def _load_raw_novel_view(
        self,
        sample_name: str,
        view_id: int,
        intr_proc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        if not self.raw_data_root or self.render_hw is None:
            return None
        raw_cam_name = self._legacy_raw_cam_name(sample_name, view_id)
        raw_frame = _sample_frame_idx(sample_name)
        if raw_cam_name is None or raw_frame is None:
            return None
        raw_path = os.path.join(self.raw_data_root, raw_cam_name, f"frame_{raw_frame:06d}.jpg")
        if not os.path.isfile(raw_path):
            return None
        raw_img = _read_img(raw_path)
        crop_box = self._legacy_crop_box()
        if crop_box is None:
            raw_img, _ = _center_crop_square(raw_img)
        else:
            x0, y0, size = crop_box
            raw_img = raw_img[y0:y0 + size, x0:x0 + size]
        raw_img = _resize_img(raw_img, self.render_hw)
        proc_hw = self.target_hw or raw_img.shape[:2]
        intr_hr = _scale_intr(intr_proc, proc_hw, self.render_hw)
        return raw_img, intr_hr

    def _load_source_pair_hr(
        self,
        sample_name: str,
        cams_proc: dict[int, dict],
    ) -> dict[int, tuple[np.ndarray, np.ndarray]] | None:
        if self.render_hw is None or self._raw_sparse_cache is None:
            return None
        frame_idx = _sample_frame_idx(sample_name)
        cam_name_0 = self._legacy_raw_cam_name(sample_name, 0)
        cam_name_1 = self._legacy_raw_cam_name(sample_name, 1)
        if frame_idx is None or cam_name_0 is None or cam_name_1 is None:
            return None

        key0 = f"{cam_name_0}/frame_{frame_idx:06d}.jpg"
        key1 = f"{cam_name_1}/frame_{frame_idx:06d}.jpg"
        img_meta0 = self._raw_sparse_cache["images"].get(key0)
        img_meta1 = self._raw_sparse_cache["images"].get(key1)
        if img_meta0 is None or img_meta1 is None:
            return None

        raw_path0 = os.path.join(self.raw_data_root, key0)
        raw_path1 = os.path.join(self.raw_data_root, key1)
        if not (os.path.isfile(raw_path0) and os.path.isfile(raw_path1)):
            return None

        cam_meta0 = self._raw_sparse_cache["cameras"][img_meta0["camera_id"]]
        cam_meta1 = self._raw_sparse_cache["cameras"][img_meta1["camera_id"]]
        img0 = _read_img(raw_path0)
        img1 = _read_img(raw_path1)
        rectified = _rectify_raw_stereo_pair(
            img0,
            cam_meta0["intr"],
            cam_meta0["dist"],
            img_meta0["extr"],
            img1,
            cam_meta1["intr"],
            cam_meta1["dist"],
            img_meta1["extr"],
            self._legacy_crop_box(),
            self.render_hw,
        )

        return {
            0: (rectified["left"][0], rectified["left"][1]),
            1: (rectified["right"][0], rectified["right"][1]),
        }

    # ── 内部辅助 ──

    def _load_all_cameras(self, sample_name: str) -> dict[int, dict]:
        """
        返回 {view_id: {"intrinsic": (3,3), "extrinsic": (3,4)}} 字典，
        包含该样本下所有存在的视角。
        - 视角 0,1：从 0_1.json 读取
        - 视角 2+： 从 {id}_intrinsic/extrinsic.npy 读取
        """
        parm_dir  = os.path.join(self.data_root, "parameter", sample_name)
        json_path = os.path.join(parm_dir, "0_1.json")

        with open(json_path) as f:
            parm = {k: np.array(v, dtype=np.float32) for k, v in json.load(f).items()}

        cams: dict[int, dict] = {
            0: {"intrinsic": parm["intr0"], "extrinsic": parm["extr0"]},
            1: {"intrinsic": parm["intr1"], "extrinsic": parm["extr1"]},
        }

        # 视角 2~N：自动探测存在的 {id}_intrinsic.npy
        vid = 2
        while True:
            intr_p = os.path.join(parm_dir, f"{vid}_intrinsic.npy")
            extr_p = os.path.join(parm_dir, f"{vid}_extrinsic.npy")
            img_p  = os.path.join(self.data_root, "img", sample_name, f"{vid}.jpg")
            if not (os.path.exists(intr_p) and os.path.exists(extr_p) and os.path.exists(img_p)):
                break
            cams[vid] = {
                "intrinsic": np.load(intr_p).astype(np.float32),
                "extrinsic": np.load(extr_p).astype(np.float32),
            }
            vid += 1

        return cams

    def _load_one(self, sample_name: str, cam_id: int, cams: dict) -> tuple:
        img_path = os.path.join(self.data_root, "img", sample_name, f"{cam_id}.jpg")
        pil_img  = Image.open(img_path)
        orig_w, orig_h = pil_img.size
        img = np.array(pil_img)
        if img.ndim == 3 and img.shape[2] == 4:
            img = img[:, :, :3]

        intr = cams[cam_id]["intrinsic"].copy()
        extr = cams[cam_id]["extrinsic"].copy()

        if self.target_hw is not None:
            tgt_h, tgt_w = self.target_hw
            if orig_h != tgt_h or orig_w != tgt_w:
                img  = cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
                intr = intr.copy()
                intr[0] *= tgt_w / orig_w
                intr[1] *= tgt_h / orig_h

        return img, intr, extr

    def _pick_legacy_views(self, cam_ids: list[int]) -> tuple[int, int, int]:
        """固定 source=0/1，novel 从 legacy_novel_ids 中选择。"""
        ids = set(cam_ids)
        if 0 not in ids or 1 not in ids:
            raise RuntimeError("legacy 数据缺少固定输入视角 0/1")

        novel_candidates = [vid for vid in self.legacy_novel_ids if vid in ids and vid not in (0, 1)]
        if not novel_candidates:
            novel_candidates = [vid for vid in sorted(ids) if vid not in (0, 1)]
        if not novel_candidates:
            raise RuntimeError(f"样本仅包含视角 {sorted(ids)}，没有可用 novel 视角")

        return 0, 1, random.choice(novel_candidates)

    # ── Dataset 接口 ──

    def __len__(self) -> int:
        boost = self.train_boost if self.phase == "train" else self.val_boost
        return len(self.sample_list) * boost

    def __getitem__(self, index: int) -> dict:
        sample_name = self.sample_list[index % len(self.sample_list)]
        cams = self._load_all_cameras(sample_name)
        source_pair_hr = self._load_source_pair_hr(sample_name, cams)

        id_l, id_r, id_novel = self._pick_legacy_views(list(cams.keys()))

        img_l, intr_l, extr_l = self._load_one(sample_name, id_l,     cams)
        img_r, intr_r, extr_r = self._load_one(sample_name, id_r,     cams)
        img_n, intr_n, extr_n = self._load_one(sample_name, id_novel, cams)
        if source_pair_hr is not None and id_l in source_pair_hr:
            img_l_hr, intr_l_hr = source_pair_hr[id_l]
        else:
            raw_l = self._load_raw_novel_view(sample_name, id_l, intr_l)
            if raw_l is None:
                img_l_hr = _resize_img(img_l, self.render_hw)
                intr_l_hr = _scale_intr(intr_l, img_l.shape[:2], self.render_hw or img_l.shape[:2])
            else:
                img_l_hr, intr_l_hr = raw_l
        if source_pair_hr is not None and id_r in source_pair_hr:
            img_r_hr, intr_r_hr = source_pair_hr[id_r]
        else:
            raw_r = self._load_raw_novel_view(sample_name, id_r, intr_r)
            if raw_r is None:
                img_r_hr = _resize_img(img_r, self.render_hw)
                intr_r_hr = _scale_intr(intr_r, img_r.shape[:2], self.render_hw or img_r.shape[:2])
            else:
                img_r_hr, intr_r_hr = raw_r

        raw_novel = self._load_raw_novel_view(sample_name, id_novel, intr_n)
        if raw_novel is None:
            img_n_hr = _resize_img(img_n, self.render_hw)
            intr_n_hr = _scale_intr(intr_n, img_n.shape[:2], self.render_hw or img_n.shape[:2])
        else:
            img_n_hr, intr_n_hr = raw_novel

        lmain = {
            "img":  _img_to_tensor(img_l, normalize=True),
            "img_hr": _img_to_tensor(img_l_hr, normalize=True),
            "intr": torch.FloatTensor(intr_l),
            "intr_hr": torch.FloatTensor(intr_l_hr),
            "extr": torch.FloatTensor(extr_l),
        }
        rmain = {
            "img":  _img_to_tensor(img_r, normalize=True),
            "img_hr": _img_to_tensor(img_r_hr, normalize=True),
            "intr": torch.FloatTensor(intr_r),
            "intr_hr": torch.FloatTensor(intr_r_hr),
            "extr": torch.FloatTensor(extr_r),
        }
        novel_view = _build_novel_view_tensors(
            intr_n_hr, extr_n, img_n_hr, self.cfg, None
        )
        novel_view["sample_name"] = sample_name

        return {
            "name":       sample_name,
            "lmain":      lmain,
            "rmain":      rmain,
            "novel_view": novel_view,
            "cam_id_l":   id_l,
            "cam_id_r":   id_r,
            "cam_id_n":   id_novel,
        }


# ──────────────────────────────────────────────────────────────────────────────
#  Combined multi-dataset
# ──────────────────────────────────────────────────────────────────────────────

def build_pag_dataset(
    cfg_dataset,
    phase: Literal["train", "val", "test"] = "train",
) -> Dataset:
    """
    根据 cfg_dataset 构建多数据集合并后的 Dataset。

    cfg_dataset 相关字段（在 pag_stage.yaml 中设置）：
      multi_train_roots: list[str]  - train 数据目录列表
      multi_val_roots:   list[str]  - val   数据目录列表（可为空，空则用 multi_train_roots 对应 val 目录）
      multi_formats:     list[str]  - 每个目录的格式："mini" | "legacy"
      min_cam_gap:       int        - mini 格式相机间隔下限
      max_cam_gap:       int        - mini 格式相机间隔上限
      legacy_novel_ids:  list[int]  - legacy 格式 novel view id 列表

    若 multi_train_roots 为空或不存在，退回使用 train_data_root / val_data_root
    （legacy 格式，与旧 StereoHumanDataset 兼容）。
    """
    is_val = phase in ("val", "test")

    # 取对应 phase 的 root 列表
    roots: list[str]
    if is_val:
        roots = list(getattr(cfg_dataset, "multi_val_roots", None) or [])
        # val_roots 为空时退回 train_roots（取各目录的 val/ 子目录）
        if not roots:
            train_roots = list(getattr(cfg_dataset, "multi_train_roots", None) or [])
            roots = [
                r.replace("/train", "/val").replace("/train/", "/val/")
                for r in train_roots
            ]
    else:
        roots = list(getattr(cfg_dataset, "multi_train_roots", None) or [])

    formats: list[str] = list(getattr(cfg_dataset, "multi_formats", None) or [])
    min_gap:  int      = getattr(cfg_dataset, "min_cam_gap", 2)
    max_gap:  int      = getattr(cfg_dataset, "max_cam_gap", 12)
    novel_ids          = list(getattr(cfg_dataset, "legacy_novel_ids", [2, 3, 4, 5]))

    datasets: list[Dataset] = []

    for idx, root in enumerate(roots):
        fmt = formats[idx] if idx < len(formats) else "mini"
        if not os.path.isdir(root):
            logging.warning(f"[PAGDataset] 跳过不存在的目录: {root}")
            continue

        if fmt == "mini":
            ds = PAGMiniSceneDataset(
                scene_root  = root,
                cfg_dataset = cfg_dataset,
                phase       = phase,
                min_cam_gap = min_gap,
                max_cam_gap = max_gap,
            )
            logging.info(f"[PAGDataset] +mini   {root}  ({len(ds.sample_list)} samples)")

        elif fmt == "legacy":
            ds = PAGLegacyDataset(
                data_root   = root,
                cfg_dataset = cfg_dataset,
                phase       = phase,
                novel_ids   = novel_ids,
            )
            logging.info(f"[PAGDataset] +legacy {root}  ({len(ds.sample_list)} samples)")
        else:
            logging.warning(f"[PAGDataset] 未知 format={fmt}，跳过 {root}")
            continue

        datasets.append(ds)

    # 没有有效 root → 退回旧的单路径 legacy 模式
    if not datasets:
        fallback_root = (
            cfg_dataset.val_data_root if is_val else cfg_dataset.train_data_root
        )
        logging.info(f"[PAGDataset] 退回 legacy 模式: {fallback_root}")
        datasets.append(
            PAGLegacyDataset(
                data_root   = fallback_root,
                cfg_dataset = cfg_dataset,
                phase       = phase,
                novel_ids   = novel_ids,
            )
        )

    if len(datasets) == 1:
        return datasets[0]

    combined = ConcatDataset(datasets)
    logging.info(
        f"[PAGDataset] 合并 {len(datasets)} 个数据集  "
        f"总 epoch 长度 = {len(combined)}"
    )
    return combined
