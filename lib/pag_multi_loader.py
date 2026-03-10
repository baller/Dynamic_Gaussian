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

        lmain = {
            "img":  _img_to_tensor(img_l, normalize=True),
            "intr": torch.FloatTensor(intr_l),
            "extr": torch.FloatTensor(extr_l),
        }
        rmain = {
            "img":  _img_to_tensor(img_r, normalize=True),
            "intr": torch.FloatTensor(intr_r),
            "extr": torch.FloatTensor(extr_r),
        }
        novel_view = _build_novel_view_tensors(
            intr_n, extr_n, img_n, self.cfg, self.target_hw
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

    与 PAGMiniSceneDataset 对齐：每次从所有可用视角（0~5）中随机选 3 个，
    2 个作为输入（lmain / rmain），1 个作为新视角 GT。
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

        img_dir = os.path.join(data_root, "img")
        self.sample_list: list[str] = sorted(os.listdir(img_dir))

        self.train_boost = getattr(cfg_dataset, "train_boost", 50)
        self.val_boost   = getattr(cfg_dataset, "val_boost",   200)

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

    def _pick_cam_triple(self, cam_ids: list[int]) -> tuple[int, int, int]:
        """
        从可用视角中找满足相邻间隔约束的有序三元组，再随机打乱分配。
        cam_ids 已排序；约束与 mini 格式一致：相邻间隔在 [min_cam_gap, max_cam_gap] 内。
        """
        if len(cam_ids) < 3:
            raise RuntimeError(f"视角数 {len(cam_ids)} < 3，无法选三视角")
        min_g = getattr(self.cfg, "min_cam_gap", 1)
        max_g = getattr(self.cfg, "max_cam_gap", 12)
        ids = sorted(cam_ids)
        triples = [
            (a, b, c)
            for a in ids for b in ids for c in ids
            if a < b < c
            and min_g <= b - a <= max_g
            and min_g <= c - b <= max_g
        ]
        if not triples:
            # 退化：宽松选3个
            triples = [
                (a, b, c)
                for a in ids for b in ids for c in ids
                if a < b < c
            ]
        a, b, c = random.choice(triples)
        perm = random.sample([a, b, c], 3)
        return perm[0], perm[1], perm[2]

    # ── Dataset 接口 ──

    def __len__(self) -> int:
        boost = self.train_boost if self.phase == "train" else self.val_boost
        return len(self.sample_list) * boost

    def __getitem__(self, index: int) -> dict:
        sample_name = self.sample_list[index % len(self.sample_list)]
        cams = self._load_all_cameras(sample_name)

        id_l, id_r, id_novel = self._pick_cam_triple(list(cams.keys()))

        img_l, intr_l, extr_l = self._load_one(sample_name, id_l,     cams)
        img_r, intr_r, extr_r = self._load_one(sample_name, id_r,     cams)
        img_n, intr_n, extr_n = self._load_one(sample_name, id_novel, cams)

        lmain = {
            "img":  _img_to_tensor(img_l, normalize=True),
            "intr": torch.FloatTensor(intr_l),
            "extr": torch.FloatTensor(extr_l),
        }
        rmain = {
            "img":  _img_to_tensor(img_r, normalize=True),
            "intr": torch.FloatTensor(intr_r),
            "extr": torch.FloatTensor(extr_r),
        }
        novel_view = _build_novel_view_tensors(
            intr_n, extr_n, img_n, self.cfg, None   # _load_one 已经 resize
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
