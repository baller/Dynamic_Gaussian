from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from colmap_read_model import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
    read_points3d_binary,
)


RAW_FLAT_IMAGE_RE = re.compile(r"^(?P<frame>\d+)_(?P<cam>cam\d+)\.(?P<ext>jpe?g|png)$", re.IGNORECASE)
RAW_FRAME_IMAGE_RE = re.compile(r"^frame_(?P<frame>\d+)\.(?P<ext>jpe?g|png)$", re.IGNORECASE)
VERIFY_THRESHOLDS = {"median_px": 0.5, "p95_px": 2.0}
BASELINE_REC_SCALE = 0.5


@dataclass
class CropResize:
    crop_x0: int
    crop_y0: int
    crop_size: int
    out_size: int

    @property
    def scale_x(self) -> float:
        return self.out_size / float(self.crop_size)

    @property
    def scale_y(self) -> float:
        return self.out_size / float(self.crop_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a COLMAP-calibrated multi-camera sequence to GPS_plus legacy format."
    )
    parser.add_argument("-d", "--data-root", type=Path, required=True, help="Raw sequence root.")
    parser.add_argument(
        "-o",
        "--processed-root",
        type=Path,
        default=None,
        help="Output root. Default: <data-root>_processed next to the input sequence.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=1024,
        help="Square output size after center crop.",
    )
    parser.add_argument(
        "-n",
        "--setsize",
        type=int,
        default=4,
        help="Number of cameras in each overlapping work set. Default: 4.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val"],
        help="Which dataset splits to export. Default: train val.",
    )
    parser.add_argument(
        "--verify-debug-count",
        type=int,
        default=6,
        help="How many stitched rectification debug images to save per work set.",
    )
    return parser.parse_args()


def default_processed_root(data_root: Path) -> Path:
    return data_root.parent / f"{data_root.name}_processed"


def sort_camera_names(raw_camera_names: List[str]) -> List[str]:
    return sorted(raw_camera_names, key=lambda name: int(re.search(r"(\d+)$", name).group(1)))


def camera_to_intrinsics(camera) -> Tuple[np.ndarray, np.ndarray]:
    params = camera.params.astype(np.float64)
    if camera.model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        intr = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif camera.model == "PINHOLE":
        fx, fy, cx, cy = params
        intr = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.zeros(5, dtype=np.float64)
    elif camera.model == "SIMPLE_RADIAL":
        f, cx, cy, k1 = params
        intr = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    elif camera.model == "RADIAL":
        f, cx, cy, k1, k2 = params
        intr = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
        dist = np.array([k1, k2, 0.0, 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported COLMAP camera model: {camera.model}")
    return intr, dist


def parse_folder_layout(data_root: Path):
    raw_cams = sort_camera_names([path.name for path in data_root.iterdir() if path.is_dir() and path.name.startswith("cam_")])
    if not raw_cams:
        return None

    raw_image_paths: Dict[int, Dict[str, Path]] = {}
    for raw_cam in raw_cams:
        cam_dir = data_root / raw_cam
        for path in cam_dir.iterdir():
            if not path.is_file():
                continue
            match = RAW_FRAME_IMAGE_RE.match(path.name)
            if match is None:
                continue
            raw_frame = int(match.group("frame"))
            raw_image_paths.setdefault(raw_frame, {})[raw_cam] = path

    if not raw_image_paths:
        return None

    mapping_path = data_root / "custom_frame_mapping.json"
    mapping_info = None
    if mapping_path.exists():
        with mapping_path.open("r") as handle:
            mapping_info = json.load(handle)

    camera_map = mapping_info.get("camera_map", {raw_cam: raw_cam for raw_cam in raw_cams}) if mapping_info else {
        raw_cam: raw_cam for raw_cam in raw_cams
    }
    complete_raw_frames = sorted(
        raw_frame
        for raw_frame, frame_paths in raw_image_paths.items()
        if all(raw_cam in frame_paths for raw_cam in raw_cams)
    )
    return {
        "layout": "folder",
        "raw_cams": tuple(raw_cams),
        "raw_image_paths": raw_image_paths,
        "complete_raw_frames": complete_raw_frames,
        "colmap_to_raw": camera_map,
        "raw_to_sparse": (
            {int(raw_idx): int(sparse_idx) for sparse_idx, raw_idx in mapping_info["frame_map"].items()}
            if mapping_info
            else {}
        ),
        "has_explicit_mapping": mapping_info is not None,
    }


def parse_flat_layout(data_root: Path):
    mapping_path = data_root / "custom_frame_mapping.json"
    if not mapping_path.exists():
        return None

    with mapping_path.open("r") as handle:
        mapping_info = json.load(handle)

    raw_cams = sort_camera_names(list(mapping_info["camera_map"].values()))
    raw_image_paths: Dict[int, Dict[str, Path]] = {}
    for path in data_root.iterdir():
        if not path.is_file():
            continue
        match = RAW_FLAT_IMAGE_RE.match(path.name)
        if match is None:
            continue
        raw_cam = match.group("cam")
        if raw_cam not in raw_cams:
            continue
        raw_frame = int(match.group("frame"))
        raw_image_paths.setdefault(raw_frame, {})[raw_cam] = path

    if not raw_image_paths:
        raise RuntimeError(f"Found {mapping_path} but no flat raw images in {data_root}")

    raw_to_sparse = {int(raw_idx): int(sparse_idx) for sparse_idx, raw_idx in mapping_info["frame_map"].items()}
    complete_raw_frames = sorted(
        raw_frame
        for raw_frame, frame_paths in raw_image_paths.items()
        if all(raw_cam in frame_paths for raw_cam in raw_cams)
    )
    return {
        "layout": "flat",
        "raw_cams": tuple(raw_cams),
        "raw_image_paths": raw_image_paths,
        "complete_raw_frames": complete_raw_frames,
        "colmap_to_raw": mapping_info["camera_map"],
        "raw_to_sparse": raw_to_sparse,
        "has_explicit_mapping": True,
    }


def detect_raw_layout(data_root: Path):
    folder_layout = parse_folder_layout(data_root)
    flat_layout = parse_flat_layout(data_root)
    if folder_layout is not None:
        return folder_layout
    if flat_layout is not None:
        return flat_layout
    raise RuntimeError(
        f"Unsupported raw layout under {data_root}. Expected either cam_*/frame_XXXXXX.jpg folders "
        "or flat XXXX_camYY.jpg files with custom_frame_mapping.json."
    )


def image_to_extrinsic(image) -> np.ndarray:
    rot = qvec2rotmat(image.qvec)
    trans = image.tvec.reshape(3, 1)
    return np.hstack([rot, trans]).astype(np.float64)


def invert_extrinsic(extr: np.ndarray) -> np.ndarray:
    rot = extr[:, :3]
    trans = extr[:, 3:]
    inv_rot = rot.T
    inv_trans = -inv_rot @ trans
    return np.hstack([inv_rot, inv_trans])


def compose_extrinsics(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    rot_l, trans_l = lhs[:, :3], lhs[:, 3:]
    rot_r, trans_r = rhs[:, :3], rhs[:, 3:]
    rot = rot_l @ rot_r
    trans = rot_l @ trans_r + trans_l
    return np.hstack([rot, trans])


def relative_extrinsic(base_extr: np.ndarray, target_extr: np.ndarray) -> np.ndarray:
    return compose_extrinsics(target_extr, invert_extrinsic(base_extr))


def average_rotations(rotations: List[np.ndarray]) -> np.ndarray:
    mean_rot = np.mean(np.stack(rotations, axis=0), axis=0)
    u_mat, _, vh_mat = np.linalg.svd(mean_rot)
    rot = u_mat @ vh_mat
    if np.linalg.det(rot) < 0.0:
        u_mat[:, -1] *= -1.0
        rot = u_mat @ vh_mat
    return rot


def build_dataset_context(data_root: Path, setsize: int):
    layout = detect_raw_layout(data_root)
    raw_cams = list(layout["raw_cams"])
    if len(raw_cams) < 2:
        raise ValueError(f"Need at least 2 cameras, got {raw_cams}")
    if setsize < 2:
        raise ValueError(f"--setsize must be >= 2, got {setsize}")
    if len(raw_cams) < setsize:
        raise ValueError(f"--setsize {setsize} exceeds camera count {len(raw_cams)}")

    sparse_dir = data_root / "sparse" / "0"
    camdata = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    imgdata = read_images_binary(str(sparse_dir / "images.bin"))
    points3d = None
    points_path = sparse_dir / "points3D.bin"
    if points_path.exists():
        points3d = read_points3d_binary(str(points_path))

    frame_entries: Dict[int, Dict[str, object]] = {}
    colmap_name_to_camera_id: Dict[str, int] = {}
    for image in imgdata.values():
        parts = image.name.split("/")
        if len(parts) != 2 or parts[0] not in layout["colmap_to_raw"]:
            continue
        raw_cam = layout["colmap_to_raw"][parts[0]]
        sparse_frame = int(parts[1].split("_")[1].split(".")[0])
        frame_entries.setdefault(sparse_frame, {})[raw_cam] = image
        colmap_name_to_camera_id[parts[0]] = image.camera_id

    camera_models = {}
    for colmap_name, raw_cam in layout["colmap_to_raw"].items():
        camera_id = colmap_name_to_camera_id[colmap_name]
        intr, dist = camera_to_intrinsics(camdata[camera_id])
        camera_models[raw_cam] = {
            "camera_id": camera_id,
            "colmap_name": colmap_name,
            "K": intr,
            "dist": dist,
            "size": (camdata[camera_id].width, camdata[camera_id].height),
        }

    sparse_complete_frames = sorted(
        sparse_frame
        for sparse_frame, frame_data in frame_entries.items()
        if all(raw_cam in frame_data for raw_cam in raw_cams)
    )

    calibration_samples = []
    frame_pairs = (
        sorted(layout["raw_to_sparse"].items())
        if layout["raw_to_sparse"]
        else [(sparse_frame, sparse_frame) for sparse_frame in sparse_complete_frames]
    )
    for raw_frame, sparse_frame in frame_pairs:
        if sparse_frame not in frame_entries:
            continue
        if raw_frame not in layout["raw_image_paths"]:
            continue
        if not all(raw_cam in frame_entries[sparse_frame] for raw_cam in raw_cams):
            continue
        if not all(raw_cam in layout["raw_image_paths"][raw_frame] for raw_cam in raw_cams):
            continue
        calibration_samples.append((raw_frame, sparse_frame))
    if not calibration_samples:
        raise RuntimeError(f"No complete sparse-aligned samples found under {data_root}")

    export_frames = (
        [raw_frame for raw_frame, _ in calibration_samples]
        if layout["has_explicit_mapping"]
        else list(layout["complete_raw_frames"])
    )
    if not export_frames:
        raise RuntimeError(f"No complete raw multi-camera frames found under {data_root}")

    base_cam = raw_cams[0]
    rig_extrinsics = {base_cam: np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)}
    for raw_cam in raw_cams[1:]:
        rel_rots = []
        rel_trans = []
        for _, sparse_frame in calibration_samples:
            base_extr = image_to_extrinsic(frame_entries[sparse_frame][base_cam])
            target_extr = image_to_extrinsic(frame_entries[sparse_frame][raw_cam])
            rel_extr = relative_extrinsic(base_extr, target_extr)
            rel_rots.append(rel_extr[:, :3])
            rel_trans.append(rel_extr[:, 3:])
        rig_extrinsics[raw_cam] = np.hstack(
            [average_rotations(rel_rots), np.mean(np.stack(rel_trans, axis=0), axis=0)]
        )

    image_size = camera_models[base_cam]["size"]
    for raw_cam in raw_cams[1:]:
        if camera_models[raw_cam]["size"] != image_size:
            raise ValueError("All cameras must share the same image size for this pipeline")

    n_sets = (len(raw_cams) - 1) // (setsize - 1)
    worksets = []
    for set_idx in range(n_sets):
        left = set_idx * (setsize - 1)
        right = (set_idx + 1) * (setsize - 1)
        cams = raw_cams[left : right + 1]
        if len(cams) < 2:
            continue
        worksets.append(
            {
                "scene": f"s{set_idx + 1}",
                "cams": tuple(cams),
                "source_pair": (cams[0], cams[-1]),
                "novel_order": tuple(cams[1:-1] + [cams[0], cams[-1]]),
            }
        )
    if not worksets:
        raise RuntimeError(f"Could not form any work set from {len(raw_cams)} cameras and setsize={setsize}")

    return {
        "layout": layout["layout"],
        "data_root": data_root,
        "raw_cams": tuple(raw_cams),
        "raw_image_paths": layout["raw_image_paths"],
        "complete_raw_frames": tuple(layout["complete_raw_frames"]),
        "export_frames": tuple(export_frames),
        "calibration_samples": tuple(calibration_samples),
        "mapping_mode": "explicit" if layout["has_explicit_mapping"] else "identity_overlap_for_calibration",
        "frame_entries": frame_entries,
        "camera_models": camera_models,
        "rig_extrinsics": rig_extrinsics,
        "image_size": image_size,
        "points3d": points3d,
        "worksets": worksets,
    }


def build_crop_resize(image_size: Tuple[int, int], out_size: int) -> CropResize:
    width, height = image_size
    crop_size = min(width, height)
    return CropResize(
        crop_x0=(width - crop_size) // 2,
        crop_y0=(height - crop_size) // 2,
        crop_size=crop_size,
        out_size=out_size,
    )


def crop_and_resize_image(image: np.ndarray, crop_resize: CropResize, interpolation: int) -> np.ndarray:
    cropped = image[
        crop_resize.crop_y0 : crop_resize.crop_y0 + crop_resize.crop_size,
        crop_resize.crop_x0 : crop_resize.crop_x0 + crop_resize.crop_size,
    ]
    if crop_resize.out_size == crop_resize.crop_size:
        return cropped
    return cv2.resize(cropped, (crop_resize.out_size, crop_resize.out_size), interpolation=interpolation)


def transform_intrinsic_for_output(intr: np.ndarray, crop_resize: CropResize) -> np.ndarray:
    out_intr = intr.astype(np.float64).copy()
    out_intr[0, 2] -= crop_resize.crop_x0
    out_intr[1, 2] -= crop_resize.crop_y0
    scale = np.diag([crop_resize.scale_x, crop_resize.scale_y, 1.0]).astype(np.float64)
    return scale @ out_intr


def transform_points_for_output(points: np.ndarray, crop_resize: CropResize) -> np.ndarray:
    out = points.astype(np.float64).copy()
    out[:, 0] -= crop_resize.crop_x0
    out[:, 1] -= crop_resize.crop_y0
    out[:, 0] *= crop_resize.scale_x
    out[:, 1] *= crop_resize.scale_y
    return out


def ensure_split_dirs(processed_root: Path, split_name: str):
    split_root = processed_root / split_name
    img_dir = split_root / "img"
    mask_dir = split_root / "mask"
    param_dir = split_root / "parameter"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    return split_root, img_dir, mask_dir, param_dir


def save_json(data: Dict[str, object], path: Path) -> None:
    with path.open("w") as handle:
        json.dump(data, handle, indent=2)


def make_sample_name(scene_name: str, raw_frame: int) -> str:
    return f"{scene_name}_{raw_frame:04d}"


def load_raw_image(context: Dict[str, object], raw_frame: int, raw_cam: str) -> np.ndarray:
    image_path = context["raw_image_paths"][raw_frame][raw_cam]
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read {image_path}")
    return image


def process_single_view_image(img: np.ndarray, intr: np.ndarray, dist: np.ndarray, crop_resize: CropResize) -> np.ndarray:
    undistorted = cv2.undistort(img, intr, dist, None)
    return crop_and_resize_image(undistorted, crop_resize, cv2.INTER_LINEAR)


def get_split_samples(samples, split_name: str):
    total = len(samples)
    split_size = total // 8
    if split_size == 0:
        raise ValueError(f"Need at least 8 samples for the fixed 7/8-1/8 split, got {total}")
    if split_name == "train":
        return samples[:-split_size]
    if split_name == "val":
        return samples[-split_size:]
    raise ValueError(split_name)


def build_rectification_config(context: Dict[str, object], workset: Dict[str, object], crop_resize: CropResize):
    left_cam, right_cam = workset["source_pair"]
    left_model = context["camera_models"][left_cam]
    right_model = context["camera_models"][right_cam]
    extr0 = context["rig_extrinsics"][left_cam]
    extr1 = context["rig_extrinsics"][right_cam]

    rel_extr = relative_extrinsic(extr0, extr1)
    rel_rot = rel_extr[:, :3]
    rel_trans = rel_extr[:, 3].astype(np.float64)
    width, height = context["image_size"]

    rect_rot0, rect_rot1, proj0, proj1, _, _, _ = cv2.stereoRectify(
        left_model["K"],
        left_model["dist"],
        right_model["K"],
        right_model["dist"],
        (width, height),
        rel_rot,
        rel_trans,
        flags=0,
    )

    map0x, map0y = cv2.initUndistortRectifyMap(
        left_model["K"],
        left_model["dist"],
        rect_rot0,
        proj0,
        (width, height),
        cv2.CV_32FC1,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        right_model["K"],
        right_model["dist"],
        rect_rot1,
        proj1,
        (width, height),
        cv2.CV_32FC1,
    )

    camera_full = {
        "intr0": proj0[:3, :3].astype(np.float64),
        "intr1": proj1[:3, :3].astype(np.float64),
        "extr0": (rect_rot0 @ extr0).astype(np.float64),
        "extr1": (rect_rot1 @ extr1).astype(np.float64),
        "Tf_x": np.array([proj1[0, 3]], dtype=np.float64),
    }
    camera_output = {
        "intr0": transform_intrinsic_for_output(camera_full["intr0"], crop_resize),
        "intr1": transform_intrinsic_for_output(camera_full["intr1"], crop_resize),
        "extr0": camera_full["extr0"],
        "extr1": camera_full["extr1"],
        "Tf_x": np.array([float(camera_full["Tf_x"][0]) * crop_resize.scale_x], dtype=np.float64),
    }

    mask = np.ones((height, width, 3), dtype=np.uint8) * 255
    mask0 = crop_and_resize_image(cv2.remap(mask, map0x, map0y, cv2.INTER_NEAREST), crop_resize, cv2.INTER_NEAREST)
    mask1 = crop_and_resize_image(cv2.remap(mask, map1x, map1y, cv2.INTER_NEAREST), crop_resize, cv2.INTER_NEAREST)

    return {
        "scene": workset["scene"],
        "source_pair": workset["source_pair"],
        "novel_order": workset["novel_order"],
        "map0x": map0x,
        "map0y": map0y,
        "map1x": map1x,
        "map1y": map1y,
        "rect_rot0": rect_rot0,
        "rect_rot1": rect_rot1,
        "proj0": proj0,
        "proj1": proj1,
        "camera_output": camera_output,
        "mask0": mask0,
        "mask1": mask1,
    }


def rectify_source_pair(img0: np.ndarray, img1: np.ndarray, rectification: Dict[str, object], crop_resize: CropResize):
    rect0 = cv2.remap(img0, rectification["map0x"], rectification["map0y"], cv2.INTER_LINEAR)
    rect1 = cv2.remap(img1, rectification["map1x"], rectification["map1y"], cv2.INTER_LINEAR)
    rect0 = crop_and_resize_image(rect0, crop_resize, cv2.INTER_LINEAR)
    rect1 = crop_and_resize_image(rect1, crop_resize, cv2.INTER_LINEAR)
    return rect0, rect1


def build_debug_canvas(left_img: np.ndarray, right_img: np.ndarray, label: str) -> np.ndarray:
    canvas = np.concatenate([left_img, right_img], axis=1)
    for row in range(0, canvas.shape[0], 128):
        cv2.line(canvas, (0, row), (canvas.shape[1] - 1, row), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def compute_sparse_depth_hint(context: Dict[str, object], workset: Dict[str, object]) -> Dict[str, object]:
    points3d = context["points3d"]
    left_cam, right_cam = workset["source_pair"]
    rel_extr = relative_extrinsic(context["rig_extrinsics"][left_cam], context["rig_extrinsics"][right_cam])
    baseline = float(np.linalg.norm(rel_extr[:, 3]))
    baseline_hint = BASELINE_REC_SCALE / baseline

    if points3d is None:
        return {
            "baseline_m": baseline,
            "baseline_hint": baseline_hint,
            "sparse_depth_stats_m": None,
            "sparse_inverse_depth_hints": None,
            "recommended_inverse_depth_init": baseline_hint,
            "recommendation_method": "baseline_only_no_points3d",
        }

    depths = []
    for _, sparse_frame in context["calibration_samples"]:
        left_entry = context["frame_entries"][sparse_frame][left_cam]
        right_entry = context["frame_entries"][sparse_frame][right_cam]
        common_ids = set(int(pid) for pid in left_entry.point3D_ids if pid != -1) & set(
            int(pid) for pid in right_entry.point3D_ids if pid != -1
        )
        if not common_ids:
            continue
        extr = image_to_extrinsic(left_entry)
        rot = extr[:, :3]
        trans = extr[:, 3:]
        for point_id in common_ids:
            point = points3d.get(point_id)
            if point is None:
                continue
            cam_point = rot @ point.xyz.reshape(3, 1) + trans
            depth = float(cam_point[2, 0])
            if depth > 0.0:
                depths.append(depth)

    if not depths:
        return {
            "baseline_m": baseline,
            "baseline_hint": baseline_hint,
            "sparse_depth_stats_m": None,
            "sparse_inverse_depth_hints": None,
            "recommended_inverse_depth_init": baseline_hint,
            "recommendation_method": "baseline_only_no_valid_sparse_depth",
        }

    depths_arr = np.array(depths, dtype=np.float64)
    depth_stats = {
        "p01": float(np.percentile(depths_arr, 1)),
        "p05": float(np.percentile(depths_arr, 5)),
        "p10": float(np.percentile(depths_arr, 10)),
        "p50": float(np.percentile(depths_arr, 50)),
        "p90": float(np.percentile(depths_arr, 90)),
        "p95": float(np.percentile(depths_arr, 95)),
        "p99": float(np.percentile(depths_arr, 99)),
    }
    sparse_hints = {
        "from_depth_p90": float(1.0 / depth_stats["p90"]),
        "from_depth_p95": float(1.0 / depth_stats["p95"]),
        "from_depth_p99": float(1.0 / depth_stats["p99"]),
    }
    recommended = max(baseline_hint, sparse_hints["from_depth_p95"])
    method = "max(baseline_hint=0.5/baseline, sparse_hint=1/depth_p95_common_points)"

    return {
        "baseline_m": baseline,
        "baseline_hint": baseline_hint,
        "sparse_depth_stats_m": depth_stats,
        "sparse_inverse_depth_hints": sparse_hints,
        "recommended_inverse_depth_init": float(recommended),
        "recommendation_method": method,
    }


def verify_rectification(
    context: Dict[str, object],
    workset: Dict[str, object],
    rectification: Dict[str, object],
    crop_resize: CropResize,
    processed_root: Path,
    debug_count: int,
):
    left_cam, right_cam = rectification["source_pair"]
    left_model = context["camera_models"][left_cam]
    right_model = context["camera_models"][right_cam]

    debug_root = processed_root / "rectify_debug"
    debug_root.mkdir(parents=True, exist_ok=True)
    sample_indices = np.linspace(
        0,
        len(context["calibration_samples"]) - 1,
        num=min(debug_count, len(context["calibration_samples"])),
        dtype=int,
    )
    debug_frames = {context["calibration_samples"][idx][0] for idx in sample_indices}

    all_dy = []
    per_frame = []
    for raw_frame, sparse_frame in context["calibration_samples"]:
        left_entry = context["frame_entries"][sparse_frame][left_cam]
        right_entry = context["frame_entries"][sparse_frame][right_cam]
        left_points = {int(pid): xy for xy, pid in zip(left_entry.xys, left_entry.point3D_ids) if pid != -1}
        right_points = {int(pid): xy for xy, pid in zip(right_entry.xys, right_entry.point3D_ids) if pid != -1}
        common_ids = sorted(set(left_points) & set(right_points))
        if not common_ids:
            continue

        pts0 = np.array([left_points[pid] for pid in common_ids], dtype=np.float64).reshape(-1, 1, 2)
        pts1 = np.array([right_points[pid] for pid in common_ids], dtype=np.float64).reshape(-1, 1, 2)
        rect_pts0 = cv2.undistortPoints(
            pts0,
            left_model["K"],
            left_model["dist"],
            R=rectification["rect_rot0"],
            P=rectification["proj0"],
        ).reshape(-1, 2)
        rect_pts1 = cv2.undistortPoints(
            pts1,
            right_model["K"],
            right_model["dist"],
            R=rectification["rect_rot1"],
            P=rectification["proj1"],
        ).reshape(-1, 2)

        proc_pts0 = transform_points_for_output(rect_pts0, crop_resize)
        proc_pts1 = transform_points_for_output(rect_pts1, crop_resize)
        valid = (
            (proc_pts0[:, 0] >= 0.0)
            & (proc_pts0[:, 0] < crop_resize.out_size)
            & (proc_pts0[:, 1] >= 0.0)
            & (proc_pts0[:, 1] < crop_resize.out_size)
            & (proc_pts1[:, 0] >= 0.0)
            & (proc_pts1[:, 0] < crop_resize.out_size)
            & (proc_pts1[:, 1] >= 0.0)
            & (proc_pts1[:, 1] < crop_resize.out_size)
        )
        if not np.any(valid):
            continue

        dy = np.abs(proc_pts0[valid, 1] - proc_pts1[valid, 1])
        all_dy.append(dy)
        per_frame.append(
            {
                "sample_name": make_sample_name(workset["scene"], raw_frame),
                "raw_frame": raw_frame,
                "sparse_frame": sparse_frame,
                "common_points": len(common_ids),
                "valid_points_after_crop": int(valid.sum()),
                "median_abs_y_px": float(np.median(dy)),
                "p95_abs_y_px": float(np.percentile(dy, 95)),
                "max_abs_y_px": float(np.max(dy)),
            }
        )

        if raw_frame in debug_frames:
            img0 = load_raw_image(context, raw_frame, left_cam)
            img1 = load_raw_image(context, raw_frame, right_cam)
            rect0, rect1 = rectify_source_pair(img0, img1, rectification, crop_resize)
            label = f"{workset['scene']} frame {raw_frame:04d}"
            canvas = build_debug_canvas(rect0, rect1, label)
            cv2.imwrite(str(debug_root / f"{workset['scene']}_{raw_frame:04d}.jpg"), canvas)

    if not all_dy:
        raise RuntimeError(f"No valid rectification correspondences for work set {workset['scene']}")

    all_dy_arr = np.concatenate(all_dy, axis=0)
    metrics = {
        "thresholds_px": VERIFY_THRESHOLDS,
        "global_metrics": {
            "median_abs_y_px": float(np.median(all_dy_arr)),
            "p95_abs_y_px": float(np.percentile(all_dy_arr, 95)),
            "p99_abs_y_px": float(np.percentile(all_dy_arr, 99)),
            "max_abs_y_px": float(np.max(all_dy_arr)),
        },
        "per_frame": per_frame,
    }
    metrics["status"] = (
        "ok"
        if metrics["global_metrics"]["median_abs_y_px"] <= VERIFY_THRESHOLDS["median_px"]
        and metrics["global_metrics"]["p95_abs_y_px"] <= VERIFY_THRESHOLDS["p95_px"]
        else "warning"
    )
    return metrics


def save_camera_json(camera: Dict[str, np.ndarray], path: Path) -> None:
    data = {key: value.tolist() for key, value in camera.items()}
    with path.open("w") as handle:
        json.dump(data, handle, indent=1)


def export_split(
    context: Dict[str, object],
    worksets: List[Dict[str, object]],
    crop_resize: CropResize,
    processed_root: Path,
    split_name: str,
) -> None:
    split_samples = get_split_samples(list(context["export_frames"]), split_name)
    _, img_dir, mask_dir, param_dir = ensure_split_dirs(processed_root, split_name)

    for workset in worksets:
        scene_name = workset["scene"]
        left_cam, right_cam = workset["source_pair"]
        rectification = workset["rectification"]
        processed_intrinsics = {
            raw_cam: transform_intrinsic_for_output(context["camera_models"][raw_cam]["K"], crop_resize)
            for raw_cam in workset["novel_order"]
        }
        rig_extrinsics = {raw_cam: context["rig_extrinsics"][raw_cam] for raw_cam in workset["novel_order"]}

        for raw_frame in tqdm(split_samples, desc=f"{split_name} {scene_name}"):
            sample_name = make_sample_name(scene_name, raw_frame)
            sample_img_dir = img_dir / sample_name
            sample_mask_dir = mask_dir / sample_name
            sample_param_dir = param_dir / sample_name
            sample_img_dir.mkdir(exist_ok=True)
            sample_mask_dir.mkdir(exist_ok=True)
            sample_param_dir.mkdir(exist_ok=True)

            left_img = load_raw_image(context, raw_frame, left_cam)
            right_img = load_raw_image(context, raw_frame, right_cam)
            rect0, rect1 = rectify_source_pair(left_img, right_img, rectification, crop_resize)
            cv2.imwrite(str(sample_img_dir / "0.jpg"), rect0.astype(np.uint8))
            cv2.imwrite(str(sample_img_dir / "1.jpg"), rect1.astype(np.uint8))
            cv2.imwrite(str(sample_mask_dir / "0.jpg"), rectification["mask0"].astype(np.uint8))
            cv2.imwrite(str(sample_mask_dir / "1.jpg"), rectification["mask1"].astype(np.uint8))
            save_camera_json(rectification["camera_output"], sample_param_dir / "0_1.json")

            for view_offset, raw_cam in enumerate(workset["novel_order"], start=2):
                img = load_raw_image(context, raw_frame, raw_cam)
                model = context["camera_models"][raw_cam]
                img_out = process_single_view_image(img, model["K"], model["dist"], crop_resize)
                cv2.imwrite(str(sample_img_dir / f"{view_offset}.jpg"), img_out.astype(np.uint8))
                np.save(str(sample_param_dir / f"{view_offset}_intrinsic.npy"), processed_intrinsics[raw_cam])
                np.save(str(sample_param_dir / f"{view_offset}_extrinsic.npy"), rig_extrinsics[raw_cam])


def main() -> None:
    args = parse_args()
    processed_root = args.processed_root or default_processed_root(args.data_root)
    context = build_dataset_context(args.data_root, setsize=args.setsize)
    crop_resize = build_crop_resize(context["image_size"], args.size)

    if len(context["export_frames"]) != len(context["calibration_samples"]):
        print(
            "Using",
            len(context["calibration_samples"]),
            "sparse-aligned frames for calibration/verification and exporting",
            len(context["export_frames"]),
            "complete raw frames.",
        )

    worksets = []
    for workset in context["worksets"]:
        rectification = build_rectification_config(context, workset, crop_resize)
        verification = verify_rectification(
            context,
            workset,
            rectification,
            crop_resize,
            processed_root,
            debug_count=args.verify_debug_count,
        )
        inverse_depth = compute_sparse_depth_hint(context, workset)
        worksets.append(
            {
                **workset,
                "rectification": rectification,
                "verification": verification,
                "inverse_depth_init": inverse_depth,
            }
        )

    for split_name in args.splits:
        export_split(context, worksets, crop_resize, processed_root, split_name)

    report = {
        "data_root": str(args.data_root),
        "processed_root": str(processed_root),
        "layout": context["layout"],
        "mapping_mode": context["mapping_mode"],
        "camera_names": list(context["raw_cams"]),
        "sample_count_total": len(context["export_frames"]),
        "calibration_sample_count": len(context["calibration_samples"]),
        "complete_raw_frame_count": len(context["complete_raw_frames"]),
        "split_counts": {split: len(get_split_samples(list(context["export_frames"]), split)) for split in args.splits},
        "image_size_before_crop": list(context["image_size"]),
        "crop_resize": {
            "crop_x0": crop_resize.crop_x0,
            "crop_y0": crop_resize.crop_y0,
            "crop_size": crop_resize.crop_size,
            "output_size": crop_resize.out_size,
        },
        "worksets": [
            {
                "scene": workset["scene"],
                "cams": list(workset["cams"]),
                "source_pair": list(workset["source_pair"]),
                "novel_order": list(workset["novel_order"]),
                "Tf_x": float(workset["rectification"]["camera_output"]["Tf_x"][0]),
                "inverse_depth_init": workset["inverse_depth_init"],
                "verification": workset["verification"],
            }
            for workset in worksets
        ],
    }

    save_json(report, processed_root / "export_report.json")

    print(f"Exported splits {args.splits} to {processed_root}")
    for workset in worksets:
        rec = workset["inverse_depth_init"]
        print(
            f"{workset['scene']} source {workset['source_pair'][0]}->{workset['source_pair'][1]}: "
            f"recommended inverse_depth_init={rec['recommended_inverse_depth_init']:.6f} "
            f"(baseline_hint={rec['baseline_hint']:.6f})"
        )


if __name__ == "__main__":
    main()
