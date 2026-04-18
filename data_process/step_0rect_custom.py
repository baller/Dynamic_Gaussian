from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

from colmap_read_model import qvec2rotmat, read_cameras_binary, read_images_binary


DEFAULT_DATA_ROOT = Path("/data/sifang/GPS_plus_data/hias_man1_s3")
DEFAULT_PROCESSED_ROOT = Path("/data/sifang/GPS_plus_data/hias_man1_s3_processed")
SOURCE_RAW_CAMS = ("cam00", "cam03")
NOVEL_RAW_ORDER = ("cam01", "cam02", "cam00", "cam03")
VERIFY_THRESHOLDS = {"median_px": 0.5, "p95_px": 2.0}
RAW_IMAGE_RE = re.compile(r"^(?P<frame>\d+)_(?P<cam>cam\d+)\.(?P<ext>jpe?g|png)$", re.IGNORECASE)


class CropResize:
    def __init__(self, crop_x0: int, crop_y0: int, crop_size: int, out_size: int):
        self.crop_x0 = crop_x0
        self.crop_y0 = crop_y0
        self.crop_size = crop_size
        self.out_size = out_size
        self.scale_x = out_size / float(crop_size)
        self.scale_y = out_size / float(crop_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--trainval", required=True, choices=("train", "val"))
    parser.add_argument(
        "-d",
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Raw custom sequence root that contains sparse/0 and 0000_camXX.jpg files.",
    )
    parser.add_argument(
        "-o",
        "--processed-root",
        type=Path,
        default=DEFAULT_PROCESSED_ROOT,
        help="Processed GPS_plus legacy root. The script writes into train/ or val/ under this root.",
    )
    parser.add_argument(
        "-s",
        "--size",
        type=int,
        default=1024,
        help="Square output resolution after center crop.",
    )
    parser.add_argument(
        "-n",
        "--setsize",
        type=int,
        default=4,
        help="Compatibility flag. This implementation expects the 4-camera hias_man1_s3 rig.",
    )
    return parser.parse_args()


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


def collect_raw_image_paths(data_root: Path, raw_cams: List[str]) -> Dict[int, Dict[str, Path]]:
    raw_image_paths: Dict[int, Dict[str, Path]] = {}
    for path in data_root.iterdir():
        if not path.is_file():
            continue
        match = RAW_IMAGE_RE.match(path.name)
        if match is None:
            continue
        raw_cam = match.group("cam")
        if raw_cam not in raw_cams:
            continue
        raw_frame = int(match.group("frame"))
        raw_image_paths.setdefault(raw_frame, {})[raw_cam] = path
    return raw_image_paths


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


def build_dataset_context(data_root: Path, setsize: int = 4) -> Dict[str, object]:
    if setsize != 4:
        raise ValueError(f"hias_man1_s3 expects --setsize 4, got {setsize}")

    mapping_path = data_root / "custom_frame_mapping.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Missing mapping file: {mapping_path}")
    with mapping_path.open("r") as handle:
        mapping_info = json.load(handle)

    colmap_to_raw = mapping_info["camera_map"]
    raw_cams = sort_camera_names(list(colmap_to_raw.values()))
    raw_to_sparse = {int(raw_idx): int(sparse_idx) for sparse_idx, raw_idx in mapping_info["frame_map"].items()}

    sparse_dir = data_root / "sparse" / "0"
    camdata = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    imgdata = read_images_binary(str(sparse_dir / "images.bin"))

    frame_entries: Dict[int, Dict[str, object]] = {}
    colmap_name_to_camera_id: Dict[str, int] = {}
    for image in imgdata.values():
        parts = image.name.split("/")
        if len(parts) != 2 or parts[0] not in colmap_to_raw:
            continue
        raw_cam = colmap_to_raw[parts[0]]
        sparse_frame = int(parts[1].split("_")[1].split(".")[0])
        frame_entries.setdefault(sparse_frame, {})[raw_cam] = image
        colmap_name_to_camera_id[parts[0]] = image.camera_id

    camera_models: Dict[str, Dict[str, object]] = {}
    for colmap_name, raw_cam in colmap_to_raw.items():
        camera_id = colmap_name_to_camera_id[colmap_name]
        intr, dist = camera_to_intrinsics(camdata[camera_id])
        camera_models[raw_cam] = {
            "camera_id": camera_id,
            "colmap_name": colmap_name,
            "K": intr,
            "dist": dist,
            "size": (camdata[camera_id].width, camdata[camera_id].height),
        }

    raw_image_paths = collect_raw_image_paths(data_root, raw_cams)
    samples: List[Tuple[int, int]] = []
    for raw_frame, sparse_frame in sorted(raw_to_sparse.items()):
        if sparse_frame not in frame_entries:
            continue
        if raw_frame not in raw_image_paths:
            continue
        if not all(raw_cam in frame_entries[sparse_frame] for raw_cam in raw_cams):
            continue
        if not all(raw_cam in raw_image_paths[raw_frame] for raw_cam in raw_cams):
            continue
        samples.append((raw_frame, sparse_frame))

    if not samples:
        raise RuntimeError(f"No complete mapped samples found under {data_root}")

    base_cam = raw_cams[0]
    rig_extrinsics: Dict[str, np.ndarray] = {base_cam: np.hstack([np.eye(3), np.zeros((3, 1))]).astype(np.float64)}
    for raw_cam in raw_cams[1:]:
        rel_rots: List[np.ndarray] = []
        rel_trans: List[np.ndarray] = []
        for _, sparse_frame in samples:
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

    source_raw_cams = tuple(SOURCE_RAW_CAMS)
    if not all(raw_cam in raw_cams for raw_cam in source_raw_cams):
        raise ValueError(f"Source cameras {source_raw_cams} not found in {raw_cams}")
    if not all(raw_cam in raw_cams for raw_cam in NOVEL_RAW_ORDER):
        raise ValueError(f"Novel view order {NOVEL_RAW_ORDER} not found in {raw_cams}")

    return {
        "data_root": data_root,
        "raw_cams": tuple(raw_cams),
        "source_raw_cams": source_raw_cams,
        "novel_raw_order": tuple(NOVEL_RAW_ORDER),
        "raw_to_sparse": raw_to_sparse,
        "samples": samples,
        "frame_entries": frame_entries,
        "raw_image_paths": raw_image_paths,
        "camera_models": camera_models,
        "rig_extrinsics": rig_extrinsics,
        "image_size": image_size,
    }


def build_crop_resize(image_size: Tuple[int, int], out_size: int) -> CropResize:
    if out_size <= 0:
        raise ValueError(f"Output size must be positive, got {out_size}")
    width, height = image_size
    crop_size = min(width, height)
    crop_x0 = (width - crop_size) // 2
    crop_y0 = (height - crop_size) // 2
    return CropResize(crop_x0=crop_x0, crop_y0=crop_y0, crop_size=crop_size, out_size=out_size)


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


def transform_camera_for_output(camera: Dict[str, np.ndarray], crop_resize: CropResize) -> Dict[str, np.ndarray]:
    out_camera = {
        "intr0": transform_intrinsic_for_output(camera["intr0"], crop_resize),
        "intr1": transform_intrinsic_for_output(camera["intr1"], crop_resize),
        "extr0": camera["extr0"].astype(np.float64).copy(),
        "extr1": camera["extr1"].astype(np.float64).copy(),
        "Tf_x": np.array([float(camera["Tf_x"][0]) * crop_resize.scale_x], dtype=np.float64),
    }
    return out_camera


def load_raw_image(context: Dict[str, object], raw_frame: int, raw_cam: str) -> np.ndarray:
    image_path = context["raw_image_paths"][raw_frame][raw_cam]
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Failed to read {image_path}")
    return image


def process_single_view_image(img: np.ndarray, intr: np.ndarray, dist: np.ndarray, crop_resize: CropResize) -> np.ndarray:
    undistorted = cv2.undistort(img, intr, dist, None)
    return crop_and_resize_image(undistorted, crop_resize, cv2.INTER_LINEAR)


def save_camera_json(camera: Dict[str, np.ndarray], path: Path) -> None:
    serializable = {key: value.tolist() for key, value in camera.items()}
    with path.open("w") as handle:
        json.dump(serializable, handle, indent=1)


def get_split_samples(samples: List[Tuple[int, int]], trainval: str) -> List[Tuple[int, int]]:
    total = len(samples)
    split_size = total // 8
    if split_size == 0:
        raise ValueError(f"Need at least 8 samples for the fixed 7/8-1/8 split, got {total}")
    if trainval == "train":
        return samples[:-split_size]
    if trainval == "val":
        return samples[-split_size:]
    raise ValueError(trainval)


def make_sample_name(raw_frame: int) -> str:
    return f"s1_{raw_frame:04d}"


def build_rectification_config(context: Dict[str, object], crop_resize: CropResize) -> Dict[str, object]:
    left_cam, right_cam = context["source_raw_cams"]
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
    camera_output = transform_camera_for_output(camera_full, crop_resize)

    white_mask = np.ones((height, width, 3), dtype=np.uint8) * 255
    mask0 = cv2.remap(white_mask, map0x, map0y, cv2.INTER_NEAREST)
    mask1 = cv2.remap(white_mask, map1x, map1y, cv2.INTER_NEAREST)
    mask0 = crop_and_resize_image(mask0, crop_resize, cv2.INTER_NEAREST)
    mask1 = crop_and_resize_image(mask1, crop_resize, cv2.INTER_NEAREST)

    return {
        "source_raw_cams": (left_cam, right_cam),
        "map0x": map0x,
        "map0y": map0y,
        "map1x": map1x,
        "map1y": map1y,
        "rect_rot0": rect_rot0,
        "rect_rot1": rect_rot1,
        "proj0": proj0,
        "proj1": proj1,
        "camera_full": camera_full,
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


def transform_points_for_output(points: np.ndarray, crop_resize: CropResize) -> np.ndarray:
    out = points.astype(np.float64).copy()
    out[:, 0] -= crop_resize.crop_x0
    out[:, 1] -= crop_resize.crop_y0
    out[:, 0] *= crop_resize.scale_x
    out[:, 1] *= crop_resize.scale_y
    return out


def build_debug_canvas(left_img: np.ndarray, right_img: np.ndarray, label: str) -> np.ndarray:
    canvas = np.concatenate([left_img, right_img], axis=1)
    for row in range(0, canvas.shape[0], 128):
        cv2.line(canvas, (0, row), (canvas.shape[1] - 1, row), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, label, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2, cv2.LINE_AA)
    return canvas


def verify_rectification(
    context: Dict[str, object],
    rectification: Dict[str, object],
    crop_resize: CropResize,
    processed_root: Path,
) -> Dict[str, object]:
    left_cam, right_cam = rectification["source_raw_cams"]
    left_model = context["camera_models"][left_cam]
    right_model = context["camera_models"][right_cam]

    metrics_per_frame = []
    all_dy = []
    debug_root = processed_root / "rectify_debug"
    debug_root.mkdir(parents=True, exist_ok=True)

    sample_indices = np.linspace(0, len(context["samples"]) - 1, num=min(6, len(context["samples"])), dtype=int)
    debug_frames = {context["samples"][idx][0] for idx in sample_indices}

    for raw_frame, sparse_frame in context["samples"]:
        left_entry = context["frame_entries"][sparse_frame][left_cam]
        right_entry = context["frame_entries"][sparse_frame][right_cam]

        left_points = {
            int(point_id): xy
            for xy, point_id in zip(left_entry.xys, left_entry.point3D_ids)
            if point_id != -1
        }
        right_points = {
            int(point_id): xy
            for xy, point_id in zip(right_entry.xys, right_entry.point3D_ids)
            if point_id != -1
        }
        common_point_ids = sorted(set(left_points) & set(right_points))
        if not common_point_ids:
            continue

        pts0 = np.array([left_points[point_id] for point_id in common_point_ids], dtype=np.float64).reshape(-1, 1, 2)
        pts1 = np.array([right_points[point_id] for point_id in common_point_ids], dtype=np.float64).reshape(-1, 1, 2)
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
        dx = proc_pts0[valid, 0] - proc_pts1[valid, 0]
        all_dy.append(dy)
        metrics_per_frame.append(
            {
                "sample_name": make_sample_name(raw_frame),
                "raw_frame": raw_frame,
                "sparse_frame": sparse_frame,
                "common_points": len(common_point_ids),
                "valid_points_after_crop": int(valid.sum()),
                "median_abs_y_px": float(np.median(dy)),
                "p95_abs_y_px": float(np.percentile(dy, 95)),
                "max_abs_y_px": float(np.max(dy)),
                "mean_disp_x_px": float(dx.mean()),
            }
        )

        if raw_frame in debug_frames:
            img0 = load_raw_image(context, raw_frame, left_cam)
            img1 = load_raw_image(context, raw_frame, right_cam)
            rect0, rect1 = rectify_source_pair(img0, img1, rectification, crop_resize)
            canvas = build_debug_canvas(rect0, rect1, make_sample_name(raw_frame))
            cv2.imwrite(str(debug_root / f"{make_sample_name(raw_frame)}.jpg"), canvas)

    if not all_dy:
        raise RuntimeError("Rectification verification found no valid 3D correspondences after crop/resize")

    all_dy_arr = np.concatenate(all_dy, axis=0)
    right_extr = context["rig_extrinsics"][right_cam]
    baseline = float(np.linalg.norm((-right_extr[:, :3].T @ right_extr[:, 3:]).reshape(-1)))

    summary = {
        "mapped_frame_count": len(metrics_per_frame),
        "thresholds_px": VERIFY_THRESHOLDS,
        "source_pair": [left_cam, right_cam],
        "image_size_before_crop": list(context["image_size"]),
        "crop_x0": crop_resize.crop_x0,
        "crop_y0": crop_resize.crop_y0,
        "crop_size": crop_resize.crop_size,
        "output_size": crop_resize.out_size,
        "baseline_m": baseline,
        "inverse_depth_init_hint": float(0.5 / baseline),
        "Tf_x": float(rectification["camera_output"]["Tf_x"][0]),
        "global_metrics": {
            "median_abs_y_px": float(np.median(all_dy_arr)),
            "p95_abs_y_px": float(np.percentile(all_dy_arr, 95)),
            "p99_abs_y_px": float(np.percentile(all_dy_arr, 99)),
            "max_abs_y_px": float(np.max(all_dy_arr)),
        },
        "per_frame": metrics_per_frame,
    }
    summary["status"] = (
        "ok"
        if summary["global_metrics"]["median_abs_y_px"] <= VERIFY_THRESHOLDS["median_px"]
        and summary["global_metrics"]["p95_abs_y_px"] <= VERIFY_THRESHOLDS["p95_px"]
        else "warning"
    )

    report_path = processed_root / "rectify_verification.json"
    with report_path.open("w") as handle:
        json.dump(summary, handle, indent=2)

    if summary["status"] != "ok":
        print(
            "WARNING: rectification residual exceeded threshold:",
            summary["global_metrics"],
        )
    else:
        print("Rectification verification passed:", summary["global_metrics"])

    print("baseline(m):", baseline)
    print("inverse_depth_init hint:", 0.5 / baseline)
    return summary


def ensure_split_dirs(processed_root: Path, trainval: str):
    split_root = processed_root / trainval
    img_dir = split_root / "img"
    mask_dir = split_root / "mask"
    param_dir = split_root / "parameter"
    img_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    param_dir.mkdir(parents=True, exist_ok=True)
    return split_root, img_dir, mask_dir, param_dir


def main() -> None:
    args = parse_args()
    context = build_dataset_context(args.data_root, setsize=args.setsize)
    crop_resize = build_crop_resize(context["image_size"], args.size)
    rectification = build_rectification_config(context, crop_resize)
    verify_rectification(context, rectification, crop_resize, args.processed_root)

    split_samples = get_split_samples(context["samples"], args.trainval)
    _, img_dir, mask_dir, param_dir = ensure_split_dirs(args.processed_root, args.trainval)

    left_cam, right_cam = rectification["source_raw_cams"]
    for raw_frame, _ in tqdm(split_samples, desc=f"export {args.trainval}"):
        sample_name = make_sample_name(raw_frame)
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

    print(f"Wrote {len(split_samples)} {args.trainval} samples to {args.processed_root / args.trainval}")


if __name__ == "__main__":
    main()
