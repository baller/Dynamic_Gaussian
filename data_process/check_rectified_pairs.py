from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check epipolar alignment on rectified stereo image pairs using local feature matches."
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help=(
            "Processed root, split root, img root, or rectify_debug root. "
            "Examples: .../sequence_processed, .../sequence_processed/train, "
            ".../sequence_processed/train/img, or .../sequence_processed/rectify_debug."
        ),
    )
    parser.add_argument(
        "--source",
        choices=("auto", "exported", "debug"),
        default="auto",
        help="Which image source to inspect. Default: auto.",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "val"),
        default="all",
        help="Which split to inspect for exported pairs. Ignored for rectify_debug. Default: all.",
    )
    parser.add_argument(
        "--matcher",
        choices=("row_features", "flow", "features"),
        default="row_features",
        help="How to establish correspondences. Default: row_features.",
    )
    parser.add_argument(
        "--detector",
        choices=("auto", "sift", "orb"),
        default="auto",
        help="Local feature detector/descriptor for --matcher features. Default: auto.",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Optional cap on checked pairs after sorting. 0 means all.",
    )
    parser.add_argument(
        "--ratio-test",
        type=float,
        default=0.75,
        help="Lowe ratio threshold. Default: 0.75.",
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=1.5,
        help="RANSAC reprojection threshold in pixels. Default: 1.5.",
    )
    parser.add_argument(
        "--min-matches",
        type=int,
        default=40,
        help="Minimum inlier matches required for a sample to be considered valid. Default: 40.",
    )
    parser.add_argument(
        "--max-corners",
        type=int,
        default=2500,
        help="Maximum corners for --matcher flow. Default: 2500.",
    )
    parser.add_argument(
        "--row-band",
        type=float,
        default=4.0,
        help="Allowed vertical search band in pixels for --matcher row_features. Default: 4.0.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path. Default: <resolved_root>/rectify_feature_check_<source>.json",
    )
    return parser.parse_args()


def has_image_suffix(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def resolve_source(root: Path, source: str) -> str:
    if source != "auto":
        return source
    if root.name == "img" or (root / "img").is_dir() or (root / "train" / "img").is_dir() or (root / "val" / "img").is_dir():
        return "exported"
    if root.name == "rectify_debug" or (root / "rectify_debug").is_dir():
        return "debug"
    raise FileNotFoundError(f"Could not auto-detect a supported source under {root}")


def resolve_export_img_roots(root: Path, split: str) -> List[Tuple[str, Path]]:
    if root.name == "img":
        split_name = root.parent.name if root.parent.name in {"train", "val"} else "unknown"
        return [(split_name, root)]

    if (root / "img").is_dir():
        split_name = root.name if root.name in {"train", "val"} else "unknown"
        return [(split_name, root / "img")]

    img_roots: List[Tuple[str, Path]] = []
    splits = ("train", "val") if split == "all" else (split,)
    for split_name in splits:
        img_root = root / split_name / "img"
        if img_root.is_dir():
            img_roots.append((split_name, img_root))

    if not img_roots:
        raise FileNotFoundError(f"Could not find exported img roots under {root}")
    return img_roots


def collect_exported_pairs(root: Path, split: str) -> Tuple[List[Dict[str, object]], Path]:
    img_roots = resolve_export_img_roots(root, split)
    pairs: List[Dict[str, object]] = []
    for split_name, img_root in img_roots:
        for sample_dir in sorted(path for path in img_root.iterdir() if path.is_dir()):
            left_path = sample_dir / "0.jpg"
            right_path = sample_dir / "1.jpg"
            if not left_path.is_file() or not right_path.is_file():
                continue
            pairs.append(
                {
                    "sample_name": f"{split_name}/{sample_dir.name}",
                    "left_path": left_path,
                    "right_path": right_path,
                    "source": "exported",
                }
            )
    if not pairs:
        raise RuntimeError(f"No exported 0.jpg/1.jpg pairs found under {root}")
    return pairs, img_roots[0][1].parent.parent if img_roots[0][1].parent.parent.exists() else root


def resolve_debug_root(root: Path) -> Path:
    if root.name == "rectify_debug":
        return root
    debug_root = root / "rectify_debug"
    if debug_root.is_dir():
        return debug_root
    raise FileNotFoundError(f"Could not find rectify_debug under {root}")


def collect_debug_pairs(root: Path) -> Tuple[List[Dict[str, object]], Path]:
    debug_root = resolve_debug_root(root)
    pairs: List[Dict[str, object]] = []
    for image_path in sorted(path for path in debug_root.iterdir() if path.is_file() and has_image_suffix(path)):
        pairs.append(
            {
                "sample_name": image_path.stem,
                "debug_path": image_path,
                "source": "debug",
            }
        )
    if not pairs:
        raise RuntimeError(f"No debug images found under {debug_root}")
    return pairs, debug_root.parent


def build_overlay_mask(image: np.ndarray) -> np.ndarray:
    mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
    blue = image[:, :, 0].astype(np.int16)
    green = image[:, :, 1].astype(np.int16)
    red = image[:, :, 2].astype(np.int16)

    green_lines = (green > 180) & (red < 120) & (blue < 120)
    yellow_text = (green > 150) & (red > 150) & (blue < 160)
    mask[green_lines | yellow_text] = 0
    return mask


def load_pair(pair_info: Dict[str, object]) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    source = pair_info["source"]
    if source == "exported":
        left = cv2.imread(str(pair_info["left_path"]))
        right = cv2.imread(str(pair_info["right_path"]))
        if left is None or right is None:
            raise FileNotFoundError(f"Failed to read exported pair for {pair_info['sample_name']}")
        return left, right, None, None

    debug_path = pair_info["debug_path"]
    canvas = cv2.imread(str(debug_path))
    if canvas is None:
        raise FileNotFoundError(f"Failed to read debug image {debug_path}")
    width = canvas.shape[1]
    if width < 2:
        raise ValueError(f"Debug image is too narrow: {debug_path}")

    mid = width // 2
    left = canvas[:, :mid]
    right = canvas[:, mid:]
    left_mask = build_overlay_mask(left)
    right_mask = build_overlay_mask(right)
    return left, right, left_mask, right_mask


def build_feature_backend(detector_name: str):
    if detector_name == "auto":
        detector_name = "sift" if hasattr(cv2, "SIFT_create") else "orb"

    if detector_name == "sift":
        if not hasattr(cv2, "SIFT_create"):
            raise RuntimeError("OpenCV was built without SIFT. Use --detector orb instead.")
        return detector_name, cv2.SIFT_create(nfeatures=4096), cv2.NORM_L2

    return detector_name, cv2.ORB_create(nfeatures=4096), cv2.NORM_HAMMING


def compute_descriptor_distances(query_desc: np.ndarray, candidate_descs: np.ndarray, detector_name: str) -> np.ndarray:
    if detector_name == "sift":
        diff = candidate_descs.astype(np.float32) - query_desc.astype(np.float32)
        return np.sqrt(np.sum(diff * diff, axis=1))

    xor = np.bitwise_xor(candidate_descs, query_desc)
    return np.unpackbits(xor, axis=1).sum(axis=1).astype(np.float32)


def detect_and_match_with_row_band(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
    detector_name: str,
    detector,
    ratio_test: float,
    row_band: float,
) -> Dict[str, np.ndarray]:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    kpts0, desc0 = detector.detectAndCompute(left_gray, left_mask)
    kpts1, desc1 = detector.detectAndCompute(right_gray, right_mask)
    if desc0 is None or desc1 is None or len(kpts0) < 8 or len(kpts1) < 8:
        raise RuntimeError("Too few features were detected.")

    right_bins: Dict[int, List[int]] = {}
    for index, keypoint in enumerate(kpts1):
        row = int(round(keypoint.pt[1]))
        right_bins.setdefault(row, []).append(index)

    accepted = []
    for query_index, keypoint in enumerate(kpts0):
        center_row = int(round(keypoint.pt[1]))
        candidate_indices: List[int] = []
        for row in range(int(np.floor(center_row - row_band)), int(np.ceil(center_row + row_band)) + 1):
            candidate_indices.extend(right_bins.get(row, []))
        if len(candidate_indices) < 2:
            continue

        candidate_descs = desc1[candidate_indices]
        distances = compute_descriptor_distances(desc0[query_index], candidate_descs, detector_name)
        order = np.argsort(distances)
        best = order[0]
        second = order[1]
        if distances[best] >= ratio_test * distances[second]:
            continue

        train_index = candidate_indices[int(best)]
        accepted.append((float(distances[best]), query_index, train_index))

    if len(accepted) < 8:
        raise RuntimeError("Too few matches survived row-band ratio filtering.")

    accepted.sort(key=lambda item: item[0])
    used_train = set()
    matches = []
    for _, query_index, train_index in accepted:
        if train_index in used_train:
            continue
        used_train.add(train_index)
        matches.append((query_index, train_index))

    if len(matches) < 8:
        raise RuntimeError("Too few unique row-band matches remained after train-side deduplication.")

    pts0 = np.float32([kpts0[query_index].pt for query_index, _ in matches])
    pts1 = np.float32([kpts1[train_index].pt for _, train_index in matches])
    dy = np.abs(pts0[:, 1] - pts1[:, 1]).astype(np.float64)
    dx = (pts1[:, 0] - pts0[:, 0]).astype(np.float64)

    return {
        "pts0": pts0,
        "pts1": pts1,
        "dy": dy,
        "dx": dx,
        "raw_keypoints_left": np.array([len(kpts0)], dtype=np.int32),
        "raw_keypoints_right": np.array([len(kpts1)], dtype=np.int32),
        "ratio_matches": np.array([len(matches)], dtype=np.int32),
    }


def track_with_optical_flow(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray | None,
    max_corners: int,
) -> Dict[str, np.ndarray]:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    pts0 = cv2.goodFeaturesToTrack(
        left_gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=8,
        blockSize=7,
        mask=left_mask,
    )
    if pts0 is None or len(pts0) < 8:
        raise RuntimeError("Too few corners were detected for optical flow.")

    lk_params = dict(
        winSize=(61, 61),
        maxLevel=5,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01),
    )
    pts1, status_fwd, err_fwd = cv2.calcOpticalFlowPyrLK(left_gray, right_gray, pts0, None, **lk_params)
    if pts1 is None or status_fwd is None or err_fwd is None:
        raise RuntimeError("Forward optical flow failed.")

    pts0_back, status_bwd, err_bwd = cv2.calcOpticalFlowPyrLK(right_gray, left_gray, pts1, None, **lk_params)
    if pts0_back is None or status_bwd is None or err_bwd is None:
        raise RuntimeError("Backward optical flow failed.")

    pts0 = pts0.reshape(-1, 2)
    pts1 = pts1.reshape(-1, 2)
    pts0_back = pts0_back.reshape(-1, 2)
    status = status_fwd.reshape(-1).astype(bool) & status_bwd.reshape(-1).astype(bool)
    if not np.any(status):
        raise RuntimeError("Optical flow produced no valid tracks.")

    fb_error = np.linalg.norm(pts0 - pts0_back, axis=1)
    fwd_error = err_fwd.reshape(-1)
    bwd_error = err_bwd.reshape(-1)

    valid = status & np.isfinite(fb_error) & np.isfinite(fwd_error) & np.isfinite(bwd_error)
    valid &= fb_error <= 1.5
    valid &= fwd_error <= np.percentile(fwd_error[status], 80)
    valid &= bwd_error <= np.percentile(bwd_error[status], 80)
    if not np.any(valid):
        raise RuntimeError("Optical flow produced no consistent tracks.")

    pts0 = pts0[valid]
    pts1 = pts1[valid]
    dy = np.abs(pts0[:, 1] - pts1[:, 1]).astype(np.float64)
    dx = (pts1[:, 0] - pts0[:, 0]).astype(np.float64)

    return {
        "pts0": pts0,
        "pts1": pts1,
        "dy": dy,
        "dx": dx,
        "raw_keypoints_left": np.array([len(status)], dtype=np.int32),
        "raw_keypoints_right": np.array([len(status)], dtype=np.int32),
        "ratio_matches": np.array([int(valid.sum())], dtype=np.int32),
    }


def detect_and_match(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray | None,
    right_mask: np.ndarray | None,
    detector_name: str,
    detector,
    norm_type: int,
    ratio_test: float,
    ransac_thresh: float,
) -> Dict[str, np.ndarray]:
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    kpts0, desc0 = detector.detectAndCompute(left_gray, left_mask)
    kpts1, desc1 = detector.detectAndCompute(right_gray, right_mask)
    if desc0 is None or desc1 is None or len(kpts0) < 8 or len(kpts1) < 8:
        raise RuntimeError("Too few features were detected.")

    matcher = cv2.BFMatcher(norm_type)
    knn_matches = matcher.knnMatch(desc0, desc1, k=2)

    good_matches = []
    for match_pair in knn_matches:
        if len(match_pair) < 2:
            continue
        first, second = match_pair
        if first.distance < ratio_test * second.distance:
            good_matches.append(first)
    if len(good_matches) < 8:
        raise RuntimeError("Too few matches survived the ratio test.")

    pts0 = np.float32([kpts0[m.queryIdx].pt for m in good_matches])
    pts1 = np.float32([kpts1[m.trainIdx].pt for m in good_matches])

    ransac_method = getattr(cv2, "USAC_MAGSAC", cv2.FM_RANSAC)
    _, inlier_mask = cv2.findFundamentalMat(pts0, pts1, ransac_method, ransac_thresh)
    if inlier_mask is None:
        raise RuntimeError("Fundamental matrix estimation failed.")

    inlier_mask = inlier_mask.ravel().astype(bool)
    pts0_inliers = pts0[inlier_mask]
    pts1_inliers = pts1[inlier_mask]
    if len(pts0_inliers) < 8:
        raise RuntimeError("Too few inlier matches survived geometric verification.")

    dy = np.abs(pts0_inliers[:, 1] - pts1_inliers[:, 1]).astype(np.float64)
    dx = (pts1_inliers[:, 0] - pts0_inliers[:, 0]).astype(np.float64)

    return {
        "pts0": pts0_inliers,
        "pts1": pts1_inliers,
        "dy": dy,
        "dx": dx,
        "raw_keypoints_left": np.array([len(kpts0)], dtype=np.int32),
        "raw_keypoints_right": np.array([len(kpts1)], dtype=np.int32),
        "ratio_matches": np.array([len(good_matches)], dtype=np.int32),
    }


def summarize_pair(pair_info: Dict[str, object], stats: Dict[str, np.ndarray]) -> Dict[str, object]:
    dy = stats["dy"]
    dx = stats["dx"]
    return {
        "sample_name": pair_info["sample_name"],
        "source": pair_info["source"],
        "inlier_matches": int(len(dy)),
        "median_abs_y_px": float(np.median(dy)),
        "p95_abs_y_px": float(np.percentile(dy, 95)),
        "p99_abs_y_px": float(np.percentile(dy, 99)),
        "max_abs_y_px": float(np.max(dy)),
        "mean_abs_y_px": float(np.mean(dy)),
        "median_disp_x_px": float(np.median(dx)),
        "mean_disp_x_px": float(np.mean(dx)),
        "std_disp_x_px": float(np.std(dx)),
        "keypoints_left": int(stats["raw_keypoints_left"][0]),
        "keypoints_right": int(stats["raw_keypoints_right"][0]),
        "ratio_matches": int(stats["ratio_matches"][0]),
    }


def summarize_all(results: Sequence[Dict[str, object]], requested_pairs: int, min_matches: int) -> Dict[str, object]:
    valid_results = [result for result in results if result.get("inlier_matches", 0) >= min_matches]
    failed_results = [result for result in results if "error" in result or result.get("inlier_matches", 0) < min_matches]

    if valid_results:
        all_dy = np.concatenate(
            [np.array([result["median_abs_y_px"], result["p95_abs_y_px"], result["p99_abs_y_px"], result["max_abs_y_px"]]) for result in valid_results]
        )
        median_dy_values = np.array([result["median_abs_y_px"] for result in valid_results], dtype=np.float64)
        p95_dy_values = np.array([result["p95_abs_y_px"] for result in valid_results], dtype=np.float64)
        max_dy_values = np.array([result["max_abs_y_px"] for result in valid_results], dtype=np.float64)
        mean_dx_values = np.array([result["mean_disp_x_px"] for result in valid_results], dtype=np.float64)
        match_counts = np.array([result["inlier_matches"] for result in valid_results], dtype=np.float64)

        aggregate = {
            "valid_pair_count": len(valid_results),
            "failed_pair_count": len(failed_results),
            "requested_pair_count": requested_pairs,
            "median_of_pair_medians_abs_y_px": float(np.median(median_dy_values)),
            "p95_of_pair_medians_abs_y_px": float(np.percentile(median_dy_values, 95)),
            "median_of_pair_p95_abs_y_px": float(np.median(p95_dy_values)),
            "p95_of_pair_p95_abs_y_px": float(np.percentile(p95_dy_values, 95)),
            "worst_pair_max_abs_y_px": float(np.max(max_dy_values)),
            "median_mean_disp_x_px": float(np.median(mean_dx_values)),
            "median_inlier_matches": float(np.median(match_counts)),
            "status_thresholds_px": {
                "median_of_pair_medians_abs_y_px": 2.0,
                "p95_of_pair_p95_abs_y_px": 3.0,
            },
            "status": (
                "ok"
                if float(np.median(median_dy_values)) <= 2.0
                and float(np.percentile(p95_dy_values, 95)) <= 3.0
                else "warning"
            ),
        }
        _ = all_dy
    else:
        aggregate = {
            "valid_pair_count": 0,
            "failed_pair_count": len(failed_results),
            "requested_pair_count": requested_pairs,
            "status": "error",
        }
    return aggregate


def default_output_path(report_root: Path, source: str) -> Path:
    return report_root / f"rectify_feature_check_{source}.json"


def main() -> None:
    args = parse_args()
    source = resolve_source(args.input_root, args.source)
    if source == "exported":
        pairs, report_root = collect_exported_pairs(args.input_root, args.split)
    else:
        pairs, report_root = collect_debug_pairs(args.input_root)

    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    detector_name = args.detector
    detector = None
    norm_type = None
    if args.matcher in {"row_features", "features"}:
        detector_name, detector, norm_type = build_feature_backend(args.detector)
    results: List[Dict[str, object]] = []
    for pair_info in pairs:
        try:
            left, right, left_mask, right_mask = load_pair(pair_info)
            if args.matcher == "row_features":
                stats = detect_and_match_with_row_band(
                    left=left,
                    right=right,
                    left_mask=left_mask,
                    right_mask=right_mask,
                    detector_name=detector_name,
                    detector=detector,
                    ratio_test=args.ratio_test,
                    row_band=args.row_band,
                )
            elif args.matcher == "flow":
                stats = track_with_optical_flow(
                    left=left,
                    right=right,
                    left_mask=left_mask,
                    max_corners=args.max_corners,
                )
            else:
                stats = detect_and_match(
                    left=left,
                    right=right,
                    left_mask=left_mask,
                    right_mask=right_mask,
                    detector_name=detector_name,
                    detector=detector,
                    norm_type=norm_type,
                    ratio_test=args.ratio_test,
                    ransac_thresh=args.ransac_thresh,
                )
            summary = summarize_pair(pair_info, stats)
            if summary["inlier_matches"] < args.min_matches:
                summary["error"] = f"Too few inlier matches after geometric verification (< {args.min_matches})."
            results.append(summary)
        except Exception as exc:
            results.append(
                {
                    "sample_name": pair_info["sample_name"],
                    "source": pair_info["source"],
                    "error": str(exc),
                }
            )

    aggregate = summarize_all(results, requested_pairs=len(pairs), min_matches=args.min_matches)
    report = {
        "input_root": str(args.input_root),
        "resolved_source": source,
        "detector": detector_name,
        "matcher": args.matcher,
        "split": args.split,
        "ratio_test": args.ratio_test,
        "ransac_thresh": args.ransac_thresh,
        "min_matches": args.min_matches,
        "aggregate": aggregate,
        "worst_samples_by_p95_abs_y_px": sorted(
            [result for result in results if "p95_abs_y_px" in result],
            key=lambda item: item["p95_abs_y_px"],
            reverse=True,
        )[:10],
        "samples": results,
    }

    output_path = args.output or default_output_path(report_root, source)
    with output_path.open("w") as handle:
        json.dump(report, handle, indent=2)

    matcher_label = args.matcher if args.matcher == "flow" else f"{args.matcher}:{detector_name}"
    print(f"Checked {len(pairs)} pairs from {source} using {matcher_label}.")
    print(f"Report written to {output_path}")
    print(json.dumps(report["aggregate"], indent=2))
    if report["worst_samples_by_p95_abs_y_px"]:
        print("Worst samples by p95 |dy|:")
        for item in report["worst_samples_by_p95_abs_y_px"][:5]:
            print(
                f"  {item['sample_name']}: "
                f"inliers={item['inlier_matches']} "
                f"median={item['median_abs_y_px']:.3f} "
                f"p95={item['p95_abs_y_px']:.3f} "
                f"max={item['max_abs_y_px']:.3f}"
            )


if __name__ == "__main__":
    main()
