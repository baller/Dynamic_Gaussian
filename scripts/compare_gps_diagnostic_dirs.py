#!/usr/bin/env python3
"""Create side-by-side summaries from two exported GPS diagnostic directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--origin-dir", required=True)
    parser.add_argument("--current-dir", required=True)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_img(path: Path) -> np.ndarray | None:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img


def label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(out, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def resize_to_height(img: np.ndarray, height: int) -> np.ndarray:
    if img.shape[0] == height:
        return img
    width = int(round(img.shape[1] * height / img.shape[0]))
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def montage_pair(origin_path: Path, current_path: Path, out_path: Path, title: str) -> bool:
    o = read_img(origin_path)
    c = read_img(current_path)
    if o is None or c is None:
        return False
    h = min(o.shape[0], c.shape[0], 768)
    o = resize_to_height(o, h)
    c = resize_to_height(c, h)
    canvas = np.concatenate([label(o, "origin GPS RAFT"), label(c, "current StereoGS")], axis=1)
    cv2.putText(canvas, title, (8, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, title, (8, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), canvas)
    return True


def q(stats: dict[str, Any], path: list[str], default=None):
    cur: Any = stats
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def build_summary(origin: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    fields = {
        "mae_rgb_0_255": ["image", "mae_rgb_0_255"],
        "hole_candidate_ratio": ["image", "hole_candidate_ratio"],
        "projection_coverage_ratio": ["projection", "coverage_ratio"],
        "projection_density_mean": ["projection", "density_mean"],
        "projection_density_p99": ["projection", "density_p99"],
        "density_on_hole_candidates_mean": ["projection", "density_on_hole_candidates_mean"],
        "left_base_opacity_p50": ["views", "left", "base_opacity", "p50"],
        "right_base_opacity_p50": ["views", "right", "base_opacity", "p50"],
        "left_base_scale_p50": ["views", "left", "base_scale_mean", "p50"],
        "right_base_scale_p50": ["views", "right", "base_scale_mean", "p50"],
        "left_sub_opacity_p50": ["views", "left", "sub_opacity", "p50"],
        "right_sub_opacity_p50": ["views", "right", "sub_opacity", "p50"],
        "left_split_sum_p50": ["views", "left", "split_sum", "p50"],
        "right_split_sum_p50": ["views", "right", "split_sum", "p50"],
    }
    rows = {}
    for name, path in fields.items():
        ov = q(origin, path)
        cv = q(current, path)
        delta = None if ov is None or cv is None else cv - ov
        ratio = None if ov in (None, 0) or cv is None else cv / ov
        rows[name] = {"origin": ov, "current": cv, "delta": delta, "ratio_current_over_origin": ratio}
    return rows


def write_report(out: Path, origin: dict[str, Any], current: dict[str, Any], summary: dict[str, Any]) -> None:
    lines = [
        "# GPS vs StereoGS Diagnostic Summary",
        "",
        f"- Origin: `{origin['method']}` step `{origin['checkpoint_step']}`",
        f"- Current: `{current['method']}` step `{current['checkpoint_step']}`",
        f"- Sample: `{current['sample_name']}`, novel view `{current['novel_id']}`",
        "",
        "## Key Metrics",
        "",
        "| Metric | Origin | Current | Current/Origin |",
        "|---|---:|---:|---:|",
    ]
    for key, row in summary.items():
        ov = row["origin"]
        cv = row["current"]
        ratio = row["ratio_current_over_origin"]
        ov_s = "n/a" if ov is None else f"{ov:.6g}"
        cv_s = "n/a" if cv is None else f"{cv:.6g}"
        ratio_s = "n/a" if ratio is None else f"{ratio:.3g}"
        lines.append(f"| `{key}` | {ov_s} | {cv_s} | {ratio_s} |")
    lines.extend(
        [
            "",
            "## First Read",
            "",
            "- Current StereoGS has higher projection coverage and much higher projected Gaussian density.",
            "- Current StereoGS base opacity median is lower than origin GPS RAFT.",
            "- Current StereoGS sub-Gaussian opacity median is very low, so many sub-Gaussians exist but contribute weakly.",
            "- If holes remain visible despite higher projection density, the likely failure point is opacity/footprint contribution rather than missing depth points.",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    origin_dir = Path(args.origin_dir)
    current_dir = Path(args.current_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    origin = load_json(origin_dir / "stats.json")
    current = load_json(current_dir / "stats.json")
    summary = build_summary(origin, current)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(out, origin, current, summary)

    pairs = [
        ("overview", "images/overview.png"),
        ("render", "images/render.png"),
        ("diff_x3", "images/diff_x3.png"),
        ("hole_candidates", "images/hole_candidates.png"),
        ("left_depth", "maps/left_depth.png"),
        ("right_depth", "maps/right_depth.png"),
        ("left_opacity", "maps/left_opacity.png"),
        ("right_opacity", "maps/right_opacity.png"),
        ("left_scale_mean", "maps/left_scale_mean.png"),
        ("right_scale_mean", "maps/right_scale_mean.png"),
        ("projection_density", "projection/all_gaussians_density.png"),
        ("projection_coverage", "projection/all_gaussians_coverage.png"),
    ]
    for name, rel in pairs:
        montage_pair(origin_dir / rel, current_dir / rel, out / f"{name}_compare.png", name)
    print(json.dumps({"out": str(out), "report": str(out / "REPORT.md")}, indent=2))


if __name__ == "__main__":
    main()
