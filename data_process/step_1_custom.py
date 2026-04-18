from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from step_0rect_custom import (
    DEFAULT_DATA_ROOT,
    DEFAULT_PROCESSED_ROOT,
    NOVEL_RAW_ORDER,
    build_crop_resize,
    build_dataset_context,
    ensure_split_dirs,
    get_split_samples,
    load_raw_image,
    make_sample_name,
    process_single_view_image,
    transform_intrinsic_for_output,
)


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


def main() -> None:
    args = parse_args()
    context = build_dataset_context(args.data_root, setsize=args.setsize)
    crop_resize = build_crop_resize(context["image_size"], args.size)
    split_samples = get_split_samples(context["samples"], args.trainval)
    _, img_dir, _, param_dir = ensure_split_dirs(args.processed_root, args.trainval)

    processed_intrinsics = {
        raw_cam: transform_intrinsic_for_output(context["camera_models"][raw_cam]["K"], crop_resize)
        for raw_cam in NOVEL_RAW_ORDER
    }
    rig_extrinsics = {raw_cam: context["rig_extrinsics"][raw_cam] for raw_cam in NOVEL_RAW_ORDER}

    for raw_frame, _ in tqdm(split_samples, desc=f"export novel views {args.trainval}"):
        sample_name = make_sample_name(raw_frame)
        sample_img_dir = img_dir / sample_name
        sample_param_dir = param_dir / sample_name
        sample_img_dir.mkdir(exist_ok=True)
        sample_param_dir.mkdir(exist_ok=True)

        for view_offset, raw_cam in enumerate(NOVEL_RAW_ORDER, start=2):
            img = load_raw_image(context, raw_frame, raw_cam)
            model = context["camera_models"][raw_cam]
            img_out = process_single_view_image(img, model["K"], model["dist"], crop_resize)
            cv2.imwrite(str(sample_img_dir / f"{view_offset}.jpg"), img_out.astype(np.uint8))
            np.save(str(sample_param_dir / f"{view_offset}_intrinsic.npy"), processed_intrinsics[raw_cam])
            np.save(str(sample_param_dir / f"{view_offset}_extrinsic.npy"), rig_extrinsics[raw_cam])

    print(f"Wrote {len(split_samples)} {args.trainval} novel-view samples to {args.processed_root / args.trainval}")


if __name__ == "__main__":
    main()
