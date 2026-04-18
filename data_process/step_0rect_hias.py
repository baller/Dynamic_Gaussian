import numpy as np
import os
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import argparse

# Data paths
data_root = '/data/sifang/GPS_plus_data/hias_man1_s2'
processed_data_root = '/data/sifang/GPS_plus_data/hias_man1_s2_processed/'

N_CAMS = 4  # cam_1 ~ cam_4, indexed 0~3


def load_cam_params(intri_extri_dir):
    """Load per-camera intrinsics and build world-space extrinsics (3x4)."""
    intrs = []
    dist_coeffs = []
    img_sizes = []
    for i in range(N_CAMS):
        with open(os.path.join(intri_extri_dir, f'cam{i}_intrinsic.json')) as f:
            d = json.load(f)
        intrs.append(np.array(d['camera_matrix'], dtype=np.float64))
        dist_coeffs.append(np.array(d['dist_coeffs'][0], dtype=np.float64))
        img_sizes.append(tuple(d['image_size']))  # (W, H)

    # cam0 is reference (identity extrinsic)
    extrs = [np.eye(4)]
    with open(os.path.join(intri_extri_dir, 'quad_extrinsics.json')) as f:
        quad = json.load(f)
    for i in range(1, N_CAMS):
        key = f'cam{i}'
        R = np.array(quad[key]['R'], dtype=np.float64)
        T = np.array(quad[key]['T'], dtype=np.float64).reshape(3, 1)
        E = np.eye(4)
        E[:3, :3] = R
        E[:3, 3:] = T
        extrs.append(E)

    # Convert to 3x4
    extrs_34 = [e[:3, :] for e in extrs]
    return intrs, dist_coeffs, extrs_34, img_sizes


def _estimate_RT_from_images(img0, img1, intr0, intr1, dist0, dist1):
    """Estimate relative R, T from image features when calibration R/T is unreliable."""
    u0 = cv2.undistort(img0, intr0, dist0)
    u1 = cv2.undistort(img1, intr1, dist1)
    sift = cv2.SIFT_create(3000)
    kp0, des0 = sift.detectAndCompute(cv2.cvtColor(u0, cv2.COLOR_BGR2GRAY), None)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(u1, cv2.COLOR_BGR2GRAY), None)
    matches = [m for m, n in cv2.BFMatcher().knnMatch(des0, des1, k=2) if m.distance < 0.7 * n.distance]
    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
    # normalize with respective K
    pts0n = cv2.undistortPoints(pts0.reshape(-1, 1, 2), intr0, None).reshape(-1, 2)
    pts1n = cv2.undistortPoints(pts1.reshape(-1, 1, 2), intr1, None).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(pts0n, pts1n, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
    inliers = mask.ravel() == 1
    _, R, T, _ = cv2.recoverPose(E, pts0n[inliers], pts1n[inliers])
    return R, T


def get_rectified_stereo_data(main_view_data, ref_view_data, img_sz):
    img0, intr0, dist0, extr0 = main_view_data
    img1, intr1, dist1, extr1 = ref_view_data

    W, H = img_sz
    R, T = _estimate_RT_from_images(img0, img1, intr0, intr1, dist0, dist1)

    R0, R1, P0, P1, _, _, _ = cv2.stereoRectify(
        intr0, np.zeros(5), intr1, np.zeros(5), (W, H), R, T, flags=0)

    # Apply rectification rotation to calibrated extrinsics (keeps world coordinate system consistent)
    new_extr0 = R0 @ extr0
    new_intr0 = P0[:3, :3]
    new_extr1 = R1 @ extr1
    new_intr1 = P1[:3, :3]

    map0x, map0y = cv2.initUndistortRectifyMap(intr0, dist0, R0, P0, (W, H), cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(intr1, dist1, R1, P1, (W, H), cv2.CV_32FC1)

    mask = np.ones((*img0.shape[:2], 3), dtype=np.uint8) * 255
    new_img0  = cv2.remap(img0,  map0x, map0y, cv2.INTER_LINEAR)
    new_mask0 = cv2.remap(mask,  map0x, map0y, cv2.INTER_LINEAR)
    new_img1  = cv2.remap(img1,  map1x, map1y, cv2.INTER_LINEAR)
    new_mask1 = cv2.remap(mask,  map1x, map1y, cv2.INTER_LINEAR)

    camera = {
        'intr0': new_intr0, 'intr1': new_intr1,
        'extr0': new_extr0, 'extr1': new_extr1,
        'Tf_x': np.array(P1[0, 3])
    }
    return {'img0': new_img0, 'mask0': new_mask0,
            'img1': new_img1, 'mask1': new_mask1,
            'camera': camera}


def save_np_to_json(parm, save_name):
    out = {k: v.tolist() for k, v in parm.items()}
    with open(save_name, 'w') as f:
        json.dump(out, f, indent=1)


def get_frame_ids(trainval):
    all_frames = sorted(
        int(p.stem.split('_')[1])
        for p in Path(os.path.join(data_root, 'cam_1')).glob('frame_*.jpg')
    )
    n = len(all_frames)
    if trainval == 'train':
        return all_frames[:-(n // 8)]
    elif trainval == 'val':
        return all_frames[-(n // 8):]
    else:
        raise ValueError(trainval)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainval', required=True)
    parser.add_argument('-n', '--setsize', type=int, default=4)
    parser.add_argument('-s', '--size', type=int, default=None, help='resize output images to SxS')
    arg = parser.parse_args()

    out_root = processed_data_root + arg.trainval
    s_set = arg.setsize

    intrs, dists, extrs, img_sizes = load_cam_params(
        os.path.join(data_root, 'intri_extri'))

    # Work sets: (0,3), overlap by 1 like original: pairs (0,1),(1,2),(2,3)
    # With s_set=4: n_set=1, cam_id_list = [0, 3]  (leftmost & rightmost)
    n_set = (N_CAMS - 1) // (s_set - 1)
    cam_id_list_s = [[i * (s_set - 1), (i + 1) * (s_set - 1)] for i in range(n_set)]

    frame_ids = get_frame_ids(arg.trainval)
    img_sz = img_sizes[0]  # all same resolution

    img_dir = Path(out_root) / 'img'
    msk_dir = Path(out_root) / 'mask'
    par_dir = Path(out_root) / 'parameter'
    for d in [img_dir, msk_dir, par_dir]:
        d.mkdir(parents=True, exist_ok=True)

    for set_i, (ci0, ci1) in enumerate(cam_id_list_s):
        scene_n = f's{set_i + 1}'
        for fid in tqdm(frame_ids, desc=f'set {set_i+1}'):
            t_img = img_dir / f'{scene_n}_{fid:04d}'
            t_msk = msk_dir / f'{scene_n}_{fid:04d}'
            t_par = par_dir / f'{scene_n}_{fid:04d}'
            for d in [t_img, t_msk, t_par]:
                d.mkdir(exist_ok=True)

            def read_img(cam_idx):
                p = os.path.join(data_root, f'cam_{cam_idx+1}', f'frame_{fid:06d}.jpg')
                return cv2.imread(p)

            mimg = read_img(ci0)
            rimg = read_img(ci1)

            rect = get_rectified_stereo_data(
                (mimg, intrs[ci0], dists[ci0], extrs[ci0]),
                (rimg, intrs[ci1], dists[ci1], extrs[ci1]),
                img_sz)

            out_sz = (arg.size, arg.size) if arg.size else None

            def maybe_resize(im):
                return cv2.resize(im, out_sz) if out_sz else im

            # adjust intrinsics if resizing
            camera = rect['camera']
            if out_sz:
                W0, H0 = img_sz
                sx, sy = out_sz[0] / W0, out_sz[1] / H0
                scale = np.diag([sx, sy, 1.0])
                camera = dict(camera)
                camera['intr0'] = scale @ camera['intr0']
                camera['intr1'] = scale @ camera['intr1']

            cv2.imwrite(str(t_img / '0.jpg'), maybe_resize(rect['img0']).astype(np.uint8))
            cv2.imwrite(str(t_img / '1.jpg'), maybe_resize(rect['img1']).astype(np.uint8))
            cv2.imwrite(str(t_msk / '0.jpg'), maybe_resize(rect['mask0']).astype(np.uint8))
            cv2.imwrite(str(t_msk / '1.jpg'), maybe_resize(rect['mask1']).astype(np.uint8))
            save_np_to_json(camera, str(t_par / '0_1.json'))

        extr0, extr1 = extrs[ci0], extrs[ci1]
        pos0 = (-extr0[:3, :3].T @ extr0[:3, 3:]).flatten()
        pos1 = (-extr1[:3, :3].T @ extr1[:3, 3:]).flatten()
        dist = np.linalg.norm(pos0 - pos1)
        print(f'set {set_i+1} baseline: {dist:.4f}m, inverse_depth_init hint: {0.5/dist:.4f}')
