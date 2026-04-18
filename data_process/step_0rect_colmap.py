import numpy as np
import os
import cv2
from pathlib import Path
import json
from tqdm import tqdm
import argparse
from colmap_read_model import read_cameras_binary, read_images_binary, qvec2rotmat

data_root = '/data/sifang/GPS_plus_data/test_s1'
processed_data_root = '/data/sifang/GPS_plus_data/test_s1_processed/'
N_CAMS = 4


def load_colmap_params(sparse_dir):
    """Returns cameras dict and frame_data[fid][cam_id] = extr (3x4)."""
    camdata = read_cameras_binary(os.path.join(sparse_dir, 'cameras.bin'))
    cameras = {}
    for cam_id, cam in camdata.items():
        f, cx, cy, k1 = cam.params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]], dtype=np.float64)
        dist = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
        cameras[cam_id] = {'K': K, 'dist': dist, 'W': cam.width, 'H': cam.height}

    imgdata = read_images_binary(os.path.join(sparse_dir, 'images.bin'))
    frame_data = {}
    for img in imgdata.values():
        parts = img.name.split('/')
        cam_id = int(parts[0].split('_')[1])
        fid = int(parts[1].split('_')[1].split('.')[0])
        R = qvec2rotmat(img.qvec)
        t = img.tvec.reshape(3, 1)
        extr = np.hstack([R, t])
        frame_data.setdefault(fid, {})[cam_id] = extr

    # keep only frames with all 4 cameras
    frame_data = {fid: v for fid, v in frame_data.items() if len(v) == N_CAMS}
    return cameras, frame_data


def _estimate_RT(img0, K0, d0, img1, K1, d1):
    u0 = cv2.undistort(img0, K0, d0)
    u1 = cv2.undistort(img1, K1, d1)
    sift = cv2.SIFT_create(3000)
    kp0, des0 = sift.detectAndCompute(cv2.cvtColor(u0, cv2.COLOR_BGR2GRAY), None)
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(u1, cv2.COLOR_BGR2GRAY), None)
    matches = [m for m, n in cv2.BFMatcher().knnMatch(des0, des1, k=2) if m.distance < 0.7 * n.distance]
    pts0 = np.float32([kp0[m.queryIdx].pt for m in matches])
    pts1 = np.float32([kp1[m.trainIdx].pt for m in matches])
    pts0n = cv2.undistortPoints(pts0.reshape(-1,1,2), K0, None).reshape(-1,2)
    pts1n = cv2.undistortPoints(pts1.reshape(-1,1,2), K1, None).reshape(-1,2)
    E, mask = cv2.findEssentialMat(pts0n, pts1n, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
    inliers = mask.ravel() == 1
    _, R, T, _ = cv2.recoverPose(E, pts0n[inliers], pts1n[inliers])
    return R, T


def get_rectified_stereo(img0, K0, d0, extr0, img1, K1, d1, extr1, W, H):
    R, T = _estimate_RT(img0, K0, d0, img1, K1, d1)
    Rr0, Rr1, P0, P1, _, _, _ = cv2.stereoRectify(K0, np.zeros(5), K1, np.zeros(5), (W, H), R, T, flags=0)

    map0x, map0y = cv2.initUndistortRectifyMap(K0, d0, Rr0, P0, (W, H), cv2.CV_32FC1)
    map1x, map1y = cv2.initUndistortRectifyMap(K1, d1, Rr1, P1, (W, H), cv2.CV_32FC1)

    mask = np.ones((*img0.shape[:2], 3), dtype=np.uint8) * 255
    return {
        'img0':   cv2.remap(img0, map0x, map0y, cv2.INTER_LINEAR),
        'mask0':  cv2.remap(mask, map0x, map0y, cv2.INTER_LINEAR),
        'img1':   cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR),
        'mask1':  cv2.remap(mask, map1x, map1y, cv2.INTER_LINEAR),
        'camera': {
            'intr0': Rr0 @ extr0,  # will be overwritten below
            'intr1': Rr1 @ extr1,
            'extr0': Rr0 @ extr0,
            'extr1': Rr1 @ extr1,
            'Tf_x':  np.array(P1[0, 3]),
            '_intr0': P0[:3, :3],
            '_intr1': P1[:3, :3],
        }
    }


def save_json(parm, path):
    with open(path, 'w') as f:
        json.dump({k: v.tolist() for k, v in parm.items()}, f, indent=1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--trainval', required=True)
    parser.add_argument('-n', '--setsize', type=int, default=4)
    parser.add_argument('-s', '--size', type=int, default=None)
    arg = parser.parse_args()

    cameras, frame_data = load_colmap_params(os.path.join(data_root, 'sparse/0'))
    all_fids = sorted(frame_data.keys())
    n = len(all_fids)
    if arg.trainval == 'train':
        fids = all_fids[:-(n // 8)]
    elif arg.trainval == 'val':
        fids = all_fids[-(n // 8):]
    else:
        raise ValueError(arg.trainval)

    out_root = Path(processed_data_root + arg.trainval)
    img_dir = out_root / 'img';   img_dir.mkdir(parents=True, exist_ok=True)
    msk_dir = out_root / 'mask';  msk_dir.mkdir(parents=True, exist_ok=True)
    par_dir = out_root / 'parameter'; par_dir.mkdir(parents=True, exist_ok=True)

    s_set = arg.setsize
    n_set = (N_CAMS - 1) // (s_set - 1)
    # source pair: cam_1(id=1) and cam_4(id=4); middle: cam_2, cam_3
    src_pairs = [(1, 4)]  # cam ids for source views (n_set=1)
    mid_cams  = [2, 3]

    W, H = cameras[1]['W'], cameras[1]['H']
    out_sz = (arg.size, arg.size) if arg.size else None

    def maybe_resize(im):
        return cv2.resize(im, out_sz) if out_sz else im

    def scale_K(K):
        if not out_sz:
            return K
        return np.diag([out_sz[0]/W, out_sz[1]/H, 1.0]) @ K

    for set_i, (ci0, ci1) in enumerate(src_pairs):
        scene_n = f's{set_i+1}'
        for fid in tqdm(fids, desc=f'set{set_i+1}'):
            t_img = img_dir / f'{scene_n}_{fid:04d}'; t_img.mkdir(exist_ok=True)
            t_msk = msk_dir / f'{scene_n}_{fid:04d}'; t_msk.mkdir(exist_ok=True)
            t_par = par_dir / f'{scene_n}_{fid:04d}'; t_par.mkdir(exist_ok=True)

            extr0 = frame_data[fid][ci0]
            extr1 = frame_data[fid][ci1]
            K0, d0 = cameras[ci0]['K'], cameras[ci0]['dist']
            K1, d1 = cameras[ci1]['K'], cameras[ci1]['dist']

            img0 = cv2.imread(os.path.join(data_root, f'cam_{ci0}', f'frame_{fid:06d}.jpg'))
            img1 = cv2.imread(os.path.join(data_root, f'cam_{ci1}', f'frame_{fid:06d}.jpg'))

            rect = get_rectified_stereo(img0, K0, d0, extr0, img1, K1, d1, extr1, W, H)

            # fix camera dict: use P[:3,:3] for intrinsics
            cam = rect['camera']
            cam_out = {
                'intr0': scale_K(cam['_intr0']),
                'intr1': scale_K(cam['_intr1']),
                'extr0': cam['extr0'],
                'extr1': cam['extr1'],
                'Tf_x':  cam['Tf_x'] * (out_sz[0]/W if out_sz else 1.0),
            }

            cv2.imwrite(str(t_img / '0.jpg'), maybe_resize(rect['img0']).astype(np.uint8))
            cv2.imwrite(str(t_img / '1.jpg'), maybe_resize(rect['img1']).astype(np.uint8))
            cv2.imwrite(str(t_msk / '0.jpg'), maybe_resize(rect['mask0']).astype(np.uint8))
            cv2.imwrite(str(t_msk / '1.jpg'), maybe_resize(rect['mask1']).astype(np.uint8))
            save_json(cam_out, str(t_par / '0_1.json'))

            # middle views
            for mid_i, cam_id in enumerate(mid_cams, start=2):
                extr = frame_data[fid][cam_id]
                K_out = scale_K(cameras[cam_id]['K'])
                np.save(str(t_par / f'{mid_i}_extrinsic.npy'), extr)
                np.save(str(t_par / f'{mid_i}_intrinsic.npy'), K_out)
                src = os.path.join(data_root, f'cam_{cam_id}', f'frame_{fid:06d}.jpg')
                img = cv2.imread(src)
                cv2.imwrite(str(t_img / f'{mid_i}.jpg'), maybe_resize(img))
