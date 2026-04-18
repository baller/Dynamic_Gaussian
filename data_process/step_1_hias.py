import os
import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
from step_0rect_hias import load_cam_params, get_frame_ids, data_root, processed_data_root, N_CAMS

# python step_1_hias.py -t train -n 4

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trainval', required=True)
parser.add_argument('-n', '--setsize', type=int, default=4)
parser.add_argument('-s', '--size', type=int, default=None, help='resize output images to SxS')
arg = parser.parse_args()

out_root = processed_data_root + arg.trainval
s_set = arg.setsize

intrs, dists, extrs, img_sizes = load_cam_params(os.path.join(data_root, 'intri_extri'))

n_set = (N_CAMS - 1) // (s_set - 1)
# supervision/novel views: all cams except the two source cams (first & last of set)
# same logic as original: indices 1..s_set-2 plus the two endpoints
cam_id_list_s = []
for i in range(n_set):
    left, right = i * (s_set - 1), (i + 1) * (s_set - 1)
    middle = list(range(left + 1, right))
    cam_id_list_s.append(middle + [left, right])

frame_ids = get_frame_ids(arg.trainval)

img_dir = Path(out_root) / 'img'
par_dir = Path(out_root) / 'parameter'
img_dir.mkdir(parents=True, exist_ok=True)
par_dir.mkdir(parents=True, exist_ok=True)

for set_i, cam_id_list in enumerate(cam_id_list_s):
    scene_n = f's{set_i + 1}'
    for cam_i, cam in enumerate(cam_id_list):
        extr = extrs[cam].copy()
        intr = intrs[cam].copy()
        for fid in tqdm(frame_ids, desc=f'set{set_i+1} cam{cam}'):
            t_img = img_dir / f'{scene_n}_{fid:04d}'
            t_par = par_dir / f'{scene_n}_{fid:04d}'
            t_img.mkdir(exist_ok=True)
            t_par.mkdir(exist_ok=True)

            np.save(str(t_par / f'{cam_i+2}_extrinsic.npy'), extr)

            out_intr = intr.copy()
            if arg.size:
                W0, H0 = img_sizes[0]
                sx, sy = arg.size / W0, arg.size / H0
                out_intr = np.diag([sx, sy, 1.0]) @ intr
            np.save(str(t_par / f'{cam_i+2}_intrinsic.npy'), out_intr)

            src = os.path.join(data_root, f'cam_{cam+1}', f'frame_{fid:06d}.jpg')
            img = cv2.imread(src)
            if arg.size:
                img = cv2.resize(img, (arg.size, arg.size))
            cv2.imwrite(str(t_img / f'{cam_i+2}.jpg'), img)
