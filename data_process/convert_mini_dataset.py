#!/usr/bin/env python3
"""
Convert mini_dataset → GPS_plus format.

No view transformation — video frames saved as-is (resized to 1024×1024).
All camera views numbered 0, 1, 2, …, N-1.
Camera intrinsics adjusted for crop+resize, extrinsics kept unchanged.
Masks are all-white (全图有效).
Split: train 70% / val 20% / test 10%.

Output structure:
  processed_mini/{dataset}/{scene}/{split}/
    img/{scene}_{frame:04d}/0.jpg … {N-1}.jpg
    mask/{scene}_{frame:04d}/0.jpg … {N-1}.jpg
    parameter/{scene}_{frame:04d}/
      cameras.json          ← all cameras' intrinsic + extrinsic
"""

import os, json, re
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm

MINI_ROOT   = '/data/sifang/GPS_plus_data/mini_dataset'
OUTPUT_ROOT = '/data/sifang/GPS_plus_data/processed_mini'
TARGET      = 1024
SPLIT_RATIO = (0.7, 0.2, 0.1)  # train / val / test

DATASETS = {
    'enerf_outdoor': ['actor1_4', 'actor2_3', 'actor5_6'],
    'mobile_stage':  ['dance3'],
    'my_zjumocap':   ['my_313', 'my_315', 'my_377', 'my_387'],
    'NHR':           ['basketball', 'sport_1_easymocap',
                      'sport_2_easymocap', 'sport_3_easymocap'],
    'renbody':       ['0008_01', '0012_01', '0012_11',
                      '0013_01', '0013_03', '0013_09',
                      '0013_11', '0019_08', '0023_06'],
}

# ───────── YML camera parsing ─────────
def find_yml_dir(scene_dir, ds_dir):
    for d in [os.path.join(scene_dir, 'optimized'), scene_dir,
              os.path.join(ds_dir, 'optimized'), ds_dir]:
        if os.path.isfile(os.path.join(d, 'intri.yml')):
            return d
    return None

def parse_cameras(yml_dir):
    fi = cv2.FileStorage(os.path.join(yml_dir, 'intri.yml'), cv2.FILE_STORAGE_READ)
    fe = cv2.FileStorage(os.path.join(yml_dir, 'extri.yml'), cv2.FILE_STORAGE_READ)
    nn = fi.getNode('names')
    names = [nn.at(i).string() for i in range(nn.size())]
    cams = {}
    for n in names:
        K = fi.getNode(f'K_{n}').mat()
        Dn = fi.getNode(f'D_{n}')
        if Dn.empty():
            Dn = fi.getNode(f'dist_{n}')
        D = Dn.mat().flatten() if not Dn.empty() else np.zeros(5)
        Rot = fe.getNode(f'Rot_{n}').mat()
        T   = fe.getNode(f'T_{n}').mat()
        E = np.zeros((3, 4))
        E[:3, :3] = Rot
        E[:3, 3:] = T.reshape(3, 1)
        cams[n] = dict(K=K, D=D, extr=E)
    fi.release(); fe.release()
    return names, cams

# ───────── video filename parsing ─────────
_VID_RE = re.compile(
    r'(?P<cam>\w+)_N(?P<N>\d+)_x(?P<x>\d+)_y(?P<y>\d+)'
    r'_W(?P<W>\d+)_H(?P<H>\d+)_FW(?P<FW>\d+)_FH(?P<FH>\d+)\.mp4$')

def build_video_map(vid_dir):
    vmap = {}
    for fname in sorted(os.listdir(vid_dir)):
        if 'mask' in fname.lower() or not fname.endswith('.mp4'):
            continue
        m = _VID_RE.match(fname)
        if m:
            vmap[m['cam']] = dict(
                fname=fname,
                N=int(m['N']), x=int(m['x']), y=int(m['y']),
                W=int(m['W']), H=int(m['H']),
                FW=int(m['FW']), FH=int(m['FH']))
    return vmap

# ───────── intrinsic adjustment (no view transform, just account for crop+resize) ─────────
def adjust_K(K_full, vinfo, vid_w, vid_h):
    """
    Adjust intrinsic from full-image (FW×FH) to the actual video frame,
    then to a 1024×1024 square output.
    """
    K = K_full.copy()
    # scale from FW×FH → W×H
    K[0, :] *= vinfo['W'] / vinfo['FW']
    K[1, :] *= vinfo['H'] / vinfo['FH']
    # crop offset
    K[0, 2] -= vinfo['x']
    K[1, 2] -= vinfo['y']
    # crop-to-square (center crop on the longer side)
    if vid_h > vid_w:
        K[1, 2] -= (vid_h - vid_w) / 2
        sq = vid_w
    elif vid_w > vid_h:
        K[0, 2] -= (vid_w - vid_h) / 2
        sq = vid_h
    else:
        sq = vid_w
    # resize to TARGET
    s = TARGET / sq
    K[0, :] *= s
    K[1, :] *= s
    return K

WHITE = np.ones((TARGET, TARGET, 3), dtype=np.uint8) * 255

def crop_sq_resize(frame):
    """Center-crop to square, resize to TARGET×TARGET."""
    h, w = frame.shape[:2]
    if h > w:
        off = (h - w) // 2
        frame = frame[off:off + w]
    elif w > h:
        off = (w - h) // 2
        frame = frame[:, off:off + h]
    return cv2.resize(frame, (TARGET, TARGET))

def get_split(fi, nf):
    t1 = int(nf * SPLIT_RATIO[0])
    t2 = int(nf * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))
    if fi < t1:
        return 'train'
    elif fi < t2:
        return 'val'
    else:
        return 'test'

# ───────── process one scene ─────────
def process_scene(ds_name, scene_name):
    ds_dir    = os.path.join(MINI_ROOT, ds_name)
    scene_dir = os.path.join(ds_dir, scene_name)
    vid_dir   = os.path.join(scene_dir, 'videos_libx265')
    out_dir   = os.path.join(OUTPUT_ROOT, ds_name, scene_name)

    yml_dir = find_yml_dir(scene_dir, ds_dir)
    if yml_dir is None:
        print(f'    [skip] no YML found'); return
    cam_names, cameras = parse_cameras(yml_dir)

    vmap = build_video_map(vid_dir)
    avail = sorted(c for c in vmap if c in cameras)
    n_cams = len(avail)
    if n_cams < 2:
        print(f'    [skip] only {n_cams} cameras'); return

    nf = vmap[avail[0]]['N']
    t1 = int(nf * SPLIT_RATIO[0])
    t2 = int(nf * (SPLIT_RATIO[0] + SPLIT_RATIO[1]))
    print(f'    cams={n_cams}  frames={nf}  train={t1}  val={t2-t1}  test={nf-t2}')

    # open all video captures
    caps = []
    vid_sizes = []
    for c in avail:
        cap = cv2.VideoCapture(os.path.join(vid_dir, vmap[c]['fname']))
        caps.append(cap)
        vid_sizes.append((int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # pre-compute adjusted intrinsics for each camera
    adj_Ks = []
    for i, c in enumerate(avail):
        adj_Ks.append(adjust_K(cameras[c]['K'], vmap[c], vid_sizes[i][0], vid_sizes[i][1]))

    # build per-camera param dict (saved once per sample)
    cam_params = {}
    for i, c in enumerate(avail):
        cam_params[str(i)] = dict(
            intrinsic=adj_Ks[i].tolist(),
            extrinsic=cameras[c]['extr'].tolist(),
            original_cam_id=c)

    for fi in tqdm(range(nf), desc=f'    {scene_name}', leave=True):
        split = get_split(fi, nf)
        name = f'{scene_name}_{fi:04d}'
        img_d = os.path.join(out_dir, split, 'img', name)
        msk_d = os.path.join(out_dir, split, 'mask', name)
        par_d = os.path.join(out_dir, split, 'parameter', name)
        os.makedirs(img_d, exist_ok=True)
        os.makedirs(msk_d, exist_ok=True)
        os.makedirs(par_d, exist_ok=True)

        all_ok = True
        for i in range(n_cams):
            ok, frame = caps[i].read()
            if not ok:
                all_ok = False; break
            img_out = crop_sq_resize(frame)
            cv2.imwrite(os.path.join(img_d, f'{i}.jpg'), img_out)
            cv2.imwrite(os.path.join(msk_d, f'{i}.jpg'), WHITE)

        if not all_ok:
            break

        # save camera parameters
        with open(os.path.join(par_d, 'cameras.json'), 'w') as f:
            json.dump(cam_params, f, indent=1)

    for cap in caps:
        cap.release()

# ───────── main ─────────
def main():
    for ds_name, scenes in DATASETS.items():
        print(f'\n{"="*60}\n  {ds_name}\n{"="*60}')
        for scene_name in scenes:
            print(f'\n  >> {scene_name}')
            process_scene(ds_name, scene_name)
    print(f'\nDone. Output -> {OUTPUT_ROOT}')

if __name__ == '__main__':
    main()
