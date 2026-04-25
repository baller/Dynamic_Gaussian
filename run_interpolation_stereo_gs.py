"""
StereoGS 自由视角插值渲染脚本

基于 run_interpolation_ffs.py，替换为 StereoGS 模型：
  - 使用 RtStereoHumanModel (depth_mode='stereo_gs') 加载模型
  - get_item_free 保留 Tf_x / ref_intr（FFS 需要用于主点偏移补偿）
  - 使用原版 pts2render 完成高斯渲染
  - 支持可选的渲染后精化 (post_refine)
  - 输出 RGB 帧、逆深度伪彩图、置信度图，并自动合成视频

用法:
    python run_interpolation_stereo_gs.py -i s1a6 \\
        --ckpt experiments/stereo_gs_verify/ckpt/stereo_gs_final.pth \\
        --config config/stereo_gs_stage.yaml \\
        --cams 22139908,22139909 --loop 30 --frames 600

    # 多段相机路径:
    python run_interpolation_stereo_gs.py -i s1a6 \\
        --ckpt experiments/stereo_gs_verify/ckpt/stereo_gs_final.pth \\
        --config config/stereo_gs_stage.yaml \\
        --cams 22139908,22139909:22139909,22139914:22139914,22139906 \\
        --sids 1,2,3 --loop 20 --frames 600
"""

from __future__ import print_function, division

import argparse
import atexit
import glob
import logging
import re
import signal
import subprocess
import sys

import numpy as np
import cv2
import os
import json
from pathlib import Path
from tqdm import tqdm

from lib.human_loader import load_json_to_np
from lib.network import RtStereoHumanModel
from config.stereo_human_config import ConfigStereoHuman as config
from lib.GaussianRender import pts2render, pts2render_cags
from lib.gs_utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov

from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp

import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from PIL import Image


# ──────────────────────────────────────────────────────────
#  视频生成工具
# ──────────────────────────────────────────────────────────

_videos_generated = False
_show_path_global: str = ''
_video_dir_global: str = ''


def get_iter_label(ckpt_path: str) -> str:
    stem = Path(ckpt_path).stem
    if 'latest' in stem.lower():
        return 'latest'
    m = re.search(r'(\d{5,8})$', stem)
    if m:
        return f'iter_{int(m.group(1))}'
    return stem


def make_videos(show_path: str, video_dir: str) -> None:
    tasks = [
        ('RGB',        '%05d.jpg',            os.path.join(video_dir, 'rgb.mp4')),
        ('depth',      '%05d_depth.png',      os.path.join(video_dir, 'depth.mp4')),
        ('confidence', '%05d_confidence.png',  os.path.join(video_dir, 'confidence.mp4')),
    ]
    for label, fmt, out_mp4 in tasks:
        ext = fmt.split('.')[-1]
        existing = sorted(glob.glob(os.path.join(show_path, f'*.{ext}')))
        if not existing:
            continue
        os.makedirs(video_dir, exist_ok=True)
        start_num = int(Path(existing[0]).stem.replace('_depth', '').replace('_confidence', ''))
        cmd = [
            'ffmpeg', '-y',
            '-framerate', '30',
            '-start_number', str(start_num),
            '-i', os.path.join(show_path, fmt),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            out_mp4,
        ]
        logging.info(f'生成 {label} 视频（共 {len(existing)} 帧）: {out_mp4}')
        ret = subprocess.run(cmd, capture_output=True)
        if ret.returncode != 0:
            logging.warning(f'{label} 视频生成失败:\n{ret.stderr.decode(errors="replace")}')
        else:
            logging.info(f'{label} 视频已保存至: {out_mp4}')


def _finalize_videos() -> None:
    global _videos_generated
    if _videos_generated or not _show_path_global:
        return
    _videos_generated = True
    logging.info('正在生成渲染结果视频...')
    make_videos(_show_path_global, _video_dir_global)


def _signal_handler(sig, frame) -> None:
    logging.info(f'收到中断信号 ({sig})，先生成视频再退出...')
    _finalize_videos()
    sys.exit(0)


# ──────────────────────────────────────────────────────────
#  相机参数读取 & 插值
# ──────────────────────────────────────────────────────────

def read_calib(calib, RS=1024):
    R = np.array(calib['R']).reshape((3, 3))
    T = np.array(calib['T']).reshape((3, 1))
    extr = np.zeros((3, 4))
    extr[:3, :3] = R
    extr[:3, 3:] = T
    intr = np.zeros((3, 3))
    intr[:3, :3] = np.array(calib['K']).reshape((3, 3))
    img_size = calib.get('imgSize', None)
    if img_size is not None and tuple(img_size) != (RS, RS):
        W, H = int(img_size[0]), int(img_size[1])
    else:
        W, H = 1500, 2048
    if W > H:
        intr[0, 2] -= (W - H) / 2
        intr[:2] *= RS / H
    else:
        intr[1, 2] -= (H - W) / 2
        intr[:2] *= RS / W
    return extr, intr


def extr_interpolate(cfg, tar_n, cam_id_list, loop_num, RS=1024):
    novel_extrs = []
    novel_intrs = []

    calib_path = os.path.join(
        cfg.dataset.local_data_root, 'test', tar_n + '_process', 'calibration_full.json')
    with open(calib_path, 'r') as f:
        calib_full = json.load(f)

    extr0, intr0 = read_calib(calib_full[cam_id_list[0]], RS)
    extr1, intr1 = read_calib(calib_full[cam_id_list[1]], RS)

    pose_0 = np.eye(4); pose_0[:3] = extr0; pose_0 = np.linalg.inv(pose_0)
    pose_1 = np.eye(4); pose_1[:3] = extr1; pose_1 = np.linalg.inv(pose_1)

    for ratio in np.linspace(0., 0.95, loop_num):
        rots = Rot.from_matrix(np.stack([pose_0[:3, :3], pose_1[:3, :3]]))
        rot = Slerp([0, 1], rots)(ratio)
        pose = np.eye(4, dtype=np.float32)
        pose[:3, :3] = rot.as_matrix()
        pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
        pose = np.linalg.inv(pose)
        novel_extrs.append(pose[:3])
        novel_intrs.append((1.0 - ratio) * intr0 + ratio * intr1)

    return novel_extrs, novel_intrs


# ──────────────────────────────────────────────────────────
#  StereoGS 推理渲染器
# ──────────────────────────────────────────────────────────

class StereoGSRenderModel(nn.Module):
    """
    StereoGS 自由视角渲染器。

    与 FFSRenderModel 的关键区别：
    - checkpoint 中可能不包含 FFS 权重（训练时冻结且不保存）
    - 支持可选的 post-refinement（渲染后精化）
    - 需要冻结 StereoGS 内部的 FFS 子模块 BN
    """

    def __init__(self, cfg, ckpt_path, novel_extrs, novel_intrs, tar_n, s_id=1):
        super().__init__()
        self.cfg = cfg
        self.tar_n = tar_n
        self.use_post_refine = getattr(cfg.stereo_gs, 'use_post_refine', False)

        self.model = RtStereoHumanModel(cfg, with_gs_render=True)

        assert os.path.exists(ckpt_path), f'Checkpoint 不存在: {ckpt_path}'
        ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt['network'], strict=False)

        ffs_missing = [k for k in missing if 'ffs_extractor' in k or 'ffs.' in k]
        other_missing = [k for k in missing if k not in ffs_missing]
        if ffs_missing:
            logging.info(f'  FFS 权重由模型初始化时加载（跳过 {len(ffs_missing)} 个 checkpoint 键）')
        if other_missing:
            logging.warning(f'  缺失 {len(other_missing)} 个非 FFS 键: {other_missing[:5]}...')
        if unexpected:
            logging.warning(f'  忽略了 {len(unexpected)} 个不匹配的键')

        self.model = self.model.cuda()
        self.model.eval()
        self._freeze_bn()

        self.novel_extrs = novel_extrs
        self.novel_intrs = novel_intrs
        self.s_id = s_id

        self.img_path = os.path.join(
            cfg.dataset.local_data_root, 'test', tar_n + '_process', 'img')
        self.msk_path = os.path.join(
            cfg.dataset.local_data_root, 'test', tar_n + '_process', 'mask')

        parm_name = os.path.join(
            cfg.dataset.local_data_root, 'test', tar_n + '_process', 'parameter',
            f'{tar_n}_s{s_id}_0000',
            f'{cfg.dataset.source_id[0]}_{cfg.dataset.source_id[1]}.json')
        camera = load_json_to_np(parm_name)

        self.Tf_x = np.array([camera['Tf_x']])
        self.intrinsics = [torch.FloatTensor(camera['intr0']),
                           torch.FloatTensor(camera['intr1'])]
        self.extrinsics = [torch.FloatTensor(camera['extr0']),
                           torch.FloatTensor(camera['extr1'])]

    def _freeze_bn(self):
        if hasattr(self.model, 'stereo_gs_model') and self.model.stereo_gs_model is not None:
            self.model.stereo_gs_model.freeze_bn()

    def _load_and_normalize(self, path):
        img = np.array(Image.open(path)).astype(np.float32)
        return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).cuda()

    def get_item_free(self, frame_id, view_id):
        src0, src1 = self.cfg.dataset.source_id
        prefix = f'{self.tar_n}_s{self.s_id}_{frame_id:04d}'

        img0 = self._load_and_normalize(f'{self.img_path}/{prefix}/{src0}.jpg')
        img1 = self._load_and_normalize(f'{self.img_path}/{prefix}/{src1}.jpg')
        msk0 = self._load_and_normalize(f'{self.msk_path}/{prefix}/{src0}.jpg')
        msk1 = self._load_and_normalize(f'{self.msk_path}/{prefix}/{src1}.jpg')

        img0 = 2 * (img0 / 255.0) - 1.0
        img1 = 2 * (img1 / 255.0) - 1.0
        msk0 /= 255.0
        msk1 /= 255.0

        intr0 = self.intrinsics[0].unsqueeze(0).cuda()
        intr1 = self.intrinsics[1].unsqueeze(0).cuda()
        extr0 = self.extrinsics[0].unsqueeze(0).cuda()
        extr1 = self.extrinsics[1].unsqueeze(0).cuda()
        Tf_x = torch.FloatTensor([self.Tf_x[0]]).cuda()

        l_view = {
            'img': img0, 'mask': msk0,
            'intr': intr0.clone(), 'ref_intr': intr1.clone(),
            'extr': extr0, 'Tf_x': Tf_x,
        }
        r_view = {
            'img': img1, 'mask': msk1,
            'intr': intr1.clone(), 'ref_intr': intr0.clone(),
            'extr': extr1, 'Tf_x': -Tf_x,
        }

        novel_intr = self.novel_intrs[view_id]
        novel_extr = self.novel_extrs[view_id]

        width, height = 1024, 1024
        R = np.array(novel_extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(novel_extr[:3, 3], np.float32)

        FovX = focal2fov(novel_intr[0, 0], width)
        FovY = focal2fov(novel_intr[1, 1], height)
        projection_matrix = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY,
            K=novel_intr, h=height, w=width).transpose(0, 1)
        world_view_transform = torch.tensor(
            getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view = {
            'height': [height], 'width': [width],
            'FovX': [torch.FloatTensor(np.array(FovX)).cuda()],
            'FovY': [torch.FloatTensor(np.array(FovY)).cuda()],
            'world_view_transform': [world_view_transform.cuda()],
            'full_proj_transform': [full_proj_transform.cuda()],
            'camera_center': [camera_center.cuda()],
        }

        return {'lmain': l_view, 'rmain': r_view, 'novel_view': novel_view}


# ──────────────────────────────────────────────────────────
#  可视化工具
# ──────────────────────────────────────────────────────────

def inv_depth_to_colormap(inv_depth_tensor, mask=None):
    d = inv_depth_tensor[0, 0].detach().cpu().float().numpy()
    if mask is not None:
        m = mask[0, 0].detach().cpu().numpy() > 0.5
        valid = d[m]
    else:
        valid = d[d > 1e-6]

    if valid.size == 0:
        return np.zeros((*d.shape, 3), dtype=np.uint8)

    lo, hi = np.percentile(valid, [2, 98])
    d_clip = np.clip(d, lo, hi)
    d_norm = ((d_clip - lo) / (hi - lo + 1e-8) * 255).astype(np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)


def confidence_to_colormap(conf_tensor):
    """置信度 (B,1,H,W) → 伪彩色 numpy (H,W,3) BGR"""
    c = conf_tensor[0, 0].detach().cpu().float().numpy()
    c_u8 = (np.clip(c, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(c_u8, cv2.COLORMAP_VIRIDIS)


# ──────────────────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='StereoGS 自由视角插值渲染')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='目标序列名称（如 s1a6）')
    parser.add_argument('--ckpt', type=str, required=True,
                        help='StereoGS checkpoint 路径')
    parser.add_argument('--config', type=str, default='config/stereo_gs_stage.yaml',
                        help='配置文件路径')
    parser.add_argument('--cams', type=str, default='22139908,22139909',
                        help='相机对，格式: cam0,cam1 或多段 cam0,cam1:cam2,cam3:...')
    parser.add_argument('--sids', type=str, default='1',
                        help='每段对应的 s_id，逗号分隔（如 1,2,3）')
    parser.add_argument('--loop', type=int, default=20,
                        help='每段插值帧数')
    parser.add_argument('--frames', type=int, default=600,
                        help='渲染总帧数')
    parser.add_argument('--start', type=int, default=0,
                        help='起始帧 ID')
    parser.add_argument('--crop', type=int, default=0,
                        help='四边裁剪像素数')
    parser.add_argument('--no-depth', dest='save_depth', action='store_false',
                        help='不保存深度图')
    parser.add_argument('--no-confidence', dest='save_confidence', action='store_false',
                        help='不保存置信度图')
    arg = parser.parse_args()

    tar_n = arg.input
    LOOP_NUM = arg.loop
    RS = 1024
    crop = arg.crop

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    cfg_obj = config()
    cfg_obj.load(arg.config)
    cfg = cfg_obj.get_cfg()

    cam_segments = arg.cams.split(':')
    cam_pairs = [seg.strip().split(',') for seg in cam_segments]
    for pair in cam_pairs:
        assert len(pair) == 2, f'每段需要恰好 2 个相机 ID，得到: {pair}'

    sid_list = [int(s) for s in arg.sids.split(',')]
    if len(sid_list) == 1:
        sid_list = sid_list * len(cam_pairs)
    assert len(sid_list) == len(cam_pairs), \
        f'--sids 数量 ({len(sid_list)}) 必须与相机段数 ({len(cam_pairs)}) 一致'

    iter_label = get_iter_label(arg.ckpt)
    ckpt_parent = Path(arg.ckpt).parent.parent
    cam_label = '_'.join(cam_pairs[0])
    result_name = f'free_{tar_n}_{cam_label}_{iter_label}'
    show_path = str(ckpt_parent / 'test_result' / result_name)
    video_dir = os.path.join(show_path, 'video')

    cfg.defrost()
    cfg.exp_name = 'stereo_gs'
    cfg.record.show_path = show_path
    cfg.restore_ckpt = arg.ckpt
    cfg.freeze()

    from_list = [cfg.dataset.source_id[0]]
    to_list = [cfg.dataset.source_id[1]]

    Path(show_path).mkdir(exist_ok=True, parents=True)
    Path(video_dir).mkdir(exist_ok=True, parents=True)

    _show_path_global = show_path
    _video_dir_global = video_dir
    atexit.register(_finalize_videos)
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    use_post_refine = getattr(cfg.stereo_gs, 'use_post_refine', False)
    use_cags = getattr(cfg.stereo_gs, 'use_cags', False)
    render_fn = pts2render_cags if use_cags else pts2render
    logging.info(f'  渲染函数:     {"pts2render_cags (CAGS)" if use_cags else "pts2render"}')

    print('=' * 55)
    print('StereoGS 自由视角插值渲染')
    print('=' * 55)
    print(f'  序列:         {tar_n}')
    print(f'  相机段数:     {len(cam_pairs)}')
    for idx, (pair, sid) in enumerate(zip(cam_pairs, sid_list)):
        print(f'    [{idx}] {pair[0]} → {pair[1]}  (s_id={sid})')
    print(f'  Checkpoint:   {arg.ckpt}')
    print(f'  插值帧数:     {LOOP_NUM}  总渲染帧: {arg.frames}')
    print(f'  Post-refine:  {use_post_refine}')
    print(f'  输出目录:     {show_path}')
    print('=' * 55)

    renderers = []
    for pair, sid in zip(cam_pairs, sid_list):
        logging.info(f'初始化渲染器: cams={pair}, s_id={sid}')
        novel_extrs, novel_intrs = extr_interpolate(cfg, tar_n, pair, LOOP_NUM, RS)
        renderer = StereoGSRenderModel(cfg, arg.ckpt, novel_extrs, novel_intrs, tar_n, sid)
        renderers.append(renderer)
        logging.info(f'  插值路径帧数: {len(novel_extrs)}')

    n_seg = len(renderers)
    total_cycle = n_seg * 2

    start_frame = arg.start
    end_frame = arg.start + arg.frames

    logging.info(f'开始渲染 [{start_frame}, {end_frame}) ...')

    for fr_i in tqdm(range(start_frame, end_frame)):
        wi_ct = fr_i // LOOP_NUM
        cycle_pos = wi_ct % total_cycle
        view_idx = fr_i % LOOP_NUM

        if cycle_pos < n_seg:
            seg_idx = cycle_pos
            rev = False
        else:
            seg_idx = total_cycle - 1 - cycle_pos
            rev = True

        if rev:
            view_idx = (LOOP_NUM - 1) - view_idx

        render = renderers[seg_idx]

        with torch.no_grad():
            try:
                data = render.get_item_free(fr_i, view_idx)
                data, _, _ = render.model(data, is_train=False)
            except FileNotFoundError as e:
                logging.warning(f'帧 {fr_i} 文件缺失，跳过: {e}')
                continue

            data = render_fn(data, bg_color=cfg.dataset.bg_color)

            if use_post_refine:
                data = render.model.stereo_gs_model.refine_rendered(data)

            tmp_novel = data['novel_view']['img_pred'][0].detach()
            tmp_novel = tmp_novel.clamp(0, 1) * 255
            tmp_novel_np = tmp_novel.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            if crop > 0:
                tmp_novel_np = tmp_novel_np[crop:RS - crop, crop:RS - crop]
            cv2.imwrite(f'{show_path}/{fr_i:05d}.jpg', tmp_novel_np[:, :, ::-1])

            if arg.save_depth and 'depth' in data['lmain']:
                depth_color = inv_depth_to_colormap(
                    data['lmain']['depth'], data['lmain'].get('mask'))
                if crop > 0:
                    depth_color = depth_color[crop:RS - crop, crop:RS - crop]
                cv2.imwrite(f'{show_path}/{fr_i:05d}_depth.png', depth_color)

            if arg.save_confidence:
                extras = data.get('_stereo_gs_extras', {})
                conf = extras.get('confidence_left', None)
                if conf is not None:
                    import torch.nn.functional as F
                    conf_full = F.interpolate(conf, size=(RS, RS), mode='bilinear', align_corners=True)
                    conf_color = confidence_to_colormap(conf_full)
                    if crop > 0:
                        conf_color = conf_color[crop:RS - crop, crop:RS - crop]
                    cv2.imwrite(f'{show_path}/{fr_i:05d}_confidence.png', conf_color)

    logging.info(f'渲染完成！输出目录: {show_path}')
    _finalize_videos()
