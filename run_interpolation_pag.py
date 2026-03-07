"""
PAG-Splat 自由视角插值渲染脚本

结构完全对齐 GPS+ run_interpolation.py，只改变模型加载部分：
  - 使用 build_pag_splat 加载 PAGSplat 模型
  - get_item_free 中去除 GPS+ 专属的 flow_init / Tf_x / ref_intr
  - 使用 pag_pts2render 完成渲染
  - novel_view 字典格式与原版完全一致 (字段用列表包裹)

用法:
    python run_interpolation_pag.py -i s1a6 \\
        --ckpt experiments/pag_splat_0304/ckpt/pag_splat_latest.pth \\
        --cams 22139908,22139909 --loop 30 --frames 600
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
from config.stereo_human_config import ConfigStereoHuman as config
from pag_splat.model import build_pag_splat
from pag_splat.render import pag_pts2render
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
    """从 checkpoint 文件名中提取 'latest' 或 iter 步数标签。"""
    stem = Path(ckpt_path).stem
    if 'latest' in stem.lower():
        return 'latest'
    m = re.search(r'(\d{5,8})$', stem)
    if m:
        return f'iter_{int(m.group(1))}'
    return stem


def make_videos(show_path: str, video_dir: str) -> None:
    """用 ffmpeg 将帧序列合成 RGB 视频和深度视频。"""
    tasks = [
        ('RGB',   '%05d.jpg',       os.path.join(video_dir, 'rgb.mp4')),
        ('depth', '%05d_depth.png', os.path.join(video_dir, 'depth.mp4')),
    ]
    for label, fmt, out_mp4 in tasks:
        ext = fmt.split('.')[-1]
        existing = sorted(glob.glob(os.path.join(show_path, f'*.{ext}')))
        if not existing:
            logging.info(f'无 {label} 帧文件，跳过视频生成')
            continue

        os.makedirs(video_dir, exist_ok=True)
        # 以实际最小帧号作为起始索引，避免 ffmpeg 从 0 找不到文件
        start_num = int(Path(existing[0]).stem.replace('_depth', ''))
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
#  相机参数读取 & 插值（完全对齐 run_interpolation.py）
# ──────────────────────────────────────────────────────────

def read_calib(calib):
    R = np.array(calib['R']).reshape((3, 3))
    T = np.array(calib['T']).reshape((3, 1))
    extr = np.zeros((3, 4))
    extr[:3, :3] = R
    extr[:3, 3:] = T
    intr = np.zeros((3, 3))
    intr[:3, :3] = np.array(calib['K']).reshape((3, 3))
    H = 2048
    W = 1500
    if W > H:
        intr[0, 2] -= (W - H) / 2
        intr[:2] *= RS / H
    else:
        intr[1, 2] -= (H - W) / 2
        intr[:2] *= RS / W
    calib = intr @ extr
    return extr, intr, calib


def extr_interpolate(RS, cam_id_list):
    novel_extr_list = []
    novel_intr_list = []

    for i in range(1):
        novel_extrs = []
        novel_intrs = []

        calib_path = os.path.join(cfg.dataset.local_data_root, 'test', tar_n + '_process', 'calibration_full.json')
        with open(calib_path, 'r') as f:
            calib_full = json.load(f)

        cam = cam_id_list[0]
        extr0, intr0, calib0 = read_calib(calib_full[cam])

        cam = cam_id_list[1]
        extr1, intr1, calib1 = read_calib(calib_full[cam])

        pose_0 = np.eye(4)
        pose_1 = np.eye(4)
        pose_0[:3] = extr0
        pose_1[:3] = extr1
        pose_0 = np.linalg.inv(pose_0)
        pose_1 = np.linalg.inv(pose_1)
        for ratio in np.linspace(0., 0.95, LOOP_NUM):
            rot_0 = pose_0[:3, :3]
            rot_1 = pose_1[:3, :3]
            rots = Rot.from_matrix(np.stack([rot_0, rot_1]))
            key_times = [0, 1]
            slerp = Slerp(key_times, rots)
            rot = slerp(ratio)
            pose = np.diag([1.0, 1.0, 1.0, 1.0])
            pose = pose.astype(np.float32)
            pose[:3, :3] = rot.as_matrix()
            pose[:3, 3] = ((1.0 - ratio) * pose_0 + ratio * pose_1)[:3, 3]
            pose = np.linalg.inv(pose)
            novel_extrs.append(pose[:3])
            novel_intrs.append((1.0 - ratio) * intr0 + ratio * intr1)

        novel_extr_list = novel_extr_list + [novel_extrs]
        novel_intr_list = novel_intr_list + [novel_intrs]

    return novel_extr_list, novel_intr_list


# ──────────────────────────────────────────────────────────
#  PAGSplat 渲染器（结构对齐 StereoHumanModel，只改模型加载）
# ──────────────────────────────────────────────────────────

class PAGSplatModel(nn.Module):
    def __init__(self, cfg, model, novel_extrs, novel_intrs, s_id=1):
        super().__init__()

        self.cfg   = cfg
        self.model = model

        self.novel_extrs = novel_extrs
        self.novel_intrs = novel_intrs
        self.img_path = os.path.join(cfg.dataset.local_data_root, 'test', tar_n + '_process', 'img')
        self.msk_path = os.path.join(cfg.dataset.local_data_root, 'test', tar_n + '_process', 'mask')

        # 读取立体对相机内外参（与 run_interpolation.py 完全一致的方式）
        parm_name = os.path.join(
            cfg.dataset.local_data_root, 'test', tar_n + '_process', 'parameter',
            tar_n + '_s%d_0000' % s_id,
            '%s_%s.json' % (str(cfg.dataset.source_id[0]), str(cfg.dataset.source_id[1]))
        )
        camera = load_json_to_np(parm_name)
        self.s_id = s_id
        extr0 = camera['extr0']
        extr1 = camera['extr1']
        intr0 = camera['intr0']
        intr1 = camera['intr1']

        self.intrinsics = [torch.FloatTensor(intr0), torch.FloatTensor(intr1)]
        self.extrinsics = [torch.FloatTensor(extr0), torch.FloatTensor(extr1)]

    def get_item_free(self, frame_id, view_id):
        # 图像读取（与 run_interpolation.py 一致，只去掉 flow_init/Tf_x/ref_intr）
        img0 = np.array(Image.open(self.img_path + '/%s_s%d_%04d/%d.jpg' % (tar_n, self.s_id, frame_id, from_list[0]))).astype(np.float32)
        img1 = np.array(Image.open(self.img_path + '/%s_s%d_%04d/%d.jpg' % (tar_n, self.s_id, frame_id, to_list[0]))).astype(np.float32)
        img0 = torch.from_numpy(img0).permute(2, 0, 1).unsqueeze(0).cuda()
        img1 = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).cuda()

        msk0 = np.array(Image.open(self.msk_path + '/%s_s%d_%04d/%d.jpg' % (tar_n, self.s_id, frame_id, from_list[0]))).astype(np.float32)
        msk1 = np.array(Image.open(self.msk_path + '/%s_s%d_%04d/%d.jpg' % (tar_n, self.s_id, frame_id, to_list[0]))).astype(np.float32)
        msk0 = torch.from_numpy(msk0).permute(2, 0, 1).unsqueeze(0).cuda()
        msk1 = torch.from_numpy(msk1).permute(2, 0, 1).unsqueeze(0).cuda()

        img0 = 2 * (img0 / 255.0) - 1.0
        img1 = 2 * (img1 / 255.0) - 1.0

        msk0 /= 255
        msk1 /= 255

        intr0_ = self.intrinsics[0].unsqueeze(0).cuda()
        intr1_ = self.intrinsics[1].unsqueeze(0).cuda()
        extr0  = self.extrinsics[0].unsqueeze(0).cuda()
        extr1  = self.extrinsics[1].unsqueeze(0).cuda()

        intr0 = intr0_.clone()
        intr1 = intr1_.clone()

        # PAGSplat 不需要 flow_init / Tf_x / ref_intr
        l_view = {
            'img':  img0,
            'mask': msk0,
            'intr': intr0,
            'extr': extr0,
        }
        r_view = {
            'img':  img1,
            'mask': msk1,
            'intr': intr1,
            'extr': extr1,
        }

        # novel_view 格式与 run_interpolation.py 完全一致（列表包裹）
        novel_intr = self.novel_intrs[0][view_id]
        novel_extr = self.novel_extrs[0][view_id]

        width, height = 1024, 1024
        R = np.array(novel_extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(novel_extr[:3, 3], np.float32)

        FovX = focal2fov(novel_intr[0, 0], width)
        FovY = focal2fov(novel_intr[1, 1], height)
        projection_matrix = getProjectionMatrix(
            znear=0.01, zfar=100.0, fovX=FovX, fovY=FovY,
            K=novel_intr, h=height, w=width
        ).transpose(0, 1)
        world_view_transform = torch.tensor(
            getWorld2View2(R, T, np.array([0.0, 0.0, 0.0]), 1.0)
        ).transpose(0, 1)
        full_proj_transform = (
            world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view = {
            'height': [height],
            'width':  [width],
            'FovX':   [torch.FloatTensor(np.array(FovX)).cuda()],
            'FovY':   [torch.FloatTensor(np.array(FovY)).cuda()],
            'world_view_transform': [world_view_transform.cuda()],
            'full_proj_transform':  [full_proj_transform.cuda()],
            'camera_center':        [camera_center.cuda()],
        }

        dict_tensor = {
            'lmain': l_view,
            'rmain': r_view,
            'novel_view': novel_view,
        }

        return dict_tensor


# ──────────────────────────────────────────────────────────
#  入口（结构对齐 run_interpolation.py）
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, required=True, help='目标序列名称')
    parser.add_argument('--ckpt',   type=str, required=True,  help='PAGSplat checkpoint 路径')
    parser.add_argument('--config', type=str, default='pag_splat/pag_stage.yaml', help='配置文件')
    parser.add_argument('--cams',   type=str, default='22139908,22139909', help='逗号分隔的两个相机 ID')
    parser.add_argument('--loop',   type=int, default=20,  help='每段插值帧数')
    parser.add_argument('--frames', type=int, default=300, help='渲染总帧数')
    parser.add_argument('--start',  type=int, default=0,   help='起始帧 ID')
    parser.add_argument('--sid',    type=int, default=1,   help='序列 s_id')
    parser.add_argument('--crop',   type=int, default=0,   help='四边裁剪像素数')
    arg = parser.parse_args()

    tar_n   = arg.input
    cam_ids = [c.strip() for c in arg.cams.split(',')]
    assert len(cam_ids) == 2, '--cams 必须指定恰好 2 个相机 ID'

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s'
    )

    cfg_obj = config()
    cfg_obj.load(arg.config)
    cfg = cfg_obj.get_cfg()

    # ── 推导输出目录：ckpt 上级目录 / test_result / free_<seq>_<cam0>_<cam1>_<iter> ──
    iter_label  = get_iter_label(arg.ckpt)
    ckpt_parent = Path(arg.ckpt).parent.parent   # e.g. experiments/pag_splat_0304
    result_name = 'free_%s_%s_%s_%s' % (tar_n, cam_ids[0], cam_ids[1], iter_label)
    show_path   = str(ckpt_parent / 'test_result' / result_name)
    video_dir   = os.path.join(show_path, 'video')

    cfg.defrost()
    cfg.exp_name         = 'pag_splat'
    cfg.record.show_path = show_path
    cfg.restore_ckpt     = arg.ckpt
    cfg.freeze()

    LOOP_NUM = arg.loop
    RS       = 1024
    crop     = arg.crop
    from_list = [cfg.dataset.source_id[0]]
    to_list   = [cfg.dataset.source_id[1]]

    Path(cfg.record.show_path).mkdir(exist_ok=True, parents=True)
    Path(video_dir).mkdir(exist_ok=True, parents=True)

    # ── 注册退出时自动生成视频（正常结束或 Ctrl+C 均触发）──
    _show_path_global = cfg.record.show_path
    _video_dir_global = video_dir
    atexit.register(_finalize_videos)
    signal.signal(signal.SIGINT,  _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    print('=' * 55)
    print('PAG-Splat 自由视角插值渲染')
    print('=' * 55)
    print(f'  序列:      {tar_n}')
    print(f'  相机对:    {cam_ids}')
    print(f'  Checkpoint:{arg.ckpt}')
    print(f'  插值帧数:  {LOOP_NUM}  总渲染帧: {arg.frames}')
    print(f'  输出目录:  {cfg.record.show_path}')
    print('=' * 55)

    pag_cfg = cfg.pagsplat

    # ── 加载 PAGSplat 模型（仅此处与 run_interpolation.py 不同）──
    logging.info('加载模型...')
    model = build_pag_splat(
        da3_checkpoint=pag_cfg.da3_checkpoint,
        feat_channels=pag_cfg.feat_channels,
        feat_stride=pag_cfg.feat_stride,
        feat_layer=pag_cfg.feat_layer,
        mlp_hidden=pag_cfg.mlp_hidden,
        enc_dims=list(pag_cfg.enc_dims),
        dec_dims=list(pag_cfg.dec_dims),
        head_ch=pag_cfg.head_ch,
        scale_max=pag_cfg.scale_max,
        device='cuda',
        ckpt_path=arg.ckpt,   # 自动检测旧/新 checkpoint 的 t12_mode
    )
    assert os.path.exists(arg.ckpt), f'Checkpoint 不存在: {arg.ckpt}'
    try:
        ckpt = torch.load(arg.ckpt, map_location='cuda', weights_only=False)
    except Exception as e:
        logging.error(f'Checkpoint 加载失败（文件可能损坏或不完整）: {arg.ckpt}\n  {e}')
        sys.exit(1)
    missing, unexpected = model.load_state_dict(ckpt['network'], strict=False)
    if len(unexpected) > 0:
        print(f'[警告] 忽略了 {len(unexpected)} 个不匹配的键')
    if len(missing) > 0:
        print(f'[警告] 缺失 {len(missing)} 个键: {missing[:5]}...')
    model = model.cuda()
    model.eval()
    # 与 GPS+ _freeze_bn 一致：保持 BN 统计量不更新
    for m in model.modules():
        if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            m.eval()
    logging.info('模型加载完成')

    # ── 插值相机路径（完全对齐 run_interpolation.py）──
    novel_extrs, novel_intrs = extr_interpolate(RS, cam_ids)
    print(len(novel_extrs[0]))

    render = PAGSplatModel(cfg, model, [novel_extrs[0]], novel_intrs, arg.sid)

    start_frame = arg.start
    end_frame   = arg.start + arg.frames

    # ── 主渲染循环（对齐 run_interpolation.py）──
    for fr_i in tqdm(range(start_frame, end_frame)):
        wi_ct    = fr_i // LOOP_NUM
        scene_id = wi_ct % 2     # 来回插值：0=正向，1=反向
        view_idx = fr_i % LOOP_NUM

        with torch.no_grad():
            try:
                if scene_id == 0:
                    data = render.get_item_free(fr_i, view_idx)
                else:
                    data = render.get_item_free(fr_i, (LOOP_NUM - 1) - view_idx)
                data = render.model(data, is_train=False)
            except FileNotFoundError as e:
                logging.warning(f'帧 {fr_i} 文件缺失，跳过: {e}')
                continue

            # 渲染（对齐 run_interpolation.py 调用 pts2render 的位置）
            data = pag_pts2render(
                data,
                bg_color=cfg.dataset.bg_color,
                min_opacity=pag_cfg.min_opacity,
            )

            # 保存渲染图（与 run_interpolation.py 保存逻辑一致）
            tmp_novel = data['novel_view']['img_pred'][0].detach()
            tmp_novel = tmp_novel.clamp(0, 1) * 255
            tmp_novel_np = tmp_novel.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            if crop > 0:
                tmp_novel_np = tmp_novel_np[crop:RS - crop, crop:RS - crop]

            out_path = '%s/%05d.jpg' % (cfg.record.show_path, fr_i)
            cv2.imwrite(out_path, tmp_novel_np[:, :, ::-1])

            # 额外保存度量深度图（PAGSplat 专属）
            if 'metric_depth_l' in data:
                d = data['metric_depth_l'].squeeze().detach().cpu().float().numpy()
                d_min, d_max = d.min(), d.max()
                if d_max > d_min:
                    d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
                else:
                    d_norm = np.zeros_like(d, dtype=np.uint8)
                depth_color = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
                if crop > 0:
                    depth_color = depth_color[crop:RS - crop, crop:RS - crop]
                cv2.imwrite('%s/%05d_depth.png' % (cfg.record.show_path, fr_i), depth_color)

    logging.info(f'渲染完成！输出目录: {cfg.record.show_path}')
    _finalize_videos()
