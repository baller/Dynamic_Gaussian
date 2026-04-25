"""
StereoGS 测试脚本

评估 StereoGS 模型在验证集上的 PSNR / SSIM / LPIPS 指标,
并保存渲染结果、深度图、置信度图等可视化。

用法:
    python test_stereo_gs.py --config config/stereo_gs_stage.yaml --ckpt experiments/stereo_gs_0324/ckpt/latest.pth
    python test_stereo_gs.py --config config/stereo_gs_stage.yaml --ckpt experiments/stereo_gs_0324/ckpt/latest.pth --save_all
"""

from __future__ import print_function, division

import argparse
import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.stereo_human_config import ConfigStereoHuman
from lib.human_loader import StereoHumanDataset
from lib.network import RtStereoHumanModel
from lib.GaussianRender import pts2render, pts2render_cags
from lib.gs_utils.loss_utils import ssim
from lib.gs_utils.image_utils import psnr

warnings.filterwarnings("ignore", category=UserWarning)

FFS_PREFIX = "stereo_gs_model.ffs_extractor."


def load_model(cfg, ckpt_path):
    model = RtStereoHumanModel(cfg, with_gs_render=True)
    model.cuda()

    ckpt = torch.load(ckpt_path, map_location='cuda', weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt['network'], strict=False)
    ffs_missing = [k for k in missing if k.startswith(FFS_PREFIX)]
    other_missing = [k for k in missing if not k.startswith(FFS_PREFIX)]
    if other_missing:
        logging.warning(f"缺少非FFS权重 ({len(other_missing)} 个): {other_missing[:5]}...")

    logging.info(f"模型加载完成 (step={ckpt.get('total_steps', '?')})")
    model.eval()
    return model


def t2np(t):
    return (t[0].detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255
            ).astype(np.uint8)


def depth_to_color(depth):
    import matplotlib.cm as cm
    d = depth[0, 0].float().cpu().numpy()
    z = 1.0 / (d + 1e-8)
    z = np.clip(z, 0, np.percentile(z[z > 0], 98) if (z > 0).any() else 10)
    z_min, z_max = float(z.min()), float(z.max())
    d_norm = (z - z_min) / (z_max - z_min + 1e-6)
    rgba = cm.inferno(d_norm)
    return (rgba[:, :, :3] * 255).astype(np.uint8)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)-8s %(message)s')

    parser = argparse.ArgumentParser(description='StereoGS Testing')
    parser.add_argument('--config', default='config/stereo_gs_stage.yaml')
    parser.add_argument('--ckpt', required=True, help='checkpoint 路径')
    parser.add_argument('--out_dir', default=None, help='输出目录')
    parser.add_argument('--save_all', action='store_true', help='保存所有样本的可视化')
    args = parser.parse_args()

    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(args.config)
    cfg = cfg_obj.get_cfg()
    use_cags = getattr(cfg.stereo_gs, 'use_cags', False)
    render_fn = pts2render_cags if use_cags else pts2render

    model = load_model(cfg, args.ckpt)
    use_post_refine = getattr(cfg.stereo_gs, 'use_post_refine', False)

    val_set = StereoHumanDataset(cfg.dataset, phase='val')
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4)
    len_val = max(1, int(len(val_loader) / val_set.val_boost))

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.ckpt), '..', 'test_results')
    os.makedirs(out_dir, exist_ok=True)

    psnr_list, ssim_list = [], []
    val_iter = iter(val_loader)

    for idx in tqdm(range(len_val), desc="Testing"):
        try:
            data = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            data = next(val_iter)

        for view in ['lmain', 'rmain', 'novel_view']:
            if view not in data:
                continue
            for item in data[view]:
                v = data[view][item]
                if isinstance(v, torch.Tensor):
                    data[view][item] = v.cuda()

        with torch.no_grad():
            data, _, _ = model(data, is_train=False)
            data = render_fn(data, bg_color=cfg.dataset.bg_color)
            if use_post_refine:
                data = model.stereo_gs_model.refine_rendered(data)

            render_novel = data['novel_view']['img_pred']
            gt_novel = data['novel_view']['img'].cuda()
            psnr_val = psnr(render_novel, gt_novel).mean().double()
            ssim_val = ssim(render_novel, gt_novel)
            psnr_list.append(psnr_val.item())
            ssim_list.append(ssim_val.item())

            if args.save_all or idx < 10:
                render_np = t2np(render_novel)
                gt_np = t2np(gt_novel)
                combined = np.concatenate([render_np, gt_np], axis=1)
                cv2.imwrite(os.path.join(out_dir, f"{idx:04d}.jpg"), combined[:, :, ::-1])

                depth_l = depth_to_color(data['lmain']['depth'])
                cv2.imwrite(os.path.join(out_dir, f"{idx:04d}_depth.jpg"), depth_l[:, :, ::-1])

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))

    logging.info("=" * 50)
    logging.info(f"Results ({len_val} samples):")
    logging.info(f"  PSNR: {avg_psnr:.4f}")
    logging.info(f"  SSIM: {avg_ssim:.4f}")
    logging.info("=" * 50)

    with open(os.path.join(out_dir, "metrics.txt"), "w") as f:
        f.write(f"PSNR: {avg_psnr:.4f}\n")
        f.write(f"SSIM: {avg_ssim:.4f}\n")
        f.write(f"Samples: {len_val}\n")
        f.write(f"Checkpoint: {args.ckpt}\n")
        f.write(f"Config: fusion={cfg.stereo_gs.fusion_mode}, "
                f"confidence={cfg.stereo_gs.confidence_mode}, "
                f"sr={cfg.stereo_gs.sr_mode}, "
                f"refine={cfg.stereo_gs.use_post_refine}\n")


if __name__ == '__main__':
    main()
