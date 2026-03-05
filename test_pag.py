"""
PAG-Splat 测试脚本

对应 GPS+ test.py，完整移植三种测试模式：
  - 单视角测试:   python test_pag.py -i s1a6 -v 3
  - 多视角测试:   python test_pag.py -i s1a6 --all-views
  - 动态插值视角: python test_pag.py -i s1a6 --dynamic --interp 60 --views 2,3,4 --loop

与 GPS+ test.py 的主要差异：
  1. 使用 build_pag_splat 加载 PAGSplat 模型
  2. 使用 pag_pts2render + move_data_to_cuda 渲染（novel_view 也送 CUDA）
  3. 额外保存度量深度图 (metric_depth) 和不确定性图 (uncertainty)
  4. 不需要 optimizer/scheduler（纯推理）
"""

from __future__ import print_function, division

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.stereo_human_config import ConfigStereoHuman
from lib.human_loader import StereoHumanDataset
from lib.gs_utils.image_utils import psnr
from lib.gs_utils.loss_utils import ssim
from pag_splat.model import build_pag_splat
from pag_splat.render import pag_pts2render, move_data_to_cuda

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ──────────────────────────────────────────────────────────
#  可视化工具
# ──────────────────────────────────────────────────────────

def depth_to_colormap(depth: torch.Tensor) -> np.ndarray:
    """度量深度张量 (1,1,H,W) 或 (1,H,W) → JET 彩色图 np.uint8 BGR"""
    d = depth.squeeze().detach().cpu().float().numpy()
    d_min, d_max = d.min(), d.max()
    if d_max > d_min:
        d_norm = ((d - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        d_norm = np.zeros_like(d, dtype=np.uint8)
    return cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)


def uncertainty_to_colormap(unc: torch.Tensor, H: int, W: int) -> np.ndarray:
    """
    不确定性张量 (1,H*W,1) → PLASMA 彩色图 np.uint8 BGR

    unc 来自 model output，已 reshape 为 (B, H*W, 1)
    """
    u = unc[0].reshape(H, W).detach().cpu().float().numpy()
    u_norm = (u.clip(0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(u_norm, cv2.COLORMAP_PLASMA)


def save_render(render_novel: torch.Tensor, path: str) -> None:
    """渲染结果 (1,3,H,W) [0,1] → BGR uint8 写入磁盘"""
    img = render_novel[0].detach().clamp(0, 1) * 255
    img = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    cv2.imwrite(path, img[:, :, ::-1])


# ──────────────────────────────────────────────────────────
#  Tester
# ──────────────────────────────────────────────────────────

class PAGSplatTester:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bs = cfg.batch_size
        pag_cfg = cfg.pagsplat

        logging.info("=== PAG-Splat Tester 初始化 ===")
        logging.info(f"Checkpoint: {cfg.restore_ckpt}")

        # ── 加载模型 ──
        self.model = build_pag_splat(
            da3_checkpoint=pag_cfg.da3_checkpoint,
            feat_channels=pag_cfg.feat_channels,
            feat_stride=pag_cfg.feat_stride,
            feat_layer=pag_cfg.feat_layer,
            mlp_hidden=pag_cfg.mlp_hidden,
            enc_dims=list(pag_cfg.enc_dims),
            dec_dims=list(pag_cfg.dec_dims),
            head_ch=pag_cfg.head_ch,
            scale_max=pag_cfg.scale_max,
            device="cuda",
        )

        assert cfg.restore_ckpt and os.path.exists(cfg.restore_ckpt), \
            f"Checkpoint 不存在: {cfg.restore_ckpt}"
        ckpt = torch.load(cfg.restore_ckpt, map_location="cuda", weights_only=False)
        missing, unexpected = self.model.load_state_dict(ckpt["network"], strict=False)
        if unexpected:
            logging.warning(f"忽略 {len(unexpected)} 个多余键")
        if missing:
            logging.warning(f"缺失 {len(missing)} 个键: {missing[:5]}")

        self.model.eval()
        self.model.prior_extractor.da3_net.eval()
        logging.info("模型加载完成")

        # ── 数据集 ──
        self.val_set = StereoHumanDataset(cfg.dataset, phase=cfg.seq_name)
        self.val_loader = DataLoader(
            self.val_set, batch_size=self.bs,
            shuffle=False, num_workers=4, pin_memory=True,
        )
        self.len_val = int(len(self.val_loader) / self.val_set.val_boost)
        self.val_iterator = iter(self.val_loader)

    # ──────────────────────────────────────────────
    #  单视角测试
    # ──────────────────────────────────────────────

    def val(self):
        """对指定单一视角渲染，保存渲染图 + 深度图 + 不确定性图。"""
        logging.info("=== 单视角测试 ===")
        torch.cuda.empty_cache()
        psnr_list, ssim_list = [], []
        bg = self.cfg.dataset.bg_color

        for idx in tqdm(range(self.len_val), desc="渲染"):
            data = self.fetch_data()
            view_id  = data["novel_view"]["view_id"][0, 0].item()
            s_name   = data["novel_view"]["sample_name"][0]
            _, _, H, W = data["lmain"]["img"].shape

            with torch.no_grad():
                data = self.model(data, is_train=False)
                data = pag_pts2render(
                    data, bg_color=bg,
                    min_opacity=self.cfg.pagsplat.min_opacity,
                )

            render_novel = data["novel_view"]["img_pred"]
            gt_novel     = data["novel_view"]["img"]

            psnr_list.append(psnr(render_novel, gt_novel).mean().item())
            ssim_list.append(ssim(render_novel, gt_novel).item())

            prefix = f"{self.cfg.record.show_path}/{s_name}_{view_id:02d}"

            # 渲染图
            save_render(render_novel, f"{prefix}.jpg")

            # GT 对比图 (拼接)
            gt_np  = (gt_novel[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            ren_np = (render_novel[0].detach().clamp(0,1).permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
            compare = np.concatenate([ren_np[:,:,::-1], gt_np[:,:,::-1]], axis=1)
            cv2.imwrite(f"{prefix}_cmp.jpg", compare)

            # 度量深度彩色图
            cv2.imwrite(f"{prefix}_depth.png",
                        depth_to_colormap(data["metric_depth_l"]))

            # 不确定性彩色图
            cv2.imwrite(f"{prefix}_uncertainty.png",
                        uncertainty_to_colormap(data["lmain"]["uncertainty"], H, W))

        val_psnr = float(np.mean(psnr_list))
        val_ssim = float(np.mean(ssim_list))
        logging.info(f"结果: PSNR={val_psnr:.4f}  SSIM={val_ssim:.4f}")

    # ──────────────────────────────────────────────
    #  多视角测试
    # ──────────────────────────────────────────────

    def val_all_views(self, views=None):
        """对所有指定视角按帧顺序渲染，输出格式: {name}_{frame:04d}_v{view:02d}.jpg"""
        logging.info("=== 多视角测试 ===")
        torch.cuda.empty_cache()
        bg = self.cfg.dataset.bg_color

        if views is None:
            views = self._detect_available_views()
        else:
            avail = self._detect_available_views(views)
            views = [v for v in views if v in avail]
        if not views:
            logging.error("无可用视角")
            return
        logging.info(f"视角列表: {views}")

        # 为每个视角建 DataLoader
        view_loaders = {}
        view_iters   = {}
        for vid in views:
            self.cfg.defrost()
            self.cfg.dataset.val_novel_id = [vid]
            self.cfg.freeze()
            ds = StereoHumanDataset(self.cfg.dataset, phase=self.cfg.seq_name)
            dl = DataLoader(ds, batch_size=self.bs, shuffle=False,
                            num_workers=4, pin_memory=True)
            view_loaders[vid] = dl
            view_iters[vid]   = iter(dl)
        len_val = int(len(next(iter(view_loaders.values()))) /
                      self.val_set.val_boost)

        frame_idx = 0
        for data_idx in tqdm(range(len_val), desc="帧"):
            for vid in views:
                try:
                    data = next(view_iters[vid])
                except StopIteration:
                    view_iters[vid] = iter(view_loaders[vid])
                    data = next(view_iters[vid])

                data = move_data_to_cuda(data)
                s_name = data["novel_view"]["sample_name"][0]
                _, _, H, W = data["lmain"]["img"].shape

                with torch.no_grad():
                    data = self.model(data, is_train=False)
                    data = pag_pts2render(data, bg_color=bg,
                                         min_opacity=self.cfg.pagsplat.min_opacity)

                prefix = (f"{self.cfg.record.show_path}/"
                          f"{s_name.split('_')[0]}_{frame_idx:04d}_v{vid:02d}")
                save_render(data["novel_view"]["img_pred"], f"{prefix}.jpg")
                cv2.imwrite(f"{prefix}_depth.png",
                            depth_to_colormap(data["metric_depth_l"]))
                cv2.imwrite(f"{prefix}_uncertainty.png",
                            uncertainty_to_colormap(
                                data["lmain"]["uncertainty"], H, W))
                frame_idx += 1

        logging.info(f"多视角测试完成，共 {frame_idx} 帧")

    # ──────────────────────────────────────────────
    #  动态视角（SLERP 插值）测试
    # ──────────────────────────────────────────────

    def val_dynamic(self, num_interp: int = 30, views=None, loop: bool = False):
        """
        固定高斯点云，在插值相机路径上渲染平滑视角序列。

        先对源视角做一次模型推理获得高斯参数，然后对同一帧数据
        仅替换 novel_view 相机矩阵，重复调用渲染器，生成视角序列。
        """
        logging.info("=== 动态视角插值测试 ===")
        torch.cuda.empty_cache()
        bg = self.cfg.dataset.bg_color

        if views is None:
            views = self._detect_available_views()
        else:
            avail = self._detect_available_views(views)
            views = [v for v in views if v in avail]
        if len(views) < 2:
            logging.error("动态视角测试至少需要 2 个视角")
            return
        logging.info(f"关键视角: {views}, 插值帧数: {num_interp}, 循环: {loop}")

        # 收集每个视角的相机参数
        view_cameras = self._collect_view_cameras(views)

        # 重建数据迭代器
        self.cfg.defrost()
        self.cfg.dataset.val_novel_id = [views[0]]
        self.cfg.freeze()
        self.val_set    = StereoHumanDataset(self.cfg.dataset, phase=self.cfg.seq_name)
        self.val_loader = DataLoader(self.val_set, batch_size=self.bs,
                                     shuffle=False, num_workers=4, pin_memory=True)
        self.len_val    = int(len(self.val_loader) / self.val_set.val_boost)
        self.val_iterator = iter(self.val_loader)

        num_segments      = len(views) if loop else len(views) - 1
        frames_per_seg    = max(1, num_interp // num_segments)
        output_frame_idx  = 0
        total_out         = self.len_val * num_interp

        with tqdm(total=total_out, desc="动态视角渲染") as pbar:
            for data_idx in range(self.len_val):
                data = self.fetch_data()
                s_name = data["novel_view"]["sample_name"][0]

                # 一次推理，获取高斯点云
                with torch.no_grad():
                    data_gs = self.model(data, is_train=False)

                for interp_idx in range(num_interp):
                    global_t = interp_idx / max(num_interp - 1, 1)
                    seg_idx  = min(int(global_t * num_segments), num_segments - 1)
                    seg_start_t = seg_idx / num_segments
                    seg_end_t   = (seg_idx + 1) / num_segments
                    local_t = (
                        (global_t - seg_start_t) / (seg_end_t - seg_start_t)
                        if seg_end_t > seg_start_t else 0.0
                    )
                    local_t = float(np.clip(local_t, 0.0, 1.0))

                    v_start = views[seg_idx]
                    v_end   = views[(seg_idx + 1) % len(views)]
                    cs, ce  = view_cameras[v_start], view_cameras[v_end]

                    # SLERP 旋转 + 线性平移插值视图变换矩阵
                    wvt = self._slerp_matrix(
                        cs["world_view_transform"],
                        ce["world_view_transform"], local_t)
                    fpt = (1 - local_t) * cs["full_proj_transform"] + \
                           local_t      * ce["full_proj_transform"]
                    cc  = (1 - local_t) * cs["camera_center"] + \
                           local_t      * ce["camera_center"]
                    FovX = (1 - local_t) * cs["FovX"] + local_t * ce["FovX"]
                    FovY = (1 - local_t) * cs["FovY"] + local_t * ce["FovY"]

                    # 替换 novel_view 相机
                    nv = data_gs["novel_view"]
                    nv["world_view_transform"] = _ensure_batch(wvt, 2)
                    nv["full_proj_transform"]  = _ensure_batch(fpt, 2)
                    nv["camera_center"]        = _ensure_batch(cc,  1)
                    nv["FovX"] = _ensure_tensor_batch(FovX)
                    nv["FovY"] = _ensure_tensor_batch(FovY)

                    with torch.no_grad():
                        result = pag_pts2render(
                            data_gs, bg_color=bg,
                            min_opacity=self.cfg.pagsplat.min_opacity,
                        )

                    fname = (f"{self.cfg.record.show_path}/"
                             f"{s_name.split('_')[0]}_{data_idx:04d}_{interp_idx:04d}.jpg")
                    save_render(result["novel_view"]["img_pred"], fname)

                    output_frame_idx += 1
                    pbar.update(1)

        logging.info(f"动态视角渲染完成，共 {output_frame_idx} 帧")

    # ──────────────────────────────────────────────
    #  辅助方法
    # ──────────────────────────────────────────────

    def fetch_data(self) -> dict:
        try:
            data = next(self.val_iterator)
        except StopIteration:
            self.val_iterator = iter(self.val_loader)
            data = next(self.val_iterator)
        return move_data_to_cuda(data)

    def _detect_available_views(self, candidate_views=None):
        """扫描数据集参数目录，返回实际存在的视角 ID 列表。"""
        if candidate_views is None:
            candidate_views = list(range(10))
        seq_name = self.cfg.seq_name
        if seq_name not in ["train", "val"]:
            data_root = os.path.join(
                self.cfg.dataset.local_data_root, "test", seq_name)
        elif seq_name == "val":
            data_root = self.cfg.dataset.val_data_root
        else:
            data_root = self.cfg.dataset.train_data_root

        param_dir = os.path.join(data_root, "parameter")
        if os.path.exists(param_dir):
            samples = sorted(os.listdir(param_dir))
            if samples:
                sp = os.path.join(param_dir, samples[0])
                avail = [v for v in candidate_views
                         if os.path.exists(os.path.join(sp, f"{v}_intrinsic.npy"))]
                if avail:
                    logging.info(f"检测到可用视角: {avail}")
                    return avail
        logging.warning("无法检测视角，使用默认 [2, 3]")
        return [2, 3]

    def _collect_view_cameras(self, views: list) -> dict:
        """为每个视角加载一条数据，提取 novel_view 相机参数。"""
        cameras = {}
        for vid in views:
            self.cfg.defrost()
            self.cfg.dataset.val_novel_id = [vid]
            self.cfg.freeze()
            ds = StereoHumanDataset(self.cfg.dataset, phase=self.cfg.seq_name)
            dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)
            for data in dl:
                data = move_data_to_cuda(data)
                nv = data["novel_view"]
                cameras[vid] = {
                    k: nv[k].clone() if isinstance(nv[k], torch.Tensor) else nv[k]
                    for k in ["world_view_transform", "full_proj_transform",
                               "camera_center", "FovX", "FovY", "width", "height"]
                }
                break
            del ds, dl
        logging.info(f"收集了 {len(cameras)} 个视角相机参数")
        return cameras

    @staticmethod
    def _slerp_matrix(mat_a: torch.Tensor, mat_b: torch.Tensor, t: float) -> torch.Tensor:
        """
        对两个 (B,4,4) 或 (4,4) 世界视图变换矩阵做 SLERP 插值。
        旋转部分用 SLERP，平移部分用线性插值。
        """
        squeeze = (mat_a.dim() == 2)
        if squeeze:
            mat_a, mat_b = mat_a.unsqueeze(0), mat_b.unsqueeze(0)
        B = mat_a.shape[0]
        result = mat_a.clone()

        for b in range(B):
            R_a = mat_a[b, :3, :3].cpu().numpy()
            R_b = mat_b[b, :3, :3].cpu().numpy()
            t_a = mat_a[b, :3, 3].cpu().numpy()
            t_b = mat_b[b, :3, 3].cpu().numpy()

            try:
                rots = R.from_matrix(np.stack([R_a, R_b]))
                slerp = Slerp([0, 1], rots)
                R_interp = slerp(t).as_matrix()
            except Exception:
                R_interp = (1 - t) * R_a + t * R_b

            t_interp = (1 - t) * t_a + t * t_b
            result[b, :3, :3] = torch.from_numpy(R_interp.astype(np.float32)).to(mat_a.device)
            result[b, :3, 3]  = torch.from_numpy(t_interp.astype(np.float32)).to(mat_a.device)

        return result.squeeze(0) if squeeze else result


# ──────────────────────────────────────────────────────────
#  工具函数
# ──────────────────────────────────────────────────────────

def _ensure_batch(t: torch.Tensor, ndim_content: int) -> torch.Tensor:
    """确保张量含 batch 维，ndim_content 是去掉 batch 后的维度数。"""
    if t.dim() == ndim_content:
        return t.unsqueeze(0)
    return t


def _ensure_tensor_batch(v) -> torch.Tensor:
    if isinstance(v, torch.Tensor):
        return v.unsqueeze(0) if v.dim() == 0 else v
    return torch.tensor([v])


# ──────────────────────────────────────────────────────────
#  入口
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    )

    parser = argparse.ArgumentParser(description="PAG-Splat 测试脚本")
    parser.add_argument("-i", "--input", type=str, required=True,
                        help="输入序列名称 (不含 _process 后缀)")
    parser.add_argument("-v", "--view", type=int, default=3,
                        help="单视角测试的目标视角 (默认: 3)")
    parser.add_argument("--all-views", action="store_true",
                        help="对所有可用视角测试")
    parser.add_argument("--dynamic", action="store_true",
                        help="动态视角插值测试")
    parser.add_argument("--views", type=str, default="auto",
                        help="视角列表，逗号分隔；auto=自动检测 (默认: auto)")
    parser.add_argument("--interp", type=int, default=60,
                        help="动态视角总插值帧数 (默认: 60)")
    parser.add_argument("--loop", action="store_true",
                        help="动态视角是否首尾循环")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="PAGSplat checkpoint 路径")
    parser.add_argument("--config", type=str, default="pag_splat/pag_stage.yaml",
                        help="配置文件路径 (默认: pag_splat/pag_stage.yaml)")
    arg = parser.parse_args()

    views_list = (None if arg.views.lower() == "auto"
                  else [int(v.strip()) for v in arg.views.split(",")])

    seq_name = arg.input + "_process"

    cfg_obj = ConfigStereoHuman()
    cfg_obj.load(arg.config)
    cfg = cfg_obj.get_cfg()

    cfg.defrost()
    if arg.dynamic:
        suffix = "dynamic"
    elif arg.all_views:
        suffix = "allviews"
    else:
        suffix = f"v{arg.view}"

    cfg.exp_name        = "pag_splat"
    cfg.record.show_path = f"experiments/pag_splat/show_{seq_name}_{suffix}"
    cfg.seq_name         = seq_name
    cfg.dataset.val_novel_id = [arg.view]
    cfg.restore_ckpt     = arg.ckpt
    cfg.freeze()

    Path(cfg.record.show_path).mkdir(exist_ok=True, parents=True)

    print("=" * 55)
    print("PAG-Splat 测试")
    print("=" * 55)
    print(f"  序列:      {seq_name}")
    print(f"  Checkpoint:{arg.ckpt}")
    print(f"  输出目录:  {cfg.record.show_path}")
    if arg.dynamic:
        print(f"  模式: 动态视角  views={arg.views}  interp={arg.interp}  loop={arg.loop}")
    elif arg.all_views:
        print(f"  模式: 多视角  views={arg.views}")
    else:
        print(f"  模式: 单视角  view={arg.view}")
    print("=" * 55)

    tester = PAGSplatTester(cfg)

    if arg.dynamic:
        tester.val_dynamic(num_interp=arg.interp, views=views_list, loop=arg.loop)
    elif arg.all_views:
        tester.val_all_views(views=views_list)
    else:
        tester.val()
