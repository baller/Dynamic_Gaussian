#!/usr/bin/env python3
"""
诊断脚本：比较 FFSFeatureExtractor._forward_ffs_with_features 与
standalone FFS forward() 对同一帧的视差输出。

运行方式（在 GPS_plus 目录下）：
  conda run -n gps_plus python scripts/diag_ffs_disparity.py
"""
import sys, os, json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

GPS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GPS_ROOT))

SAMPLE_DIR = Path("/data/sifang/GPS_plus_data/processed_hias/hias_man1_s1_s0/train")
IMG_SAMPLE  = "hias_man1_s1_s0_0001"
FFS_ROOT    = Path("/data/sifang/Fast-FoundationStereo")
FFS_MODEL   = FFS_ROOT / "weights/23-36-37/model_best_bp2_serialize.pth"

# ─── 加载配置 ───────────────────────────────────────────────────────
from config.stereo_human_config import ConfigStereoHuman
cfg = ConfigStereoHuman()
cfg.load(str(GPS_ROOT / "config/stereo_gs_stage.yaml"))
cfg = cfg.get_cfg()

# ─── 加载图像和相机参数 ─────────────────────────────────────────────
img0 = np.array(Image.open(SAMPLE_DIR / f"img/{IMG_SAMPLE}/0.jpg"))
img1 = np.array(Image.open(SAMPLE_DIR / f"img/{IMG_SAMPLE}/1.jpg"))
with open(SAMPLE_DIR / f"parameter/{IMG_SAMPLE}/0_1.json") as f:
    parm = {k: np.array(v) for k, v in json.load(f).items()}

cx_L = float(parm["intr0"][0, 2])
cx_R = float(parm["intr1"][0, 2])
Tf_x = float(parm["Tf_x"])
cx_shift_raw = cx_R - cx_L
cx_shift_int = int(round(cx_shift_raw))
print(f"cx_L={cx_L:.2f}  cx_R={cx_R:.2f}  cx_shift={cx_shift_raw:.2f} → {cx_shift_int}")
print(f"Tf_x={Tf_x:.4f}")

def to_tensor(img):
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return (t * 2.0 - 1.0).cuda()   # → [-1, 1]

left_t  = to_tensor(img0)
right_t = to_tensor(img1)

# ─── 方法 A: 用 FFSFeatureExtractor（训练路径）─────────────────────
print("\n=== 方法A: FFSFeatureExtractor ===")
from lib.stereo_gs.ffs_feature_extractor import FFSFeatureExtractor
extractor = FFSFeatureExtractor(cfg)

data = {
    'lmain': {
        'img':      left_t,
        'intr':     torch.FloatTensor(parm["intr0"]).unsqueeze(0).cuda(),
        'ref_intr': torch.FloatTensor(parm["intr1"]).unsqueeze(0).cuda(),
        'extr':     torch.FloatTensor(parm["extr0"]).unsqueeze(0).cuda(),
        'Tf_x':     torch.FloatTensor([Tf_x]).cuda(),
    },
    'rmain': {
        'img':      right_t,
        'intr':     torch.FloatTensor(parm["intr1"]).unsqueeze(0).cuda(),
        'ref_intr': torch.FloatTensor(parm["intr0"]).unsqueeze(0).cuda(),
        'extr':     torch.FloatTensor(parm["extr1"]).unsqueeze(0).cuda(),
        'Tf_x':     torch.FloatTensor([-Tf_x]).cuda(),
    },
}
with torch.no_grad():
    feats = extractor(data, return_both_views=False)
disp_A = feats['left'].disparity[0, 0].cpu().numpy()
inv_depth_A = disp_A / abs(Tf_x)
depth_A = 1.0 / (inv_depth_A + 1e-8)
print(f"  disp_A    : mean={disp_A.mean():.2f}, min={disp_A.min():.2f}, max={disp_A.max():.2f}")
print(f"  depth_A   : mean={depth_A.mean():.2f}m, min={depth_A.min():.2f}m, max={depth_A.max():.2f}m")

# ─── 方法 B: 直接调用 standalone FFS（验证路径）────────────────────
print("\n=== 方法B: standalone FFS.forward() ===")
from lib.ffs_depth import _load_ffs_model
ffs_model = _load_ffs_model(str(FFS_MODEL), str(FFS_ROOT))
ffs_model.eval().cuda()

def run_ffs_direct(left_img, right_img):
    """仿照 batch_ffs_hias_val 直接调用"""
    from lib.ffs_depth import _InputPadder
    t0 = torch.from_numpy(left_img).float().cuda().permute(2,0,1).unsqueeze(0)
    t1 = torch.from_numpy(right_img).float().cuda().permute(2,0,1).unsqueeze(0)
    padder = _InputPadder(t0.shape, divis_by=32)
    t0p, t1p = padder.pad(t0, t1)
    with torch.amp.autocast('cuda', enabled=True, dtype=torch.float16):
        disp = ffs_model.forward(t0p, t1p, iters=8, test_mode=True,
                                  optimize_build_volume='pytorch1')
    disp = padder.unpad(disp.float())
    return disp[0, 0].cpu().numpy()

with torch.no_grad():
    disp_B = run_ffs_direct(img0, img1)  # 无 cx 补偿（同 batch_ffs_hias_val）

inv_depth_B = disp_B / abs(Tf_x)
depth_B_raw = 1.0 / (inv_depth_B + 1e-8)
# standalone 用 baseline 直接换算
import numpy as np
extr0 = parm["extr0"]
extr1 = parm["extr1"]
R_rel = extr1[:, :3] @ extr0[:, :3].T
t_rel = extr1[:, 3] - R_rel @ extr0[:, 3]
baseline = float(np.linalg.norm(t_rel))
depth_B_bl = baseline / (disp_B + 1e-6)   # standalone 的换算方式
print(f"  disp_B    : mean={disp_B.mean():.2f}, min={disp_B.min():.2f}, max={disp_B.max():.2f}")
print(f"  depth_B(Tf_x): mean={depth_B_raw.mean():.2f}m  (应偏大 {abs(Tf_x)/(baseline*cx_L)*0:.2f})")
print(f"  depth_B(baseline): mean={depth_B_bl.mean():.2f}m")

# ─── 比较两者视差 ───────────────────────────────────────────────────
print("\n=== 差异分析 ===")
print(f"  A/B视差差值: mean={np.mean(disp_A - disp_B):.2f}, std={np.std(disp_A - disp_B):.2f}")
print(f"  理论cx补偿量: cx_shift_int = {cx_shift_int} px")
print(f"  实际差均值应≈{cx_shift_int} 说明A做了补偿, ≈0 说明A没做补偿")

# ─── 保存可视化 ─────────────────────────────────────────────────────
import cv2
out_dir = Path("/tmp/diag_ffs")
out_dir.mkdir(exist_ok=True)

def vis_depth(d, fname):
    d_clip = np.clip(d, 0, np.percentile(d[d>0], 98))
    d_norm = ((d_clip - d_clip.min()) / (d_clip.max() - d_clip.min() + 1e-6) * 255).astype(np.uint8)
    cv2.imwrite(str(out_dir / fname), cv2.applyColorMap(d_norm, cv2.COLORMAP_INFERNO))

vis_depth(depth_A,     "depth_A_featureextractor.png")
vis_depth(depth_B_raw, "depth_B_standalone.png")
vis_depth(np.abs(disp_A - disp_B), "disp_diff.png")
print(f"\n可视化已保存至 {out_dir}")
