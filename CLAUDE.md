# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPS-Gaussian+ is a feed-forward 3D Gaussian Splatting framework for real-time human-scene rendering from sparse views. Given a rectified stereo pair, the network predicts per-pixel 3D Gaussian parameters (position, rotation, scale, opacity) in a single forward pass, then renders novel views via differentiable Gaussian rasterization. Colors come directly from input pixels — the network only learns geometry and Gaussian shape.

## Commands

### Environment Setup
```bash
conda env create --file enviroment.yml
conda activate gps_plus
# Install diff-gaussian-rasterization (required for rendering)
cd gaussian-splatting && pip install -e submodules/diff-gaussian-rasterization && cd ..
```

### Training (multiple depth estimation modes)
```bash
# Original RAFT-Stereo mode
python train.py --config config/stage.yaml

# Fast-FoundationStereo mode (FFS frozen, trains img_encoder + loftr + GSRegresser)
python train_ffs.py --config config/ffs_stage.yaml

# StereoGS mode (FFS frozen, trains adapters + fusion + confidence + Gaussian decoder)
python train_stereo_gs.py --config config/stereo_gs_stage.yaml

# PAG-Splat mode (DA3 frozen, trains prior extractor + warping + Gaussian decoder)
python train_pag.py --config pag_splat/pag_stage.yaml
```

### Testing
```bash
python test.py -i <sequence_name> -v <view_id>          # Single view
python test.py -i <sequence_name> --all-views            # All available views
python test.py -i <sequence_name> --dynamic              # Dynamic view interpolation
python test_stereo_gs.py --config config/stereo_gs_stage.yaml --ckpt <path>
```

### Free-view Rendering
```bash
python run_interpolation.py -i <sequence_name>           # RAFT/DA3 mode
python run_interpolation_ffs.py -i <sequence_name>       # FFS mode
python run_interpolation_pag.py -i <sequence_name>       # PAG-Splat mode
python run_interpolation_stereo_gs.py -i <sequence_name> # StereoGS mode
```

### Ablation Experiments
```bash
bash run_ablation.sh              # Run all ablations
bash run_ablation.sh fusion_none  # Run specific ablation
```

### Data Preprocessing
```bash
cd data_process
python step_0rect.py -i s1a1 -t train    # THumanMV rectification
python step_1.py -i s1a1 -t train        # Novel view processing
python step_0rect_custom.py -t train -n 4 # Custom data rectification
python step_1_custom.py -t train -n 4     # Custom data processing
```

## Architecture

### Core Data Flow (RAFT mode)
```
Stereo pair → UnetExtractor (shared, 3-level) → LoFTR cross-attention (row-wise epipolar)
  → RAFT-Stereo (3 GRU iters, 1D correlation) → flow2depth (inverse depth)
  → GSRegresser (dual encoder + 4 heads: rot/scale/opacity/Δdepth)
  → depth2pc (inverse depth → 3D world coords)
  → pts2render (merge L/R valid points → diff_gaussian_rasterization)
```

### Depth Estimation Modes (controlled by `depth_mode` in config)

| Mode | Module | Trainable | Key Difference |
|------|--------|-----------|----------------|
| `raft` | RAFT-Stereo + LoFTR | All | Original pipeline, flow-based stereo matching |
| `da3` | Depth-Anything-3 | img_encoder + LoFTR + GSRegresser | Monocular depth, scale-aligned |
| `ffs` | Fast-FoundationStereo | img_encoder + LoFTR + GSRegresser | FFS frozen, stereo matching |
| `stereo_gs` | FFS + StereoGS adapters | Adapters + fusion + decoder | FFS frozen, adds cross-view fusion + confidence + CAGS adaptive splitting |
| `pag` (via train_pag.py) | DA3 + PAG modules | Prior extractor + warping + decoder | DA3 frozen, adds depth GRU refinement + surface warping |

### Key Modules

- **`lib/network.py`** — `RtStereoHumanModel`: Main model, dispatches to depth mode-specific forward paths
- **`core/`** — RAFT-Stereo core: `extractor.py` (UnetExtractor), `raft_stereo_human.py`, `corr.py` (1D epipolar correlation), `update.py` (GRU)
- **`lib/gs_parm_network.py`** — `GSRegresser`: Dual-encoder (image + depth) → multi-scale decoder → 4 prediction heads
- **`lib/attention_module.py`** — `LocalFeatureTransformer`: LoFTR-style row-wise cross-attention (exploits epipolar constraint)
- **`lib/GaussianRender.py`** — `pts2render` / `pts2render_cags`: Bridge from Gaussian parameters to rasterization
- **`lib/stereo_gs/`** — StereoGS-specific modules: feature adapter, cross-view fusion, confidence extractor, CAGS adaptive splitter, Gaussian decoder/upsampler
- **`pag_splat/`** — PAG-Splat: prior extractor, scale alignment, depth GRU refinement, surface warping, Gaussian decoder
- **`gaussian_renderer/`** — Wrapper around `diff_gaussian_rasterization`
- **`config/stereo_human_config.py`** — YACS `CfgNode` schema for all configurable parameters

### Important Design Details

- **Inverse depth representation**: Pipeline works in inverse depth space; `depth2pc()` converts via `z = 1/(depth + ε)`
- **Gaussian scale clamp**: `scale ≤ 0.002` keeps Gaussians pixel-sized
- **Color from pixels**: No color prediction — RGB comes directly from input images (`img * 0.5 + 0.5`)
- **Loss**: `0.8×L1 + 0.2×(1-SSIM) + 0.5×Chamfer` (Chamfer requires pytorch3d, set `if_chamfer=False` to skip)
- **Source images normalized to [-1,1]**, GT novel views to [0,1]
- **Tf_x sign**: Positive for left view, negative for right view
- **flow_init**: Computed from constant inverse depth (0.3) as warm start for RAFT

### Configuration System

Uses YACS (`CfgNode`). Each training mode has its own YAML config:
- `config/stage.yaml` — RAFT/DA3/FFS modes (switch via `depth_mode`)
- `config/ffs_stage.yaml` — FFS-specific defaults
- `config/stereo_gs_stage.yaml` — StereoGS with fusion/confidence/CAGS settings
- `pag_splat/pag_stage.yaml` — PAG-Splat settings
- `config/ablation/` — Ablation experiment configs

### External Dependencies

- **Depth-Anything-3**: Expected at sibling directory `../Depth-Anything-3/` (DA3 mode)
- **Fast-FoundationStereo**: Path configured via `ffs.ffs_root` in YAML (FFS/StereoGS modes)
- **diff-gaussian-rasterization**: Built from `gaussian-splatting/submodules/` submodule
- **pytorch3d**: Optional, for Chamfer distance loss

### Data Format

Processed data structure: `img/` (0.jpg=left, 1.jpg=right, 2-5.jpg=novel views), `mask/`, `parameter/` (0_1.json with intr/extr/Tf_x, per-view .npy files). See `About_this_repo.md` for full specification.
