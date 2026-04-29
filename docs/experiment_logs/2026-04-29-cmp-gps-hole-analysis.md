# 2026-04-29 cmp_gps Hole Analysis

## Goal

Compare `/data/sifang/GPS_plus/experiments/stereo_gs_hias_gps_data_wavelet_deepseekfixbug` against `/data/sifang/gps_plus_origin/experiments/origin_gps_raft_0429` to investigate why the current StereoGS method shows holes around table legs and human leg edges even when its depth appears better.

## Code Changes

- Added `scripts/export_gps_diagnostics.py`.
  - Restores a method checkpoint from a specified repo snapshot.
  - Runs one validation sample and novel view.
  - Exports RGB panels, depth/opacity/scale/confidence/split maps, sampled point-cloud PLY, sampled Gaussian PLY, full compressed NPZ tensors, projection density maps, hole-candidate masks, and `stats.json`.
- Added `docs/superpowers/plans/2026-04-29-cmp-gps-hole-analysis.md`.

## Planned Experiment

Output root:

```text
/data/sifang/cmp_gps
```

Primary sample:

```text
sample_name=s3a5_s1_0050
novel_id=2
```

Methods:

```text
origin_gps_raft
stereo_gs_wavelet
```

## Run Log

### 12:2x - Exporter setup

Created and syntax-checked:

```text
scripts/export_gps_diagnostics.py
scripts/compare_gps_diagnostic_dirs.py
```

The first origin run against the experiment backup root failed because the backup omitted `lib/gs_utils`. I reran using the live origin repo root while keeping the experiment config/checkpoint:

```text
repo_root=/data/sifang/gps_plus_origin
config=/data/sifang/gps_plus_origin/experiments/origin_gps_raft_0429/file/config/stage.yaml
checkpoint=/data/sifang/gps_plus_origin/experiments/origin_gps_raft_0429/ckpt/origin_gps_raft_latest.pth
```

The current StereoGS experiment backup also omits auxiliary package directories, so I used the current repo root while keeping the experiment config/checkpoint:

```text
repo_root=/data/sifang/GPS_plus
config=/data/sifang/GPS_plus/experiments/stereo_gs_hias_gps_data_wavelet_deepseekfixbug/file/config/stereo_gs_stage.yaml
checkpoint=/data/sifang/GPS_plus/experiments/stereo_gs_hias_gps_data_wavelet_deepseekfixbug/ckpt/latest.pth
```

### Exported Outputs

```text
/data/sifang/cmp_gps/origin_gps_raft_s3a5_s1_0050_v2
/data/sifang/cmp_gps/stereo_gs_wavelet_s3a5_s1_0050_v2
/data/sifang/cmp_gps/comparison_s3a5_s1_0050_v2
```

Each method directory contains:

```text
images/
maps/
npz/
ply/
projection/
stats.json
```

The comparison directory contains side-by-side montages, `summary.json`, and `REPORT.md`.

### Key Metrics

From `/data/sifang/cmp_gps/comparison_s3a5_s1_0050_v2/REPORT.md`:

```text
origin_gps_raft checkpoint_step=4500
stereo_gs_wavelet checkpoint_step=30000
sample=s3a5_s1_0050 novel_id=2

mae_rgb_0_255: origin 6.87902, current 6.76757
hole_candidate_ratio: origin 0.025795, current 0.0218801
projection_coverage_ratio: origin 0.917001, current 0.956472
projection_density_mean: origin 1.94454, current 7.80896
projection_density_p99: origin 6, current 20
left_base_opacity_p50: origin 0.494585, current 0.349905
right_base_opacity_p50: origin 0.322387, current 0.288336
left_base_scale_p50: origin 0.002, current 0.00230572
right_base_scale_p50: origin 0.002, current 0.00230834
left_sub_opacity_p50: current 0.00918196
right_sub_opacity_p50: current 0.00787192
```

## Initial Hypotheses

1. StereoGS depth may be sharper and more accurate, but Gaussian footprint coverage may be thinner, causing novel-view projection gaps around narrow structures.
2. The current method may predict lower opacity or smaller scale around low-confidence/edge pixels.
3. CAGS sub-Gaussians may activate on edges but still not contribute enough opacity/coverage to fill table-leg and leg-edge gaps.
4. If projected density is high at hole candidates but render is black, opacity/color is the likely failure point; if projected density is low, footprint/geometry coverage is the likely failure point.

## Current Evidence

The current StereoGS export has higher projected Gaussian coverage and density than origin GPS RAFT on the same sample/view. That weakens the hypothesis that holes are primarily caused by missing source points or globally missing projections.

The stronger signal is opacity contribution:

- Base opacity median is lower in current StereoGS than in origin GPS RAFT.
- CAGS creates many sub-Gaussians, but sub-Gaussian opacity median is only about `0.008-0.009`.
- Projection density is higher in current StereoGS, but visual holes/edge cracks can still appear if those projected Gaussians carry weak opacity or insufficient local footprint contribution.

Working root-cause hypothesis after this export:

```text
Current StereoGS has enough or more projected points, but thin structures are under-composited because opacity/sub-opacity is too weak and/or edge footprints are not contributing strongly enough.
```

## Next Targeted Test

Run the same exporter with controlled inference-only perturbations:

1. Scale multiplier sweep: render current StereoGS with `scale_maps *= 1.25, 1.5, 2.0`.
2. Opacity multiplier sweep: render current StereoGS with `opacity_maps *= 1.25, 1.5, 2.0` clamped to 1.
3. Sub-opacity multiplier sweep for CAGS only.
4. Base-only vs CAGS-all render to quantify whether CAGS helps or hurts the visible holes.

If opacity/scale perturbations remove holes without changing depth, the root cause is confirmed as Gaussian contribution/footprint rather than depth.

## Perturbation Test Results

I added inference-only perturbation options to `scripts/export_gps_diagnostics.py`:

```text
--scale-mult
--opacity-mult
--sub-opacity-mult
--drop-sub-gaussians
```

Exported:

```text
/data/sifang/cmp_gps/stereo_gs_scale1p5_s3a5_s1_0050_v2
/data/sifang/cmp_gps/stereo_gs_opacity1p5_s3a5_s1_0050_v2
/data/sifang/cmp_gps/stereo_gs_subopacity5_s3a5_s1_0050_v2
/data/sifang/cmp_gps/stereo_gs_no_sub_s3a5_s1_0050_v2
```

Summary:

```text
name          mae      hole_ratio  coverage  density_mean  left_op50  right_op50  left_sub_op50  right_sub_op50
origin        6.87902  0.0257950   0.917001  1.94454       0.494585   0.322387    n/a            n/a
stereo_base   6.76757  0.0218801   0.956472  7.80896       0.349905   0.288336    0.00918196     0.00787192
scale1.5      7.63204  0.0211353   0.956472  7.80896       0.349905   0.288336    0.00918196     0.00787192
opacity1.5    6.94770  0.0216475   0.956472  7.80896       0.524857   0.432505    0.00918196     0.00787192
subopacity5   7.05440  0.0209684   0.956472  7.80896       0.349905   0.288336    0.0459098      0.0393596
no_sub        8.11925  0.0288229   0.923818  1.95256       0.349905   0.288336    n/a            n/a
```

Interpretation:

- Dropping CAGS sub-Gaussians degrades MAE and coverage, so CAGS is helping rather than being the primary cause of holes.
- Increasing sub-Gaussian opacity improves the simple hole-candidate ratio the most among the tested perturbations.
- Increasing scale also reduces the simple hole-candidate ratio, but worsens MAE more, suggesting scale inflation may cover cracks at the cost of blur/bleeding.
- Increasing base opacity restores origin-like opacity medians but does not improve the simple hole metric as much as sub-opacity, suggesting thin/edge structures are particularly tied to weak CAGS child contribution.

Updated root-cause hypothesis:

```text
The current method has enough projected geometry. The visible holes are most likely caused by under-composited edge/thin-structure Gaussians, especially very low-opacity CAGS sub-Gaussians. Scale inflation can mask this but introduces broader render error.
```
