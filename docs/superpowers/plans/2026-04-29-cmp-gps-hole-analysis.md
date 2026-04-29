# GPS Hole Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export comparable intermediate artifacts for the current StereoGS method and the original GPS RAFT method to diagnose table-leg and human-leg edge holes.

**Architecture:** Use a standalone diagnostic exporter that imports the requested repository snapshot in a clean Python process, restores one checkpoint, runs one validation sample, renders the novel view, and writes normalized visualizations plus sampled/full data products. Keep experiment notes in `docs/experiment_logs` and write generated artifacts under `/data/sifang/cmp_gps`.

**Tech Stack:** Python, PyTorch, OpenCV, NumPy, existing GPS/Gaussian renderer modules.

---

### Task 1: Exporter Script

**Files:**
- Create: `scripts/export_gps_diagnostics.py`

- [x] **Step 1: Create a standalone exporter**

The script accepts `--repo-root`, `--config`, `--ckpt`, `--method-name`, `--sample-name`, `--novel-id`, and `--out`.

- [x] **Step 2: Save diagnostics**

For each method, write `images/`, `maps/`, `npz/`, `ply/`, `projection/`, and `stats.json`.

### Task 2: Experiment Run

**Files:**
- Create/update: `docs/experiment_logs/2026-04-29-cmp-gps-hole-analysis.md`
- Output: `/data/sifang/cmp_gps`

- [ ] **Step 1: Run original GPS RAFT export**

Run from `/data/sifang/GPS_plus`:

```bash
python scripts/export_gps_diagnostics.py \
  --repo-root /data/sifang/gps_plus_origin/experiments/origin_gps_raft_0429/file \
  --config /data/sifang/gps_plus_origin/experiments/origin_gps_raft_0429/file/config/stage.yaml \
  --ckpt /data/sifang/gps_plus_origin/experiments/origin_gps_raft_0429/ckpt/origin_gps_raft_latest.pth \
  --method-name origin_gps_raft \
  --sample-name s3a5_s1_0050 \
  --novel-id 2 \
  --out /data/sifang/cmp_gps/origin_gps_raft_s3a5_s1_0050_v2
```

- [ ] **Step 2: Run current StereoGS export**

```bash
python scripts/export_gps_diagnostics.py \
  --repo-root /data/sifang/GPS_plus/experiments/stereo_gs_hias_gps_data_wavelet_deepseekfixbug/file \
  --config /data/sifang/GPS_plus/experiments/stereo_gs_hias_gps_data_wavelet_deepseekfixbug/file/config/stereo_gs_stage.yaml \
  --ckpt /data/sifang/GPS_plus/experiments/stereo_gs_hias_gps_data_wavelet_deepseekfixbug/ckpt/latest.pth \
  --method-name stereo_gs_wavelet \
  --sample-name s3a5_s1_0050 \
  --novel-id 2 \
  --out /data/sifang/cmp_gps/stereo_gs_wavelet_s3a5_s1_0050_v2
```

- [ ] **Step 3: Inspect `stats.json` and key visualizations**

Compare `hole_candidate_ratio`, projection coverage, density on hole candidates, opacity quantiles, scale quantiles, and sub-Gaussian counts.
