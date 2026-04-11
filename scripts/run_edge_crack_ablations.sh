#!/bin/bash
# Edge Crack 诊断消融实验 — 4 组独立实验，每组 20k steps
# GPU 0, conda env: gps_plus_env
set -e

cd /data/sifang/GPS_plus

echo "=========================================="
echo "Ablation 1/4: max_scale 0.003 -> 0.01"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python train_stereo_gs.py \
    --config config/ablation/edge_max_scale.yaml \
    --exp_name stereo_gs_abl_edge_max_scale

echo "=========================================="
echo "Ablation 2/4: max_pos_offset 0.002 -> 0.01"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python train_stereo_gs.py \
    --config config/ablation/edge_max_offset.yaml \
    --exp_name stereo_gs_abl_edge_max_offset

echo "=========================================="
echo "Ablation 3/4: PostRefinement enabled"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python train_stereo_gs.py \
    --config config/ablation/edge_post_refine.yaml \
    --exp_name stereo_gs_abl_edge_post_refine

echo "=========================================="
echo "Ablation 4/4: warp padding zeros -> border"
echo "=========================================="
CUDA_VISIBLE_DEVICES=0 python train_stereo_gs.py \
    --config config/ablation/edge_warp_border.yaml \
    --exp_name stereo_gs_abl_edge_warp_border

echo "=========================================="
echo "All 4 ablation experiments completed!"
echo "=========================================="
