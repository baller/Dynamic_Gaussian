#!/bin/bash
# StereoGS 消融实验批量运行脚本
#
# 用法:
#   bash run_ablation.sh                    # 运行全部消融
#   bash run_ablation.sh fusion_none        # 只运行指定消融
#
# 消融矩阵:
#   1. 跨视图融合:   none / warp_attention / occlusion_aware (默认)
#   2. 超分方案:     convex (默认) / split
#   3. 置信度引导:   无 (alpha=beta=1) / 有 (默认)
#   4. 渲染后精化:   无 / 有 (默认)
#   5. 完整方案:     config/stereo_gs_stage.yaml (默认)

set -e

CONFIGS=(
    "config/stereo_gs_stage.yaml"
    "config/ablation/fusion_none.yaml"
    "config/ablation/fusion_warp_attention.yaml"
    "config/ablation/sr_split.yaml"
    "config/ablation/no_confidence.yaml"
    "config/ablation/no_refine.yaml"
)

FILTER="${1:-}"

for cfg in "${CONFIGS[@]}"; do
    name=$(basename "$cfg" .yaml)
    if [ -n "$FILTER" ] && [ "$name" != "$FILTER" ]; then
        continue
    fi
    echo "=========================================="
    echo "Running: $name ($cfg)"
    echo "=========================================="
    python train_stereo_gs.py --config "$cfg" --auto_resume
    echo ""
done

echo "All ablation experiments completed."
