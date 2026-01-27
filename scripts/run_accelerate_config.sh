#!/bin/bash
# GPS_plus Accelerate多卡训练 - 使用配置文件启动
# 
# 使用方法:
#   ./scripts/run_accelerate_config.sh                           # 默认配置
#   ./scripts/run_accelerate_config.sh my_accelerate_config.yaml # 自定义配置

set -e

# ===================== 配置参数 =====================
# Accelerate配置文件（默认使用项目根目录的配置）
ACCELERATE_CONFIG=${1:-accelerate_config.yaml}

# 训练配置文件
TRAIN_CONFIG="config/stage.yaml"

# 梯度累积步数
GRADIENT_ACCUMULATION=1

# 随机种子
SEED=1314

# ===================== 环境检查 =====================
echo "=============================================="
echo "GPS_plus Accelerate 多卡训练 (配置文件模式)"
echo "=============================================="
echo "Accelerate配置: $ACCELERATE_CONFIG"
echo "训练配置: $TRAIN_CONFIG"
echo "=============================================="

# 检查accelerate配置文件
if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "错误: Accelerate配置文件不存在: $ACCELERATE_CONFIG"
    echo "提示: 可以运行 'accelerate config' 生成配置文件"
    exit 1
fi

# ===================== 运行训练 =====================
echo ""
echo "开始训练..."
echo ""

CUDA_VISIBLE_DEVICES=3 accelerate launch \
    --config_file $ACCELERATE_CONFIG \
    train_accelerate.py \
    --config $TRAIN_CONFIG \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --mixed_precision fp16 \
    --seed $SEED

echo ""
echo "=============================================="
echo "训练完成！"
echo "=============================================="
