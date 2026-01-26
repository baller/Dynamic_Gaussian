#!/bin/bash
# GPS_plus Accelerate多卡训练启动脚本
# 
# 使用方法:
#   ./scripts/run_accelerate.sh                    # 使用默认配置（4卡）
#   ./scripts/run_accelerate.sh 2                  # 使用2张GPU
#   ./scripts/run_accelerate.sh 4 fp16             # 4卡 + FP16混合精度
#   ./scripts/run_accelerate.sh 4 bf16 2           # 4卡 + BF16 + 梯度累积2步

set -e

# ===================== 配置参数 =====================
# GPU数量（默认4）
NUM_GPUS=${1:-4}

# 混合精度模式: no, fp16, bf16（默认no）
MIXED_PRECISION=${2:-no}

# 梯度累积步数（默认1）
GRADIENT_ACCUMULATION=${3:-1}

# 配置文件路径
CONFIG_FILE="config/stage.yaml"

# 随机种子
SEED=1314

# 指定GPU设备（可选，默认使用前NUM_GPUS张卡）
# 如需指定特定GPU，取消注释并修改下面的行
# export CUDA_VISIBLE_DEVICES="0,1,2,3"

# ===================== 环境检查 =====================
echo "=============================================="
echo "GPS_plus Accelerate 多卡训练"
echo "=============================================="
echo "GPU数量: $NUM_GPUS"
echo "混合精度: $MIXED_PRECISION"
echo "梯度累积: $GRADIENT_ACCUMULATION"
echo "配置文件: $CONFIG_FILE"
echo "=============================================="

# 检查accelerate是否安装
if ! command -v accelerate &> /dev/null; then
    echo "错误: accelerate未安装，请运行: pip install accelerate"
    exit 1
fi

# 检查配置文件是否存在
if [ ! -f "$CONFIG_FILE" ]; then
    echo "错误: 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 检查训练脚本是否存在
if [ ! -f "train_accelerate.py" ]; then
    echo "错误: 训练脚本不存在: train_accelerate.py"
    exit 1
fi

# ===================== 运行训练 =====================
echo ""
echo "开始训练..."
echo ""

# 方式1: 使用命令行参数直接启动
accelerate launch \
    --multi_gpu \
    --num_processes $NUM_GPUS \
    --mixed_precision $MIXED_PRECISION \
    train_accelerate.py \
    --config $CONFIG_FILE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION \
    --mixed_precision $MIXED_PRECISION \
    --seed $SEED

echo ""
echo "=============================================="
echo "训练完成！"
echo "=============================================="
