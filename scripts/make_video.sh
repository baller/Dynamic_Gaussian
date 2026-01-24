#!/bin/bash

# 将图片序列合成视频的脚本
# 用法: ./scripts/make_video.sh <图片目录> [帧率] [输出文件名]

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 显示帮助信息
show_help() {
    echo "用法: $0 <图片目录> [帧率] [输出文件名]"
    echo ""
    echo "参数:"
    echo "  图片目录      必需，包含图片序列的目录路径"
    echo "  帧率          可选，视频帧率 (默认: 30)"
    echo "  输出文件名    可选，输出视频文件名 (默认: output_video.mp4)"
    echo ""
    echo "示例:"
    echo "  $0 experiments/gps_plus_da3/show_free_s1a6"
    echo "  $0 experiments/gps_plus_da3/show_free_s1a6 60"
    echo "  $0 experiments/gps_plus_da3/show_free_s1a6 30 result.mp4"
    echo ""
    exit 1
}

# 检查参数
if [ $# -lt 1 ]; then
    echo -e "${RED}错误: 缺少必需参数${NC}"
    show_help
fi

# 解析参数
IMAGE_DIR="$1"
FPS="${2:-30}"
OUTPUT_NAME="${3:-output_video.mp4}"

# 检查图片目录是否存在
if [ ! -d "$IMAGE_DIR" ]; then
    echo -e "${RED}错误: 图片目录不存在: $IMAGE_DIR${NC}"
    exit 1
fi

# 统计图片数量
IMAGE_COUNT=$(find "$IMAGE_DIR" -maxdepth 1 -name "*.jpg" -o -name "*.png" | wc -l)

if [ "$IMAGE_COUNT" -eq 0 ]; then
    echo -e "${RED}错误: 目录中没有找到图片文件 (*.jpg 或 *.png)${NC}"
    exit 1
fi

echo -e "${GREEN}==================== 视频生成配置 ====================${NC}"
echo -e "图片目录: ${YELLOW}$IMAGE_DIR${NC}"
echo -e "图片数量: ${YELLOW}$IMAGE_COUNT 张${NC}"
echo -e "帧率: ${YELLOW}$FPS FPS${NC}"
echo -e "输出文件: ${YELLOW}$IMAGE_DIR/$OUTPUT_NAME${NC}"
echo -e "${GREEN}====================================================${NC}"
echo ""

# 检查ffmpeg是否安装
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}错误: ffmpeg 未安装，请先安装 ffmpeg${NC}"
    exit 1
fi

# 选择可用编码器
ENCODER="libx264"
ENCODER_OPTS=("-crf" "18" "-preset" "medium")

if ! ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "libx264"; then
    if ffmpeg -hide_banner -encoders 2>/dev/null | grep -q "libopenh264"; then
        ENCODER="libopenh264"
        ENCODER_OPTS=("-b:v" "4M")
        echo -e "${YELLOW}提示: 未检测到 libx264，使用 libopenh264 编码${NC}"
    else
        ENCODER="h264"
        ENCODER_OPTS=()
        echo -e "${YELLOW}提示: 未检测到 libx264/libopenh264，使用默认 h264 编码${NC}"
    fi
fi

make_video_for_dir() {
    local dir="$1"
    local output="$2"
    local pattern="*.jpg"
    local count

    count=$(find "$dir" -maxdepth 1 -name "*.jpg" -o -name "*.png" | wc -l)
    if [ "$count" -eq 0 ]; then
        echo -e "${YELLOW}跳过: 目录无图片 $dir${NC}"
        return
    fi

    if ! find "$dir" -maxdepth 1 -name "*.jpg" | grep -q .; then
        pattern="*.png"
    fi

    echo -e "${GREEN}开始生成视频: ${YELLOW}$dir/$output${NC}"
    (
        cd "$dir" || exit 1
        ffmpeg -framerate "$FPS" \
            -pattern_type glob -i "$pattern" \
            -c:v "$ENCODER" \
            -pix_fmt yuv420p \
            "${ENCODER_OPTS[@]}" \
            "$output" \
            -y
    )
}

# 生成主视频
make_video_for_dir "$IMAGE_DIR" "$OUTPUT_NAME"
MAIN_STATUS=$?

# 如果存在原视频目录，则额外生成原视频
ORIGIN_DIR="$IMAGE_DIR/origin"
if [ -d "$ORIGIN_DIR" ]; then
    make_video_for_dir "$ORIGIN_DIR" "origin_${OUTPUT_NAME}"
    ORIGIN_STATUS=$?
else
    ORIGIN_STATUS=0
fi

# 检查是否成功
if [ $MAIN_STATUS -eq 0 ] && [ $ORIGIN_STATUS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}==================== 生成成功! ====================${NC}"
    VIDEO_SIZE=$(du -h "$IMAGE_DIR/$OUTPUT_NAME" | cut -f1)
    VIDEO_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$IMAGE_DIR/$OUTPUT_NAME" 2>/dev/null | awk '{printf "%.2f", $1}')
    echo -e "视频路径: ${YELLOW}$IMAGE_DIR/$OUTPUT_NAME${NC}"
    echo -e "文件大小: ${YELLOW}$VIDEO_SIZE${NC}"
    echo -e "视频时长: ${YELLOW}${VIDEO_DURATION}秒${NC}"
    echo -e "分辨率: ${YELLOW}$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$IMAGE_DIR/$OUTPUT_NAME" 2>/dev/null)${NC}"

    if [ -d "$ORIGIN_DIR" ]; then
        ORIGIN_VIDEO="$ORIGIN_DIR/origin_${OUTPUT_NAME}"
        if [ -f "$ORIGIN_VIDEO" ]; then
            ORIGIN_SIZE=$(du -h "$ORIGIN_VIDEO" | cut -f1)
            ORIGIN_DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$ORIGIN_VIDEO" 2>/dev/null | awk '{printf "%.2f", $1}')
            echo -e "原视频路径: ${YELLOW}$ORIGIN_VIDEO${NC}"
            echo -e "原视频大小: ${YELLOW}$ORIGIN_SIZE${NC}"
            echo -e "原视频时长: ${YELLOW}${ORIGIN_DURATION}秒${NC}"
            echo -e "原视频分辨率: ${YELLOW}$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 "$ORIGIN_VIDEO" 2>/dev/null)${NC}"
        fi
    fi
    echo -e "${GREEN}====================================================${NC}"
else
    echo -e "${RED}错误: 视频生成失败${NC}"
    exit 1
fi
