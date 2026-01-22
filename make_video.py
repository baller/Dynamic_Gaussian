"""
将测试结果图片转换为视频

用法:
    python make_video.py -i <输入目录> -o <输出视频路径> [选项]

示例:
    # 基本用法 - 标准测试结果
    python make_video.py -i experiments/gps_plus_da3/show_s1a6_process -o output.mp4
    
    # 动态视角测试结果
    python make_video.py -i experiments/gps_plus_da3/show_s1a6_process_dynamic -o output.mp4 --type dynamic
    
    # 多视角测试结果
    python make_video.py -i experiments/gps_plus_da3/show_s1a6_process_allviews -o output.mp4 --type allviews
    
    # 指定帧率
    python make_video.py -i experiments/gps_plus_da3/show_s1a6_process -o output.mp4 --fps 30
    
    # 只处理渲染结果（不包含深度图）
    python make_video.py -i experiments/gps_plus_da3/show_s1a6_process -o output.mp4 --mode render
    
    # 只处理深度图
    python make_video.py -i experiments/gps_plus_da3/show_s1a6_process -o output.mp4 --mode depth
    
    # 并排显示渲染和深度图
    python make_video.py -i experiments/gps_plus_da3/show_s1a6_process -o output.mp4 --mode side_by_side
"""

import argparse
import os
import re
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Tuple, Optional


def natural_sort_key(s: str) -> List:
    """自然排序键函数，使得文件名按数字顺序排列"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', s)]


def find_images(input_dir: str, pattern: str = None) -> List[str]:
    """
    在目录中查找图片文件
    
    Args:
        input_dir: 输入目录
        pattern: 文件名模式（正则表达式），None表示所有图片
        
    Returns:
        排序后的图片路径列表
    """
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_files = []
    
    for f in os.listdir(input_dir):
        ext = os.path.splitext(f)[1].lower()
        if ext in image_extensions:
            if pattern is None or re.search(pattern, f):
                image_files.append(os.path.join(input_dir, f))
    
    # 自然排序
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    return image_files


def get_render_and_depth_pairs(input_dir: str, output_type: str = 'standard') -> List[Tuple[str, Optional[str]]]:
    """
    获取渲染图和深度图的配对
    
    Args:
        input_dir: 输入目录
        output_type: 输出类型 ('standard', 'dynamic', 'allviews')
        
    Returns:
        (渲染图路径, 深度图路径或None) 的列表
    """
    # 查找所有渲染结果（不包含depth的jpg文件）
    render_files = []
    depth_files = {}
    
    for f in os.listdir(input_dir):
        filepath = os.path.join(input_dir, f)
        if f.endswith('.jpg') and 'depth' not in f.lower():
            render_files.append(filepath)
        elif 'depth' in f.lower() and (f.endswith('.png') or f.endswith('.jpg')):
            # 提取基础名称用于匹配
            # 标准格式: s1a6_s1_0000_02_depth_lmain_color.png -> s1a6_s1_0000_02
            # 动态格式: s1a6_s1_0000_dynamic_0001_depth.png -> s1a6_s1_0000_dynamic_0001
            base_name = f.split('_depth')[0]
            depth_files[base_name] = filepath
    
    # 排序渲染文件
    render_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    # 配对
    pairs = []
    for render_path in render_files:
        base_name = os.path.splitext(os.path.basename(render_path))[0]
        depth_path = depth_files.get(base_name)
        pairs.append((render_path, depth_path))
    
    return pairs


def find_dynamic_images(input_dir: str, include_depth: bool = False) -> List[str]:
    """
    查找动态视角测试的图片文件
    
    Args:
        input_dir: 输入目录
        include_depth: 是否包含深度图
        
    Returns:
        排序后的图片路径列表
    """
    image_files = []
    
    for f in os.listdir(input_dir):
        filepath = os.path.join(input_dir, f)
        
        if include_depth:
            # 只查找深度图
            if 'dynamic' in f and 'depth' in f and (f.endswith('.png') or f.endswith('.jpg')):
                image_files.append(filepath)
        else:
            # 只查找渲染结果（包含dynamic但不包含depth的jpg文件）
            if 'dynamic' in f and 'depth' not in f and f.endswith('.jpg'):
                image_files.append(filepath)
    
    # 自然排序
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    return image_files


def find_allviews_images(input_dir: str, view_id: int = None, include_depth: bool = False) -> List[str]:
    """
    查找多视角测试的图片文件
    
    支持两种文件名格式:
    - 新格式: s1a6_0000_v02.jpg (帧序号_视角)
    - 旧格式: s1a6_s1_0000_02.jpg
    
    Args:
        input_dir: 输入目录
        view_id: 指定视角ID，None表示所有视角
        include_depth: 是否包含深度图
        
    Returns:
        排序后的图片路径列表
    """
    image_files = []
    
    for f in os.listdir(input_dir):
        filepath = os.path.join(input_dir, f)
        
        if include_depth:
            if 'depth' in f and (f.endswith('.png') or f.endswith('.jpg')):
                if view_id is not None:
                    # 新格式: xxx_0000_v02_depth.png
                    match = re.search(r'_v(\d{2})_depth', f)
                    if match and int(match.group(1)) == view_id:
                        image_files.append(filepath)
                    else:
                        # 旧格式: xxx_02_depth_lmain_color.png
                        match = re.search(r'_(\d{2})_depth', f)
                        if match and int(match.group(1)) == view_id:
                            image_files.append(filepath)
                else:
                    image_files.append(filepath)
        else:
            if f.endswith('.jpg') and 'depth' not in f:
                if view_id is not None:
                    # 新格式: xxx_0000_v02.jpg
                    match = re.search(r'_v(\d{2})\.jpg$', f)
                    if match and int(match.group(1)) == view_id:
                        image_files.append(filepath)
                    else:
                        # 旧格式: xxx_02.jpg
                        match = re.search(r'_(\d{2})\.jpg$', f)
                        if match and int(match.group(1)) == view_id:
                            image_files.append(filepath)
                else:
                    image_files.append(filepath)
    
    # 自然排序
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    return image_files


def find_sequential_views_images(input_dir: str, include_depth: bool = False) -> List[str]:
    """
    查找顺序视角测试的图片文件 (按帧-视角顺序排列)
    
    文件名格式: s1a6_0000_v02.jpg (序列_帧号_视角)
    
    Args:
        input_dir: 输入目录
        include_depth: 是否包含深度图
        
    Returns:
        按帧-视角顺序排列的图片路径列表
    """
    image_files = []
    
    for f in os.listdir(input_dir):
        filepath = os.path.join(input_dir, f)
        
        # 匹配新格式: xxx_0000_v02.jpg 或 xxx_0000_v02_depth.png
        if include_depth:
            if '_v' in f and 'depth' in f and (f.endswith('.png') or f.endswith('.jpg')):
                image_files.append(filepath)
        else:
            if '_v' in f and f.endswith('.jpg') and 'depth' not in f:
                image_files.append(filepath)
    
    # 自然排序 - 按帧号和视角号排序
    image_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
    
    return image_files


def create_allviews_video(
    input_dir: str,
    output_path: str,
    fps: int = 60,
    codec: str = 'mp4v',
    views: List[int] = None
) -> None:
    """
    创建多视角循环视频 - 按视角分组然后循环播放
    
    Args:
        input_dir: 输入目录
        output_path: 输出视频路径
        fps: 帧率
        codec: 视频编码器
        views: 视角列表
    """
    if views is None:
        views = [0, 1, 2, 3]
    
    # 收集每个视角的图片
    view_images = {}
    for view_id in views:
        images = find_allviews_images(input_dir, view_id=view_id, include_depth=False)
        if images:
            view_images[view_id] = images
            print(f"视角 {view_id}: {len(images)} 张图片")
    
    if not view_images:
        print("没有找到多视角图片！")
        return
    
    # 获取每个视角的帧数（取最小值）
    min_frames = min(len(imgs) for imgs in view_images.values())
    print(f"每个视角取 {min_frames} 帧")
    
    # 按帧索引组织，每一帧包含所有视角
    all_frames = []
    for frame_idx in range(min_frames):
        for view_id in sorted(view_images.keys()):
            if frame_idx < len(view_images[view_id]):
                all_frames.append(view_images[view_id][frame_idx])
    
    create_video_from_images(all_frames, output_path, fps, codec)


def create_video_from_images(
    image_paths: List[str],
    output_path: str,
    fps: int = 15,
    codec: str = 'mp4v'
) -> None:
    """
    从图片列表创建视频
    
    Args:
        image_paths: 图片路径列表
        output_path: 输出视频路径
        fps: 帧率
        codec: 视频编码器
    """
    if not image_paths:
        print("没有找到图片！")
        return
    
    # 读取第一张图片获取尺寸
    first_img = cv2.imread(image_paths[0])
    if first_img is None:
        print(f"无法读取图片: {image_paths[0]}")
        return
    
    height, width = first_img.shape[:2]
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"创建视频: {output_path}")
    print(f"分辨率: {width}x{height}, 帧率: {fps}, 总帧数: {len(image_paths)}")
    
    for img_path in tqdm(image_paths, desc="写入帧"):
        img = cv2.imread(img_path)
        if img is None:
            print(f"警告: 无法读取图片 {img_path}，跳过")
            continue
        
        # 确保尺寸一致
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        out.write(img)
    
    out.release()
    print(f"视频已保存: {output_path}")


def create_side_by_side_video(
    pairs: List[Tuple[str, Optional[str]]],
    output_path: str,
    fps: int = 15,
    codec: str = 'mp4v'
) -> None:
    """
    创建并排显示渲染和深度图的视频
    
    Args:
        pairs: (渲染图路径, 深度图路径) 的列表
        output_path: 输出视频路径
        fps: 帧率
        codec: 视频编码器
    """
    if not pairs:
        print("没有找到图片对！")
        return
    
    # 读取第一张图片获取尺寸
    first_render = cv2.imread(pairs[0][0])
    if first_render is None:
        print(f"无法读取图片: {pairs[0][0]}")
        return
    
    height, width = first_render.shape[:2]
    
    # 并排显示，总宽度翻倍
    total_width = width * 2
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, height))
    
    print(f"创建并排视频: {output_path}")
    print(f"分辨率: {total_width}x{height}, 帧率: {fps}, 总帧数: {len(pairs)}")
    
    for render_path, depth_path in tqdm(pairs, desc="写入帧"):
        render_img = cv2.imread(render_path)
        if render_img is None:
            print(f"警告: 无法读取渲染图 {render_path}，跳过")
            continue
        
        # 确保渲染图尺寸一致
        if render_img.shape[:2] != (height, width):
            render_img = cv2.resize(render_img, (width, height))
        
        # 处理深度图
        if depth_path and os.path.exists(depth_path):
            depth_img = cv2.imread(depth_path)
            if depth_img is not None:
                if depth_img.shape[:2] != (height, width):
                    depth_img = cv2.resize(depth_img, (width, height))
            else:
                depth_img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            depth_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 并排拼接
        combined = np.hstack([render_img, depth_img])
        
        out.write(combined)
    
    out.release()
    print(f"视频已保存: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='将测试结果图片转换为视频')
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入目录路径')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='输出视频路径（默认为输入目录名.mp4）')
    parser.add_argument('--fps', type=int, default=15,
                        help='视频帧率（默认15）')
    parser.add_argument('--mode', type=str, default='render',
                        choices=['render', 'depth', 'side_by_side', 'all'],
                        help='视频模式: render(仅渲染), depth(仅深度), side_by_side(并排), all(分别生成)')
    parser.add_argument('--type', type=str, default='standard',
                        choices=['standard', 'dynamic', 'allviews', 'sequential'],
                        help='输入类型: standard(标准测试), dynamic(动态视角), allviews(多视角), sequential(顺序视角)')
    parser.add_argument('--views', type=str, default='0,1,2,3',
                        help='多视角模式下要处理的视角列表（默认0,1,2,3）')
    parser.add_argument('--codec', type=str, default='mp4v',
                        help='视频编码器（默认mp4v）')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.isdir(args.input):
        print(f"错误: 输入目录不存在: {args.input}")
        return
    
    # 解析视角列表
    views_list = [int(v.strip()) for v in args.views.split(',')]
    
    # 设置默认输出路径
    if args.output is None:
        base_name = os.path.basename(args.input.rstrip('/'))
        args.output = f"{base_name}.mp4"
    
    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"=" * 50)
    print(f"输入目录: {args.input}")
    print(f"输出类型: {args.type}")
    print(f"视频模式: {args.mode}")
    print(f"帧率: {args.fps}")
    print(f"=" * 50)
    
    # 根据输入类型处理
    if args.type == 'dynamic':
        # 动态视角测试结果
        if args.mode == 'render' or args.mode == 'all':
            render_files = find_dynamic_images(args.input, include_depth=False)
            output_path = args.output if args.mode == 'render' else args.output.replace('.mp4', '_render.mp4')
            create_video_from_images(render_files, output_path, args.fps, args.codec)
        
        if args.mode == 'depth' or args.mode == 'all':
            depth_files = find_dynamic_images(args.input, include_depth=True)
            output_path = args.output.replace('.mp4', '_depth.mp4') if args.output.endswith('.mp4') else args.output + '_depth.mp4'
            if depth_files:
                create_video_from_images(depth_files, output_path, args.fps, args.codec)
            else:
                print("没有找到动态视角深度图")
        
        if args.mode == 'side_by_side' or args.mode == 'all':
            pairs = get_render_and_depth_pairs(args.input, output_type='dynamic')
            # 过滤只包含dynamic的文件
            dynamic_pairs = [(r, d) for r, d in pairs if 'dynamic' in os.path.basename(r)]
            output_path = args.output.replace('.mp4', '_combined.mp4') if args.output.endswith('.mp4') else args.output + '_combined.mp4'
            if dynamic_pairs:
                create_side_by_side_video(dynamic_pairs, output_path, args.fps, args.codec)
            else:
                print("没有找到动态视角图片对")
    
    elif args.type == 'allviews':
        # 多视角测试结果
        if args.mode == 'render' or args.mode == 'all':
            # 创建多视角循环视频
            output_path = args.output if args.mode == 'render' else args.output.replace('.mp4', '_render.mp4')
            create_allviews_video(args.input, output_path, args.fps, args.codec, views_list)
        
        if args.mode == 'depth' or args.mode == 'all':
            # 按视角收集深度图
            all_depth_files = []
            for view_id in views_list:
                depth_files = find_allviews_images(args.input, view_id=view_id, include_depth=True)
                all_depth_files.extend(depth_files)
            
            if all_depth_files:
                output_path = args.output.replace('.mp4', '_depth.mp4') if args.output.endswith('.mp4') else args.output + '_depth.mp4'
                all_depth_files.sort(key=lambda x: natural_sort_key(os.path.basename(x)))
                create_video_from_images(all_depth_files, output_path, args.fps, args.codec)
            else:
                print("没有找到多视角深度图")
        
        if args.mode == 'side_by_side' or args.mode == 'all':
            pairs = get_render_and_depth_pairs(args.input, output_type='allviews')
            output_path = args.output.replace('.mp4', '_combined.mp4') if args.output.endswith('.mp4') else args.output + '_combined.mp4'
            if pairs:
                create_side_by_side_video(pairs, output_path, args.fps, args.codec)
            else:
                print("没有找到多视角图片对")
    
    elif args.type == 'sequential':
        # 顺序视角测试结果 (按帧-视角顺序排列)
        if args.mode == 'render' or args.mode == 'all':
            render_files = find_sequential_views_images(args.input, include_depth=False)
            if not render_files:
                # 如果没有找到新格式，尝试旧格式
                render_files = find_allviews_images(args.input, include_depth=False)
            output_path = args.output if args.mode == 'render' else args.output.replace('.mp4', '_render.mp4')
            create_video_from_images(render_files, output_path, args.fps, args.codec)
        
        if args.mode == 'depth' or args.mode == 'all':
            depth_files = find_sequential_views_images(args.input, include_depth=True)
            if not depth_files:
                depth_files = find_allviews_images(args.input, include_depth=True)
            if depth_files:
                output_path = args.output.replace('.mp4', '_depth.mp4') if args.output.endswith('.mp4') else args.output + '_depth.mp4'
                create_video_from_images(depth_files, output_path, args.fps, args.codec)
            else:
                print("没有找到顺序视角深度图")
        
        if args.mode == 'side_by_side' or args.mode == 'all':
            # 获取渲染图和深度图的配对
            render_files = find_sequential_views_images(args.input, include_depth=False)
            depth_files = find_sequential_views_images(args.input, include_depth=True)
            
            # 构建配对
            depth_dict = {}
            for d in depth_files:
                base = os.path.basename(d).replace('_depth.png', '').replace('_depth.jpg', '')
                depth_dict[base] = d
            
            pairs = []
            for r in render_files:
                base = os.path.splitext(os.path.basename(r))[0]
                d = depth_dict.get(base)
                pairs.append((r, d))
            
            output_path = args.output.replace('.mp4', '_combined.mp4') if args.output.endswith('.mp4') else args.output + '_combined.mp4'
            if pairs:
                create_side_by_side_video(pairs, output_path, args.fps, args.codec)
            else:
                print("没有找到顺序视角图片对")
    
    else:
        # 标准测试结果
        if args.mode == 'render':
            render_files = find_images(args.input, pattern=r'^(?!.*depth).*\.jpg$')
            create_video_from_images(render_files, args.output, args.fps, args.codec)
            
        elif args.mode == 'depth':
            depth_files = find_images(args.input, pattern=r'depth.*\.(png|jpg)$')
            output_path = args.output.replace('.mp4', '_depth.mp4') if args.output.endswith('.mp4') else args.output + '_depth.mp4'
            create_video_from_images(depth_files, output_path, args.fps, args.codec)
            
        elif args.mode == 'side_by_side':
            pairs = get_render_and_depth_pairs(args.input)
            output_path = args.output.replace('.mp4', '_combined.mp4') if args.output.endswith('.mp4') else args.output + '_combined.mp4'
            create_side_by_side_video(pairs, output_path, args.fps, args.codec)
            
        elif args.mode == 'all':
            # 生成所有类型的视频
            # 渲染视频
            render_files = find_images(args.input, pattern=r'^(?!.*depth).*\.jpg$')
            render_output = args.output.replace('.mp4', '_render.mp4') if args.output.endswith('.mp4') else args.output + '_render.mp4'
            create_video_from_images(render_files, render_output, args.fps, args.codec)
            
            # 深度视频
            depth_files = find_images(args.input, pattern=r'depth.*\.(png|jpg)$')
            if depth_files:
                depth_output = args.output.replace('.mp4', '_depth.mp4') if args.output.endswith('.mp4') else args.output + '_depth.mp4'
                create_video_from_images(depth_files, depth_output, args.fps, args.codec)
            
            # 并排视频
            pairs = get_render_and_depth_pairs(args.input)
            combined_output = args.output.replace('.mp4', '_combined.mp4') if args.output.endswith('.mp4') else args.output + '_combined.mp4'
            create_side_by_side_video(pairs, combined_output, args.fps, args.codec)


if __name__ == '__main__':
    main()
