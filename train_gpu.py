#!/usr/bin/env python3
"""
占用指定显卡90%显存的脚本

用法:
    python occupy_gpu.py --gpu 0              # 占用GPU 0的90%显存
    python occupy_gpu.py --gpu 1 --ratio 0.8  # 占用GPU 1的80%显存
    python occupy_gpu.py --gpu 0,1            # 同时占用GPU 0和1
"""

import argparse
import time
import torch


def get_gpu_memory_info(device_id: int) -> tuple[int, int]:
    """获取GPU显存信息（总量和已用量），单位：字节"""
    torch.cuda.set_device(device_id)
    total = torch.cuda.get_device_properties(device_id).total_memory
    reserved = torch.cuda.memory_reserved(device_id)
    return total, reserved


def occupy_gpu_memory(device_id: int, ratio: float = 0.9) -> torch.Tensor:
    """
    占用指定GPU的显存
    
    Args:
        device_id: GPU设备ID
        ratio: 占用比例，默认0.9（90%）
    
    Returns:
        占用显存的tensor
    """
    torch.cuda.set_device(device_id)
    
    total_memory, reserved_memory = get_gpu_memory_info(device_id)
    available_memory = total_memory - reserved_memory
    
    # 计算需要分配的显存（留一点余量避免OOM）
    target_memory = int(total_memory * ratio)
    allocate_memory = target_memory - reserved_memory
    
    if allocate_memory <= 0:
        print(f"GPU {device_id}: 已占用足够显存，无需额外分配")
        return None
    
    # 计算需要分配的float32元素数量（每个4字节）
    num_elements = allocate_memory // 4
    
    # 分配显存
    tensor = torch.empty(num_elements, dtype=torch.float32, device=f'cuda:{device_id}')
    
    # 打印信息
    total_gb = total_memory / (1024 ** 3)
    allocated_gb = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
    
    print(f"GPU {device_id} ({torch.cuda.get_device_name(device_id)}):")
    print(f"  总显存: {total_gb:.2f} GB")
    print(f"  已占用: {allocated_gb:.2f} GB ({allocated_gb/total_gb*100:.1f}%)")
    
    return tensor


def main():
    parser = argparse.ArgumentParser(description='占用指定GPU的显存')
    parser.add_argument('--gpu', type=str, default='1', 
                        help='GPU设备ID，多个用逗号分隔，如 "0,1,2"')
    parser.add_argument('--ratio', type=float, default=0.9,
                        help='显存占用比例，默认0.9（90%%）')
    args = parser.parse_args()
    
    # 解析GPU ID
    gpu_ids = [int(x.strip()) for x in args.gpu.split(',')]
    
    print(f"准备占用 GPU {gpu_ids} 的 {args.ratio*100:.0f}% 显存...\n")
    
    # 占用每个GPU的显存
    tensors = []
    for gpu_id in gpu_ids:
        if gpu_id >= torch.cuda.device_count():
            print(f"错误: GPU {gpu_id} 不存在，可用GPU数量: {torch.cuda.device_count()}")
            continue
        tensor = occupy_gpu_memory(gpu_id, args.ratio)
        if tensor is not None:
            tensors.append(tensor)
        print()
    
    if not tensors:
        print("没有成功占用任何GPU显存")
        return
    
    print("=" * 50)
    print("显存占用中... 按 Ctrl+C 释放显存并退出")
    print("=" * 50)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n正在释放显存...")
        del tensors
        torch.cuda.empty_cache()
        print("显存已释放，程序退出")


if __name__ == '__main__':
    main()
