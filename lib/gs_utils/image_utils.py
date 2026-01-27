#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    标准公式: PSNR = 20 * log10(MAX_I / sqrt(MSE))
    对于值域[0,1]的图像: PSNR = 20 * log10(1 / sqrt(MSE))
    
    Args:
        img1: 图像1 [B, C, H, W] 或 [B, C, H, W]
        img2: 图像2 [B, C, H, W] 或 [B, C, H, W]
        
    Returns:
        psnr: [B, 1] PSNR值（dB）
    """
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    # 数值稳定性：避免MSE=0时出现inf
    mse = torch.clamp(mse, min=1e-10)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
