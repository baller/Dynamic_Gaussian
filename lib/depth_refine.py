"""
DepthRefineNet: 轻量级可学习深度细化模块

在 DA3 输出深度后、点云生成之前，利用图像信息细化深度边缘。
这是将 DA3 "转化为自己工作"的关键模块。

学术故事: "Stereo-Guided Depth Refinement"
- DA3 提供强语义先验 (平滑、无空洞)
- 本模块利用图像梯度信息细化边缘锯齿
- 残差模式: refined_depth = da3_depth + residual * scale
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock2d(nn.Module):
    """简单的 2D 残差块"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class DepthRefineNet(nn.Module):
    """
    轻量级深度细化网络

    输入: DA3 raw depth [B, 1, H, W] + RGB image [B, 3, H, W]
    输出: 细化后的深度 [B, 1, H, W] (残差模式)

    Args:
        cfg: 配置对象，需包含 cfg.depth_refine 节点
    """

    def __init__(self, cfg):
        super().__init__()
        refine_cfg = cfg.depth_refine
        ch = getattr(refine_cfg, 'channels', 64)
        n_blocks = getattr(refine_cfg, 'num_blocks', 3)
        self.res_scale = getattr(refine_cfg, 'residual_scale', 0.1)

        # Encoder: RGB(3) + Depth(1) = 4 channels
        self.encoder = nn.Sequential(
            nn.Conv2d(4, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        # Residual blocks
        self.blocks = nn.ModuleList([
            ResidualBlock2d(ch) for _ in range(n_blocks)
        ])

        # Output head: depth residual (1 channel)
        self.head = nn.Sequential(
            nn.Conv2d(ch, ch // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // 2, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        """初始化权重，让残差初始接近零"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # 输出 head 的最后一层初始化为零，确保初始时 residual ≈ 0
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, depth, image):
        """
        Args:
            depth: [B, 1, H, W] DA3 原始深度
            image: [B, 3, H, W] RGB 图像 (归一化到 [0, 1])

        Returns:
            refined_depth: [B, 1, H, W] 细化后的深度
        """
        x = torch.cat([image, depth], dim=1)  # [B, 4, H, W]
        x = self.encoder(x)
        for blk in self.blocks:
            x = blk(x)
        residual = self.head(x) * self.res_scale  # 缩放残差
        return depth + residual
