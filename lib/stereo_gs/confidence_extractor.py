"""
代价体置信度提取 — 从 FFS 的 softmax 概率分布中提取匹配置信度。

置信度指标:
  - peak:    softmax 最大概率 ∈ [0,1], 越高越可靠
  - entropy: 分布熵 (归一化), 越低越可靠, 转换为 1-entropy 使高=可靠
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceExtractor(nn.Module):
    """
    从 FFS 代价体概率分布中提取逐像素匹配置信度。

    输出 (B, 1, H/4, W/4) 的置信度图, 范围 [0, 1], 高值=高置信。
    """

    def __init__(self, mode: str = 'peak'):
        """
        Args:
            mode: 'peak' | 'entropy' | 'learned'
                - peak:    直接取 softmax 最大概率
                - entropy: 1 - 归一化熵
                - learned: peak + entropy 拼接后用小网络输出单通道
        """
        super().__init__()
        self.mode = mode
        if mode == 'learned':
            self.head = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid(),
            )

    def forward(self, cost_prob: torch.Tensor) -> torch.Tensor:
        """
        Args:
            cost_prob: (B, D, H/4, W/4) softmax 概率分布

        Returns:
            confidence: (B, 1, H/4, W/4)
        """
        peak = cost_prob.max(dim=1, keepdim=True)[0]

        if self.mode == 'peak':
            return peak

        eps = 1e-8
        log_prob = torch.log(cost_prob + eps)
        entropy = -(cost_prob * log_prob).sum(dim=1, keepdim=True)
        max_entropy = torch.log(torch.tensor(cost_prob.shape[1], dtype=torch.float32, device=cost_prob.device))
        norm_entropy = entropy / (max_entropy + eps)
        inv_entropy = 1.0 - norm_entropy

        if self.mode == 'entropy':
            return inv_entropy

        combined = torch.cat([peak, inv_entropy], dim=1)
        return self.head(combined)
