"""
Transformer + MoE 高斯参数预测网络

替换原有的 UNet 解码器，使用 Transformer 架构提升表达能力：
- Window Self-Attention: 处理局部上下文
- Cross-View Attention: 增强跨视图一致性
- MoE MLP: 不同专家处理不同材质/纹理区域
- Multi-Layer Gaussian Output: 支持动态高斯分配

架构特点：
1. 每个 Transformer 块包含: Window Attention -> (Cross-View Attention) -> MoE MLP
2. 输出多层高斯参数: [B, 14, L, H, W] (L=num_gaussian_layers)
3. MoE 路由器决定每个像素使用哪些专家

参考文献：
- Swin Transformer (窗口注意力)
- Switch Transformer (MoE)
- SHARP (多层高斯)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, List, Dict, Optional
import logging
import math

logger = logging.getLogger(__name__)

class GaussianTransformerMoE(nn.Module):
    """
    Transformer + MoE 高斯参数预测网络
    
    Args:
        cfg: 配置对象
    """
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 从配置读取参数
        gs_cfg = getattr(cfg, 'gs_transformer', None)
        moe_cfg = getattr(cfg, 'moe', None)
        
        if gs_cfg is not None:
            self.hidden_dim = getattr(gs_cfg, 'hidden_dim', 512)
            self.num_layers = getattr(gs_cfg, 'num_layers', 6)
            self.num_heads = getattr(gs_cfg, 'num_heads', 8)
            self.window_size = getattr(gs_cfg, 'window_size', 8)
            self.num_gaussian_layers = getattr(gs_cfg, 'num_gaussian_layers', 3)
            self.max_scale = getattr(gs_cfg, 'max_scale', 0.002)
            self.max_depth_offset = getattr(gs_cfg, 'max_depth_offset', 0.5)
            self.use_gradient_checkpoint = getattr(gs_cfg, 'use_gradient_checkpoint', True)
        else:
            self.hidden_dim = 512
            self.num_layers = 6
            self.num_heads = 8
            self.window_size = 8
            self.num_gaussian_layers = 3
            self.max_scale = 0.002
            self.max_depth_offset = 0.5
            self.use_gradient_checkpoint = True
        
        if moe_cfg is not None:
            self.num_experts = getattr(moe_cfg, 'num_experts', 8)
            self.top_k = getattr(moe_cfg, 'top_k', 2)
            self.load_balance_weight = getattr(moe_cfg, 'load_balance_weight', 0.01)
        else:
            self.num_experts = 8
            self.top_k = 2
            self.load_balance_weight = 0.01
        
        # DA3 特征维度
        self.da3_feat_dim = 1024
        
        # 输入投影
        # 输入: DA3 特征 + 深度
        self.input_proj = nn.Sequential(
            nn.Conv2d(self.da3_feat_dim + 1, self.hidden_dim, 3, padding=1),
            nn.GroupNorm(8, self.hidden_dim),
            nn.GELU(),
        )
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, self.hidden_dim, 64, 64))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # Transformer 块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockMoE(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                num_experts=self.num_experts,
                top_k=self.top_k,
                use_cross_view=(i % 2 == 1),  # 交替使用跨视图注意力
                dropout=0.1,
                load_balance_coef=self.load_balance_weight,
            )
            for i in range(self.num_layers)
        ])
        
        # 输出头 - 多层高斯参数
        L = self.num_gaussian_layers
        self.output_heads = nn.ModuleDict({
            'rotation': nn.Conv2d(self.hidden_dim, 4 * L, 1),      # 四元数
            'scale': nn.Conv2d(self.hidden_dim, 3 * L, 1),         # 缩放
            'opacity': nn.Conv2d(self.hidden_dim, 1 * L, 1),       # 不透明度
            'depth_offset': nn.Conv2d(self.hidden_dim, 1 * L, 1),  # 深度偏移
            'texture_complexity': nn.Conv2d(self.hidden_dim, 1, 1), # 纹理复杂度
        })
        
        self._init_output_heads()
        
        logger.info(f"[GaussianTransformerMoE] 初始化完成: "
                   f"hidden_dim={self.hidden_dim}, num_layers={self.num_layers}, "
                   f"num_experts={self.num_experts}, num_gaussian_layers={self.num_gaussian_layers}")
    
    def _init_output_heads(self):
        """初始化输出头，使用较小的初始值"""
        for name, head in self.output_heads.items():
            if isinstance(head, nn.Conv2d):
                nn.init.zeros_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)

    def _transformer_block_forward(self, block, x, x_other=None):
        """可被checkpoint包装的transformer block前向"""
        if getattr(block, 'use_cross_view', False) and x_other is not None:
            out, router_weights, lb_loss = block(x, x_other)
        else:
            out, router_weights, lb_loss = block(x)
        return out, lb_loss if lb_loss is not None else torch.tensor(0.0, device=x.device)
    
    def forward(
        self, 
        features: torch.Tensor, 
        depth: torch.Tensor, 
        features_other: Optional[torch.Tensor] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[torch.Tensor]]:
        """
        前向传播
        
        Args:
            features: [B, C, H, W] - DA3 图像特征
            depth: [B, 1, H, W] - 融合深度图
            features_other: [B, C, H, W] - 另一视图的特征 (用于跨视图注意力)
            
        Returns:
            gaussian_params: dict - 高斯参数
            router_weights: list - MoE 路由权重 (用于分析和loss)
        """
        B, _, H, W = features.shape
        L = self.num_gaussian_layers
        
        # 上采样特征到深度分辨率（如果需要）
        if features.shape[-2:] != depth.shape[-2:]:
            features = F.interpolate(features, size=depth.shape[-2:], mode='bilinear', align_corners=False)
            if features_other is not None:
                features_other = F.interpolate(features_other, size=depth.shape[-2:], mode='bilinear', align_corners=False)
        
        H, W = depth.shape[-2:]
        
        # 输入融合
        x = torch.cat([features, depth], dim=1)  # [B, C+1, H, W]
        x = self.input_proj(x)  # [B, hidden_dim, H, W]
        
        # 添加位置编码
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear', align_corners=False)
        x = x + pos
        
        # 准备跨视图特征
        x_other = None
        if features_other is not None:
            x_other = torch.cat([features_other, depth], dim=1)
            x_other = self.input_proj(x_other) + pos
        
        # Transformer 处理
        all_router_weights = []
        all_load_balance_loss = []
        
        for block in self.transformer_blocks:
            if self.use_gradient_checkpoint and self.training:
                x, lb_loss = checkpoint(
                    self._transformer_block_forward, block, x, x_other,
                    use_reentrant=False
                )
                router_weights = None
            else:
                if block.use_cross_view and x_other is not None:
                    x, router_weights, lb_loss = block(x, x_other)
                else:
                    x, router_weights, lb_loss = block(x)
            
            if router_weights is not None:
                all_router_weights.append(router_weights)
            if lb_loss is not None:
                all_load_balance_loss.append(lb_loss)
        
        # 输出头
        rotation = self.output_heads['rotation'](x)  # [B, 4*L, H, W]
        scale = self.output_heads['scale'](x)        # [B, 3*L, H, W]
        opacity = self.output_heads['opacity'](x)    # [B, L, H, W]
        depth_offset = self.output_heads['depth_offset'](x)  # [B, L, H, W]
        texture_complexity = self.output_heads['texture_complexity'](x)  # [B, 1, H, W]
        
        # 重塑为多层格式: [B, C, L, H, W]
        rotation = rotation.view(B, 4, L, H, W)
        scale = scale.view(B, 3, L, H, W)
        opacity = opacity.view(B, 1, L, H, W)
        depth_offset = depth_offset.view(B, 1, L, H, W)
        
        # 应用激活函数
        rotation = F.normalize(rotation, dim=1)  # 归一化四元数
        scale = F.softplus(scale)  # 确保正值
        scale = torch.clamp(scale, max=self.max_scale)  # 限制最大值
        opacity = torch.sigmoid(opacity)  # [0, 1]
        depth_offset = torch.tanh(depth_offset) * self.max_depth_offset  # [-max, +max]
        texture_complexity = torch.sigmoid(texture_complexity)  # [0, 1]
        
        gaussian_params = {
            'rotation': rotation,
            'scale': scale,
            'opacity': opacity,
            'depth_offset': depth_offset,
            'texture_complexity': texture_complexity,
        }
        
        # 计算总的 load balance loss
        if all_load_balance_loss:
            gaussian_params['load_balance_loss'] = torch.stack(all_load_balance_loss).mean()
        else:
            gaussian_params['load_balance_loss'] = torch.tensor(0.0, device=x.device)
        
        return gaussian_params, all_router_weights


class TransformerBlockMoE(nn.Module):
    """
    带 MoE 的 Transformer 块
    
    结构: Window Attention -> (Cross-View Attention) -> MoE MLP
    """
    
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        window_size: int = 8, 
        num_experts: int = 8, 
        top_k: int = 2, 
        use_cross_view: bool = False,
        dropout: float = 0.0,
        load_balance_coef: float = 0.01
    ):
        super().__init__()
        self.use_cross_view = use_cross_view
        self.window_size = window_size
        self.dim = dim
        
        # Window Self-Attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = WindowAttention(dim, num_heads, window_size, dropout=dropout)
        
        # Cross-View Attention (可选)
        if use_cross_view:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        
        # MoE MLP
        self.norm2 = nn.LayerNorm(dim)
        self.moe_mlp = MoEMLP(
            dim=dim,
            hidden_dim=dim * 4,
            num_experts=num_experts,
            top_k=top_k,
            load_balance_coef=load_balance_coef
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        x_other: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: [B, C, H, W] - 输入特征
            x_other: [B, C, H, W] - 另一视图特征 (可选)
            
        Returns:
            x: [B, C, H, W] - 输出特征
            router_weights: [B, H*W, num_experts] - MoE 路由权重
            load_balance_loss: scalar - 负载均衡损失
        """
        B, C, H, W = x.shape
        
        # Window Self-Attention
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        x_flat = x_flat + self.dropout(self.self_attn(self.norm1(x_flat), H, W))
        
        # Cross-View Attention
        if self.use_cross_view and x_other is not None:
            x_other_flat = x_other.flatten(2).permute(0, 2, 1)  # [B, HW, C]
            x_cross, _ = self.cross_attn(
                self.norm_cross(x_flat),
                x_other_flat,
                x_other_flat
            )
            x_flat = x_flat + self.dropout(x_cross)
        
        # MoE MLP
        x_flat, router_weights, load_balance_loss = self.moe_mlp(self.norm2(x_flat))
        
        # 恢复空间维度
        x = x_flat.permute(0, 2, 1).view(B, C, H, W)
        
        return x, router_weights, load_balance_loss


class WindowAttention(nn.Module):
    """
    窗口注意力模块
    
    使用 Swin Transformer 风格的窗口注意力减少计算复杂度
    """
    
    def __init__(self, dim: int, num_heads: int = 8, window_size: int = 8, dropout: float = 0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # 计算相对位置索引
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), 
            torch.arange(window_size),
            indexing='ij'
        ))
        coords_flatten = coords.flatten(1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: [B, N, C] - 输入特征 (N = H * W)
            H, W: 空间尺寸
            
        Returns:
            out: [B, N, C] - 输出特征
        """
        B, N, C = x.shape
        ws = self.window_size
        
        # 重塑为空间格式
        x = x.view(B, H, W, C)
        
        # Padding 到窗口大小的倍数
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        
        # 划分窗口: [B, Hp, Wp, C] -> [B*num_windows, ws, ws, C]
        x = x.view(B, Hp // ws, ws, Wp // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, ws * ws, C)  # [B*num_windows, ws*ws, C]
        
        num_windows = x.shape[0]
        
        # 注意力计算
        qkv = self.qkv(x).reshape(num_windows, ws * ws, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, num_windows, heads, ws*ws, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [num_windows, heads, ws*ws, ws*ws]
        
        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(ws * ws, ws * ws, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(num_windows, ws * ws, C)
        x = self.proj(x)
        
        # 还原窗口: [B*num_windows, ws*ws, C] -> [B, Hp, Wp, C]
        x = x.view(B, Hp // ws, Wp // ws, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, Hp, Wp, C)
        
        # 移除 padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :H, :W, :].contiguous()
        
        # 展平: [B, H, W, C] -> [B, N, C]
        x = x.view(B, N, C)
        
        return x


class MoEMLP(nn.Module):
    """
    Mixture of Experts MLP
    
    不同专家负责处理不同类型的区域：
    - 专家1: 平滑区域 (皮肤、背景)
    - 专家2: 高频纹理 (毛发、布料)
    - 专家3: 边缘区域
    - 专家4: 反射表面
    ...
    
    Args:
        dim: 输入/输出维度
        hidden_dim: MLP 隐藏层维度
        num_experts: 专家数量
        top_k: 每个 token 激活的专家数量
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        num_experts: int = 8,
        top_k: int = 2,
        load_balance_coef: float = 0.01
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由器
        self.router = nn.Linear(dim, num_experts, bias=False)
        
        # 专家网络 - 使用单独的线性层
        self.expert_w1 = nn.Parameter(torch.empty(num_experts, dim, hidden_dim))
        self.expert_w2 = nn.Parameter(torch.empty(num_experts, hidden_dim, dim))
        self.expert_b1 = nn.Parameter(torch.zeros(num_experts, hidden_dim))
        self.expert_b2 = nn.Parameter(torch.zeros(num_experts, dim))
        
        # 初始化
        nn.init.kaiming_uniform_(self.expert_w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.expert_w2, a=math.sqrt(5))
        
        # 负载均衡损失系数
        self.load_balance_coef = load_balance_coef
        
    def forward(
        self, 
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: [B, N, C] - 输入特征
            
        Returns:
            output: [B, N, C] - 输出特征
            router_weights: [B, N, num_experts] - 路由权重
            load_balance_loss: scalar - 负载均衡损失
        """
        B, N, C = x.shape
        
        # 计算路由分数
        router_logits = self.router(x)  # [B, N, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)  # [B, N, num_experts]
        
        # 选择 top-k 专家
        top_k_weights, top_k_indices = router_weights.topk(self.top_k, dim=-1)  # [B, N, top_k]
        top_k_weights = top_k_weights / (top_k_weights.sum(dim=-1, keepdim=True) + 1e-8)  # 归一化
        
        # 计算负载均衡损失
        load_balance_loss = self._compute_load_balance_loss(router_weights, top_k_indices)
        
        # 专家计算
        # 为了效率，我们使用批量矩阵乘法
        output = torch.zeros_like(x)
        
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # [B, N]
            expert_weight = top_k_weights[:, :, k:k+1]  # [B, N, 1]
            
            # 收集每个专家的输入
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # [B, N]
                if not mask.any():
                    continue
                
                # 获取该专家的输入
                expert_input = x[mask]  # [num_selected, C]
                
                # MLP: GELU(xW1 + b1)W2 + b2
                h = F.gelu(expert_input @ self.expert_w1[e] + self.expert_b1[e])
                expert_output = h @ self.expert_w2[e] + self.expert_b2[e]
                
                # 加权累加到输出
                weight = expert_weight[mask]  # [num_selected, 1]
                output[mask] = output[mask] + weight * expert_output
        
        # 残差连接
        output = x + output
        
        return output, router_weights, load_balance_loss
    
    def _compute_load_balance_loss(
        self, 
        router_weights: torch.Tensor, 
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        计算负载均衡损失
        
        目标: 让每个专家被均匀使用
        
        Args:
            router_weights: [B, N, num_experts] - 路由权重
            top_k_indices: [B, N, top_k] - top-k 专家索引
            
        Returns:
            loss: scalar - 负载均衡损失
        """
        B, N, E = router_weights.shape
        
        # 计算每个专家的平均路由概率
        mean_routing_prob = router_weights.mean(dim=(0, 1))  # [num_experts]
        
        # 计算每个专家被选中的比例
        expert_counts = torch.zeros(E, device=router_weights.device)
        for e in range(E):
            expert_counts[e] = (top_k_indices == e).float().sum()
        expert_fraction = expert_counts / (B * N * self.top_k + 1e-8)
        
        # 负载均衡损失: 路由概率 * 选中比例 的变异系数
        # 理想情况下两者都应该接近 1/num_experts
        loss = E * (mean_routing_prob * expert_fraction).sum()
        
        return loss * self.load_balance_coef


class GaussianTransformerMoESimple(nn.Module):
    """
    简化版的 Transformer + MoE 高斯预测网络
    
    与原 GSRegresser 接口兼容，用于渐进式替换
    
    Args:
        cfg: 配置对象
        rgb_dim: RGB 输入通道数
        depth_dim: 深度输入通道数
    """
    
    def __init__(self, cfg, rgb_dim: int = 3, depth_dim: int = 1):
        super().__init__()
        self.cfg = cfg
        
        # 初始化 MoE 损失存储
        self.last_moe_balance_loss = None
        
        # 读取配置
        gs_cfg = getattr(cfg, 'gs_transformer', None)
        moe_cfg = getattr(cfg, 'moe', None)
        
        if gs_cfg is not None:
            self.hidden_dim = getattr(gs_cfg, 'hidden_dim', 256)
            self.num_layers = getattr(gs_cfg, 'num_layers', 4)
            self.num_heads = getattr(gs_cfg, 'num_heads', 8)
            self.num_gaussian_layers = getattr(gs_cfg, 'num_gaussian_layers', 3)
            self.max_scale = getattr(gs_cfg, 'max_scale', 0.002)
            self.use_gradient_checkpoint = getattr(gs_cfg, 'use_gradient_checkpoint', True)
        else:
            self.hidden_dim = 256
            self.num_layers = 4
            self.num_heads = 8
            self.num_gaussian_layers = 3
            self.max_scale = 0.002
            self.use_gradient_checkpoint = True
        
        if moe_cfg is not None:
            self.num_experts = getattr(moe_cfg, 'num_experts', 8)
            self.top_k = getattr(moe_cfg, 'top_k', 2)
            self.load_balance_weight = getattr(moe_cfg, 'load_balance_weight', 0.01)
        else:
            self.num_experts = 8
            self.top_k = 2
            self.load_balance_weight = 0.01
        
        # 获取 RAFT encoder 维度
        encoder_dims = cfg.raft.encoder_dims  # [32, 48, 96]
        
        # Transformer 下采样率 (降低分辨率以节省显存)
        self.downsample_factor = getattr(gs_cfg, 'downsample_factor', 4) if gs_cfg else 4
        
        # 多尺度特征融合 + 下采样
        self.feat_fusion = nn.Sequential(
            nn.Conv2d(sum(encoder_dims) + depth_dim + rgb_dim, self.hidden_dim, 3, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # 下采样层 (关键：降低 Transformer 输入分辨率)
        if self.downsample_factor > 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, stride=2, padding=1) if self.downsample_factor >= 4 else nn.Identity(),
                nn.BatchNorm2d(self.hidden_dim) if self.downsample_factor >= 4 else nn.Identity(),
                nn.ReLU(inplace=True) if self.downsample_factor >= 4 else nn.Identity(),
            )
        else:
            self.downsample = nn.Identity()
        
        # 轻量级 Transformer 块
        self.transformer_blocks = nn.ModuleList([
            TransformerBlockMoE(
                dim=self.hidden_dim,
                num_heads=self.num_heads,
                window_size=8,
                num_experts=self.num_experts,
                top_k=self.top_k,
                use_cross_view=False,  # 简化版不使用跨视图
                dropout=0.1,
                load_balance_coef=self.load_balance_weight,
            )
            for _ in range(self.num_layers)
        ])
        
        # 输出头 - 多层高斯参数
        # L = num_gaussian_layers，从配置读取
        L = self.num_gaussian_layers
        self.head_dim = cfg.gsnet.parm_head_dim
        
        logger.info(f"[GaussianTransformerMoESimple] 初始化多层输出头: "
                   f"num_gaussian_layers={L}, head_dim={self.head_dim}")
        
        # 旋转: [B, 4*L, H, W] -> [B, 4, L, H, W]
        self.rot_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4 * L, kernel_size=1),
        )
        
        # 缩放: [B, 3*L, H, W] -> [B, 3, L, H, W]
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3 * L, kernel_size=1),
        )
        
        # 不透明度: [B, L, H, W] -> [B, 1, L, H, W]
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, L, kernel_size=1),
        )
        
        # 深度偏移: [B, L, H, W] -> [B, 1, L, H, W]
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, L, kernel_size=1),
        )
        
        # 纹理复杂度: [B, 1, H, W] - 用于动态分配
        self.texture_head = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _transformer_block_forward(self, block, x, x_other=None):
        """可被checkpoint包装的transformer block前向"""
        if getattr(block, 'use_cross_view', False) and x_other is not None:
            out, router_weights, lb_loss = block(x, x_other)
        else:
            out, router_weights, lb_loss = block(x)
        return out, lb_loss if lb_loss is not None else torch.tensor(0.0, device=x.device)
    
    def forward(
        self, 
        img: torch.Tensor, 
        depth: torch.Tensor, 
        img_feat: Tuple[torch.Tensor, ...]
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播 - 输出多层高斯参数
        
        Args:
            img: [B, 3, H, W] - 输入图像
            depth: [B, 1, H, W] - 深度图
            img_feat: tuple of [B, C, H', W'] - 多尺度图像特征
            
        Returns:
            dict containing:
                - rotation: [B, 4, L, H, W] - 旋转四元数
                - scale: [B, 3, L, H, W] - 缩放
                - opacity: [B, 1, L, H, W] - 不透明度
                - depth_offset: [B, 1, L, H, W] - 深度偏移
                - texture_complexity: [B, 1, H, W] - 纹理复杂度
                - load_balance_loss: scalar - MoE 负载均衡损失
        """
        B, _, H, W = img.shape
        L = self.num_gaussian_layers
        
        # 上采样所有特征到输入分辨率
        feats = []
        for feat in img_feat:
            if feat.shape[-2:] != (H, W):
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            feats.append(feat)
        
        # 融合所有特征
        x = torch.cat(feats + [depth, img], dim=1)  # [B, sum(C)+1+3, H, W]
        x = self.feat_fusion(x)  # [B, hidden_dim, H, W]
        
        # 下采样 (关键：降低 Transformer 输入分辨率以节省显存)
        x = self.downsample(x)  # [B, hidden_dim, H/ds, W/ds]
        
        # Transformer 处理 (使用梯度检查点)
        total_lb_loss = 0.0
        for block in self.transformer_blocks:
            if self.use_gradient_checkpoint and self.training:
                x, lb_loss = checkpoint(
                    self._transformer_block_forward, block, x,
                    use_reentrant=False
                )
                total_lb_loss = total_lb_loss + lb_loss
            else:
                x, _, lb_loss = block(x)
                if lb_loss is not None:
                    total_lb_loss = total_lb_loss + lb_loss
        
        # 存储 MoE 负载均衡损失供外部访问
        self.last_moe_balance_loss = total_lb_loss
        
        # 上采样回原始分辨率
        if self.downsample_factor > 1:
            x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False)
        
        # 输出头 - 多层高斯参数
        rot_out = self.rot_head(x)  # [B, 4*L, H, W]
        scale_out = self.scale_head(x)  # [B, 3*L, H, W]
        opacity_out = self.opacity_head(x)  # [B, L, H, W]
        depth_out = self.depth_head(x)  # [B, L, H, W]
        texture_out = self.texture_head(x)  # [B, 1, H, W]
        
        # 重塑为多层格式: [B, C, L, H, W]
        rotation = rot_out.view(B, 4, L, H, W)
        scale = scale_out.view(B, 3, L, H, W)
        opacity = opacity_out.view(B, L, H, W).unsqueeze(1)  # [B, 1, L, H, W]
        depth_offset = depth_out.view(B, L, H, W).unsqueeze(1)  # [B, 1, L, H, W]
        
        # 应用激活函数
        # 四元数归一化 - 对每层分别处理
        rotation = F.normalize(rotation, dim=1)
        
        # 缩放 - 确保正值，从配置读取最大值
        scale = F.softplus(scale)
        scale = torch.clamp(scale, max=self.max_scale)
        
        # 不透明度 - [0, 1]
        opacity = torch.sigmoid(opacity)
        
        # 深度偏移 - 从配置读取范围
        max_depth_offset = getattr(
            getattr(self.cfg, 'gs_transformer', None), 
            'max_depth_offset', 0.5
        )
        depth_offset = torch.tanh(depth_offset) * max_depth_offset
        
        return {
            'rotation': rotation,           # [B, 4, L, H, W]
            'scale': scale,                 # [B, 3, L, H, W]
            'opacity': opacity,             # [B, 1, L, H, W]
            'depth_offset': depth_offset,   # [B, 1, L, H, W]
            'texture_complexity': texture_out,  # [B, 1, H, W]
            'load_balance_loss': total_lb_loss,
        }


def create_gaussian_transformer_moe(cfg) -> GaussianTransformerMoE:
    """
    创建 Transformer+MoE 高斯预测网络
    
    Args:
        cfg: 配置对象
        
    Returns:
        GaussianTransformerMoE 实例
    """
    return GaussianTransformerMoE(cfg)


def create_gaussian_transformer_moe_simple(cfg, rgb_dim: int = 3, depth_dim: int = 1) -> GaussianTransformerMoESimple:
    """
    创建简化版 Transformer+MoE 高斯预测网络
    
    Args:
        cfg: 配置对象
        rgb_dim: RGB 通道数
        depth_dim: 深度通道数
        
    Returns:
        GaussianTransformerMoESimple 实例
    """
    return GaussianTransformerMoESimple(cfg, rgb_dim, depth_dim)
