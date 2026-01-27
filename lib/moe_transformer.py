"""
MoE Transformer 高斯参数预测模块

基于 Transformer 架构的 Mixture of Experts 高斯参数预测器。
输入 DINO 特征和深度图，输出高斯参数（rotation, scale, opacity, depth_residual）。

架构设计:
- 共享专家 (Shared Expert): 所有输入都会经过
- 路由专家 (Routed Experts): 背景专家、人体专家等
- 动态路由: 基于输入特征学习路由权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TransformerExpert(nn.Module):
    """
    单个 Transformer 专家模块
    
    使用 Transformer Encoder 处理 patch 级特征，预测高斯参数。
    
    Args:
        dim_in: 输入特征维度 (DINO dim + depth embed dim)
        dim_hidden: Transformer 隐藏层维度
        num_heads: 注意力头数
        num_layers: Transformer 层数
        dropout: Dropout 率
    """
    
    def __init__(
        self, 
        dim_in: int = 1088,  # DINO 1024 + Depth 64
        dim_hidden: int = 512, 
        num_heads: int = 8, 
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        
        # 输入投影
        self.input_proj = nn.Linear(dim_in, dim_hidden)
        
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden,
            nhead=num_heads,
            dim_feedforward=dim_hidden * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for better stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 输出头 - 预测高斯参数
        # rotation: 四元数 [4]
        self.rot_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.GELU(),
            nn.Linear(dim_hidden // 2, 4)
        )
        
        # scale: xyz 尺度 [3]
        self.scale_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.GELU(),
            nn.Linear(dim_hidden // 2, 3),
            nn.Softplus(beta=1)
        )
        
        # opacity: 不透明度 [1]
        self.opacity_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.GELU(),
            nn.Linear(dim_hidden // 2, 1),
            nn.Sigmoid()
        )
        
        # depth_residual: 深度残差 [1]
        self.depth_res_head = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden // 2),
            nn.GELU(),
            nn.Linear(dim_hidden // 2, 1),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            features: [B, N_patches, dim_in] 输入特征
            
        Returns:
            dict: 包含 rotation, scale, opacity, depth_residual
        """
        # 输入投影
        x = self.input_proj(features)  # [B, N, dim_hidden]
        
        # Transformer 编码
        x = self.transformer(x)  # [B, N, dim_hidden]
        
        # 预测各参数
        rotation = self.rot_head(x)
        rotation = F.normalize(rotation, dim=-1)  # 归一化四元数
        
        scale = self.scale_head(x)
        scale = torch.clamp(scale, max=0.002)  # 限制最大尺度
        
        opacity = self.opacity_head(x)
        
        depth_residual = self.depth_res_head(x) * 0.5  # 缩放深度残差
        
        return {
            'rotation': rotation,      # [B, N, 4]
            'scale': scale,            # [B, N, 3]
            'opacity': opacity,        # [B, N, 1]
            'depth_residual': depth_residual  # [B, N, 1]
        }


class DepthEmbedding(nn.Module):
    """
    深度图嵌入模块
    
    将深度图转换为与 DINO patch 对齐的特征表示。
    
    Args:
        out_dim: 输出嵌入维度
        patch_size: patch 大小，应与 DINO 一致 (14)
    """
    
    def __init__(self, out_dim: int = 64, patch_size: int = 14):
        super().__init__()
        
        self.patch_size = patch_size
        self.out_dim = out_dim
        
        # 深度 patch 嵌入
        self.depth_embed = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.Conv2d(32, out_dim, kernel_size=1),
        )
    
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            depth: [B, 1, H, W] 深度图
            
        Returns:
            depth_embed: [B, N_patches, out_dim] patch 级深度嵌入
        """
        # 深度 patch 化
        x = self.depth_embed(depth)  # [B, out_dim, H/patch_size, W/patch_size]
        B, C, H, W = x.shape
        
        # 转换为序列格式
        x = x.flatten(2).permute(0, 2, 1)  # [B, N_patches, out_dim]
        
        return x


class MoERouter(nn.Module):
    """
    MoE 路由器
    
    基于输入特征学习路由权重，决定每个 patch 应该由哪个专家处理。
    
    Args:
        dim_in: 输入特征维度
        num_experts: 路由专家数量（不包括共享专家）
        hidden_dim: 隐藏层维度
    """
    
    def __init__(
        self, 
        dim_in: int, 
        num_experts: int = 2, 
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.num_experts = num_experts
        
        # 路由网络
        self.router = nn.Sequential(
            nn.Linear(dim_in, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts)
        )
        
        # 可学习的温度参数
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            features: [B, N_patches, dim_in] 输入特征
            
        Returns:
            router_weights: [B, N_patches, num_experts] 路由权重
        """
        logits = self.router(features)  # [B, N, num_experts]
        
        # 使用温度缩放的 softmax
        temp = torch.clamp(self.temperature, min=0.1)
        weights = F.softmax(logits / temp, dim=-1)
        
        return weights


class MoETransformerRegresser(nn.Module):
    """
    MoE Transformer 高斯参数回归器
    
    输入 DINO 特征和深度图，通过 MoE 架构预测高斯参数。
    
    Args:
        cfg: 配置对象
        dino_dim: DINO 特征维度 (默认 1024 for DA3-LARGE)
        depth_embed_dim: 深度嵌入维度
        num_experts: 路由专家数量
        use_shared_expert: 是否使用共享专家
        shared_expert_weight: 共享专家权重
    """
    
    def __init__(
        self,
        cfg,
        dino_dim: int = 1024,
        depth_embed_dim: int = 64,
        num_experts: int = 2,
        use_shared_expert: bool = True,
        shared_expert_weight: float = 0.3
    ):
        super().__init__()
        
        self.cfg = cfg
        self.dino_dim = dino_dim
        self.depth_embed_dim = depth_embed_dim
        self.num_experts = num_experts
        self.use_shared_expert = use_shared_expert
        self.shared_expert_weight = shared_expert_weight
        
        # 从配置中获取参数
        moe_cfg = getattr(cfg, 'moe', None)
        if moe_cfg is not None:
            transformer_cfg = getattr(moe_cfg, 'transformer', None)
            if transformer_cfg is not None:
                self.dim_hidden = getattr(transformer_cfg, 'dim_hidden', 512)
                self.num_heads = getattr(transformer_cfg, 'num_heads', 8)
                self.num_layers = getattr(transformer_cfg, 'num_layers', 4)
            else:
                self.dim_hidden = 512
                self.num_heads = 8
                self.num_layers = 4
            self.shared_expert_weight = getattr(moe_cfg, 'shared_expert_weight', shared_expert_weight)
        else:
            self.dim_hidden = 512
            self.num_heads = 8
            self.num_layers = 4
        
        # 深度嵌入模块
        self.depth_embed = DepthEmbedding(out_dim=depth_embed_dim, patch_size=14)
        
        # 特征融合维度
        fused_dim = dino_dim + depth_embed_dim  # 1024 + 64 = 1088
        
        # MoE 路由器
        self.router = MoERouter(
            dim_in=fused_dim,
            num_experts=num_experts,
            hidden_dim=256
        )
        
        # 共享专家
        if use_shared_expert:
            self.shared_expert = TransformerExpert(
                dim_in=fused_dim,
                dim_hidden=self.dim_hidden,
                num_heads=self.num_heads,
                num_layers=self.num_layers
            )
        
        # 路由专家列表
        self.routed_experts = nn.ModuleList([
            TransformerExpert(
                dim_in=fused_dim,
                dim_hidden=self.dim_hidden,
                num_heads=self.num_heads,
                num_layers=self.num_layers
            ) for _ in range(num_experts)
        ])
        
        # 专家名称（用于日志和可视化）
        self.expert_names = ['bg', 'human'] if num_experts == 2 else [f'expert_{i}' for i in range(num_experts)]
    
    def forward(
        self, 
        dino_features: torch.Tensor, 
        depth: torch.Tensor,
        bg_update_signal: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            dino_features: [B, N_patches, 1024] 来自 DA3 的 DINO 特征
            depth: [B, 1, H, W] 来自 DA3 的深度图
            bg_update_signal: 背景更新信号（训练/推理策略用）
            
        Returns:
            dict: 包含:
                - fused_params: 融合后的高斯参数
                - expert_params: 各专家的参数列表
                - router_weights: 路由权重
                - bg_update_signal: 传递的背景更新信号
        """
        B = dino_features.shape[0]
        N_dino = dino_features.shape[1]
        
        # 深度嵌入
        depth_embed = self.depth_embed(depth)  # [B, N_depth, depth_embed_dim]
        N_depth = depth_embed.shape[1]
        
        # 处理 patch 数量不匹配的情况
        if N_depth != N_dino:
            # 计算 DINO 的空间尺寸（假设接近正方形）
            H_dino = int(N_dino ** 0.5)
            W_dino = N_dino // H_dino
            if H_dino * W_dino != N_dino:
                # 尝试其他分解
                for h in range(int(N_dino ** 0.5), 0, -1):
                    if N_dino % h == 0:
                        H_dino = h
                        W_dino = N_dino // h
                        break
            
            # 计算深度嵌入的空间尺寸
            H_depth = int(N_depth ** 0.5)
            W_depth = N_depth // H_depth
            if H_depth * W_depth != N_depth:
                for h in range(int(N_depth ** 0.5), 0, -1):
                    if N_depth % h == 0:
                        H_depth = h
                        W_depth = N_depth // h
                        break
            
            # 重塑深度嵌入为空间格式并插值
            depth_embed_spatial = depth_embed.view(B, H_depth, W_depth, -1)
            depth_embed_spatial = depth_embed_spatial.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            
            # 插值到 DINO 尺寸
            depth_embed_spatial = F.interpolate(
                depth_embed_spatial,
                size=(H_dino, W_dino),
                mode='bilinear',
                align_corners=False
            )
            
            # 转回序列格式
            depth_embed = depth_embed_spatial.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
            depth_embed = depth_embed.view(B, H_dino * W_dino, -1)  # [B, N_dino, C]
        
        # 特征融合
        fused = torch.cat([dino_features, depth_embed], dim=-1)  # [B, N, fused_dim]
        
        # 路由
        router_weights = self.router(fused)  # [B, N, num_experts]
        
        # 各专家预测
        expert_params_list = []
        for expert in self.routed_experts:
            params = expert(fused)
            expert_params_list.append(params)
        
        # 共享专家预测
        shared_params = None
        if self.use_shared_expert:
            shared_params = self.shared_expert(fused)
        
        # 加权融合参数
        fused_params = self._fuse_expert_params(
            expert_params_list, 
            router_weights, 
            shared_params
        )
        
        return {
            'fused_params': fused_params,
            'expert_params': expert_params_list,
            'shared_params': shared_params,
            'router_weights': router_weights,
            'bg_update_signal': bg_update_signal
        }
    
    def _fuse_expert_params(
        self,
        expert_params_list: list,
        router_weights: torch.Tensor,
        shared_params: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        融合专家参数
        
        Args:
            expert_params_list: 各路由专家的参数列表
            router_weights: [B, N, num_experts] 路由权重
            shared_params: 共享专家参数（可选）
            
        Returns:
            fused_params: 融合后的参数
        """
        param_keys = ['rotation', 'scale', 'opacity', 'depth_residual']
        fused_params = {}
        
        for key in param_keys:
            # 路由专家的加权和
            routed_sum = torch.zeros_like(expert_params_list[0][key])
            for i, params in enumerate(expert_params_list):
                # router_weights[:, :, i:i+1] -> [B, N, 1]
                weight = router_weights[:, :, i:i+1]
                # params[key] -> [B, N, D]
                if params[key].dim() == 3:
                    routed_sum = routed_sum + weight * params[key]
                else:
                    routed_sum = routed_sum + weight.squeeze(-1) * params[key]
            
            # 加入共享专家
            if self.use_shared_expert and shared_params is not None:
                fused_params[key] = (
                    self.shared_expert_weight * shared_params[key] +
                    (1 - self.shared_expert_weight) * routed_sum
                )
            else:
                fused_params[key] = routed_sum
        
        return fused_params
    
    def get_expert_params_by_name(self, output: Dict, expert_name: str) -> Optional[Dict[str, torch.Tensor]]:
        """
        根据名称获取专家参数
        
        Args:
            output: forward 输出
            expert_name: 专家名称 ('bg', 'human', 'shared', etc.)
            
        Returns:
            专家参数字典或 None
        """
        if expert_name == 'shared':
            return output.get('shared_params')
        
        try:
            idx = self.expert_names.index(expert_name)
            return output['expert_params'][idx]
        except (ValueError, IndexError):
            return None


def create_moe_transformer_regresser(cfg) -> MoETransformerRegresser:
    """
    创建 MoE Transformer 回归器
    
    Args:
        cfg: 配置对象
        
    Returns:
        MoETransformerRegresser 实例
    """
    moe_cfg = getattr(cfg, 'moe', None)
    
    if moe_cfg is not None:
        num_experts = getattr(moe_cfg, 'num_experts', 2)
        use_shared_expert = getattr(moe_cfg, 'use_shared_expert', True)
        shared_expert_weight = getattr(moe_cfg, 'shared_expert_weight', 0.3)
    else:
        num_experts = 2
        use_shared_expert = True
        shared_expert_weight = 0.3
    
    return MoETransformerRegresser(
        cfg=cfg,
        dino_dim=1024,
        depth_embed_dim=64,
        num_experts=num_experts,
        use_shared_expert=use_shared_expert,
        shared_expert_weight=shared_expert_weight
    )
