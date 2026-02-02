"""
GSTransformer: 基于交替注意力 + DPT Head 的高斯参数预测网络

替代原 GSRegresser (CNN-based)，支持：
- 左右视图交互 (Global Attention)
- 多尺度特征融合 (DPT Head)
- 像素级高斯预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class PatchEmbed(nn.Module):
    """将图像转换为 patch tokens"""
    
    def __init__(self, patch_size=14, in_channels=4, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: [B, C, H, W] -> [B, N, embed_dim]
        x = self.proj(x)  # [B, embed_dim, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
        x = self.norm(x)
        return x


class MemoryEfficientAttention(nn.Module):
    """
    使用 PyTorch SDPA 的高效注意力实现
    
    自动使用 Flash Attention (Ampere+) 或 Memory Efficient Attention
    显存占用从 O(n²) 降低到 O(n)
    """
    
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = attn_drop
    
    def forward(self, x):
        B, N, C = x.shape
        
        # 计算 Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)
        
        # 使用 PyTorch 内置的 scaled_dot_product_attention
        # 自动选择最优实现: Flash Attention, Memory Efficient, 或 Math
        x = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=False,
        )
        
        # [B, num_heads, N, head_dim] -> [B, N, C]
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block with Memory Efficient Attention + Layer Scale
    
    使用 PyTorch SDPA，自动启用 Flash Attention (如果硬件支持)
    """
    
    def __init__(self, dim, num_heads, mlp_ratio=4.0, qkv_bias=True,
                 drop=0., attn_drop=0., init_values=0.01):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MemoryEfficientAttention(dim, num_heads, qkv_bias, attn_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )
        # Layer Scale
        self.gamma1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma2 = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x):
        x_norm = self.norm1(x)
        x = x + self.gamma1 * self.attn(x_norm)
        x = x + self.gamma2 * self.mlp(self.norm2(x))
        return x


class FeatureFusionBlock(nn.Module):
    """DPT 风格的特征融合块"""
    
    def __init__(self, features, has_residual=True):
        super().__init__()
        self.has_residual = has_residual
        
        if has_residual:
            self.residual_conv = nn.Sequential(
                nn.ReLU(inplace=False),
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, kernel_size=3, padding=1),
            )
        
        self.output_conv = nn.Conv2d(features, features, kernel_size=1)
    
    def forward(self, x, residual=None, target_size=None):
        if residual is not None and self.has_residual:
            x = x + self.residual_conv(residual)
        
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
        
        x = self.output_conv(x)
        return x


class GSTransformer(nn.Module):
    """
    基于交替注意力 + DPT Head 的高斯参数预测网络
    
    替代原 GSRegresser，支持左右视图交互和像素级预测
    
    Architecture:
        1. Patch Embedding: RGB+Depth -> tokens
        2. Alternating Attention: Frame(独立) -> Global(交叉) x N
        3. DPT Head: 多尺度特征融合
        4. Output Heads: rot, scale, opacity, depth
    
    显存优化:
        - Flash Attention (PyTorch SDPA): 自动启用，显存 O(n²) -> O(n)
        - Gradient Checkpointing: 可选，用时间换空间，进一步减少 ~50% 显存
    """
    
    def __init__(self, cfg, in_channels=4, embed_dim=768, depth=16,
                 num_heads=12, patch_size=14, mlp_ratio=4.0,
                 use_checkpoint=False):
        super().__init__()
        
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint  # Gradient Checkpointing
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)
        
        # 2. View Token (区分左右视图)
        self.view_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim) * 0.02)
        
        # 3. 交替注意力 Blocks
        self.frame_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.global_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # 4. 中间层索引 (用于 DPT)
        # 保存 4 个等间距层的特征
        self.intermediate_layers = [
            depth // 4 - 1,      # layer 4 (0-indexed: 3)
            depth // 2 - 1,      # layer 8 (0-indexed: 7)
            3 * depth // 4 - 1,  # layer 12 (0-indexed: 11)
            depth - 1            # layer 16 (0-indexed: 15)
        ]
        
        # 5. DPT Head
        dpt_features = 256
        out_channels = [256, 512, 1024, 1024]
        
        # 特征拼接后的归一化 (frame + global concat)
        self.norm = nn.LayerNorm(embed_dim * 2)
        
        # Project layers: 将 transformer 特征投影到不同通道数
        self.projects = nn.ModuleList([
            nn.Conv2d(embed_dim * 2, oc, kernel_size=1)
            for oc in out_channels
        ])
        
        # Resize layers: 调整不同层的空间分辨率
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),  # 4x 上采样
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),  # 2x 上采样
            nn.Identity(),  # 保持
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),  # 0.5x 下采样
        ])
        
        # Layer RN: 统一到 dpt_features 通道
        self.layer_rns = nn.ModuleList([
            nn.Conv2d(out_channels[i], dpt_features, kernel_size=3, padding=1, bias=False)
            for i in range(4)
        ])
        
        # RefineNet fusion blocks
        self.refinenets = nn.ModuleList([
            FeatureFusionBlock(dpt_features, has_residual=(i < 3))
            for i in range(4)
        ])
        
        # 6. Output Heads
        head_features = 32
        self.output_conv = nn.Sequential(
            nn.Conv2d(dpt_features, dpt_features // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dpt_features // 2, head_features, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Rotation head (四元数)
        self.rot_head = nn.Conv2d(head_features, 4, kernel_size=1)
        
        # Scale head
        self.scale_head = nn.Sequential(
            nn.Conv2d(head_features, 3, kernel_size=1),
            nn.Softplus(beta=1),
        )
        
        # Opacity head
        self.opacity_head = nn.Sequential(
            nn.Conv2d(head_features, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        
        # XYZ residual head (3 channels for x, y, z)
        self.xyz_head = nn.Sequential(
            nn.Conv2d(head_features, 3, kernel_size=1),
            nn.Tanh(),
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, img, depth, img_feat=None):
        """
        Args:
            img: [B*2, 3, H, W] (左右视图拼接)
            depth: [B*2, 1, H, W]
            img_feat: 未使用，保持接口兼容
        
        Returns:
            rot_maps: [B*2, 4, H, W]
            scale_maps: [B*2, 3, H, W]
            opacity_maps: [B*2, 1, H, W]
            depth_maps: [B*2, 1, H, W]
        """
        B2, _, H_orig, W_orig = img.shape
        B = B2 // 2
        
        # 计算需要的 padding，使 H, W 成为 patch_size 的倍数
        pad_h = (self.patch_size - H_orig % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W_orig % self.patch_size) % self.patch_size
        
        if pad_h > 0 or pad_w > 0:
            # Padding: (left, right, top, bottom)
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            depth = F.pad(depth, (0, pad_w, 0, pad_h), mode='reflect')
        
        _, _, H, W = img.shape
        
        # 1. 拼接 RGB + Depth
        x = torch.cat([img, depth], dim=1)  # [B*2, 4, H, W]
        
        # 2. Patch Embedding
        tokens = self.patch_embed(x)  # [B*2, P, C]
        _, P, C = tokens.shape
        
        patch_h, patch_w = H // self.patch_size, W // self.patch_size
        
        # 3. 添加 View Token
        tokens = tokens.view(B, 2, P, C)
        view_token = self.view_token.expand(B, -1, P, -1)  # [B, 2, P, C]
        tokens = tokens + view_token
        
        # 4. 交替注意力 (可选 Gradient Checkpointing)
        frame_intermediates = []
        global_intermediates = []
        
        for i in range(self.depth):
            # Frame Attention: 左右视图独立
            tokens_flat = tokens.view(B * 2, P, C)
            if self.use_checkpoint and self.training:
                tokens_flat = checkpoint(self.frame_blocks[i], tokens_flat, use_reentrant=False)
            else:
                tokens_flat = self.frame_blocks[i](tokens_flat)
            tokens = tokens_flat.view(B, 2, P, C)
            
            # Global Attention: 左右视图合并
            tokens_global = tokens.view(B, 2 * P, C)
            if self.use_checkpoint and self.training:
                tokens_global = checkpoint(self.global_blocks[i], tokens_global, use_reentrant=False)
            else:
                tokens_global = self.global_blocks[i](tokens_global)
            tokens = tokens_global.view(B, 2, P, C)
            
            # 保存中间层
            if i in self.intermediate_layers:
                frame_intermediates.append(tokens_flat.clone())
                global_intermediates.append(tokens.view(B * 2, P, C).clone())
        
        # 5. DPT Head 处理
        dpt_features = []
        for idx in range(len(self.intermediate_layers)):
            # 拼接 frame 和 global 特征
            frame_feat = frame_intermediates[idx]  # [B*2, P, C]
            global_feat = global_intermediates[idx]  # [B*2, P, C]
            feat = torch.cat([frame_feat, global_feat], dim=-1)  # [B*2, P, 2C]
            
            feat = self.norm(feat)
            
            # Reshape 到 2D
            feat = feat.permute(0, 2, 1).view(B * 2, C * 2, patch_h, patch_w)
            
            # Project 和 Resize
            feat = self.projects[idx](feat)
            feat = self.resize_layers[idx](feat)
            
            dpt_features.append(feat)
        
        # 6. RefineNet 融合 (自底向上)
        layer_rns = [self.layer_rns[i](dpt_features[i]) for i in range(4)]
        
        # 从最深层开始融合
        out = self.refinenets[3](layer_rns[3], target_size=layer_rns[2].shape[2:])
        out = self.refinenets[2](out, layer_rns[2], target_size=layer_rns[1].shape[2:])
        out = self.refinenets[1](out, layer_rns[1], target_size=layer_rns[0].shape[2:])
        out = self.refinenets[0](out, layer_rns[0])
        
        # 7. 上采样到 padded 分辨率
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
        
        # 8. Output Heads
        out = self.output_conv(out)
        
        # Rotation (归一化四元数)
        rot_out = self.rot_head(out)
        rot_out = F.normalize(rot_out, dim=1)
        
        # Scale (限制最大值)
        scale_out = self.scale_head(out)
        scale_out = torch.clamp_max(scale_out, 0.002)
        
        # Opacity
        opacity_out = self.opacity_head(out)
        
        # XYZ residual (3 channels)
        xyz_out = self.xyz_head(out) * 0.1  # smaller scale for xyz residuals
        
        # 9. 裁剪回原始分辨率 (移除 padding)
        if pad_h > 0 or pad_w > 0:
            rot_out = rot_out[:, :, :H_orig, :W_orig].contiguous()
            scale_out = scale_out[:, :, :H_orig, :W_orig].contiguous()
            opacity_out = opacity_out[:, :, :H_orig, :W_orig].contiguous()
            xyz_out = xyz_out[:, :, :H_orig, :W_orig].contiguous()
        
        return rot_out, scale_out, opacity_out, xyz_out
