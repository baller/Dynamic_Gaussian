"""
GSTransformer: 基于交替注意力 + DPT Head 的高斯参数预测网络

替代原 GSRegresser (CNN-based)，支持：
- DA3 DINOv2 特征融合 (门控机制)
- 左右视图交互 (Global Attention)
- 可学习 2D 位置编码
- 多尺度特征融合 (DPT Head + DA3 skip connections)
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
    显存占用从 O(n^2) 降低到 O(n)
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
        1. Patch Embedding: RGB+Depth -> local tokens
        2. DA3 Feature Fusion: 门控融合 DINOv2 语义特征
        3. Positional Encoding: 可学习 2D 位置编码
        4. Alternating Attention: Frame(独立) -> Global(交叉) x N
        5. DPT Head: 多尺度特征融合 + DA3 skip connections
        6. Output Heads: rot, scale, opacity, xyz_residual

    显存优化:
        - Flash Attention (PyTorch SDPA): 自动启用，显存 O(n^2) -> O(n)
        - Gradient Checkpointing: 可选，用时间换空间，进一步减少 ~50% 显存
    """

    def __init__(self, cfg, in_channels=4, embed_dim=768, depth=16,
                 num_heads=12, patch_size=14, mlp_ratio=4.0,
                 use_checkpoint=False):
        super().__init__()

        self.cfg = cfg
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint  # Gradient Checkpointing

        # 从配置读取参数约束
        self.xyz_res_scale = getattr(cfg.gsnet, 'xyz_res_scale', 0.1)
        self.scale_max = getattr(cfg.gsnet, 'scale_max', 0.002)

        # DA3 特征融合配置
        self.use_da3_features = getattr(cfg.gsnet, 'use_da3_features', True)
        da3_feat_dim = getattr(cfg.gsnet, 'da3_feat_dim', 1024)

        # 1. Patch Embedding (RGB + Depth -> local tokens)
        self.patch_embed = PatchEmbed(patch_size, in_channels, embed_dim)

        # 2. DA3 特征融合模块
        if self.use_da3_features:
            # 投影器: 将 DINOv2 特征投影到 embed_dim
            self.da3_proj = nn.Sequential(
                nn.Linear(da3_feat_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            # 门控融合: 自适应混合 local patch tokens 和 DA3 语义 tokens
            self.fusion_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Sigmoid(),
            )

        # 3. 可学习 2D 位置编码 (支持可变分辨率插值)
        max_h = 1024 // patch_size  # 73 for patch_size=14
        max_w = 1024 // patch_size
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_h * max_w, embed_dim) * 0.02
        )
        self.pos_embed_h = max_h
        self.pos_embed_w = max_w

        # 4. View Token (区分左右视图)
        self.view_token = nn.Parameter(torch.randn(1, 2, 1, embed_dim) * 0.02)

        # 5. 交替注意力 Blocks
        self.frame_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        self.global_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # 6. 中间层索引 (用于 DPT)
        # 保存 4 个等间距层的特征
        self.intermediate_layers = [
            depth // 4 - 1,      # layer ~25%
            depth // 2 - 1,      # layer ~50%
            3 * depth // 4 - 1,  # layer ~75%
            depth - 1            # last layer
        ]

        # 7. DPT Head
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
            nn.ConvTranspose2d(out_channels[0], out_channels[0], kernel_size=4, stride=4),  # 4x upsample
            nn.ConvTranspose2d(out_channels[1], out_channels[1], kernel_size=2, stride=2),  # 2x upsample
            nn.Identity(),  # keep
            nn.Conv2d(out_channels[3], out_channels[3], kernel_size=3, stride=2, padding=1),  # 0.5x downsample
        ])

        # Layer RN: 统一到 dpt_features 通道
        self.layer_rns = nn.ModuleList([
            nn.Conv2d(out_channels[i], dpt_features, kernel_size=3, padding=1, bias=False)
            for i in range(4)
        ])

        # DA3 skip connection projections (4 layers: da3_feat_dim -> dpt_features)
        if self.use_da3_features:
            self.da3_skip_projs = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(da3_feat_dim, dpt_features, kernel_size=1, bias=False),
                    nn.BatchNorm2d(dpt_features),
                    nn.ReLU(inplace=True),
                )
                for _ in range(4)
            ])

        # RefineNet fusion blocks
        self.refinenets = nn.ModuleList([
            FeatureFusionBlock(dpt_features, has_residual=(i < 3))
            for i in range(4)
        ])

        # 8. Output Heads
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

    def _interpolate_pos_embed(self, pos_embed, target_h, target_w):
        """
        对位置编码进行 2D 插值以适配不同分辨率

        Args:
            pos_embed: [1, max_h*max_w, C]
            target_h: 目标 patch grid 高度
            target_w: 目标 patch grid 宽度

        Returns:
            interpolated: [1, target_h*target_w, C]
        """
        N = pos_embed.shape[1]
        if target_h * target_w == N and target_h == self.pos_embed_h:
            return pos_embed

        C = pos_embed.shape[2]
        # Reshape to 2D: [1, max_h, max_w, C] -> [1, C, max_h, max_w]
        pos_2d = pos_embed.reshape(1, self.pos_embed_h, self.pos_embed_w, C)
        pos_2d = pos_2d.permute(0, 3, 1, 2)

        # Bicubic interpolation
        pos_2d = F.interpolate(
            pos_2d, size=(target_h, target_w),
            mode='bicubic', align_corners=False
        )

        # Reshape back: [1, C, target_h, target_w] -> [1, target_h*target_w, C]
        pos_2d = pos_2d.permute(0, 2, 3, 1).reshape(1, target_h * target_w, C)
        return pos_2d

    def forward(self, img, depth, da3_features=None):
        """
        Args:
            img: [B*2, 3, H, W] (左右视图拼接)
            depth: [B*2, 1, H, W]
            da3_features: list of [B*2, C, H', W'] DA3 DINOv2 多层特征 (可选)

        Returns:
            rot_maps: [B*2, 4, H, W]
            scale_maps: [B*2, 3, H, W]
            opacity_maps: [B*2, 1, H, W]
            xyz_res_maps: [B*2, 3, H, W]
        """
        B2, _, H_orig, W_orig = img.shape
        B = B2 // 2

        # 计算需要的 padding，使 H, W 成为 patch_size 的倍数
        pad_h = (self.patch_size - H_orig % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W_orig % self.patch_size) % self.patch_size

        if pad_h > 0 or pad_w > 0:
            img = F.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            depth = F.pad(depth, (0, pad_w, 0, pad_h), mode='reflect')

        _, _, H, W = img.shape
        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        # 1. Local Patch Embedding (RGB + Depth)
        x = torch.cat([img, depth], dim=1)  # [B*2, 4, H, W]
        local_tokens = self.patch_embed(x)  # [B*2, P, C]
        _, P, C = local_tokens.shape

        # 2. DA3 特征融合 (门控机制)
        if self.use_da3_features and da3_features is not None and len(da3_features) > 0:
            # 使用最后一层 DA3 特征 (语义最强)
            da3_last = da3_features[-1]  # [B*2, da3_feat_dim, H', W']

            # 确保 DA3 特征的空间尺寸与 patch grid 匹配
            if da3_last.shape[2:] != (patch_h, patch_w):
                da3_last = F.interpolate(
                    da3_last, size=(patch_h, patch_w),
                    mode='bilinear', align_corners=False
                )

            # 转为 token 序列: [B*2, C_da3, ph, pw] -> [B*2, P, C_da3]
            da3_tokens = da3_last.flatten(2).transpose(1, 2)
            # 投影到 embed_dim
            da3_tokens = self.da3_proj(da3_tokens)  # [B*2, P, embed_dim]

            # 门控融合
            gate = self.fusion_gate(
                torch.cat([local_tokens, da3_tokens], dim=-1)
            )  # [B*2, P, embed_dim], values in [0, 1]
            tokens = gate * local_tokens + (1 - gate) * da3_tokens
        else:
            tokens = local_tokens

        # 3. 添加位置编码
        pos = self._interpolate_pos_embed(self.pos_embed, patch_h, patch_w)
        tokens = tokens + pos  # [B*2, P, C]

        # 4. 添加 View Token
        tokens = tokens.view(B, 2, P, C)
        view_token = self.view_token.expand(B, -1, P, -1)  # [B, 2, P, C]
        tokens = tokens + view_token

        # 5. 交替注意力 (可选 Gradient Checkpointing)
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

        # 6. DPT Head 处理
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

        # 7. RefineNet 融合 (自底向上) + DA3 skip connections
        layer_rns = [self.layer_rns[i](dpt_features[i]) for i in range(4)]

        # 注入 DA3 多层 skip connections
        if self.use_da3_features and da3_features is not None and len(da3_features) >= 4:
            for i in range(4):
                da3_skip = da3_features[i]  # [B*2, da3_feat_dim, H', W']
                da3_skip_proj = self.da3_skip_projs[i](da3_skip)  # [B*2, dpt_features, H', W']
                # 对齐空间尺寸
                if da3_skip_proj.shape[2:] != layer_rns[i].shape[2:]:
                    da3_skip_proj = F.interpolate(
                        da3_skip_proj, size=layer_rns[i].shape[2:],
                        mode='bilinear', align_corners=False
                    )
                layer_rns[i] = layer_rns[i] + da3_skip_proj

        # 从最深层开始融合
        out = self.refinenets[3](layer_rns[3], target_size=layer_rns[2].shape[2:])
        out = self.refinenets[2](out, layer_rns[2], target_size=layer_rns[1].shape[2:])
        out = self.refinenets[1](out, layer_rns[1], target_size=layer_rns[0].shape[2:])
        out = self.refinenets[0](out, layer_rns[0])

        # 8. 上采样到 padded 分辨率
        out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)

        # 9. Output Heads
        out = self.output_conv(out)

        # Rotation (归一化四元数)
        rot_out = self.rot_head(out)
        rot_out = F.normalize(rot_out, dim=1)

        # Scale (限制最大值, 从配置读取)
        scale_out = self.scale_head(out)
        scale_out = torch.clamp_max(scale_out, self.scale_max)

        # Opacity
        opacity_out = self.opacity_head(out)

        # XYZ residual (从配置读取缩放因子)
        xyz_out = self.xyz_head(out) * self.xyz_res_scale

        # 10. 裁剪回原始分辨率 (移除 padding)
        if pad_h > 0 or pad_w > 0:
            rot_out = rot_out[:, :, :H_orig, :W_orig].contiguous()
            scale_out = scale_out[:, :, :H_orig, :W_orig].contiguous()
            opacity_out = opacity_out[:, :, :H_orig, :W_orig].contiguous()
            xyz_out = xyz_out[:, :, :H_orig, :W_orig].contiguous()

        return rot_out, scale_out, opacity_out, xyz_out
