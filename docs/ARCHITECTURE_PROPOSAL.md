# GPS-Gaussian+ 架构改进方案

## 1. 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GPS-Gaussian++ (改进版)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐      ┌─────────────┐                                      │
│  │  左视图 IL  │      │  右视图 IR  │                                      │
│  └──────┬──────┘      └──────┬──────┘                                      │
│         │                    │                                             │
│         ▼                    ▼                                             │
│  ┌──────────────────────────────────────┐                                  │
│  │    Depth Anything V3 Backbone        │  ← DINOv2 编码器                  │
│  │    (共享权重)                         │                                  │
│  └──────┬───────────────────┬───────────┘                                  │
│         │                   │                                              │
│    ┌────┴────┐         ┌────┴────┐                                         │
│    │ DL, FL  │         │ DR, FR  │  ← 深度 + 多尺度特征                     │
│    └────┬────┘         └────┬────┘                                         │
│         │                   │                                              │
│         ▼                   ▼                                              │
│  ┌──────────────────────────────────────┐                                  │
│  │    深度联合优化模块 (Depth Fusion)    │  ← 跨视图一致性 + 尺度校正        │
│  │    - Cross-Attention                 │                                  │
│  │    - Scale Regressor                 │                                  │
│  └──────────────┬───────────────────────┘                                  │
│                 │                                                          │
│                 ▼                                                          │
│          ┌──────────────┐                                                  │
│          │ 融合深度 D_fused │                                               │
│          └──────┬───────┘                                                  │
│                 │                                                          │
│                 ▼                                                          │
│  ┌──────────────────────────────────────┐                                  │
│  │    Transformer + MoE 高斯预测头      │                                   │
│  │    - Window Self-Attention           │                                  │
│  │    - Cross-View Attention            │                                  │
│  │    - MoE MLP (专家路由)              │                                   │
│  │    - Multi-Layer Gaussian Output     │                                  │
│  └──────────────┬───────────────────────┘                                  │
│                 │                                                          │
│                 ▼                                                          │
│  ┌──────────────────────────────────────┐                                  │
│  │    动态高斯分配模块                    │  ← 纹理自适应高斯数量              │
│  │    - Opacity-based Pruning           │                                  │
│  │    - Confidence-weighted Masking     │                                  │
│  └──────────────┬───────────────────────┘                                  │
│                 │                                                          │
│                 ▼                                                          │
│  ┌──────────────────────────────────────┐                                  │
│  │    可微高斯渲染器                      │                                  │
│  │    diff-gaussian-rasterization       │                                  │
│  └──────────────────────────────────────┘                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 2. 模块详细设计

### 2.1 Depth Anything V3 特征提取器

**文件**: `lib/da3_encoder.py`

```python
import torch
import torch.nn as nn
from depth_anything_3.api import DepthAnything3

class DA3FeatureExtractor(nn.Module):
    """
    使用 Depth Anything V3 同时提取深度和图像特征
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model_size = cfg.da3.model_size  # 'large', 'giant' 等
        self.export_layers = cfg.da3.export_feat_layers  # [11, 15, 19, 23]
        
        # 加载预训练模型
        self.da3 = DepthAnything3.from_pretrained(
            f"depth-anything/DA3-{self.model_size.upper()}"
        )
        
        # 冻结或微调策略
        if cfg.da3.freeze_backbone:
            for param in self.da3.parameters():
                param.requires_grad = False
        
    def forward(self, images):
        """
        Args:
            images: [B, 3, H, W] - 输入图像
        Returns:
            depth: [B, 1, H, W] - 预测深度
            features: dict - 多尺度特征 {layer_idx: [B, H//14, W//14, C]}
        """
        # DA3 期望输入 [B, S, 3, H, W]，单视图时 S=1
        images_4d = images.unsqueeze(1)  # [B, 1, 3, H, W]
        
        output = self.da3.model.forward(
            images_4d,
            export_feat_layers=self.export_layers
        )
        
        depth = output.depth.squeeze(1)  # [B, H, W] -> 需要加维度
        features = output.aux  # dict of features
        
        return depth.unsqueeze(1), features
```

### 2.2 深度联合优化模块

**文件**: `lib/depth_fusion.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthFusionModule(nn.Module):
    """
    深度联合优化模块：融合左右视图深度，解决尺度不一致问题
    
    关键功能：
    1. 跨视图 Cross-Attention 增强一致性
    2. 稀疏立体匹配计算尺度校正系数
    3. 融合输出带绝对尺度的深度图
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.feat_dim = cfg.depth_fusion.feat_dim  # 特征维度
        self.num_heads = cfg.depth_fusion.num_heads  # 注意力头数
        
        # 特征投影
        self.depth_proj = nn.Conv2d(1, self.feat_dim, 3, padding=1)
        self.feat_proj = nn.Conv2d(cfg.da3.feat_dim, self.feat_dim, 1)
        
        # Cross-View Attention
        self.cross_attention = CrossViewAttention(
            dim=self.feat_dim,
            num_heads=self.num_heads,
            dropout=cfg.depth_fusion.dropout
        )
        
        # 稀疏相关性计算（用于尺度校正）
        self.sparse_corr = SparseCorrelationModule(
            cfg.depth_fusion.num_sparse_points
        )
        
        # 尺度回归网络
        self.scale_regressor = nn.Sequential(
            nn.Linear(cfg.depth_fusion.num_sparse_points * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # scale, shift
        )
        
        # 深度融合头
        self.fusion_head = nn.Sequential(
            nn.Conv2d(self.feat_dim * 2 + 2, self.feat_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim // 2, 1, 1),
        )
        
    def forward(self, depth_l, depth_r, feat_l, feat_r, intrinsics, baseline):
        """
        Args:
            depth_l: [B, 1, H, W] - 左视图深度 (相对深度)
            depth_r: [B, 1, H, W] - 右视图深度 (相对深度)
            feat_l: [B, C, H', W'] - 左视图特征
            feat_r: [B, C, H', W'] - 右视图特征
            intrinsics: [B, 3, 3] - 相机内参
            baseline: [B] - 基线距离
        Returns:
            depth_fused: [B, 1, H, W] - 融合后的绝对深度
            confidence: [B, 1, H, W] - 置信度图
        """
        B, _, H, W = depth_l.shape
        
        # 1. 特征投影
        depth_feat_l = self.depth_proj(depth_l)
        depth_feat_r = self.depth_proj(depth_r)
        
        # 上采样特征到深度分辨率
        feat_l_up = F.interpolate(self.feat_proj(feat_l), size=(H, W), mode='bilinear')
        feat_r_up = F.interpolate(self.feat_proj(feat_r), size=(H, W), mode='bilinear')
        
        # 融合深度特征和图像特征
        combined_l = depth_feat_l + feat_l_up
        combined_r = depth_feat_r + feat_r_up
        
        # 2. Cross-View Attention
        enhanced_l, enhanced_r = self.cross_attention(combined_l, combined_r)
        
        # 3. 稀疏相关性计算尺度校正
        sparse_disp = self.sparse_corr(feat_l, feat_r)  # [B, N, 2]
        
        # 使用稀疏视差计算尺度系数
        # disp = baseline * fx / depth => depth = baseline * fx / disp
        scale_shift = self.scale_regressor(sparse_disp.flatten(1))
        scale = scale_shift[:, 0:1].unsqueeze(-1).unsqueeze(-1) + 1.0  # 残差学习
        shift = scale_shift[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        
        # 4. 应用尺度校正
        depth_l_metric = depth_l * scale + shift
        depth_r_metric = depth_r * scale + shift
        
        # 5. 深度融合
        fusion_input = torch.cat([
            enhanced_l, enhanced_r,
            depth_l_metric, depth_r_metric
        ], dim=1)
        
        depth_fused = self.fusion_head(fusion_input)
        depth_fused = F.relu(depth_fused)  # 深度必须为正
        
        # 计算置信度 (基于左右一致性)
        confidence = self._compute_confidence(depth_l_metric, depth_r_metric)
        
        return depth_fused, confidence
    
    def _compute_confidence(self, depth_l, depth_r):
        """基于左右视图一致性计算置信度"""
        # 简化版：使用深度差异的倒数作为置信度
        diff = torch.abs(depth_l - depth_r)
        confidence = 1.0 / (1.0 + diff * 10.0)
        return confidence


class CrossViewAttention(nn.Module):
    """跨视图注意力模块"""
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv_l = nn.Linear(dim, dim * 3)
        self.qkv_r = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, feat_l, feat_r):
        B, C, H, W = feat_l.shape
        
        # 展平空间维度
        feat_l = feat_l.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        feat_r = feat_r.flatten(2).permute(0, 2, 1)
        
        # 计算 Q, K, V
        qkv_l = self.qkv_l(feat_l).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        qkv_r = self.qkv_r(feat_r).reshape(B, -1, 3, self.num_heads, C // self.num_heads)
        
        q_l, k_l, v_l = qkv_l.permute(2, 0, 3, 1, 4).unbind(0)
        q_r, k_r, v_r = qkv_r.permute(2, 0, 3, 1, 4).unbind(0)
        
        # 跨视图注意力: L query, R key/value
        attn_l2r = (q_l @ k_r.transpose(-2, -1)) * self.scale
        attn_l2r = attn_l2r.softmax(dim=-1)
        out_l = (attn_l2r @ v_r).transpose(1, 2).reshape(B, -1, C)
        
        # 跨视图注意力: R query, L key/value
        attn_r2l = (q_r @ k_l.transpose(-2, -1)) * self.scale
        attn_r2l = attn_r2l.softmax(dim=-1)
        out_r = (attn_r2l @ v_l).transpose(1, 2).reshape(B, -1, C)
        
        # 残差连接
        out_l = self.proj(out_l) + feat_l
        out_r = self.proj(out_r) + feat_r
        
        # 恢复空间维度
        out_l = out_l.permute(0, 2, 1).reshape(B, C, H, W)
        out_r = out_r.permute(0, 2, 1).reshape(B, C, H, W)
        
        return out_l, out_r


class SparseCorrelationModule(nn.Module):
    """稀疏相关性计算，用于尺度校正"""
    def __init__(self, num_points):
        super().__init__()
        self.num_points = num_points
        
    def forward(self, feat_l, feat_r):
        """计算稀疏点的视差"""
        B, C, H, W = feat_l.shape
        
        # 选择高梯度点作为关键点
        grad_l = self._compute_gradient_magnitude(feat_l)
        
        # 选择 top-k 点
        _, indices = grad_l.flatten(1).topk(self.num_points, dim=1)
        
        # 在这些点上计算相关性得到视差
        disparities = []
        for b in range(B):
            pts = indices[b]
            y_coords = pts // W
            x_coords = pts % W
            
            # 对每个点搜索最佳匹配 (沿水平方向)
            disp_b = []
            for i in range(len(pts)):
                y, x = y_coords[i], x_coords[i]
                feat_patch = feat_l[b, :, y, x]  # [C]
                
                # 计算与右图同一行的相关性
                corr = (feat_r[b, :, y, :] * feat_patch.unsqueeze(1)).sum(0)  # [W]
                best_x = corr.argmax()
                disp = x - best_x
                disp_b.append(disp.float())
            
            disparities.append(torch.stack(disp_b))
        
        disparities = torch.stack(disparities)  # [B, N]
        
        # 返回视差和对应的深度
        return disparities.unsqueeze(-1).repeat(1, 1, 2)
    
    def _compute_gradient_magnitude(self, feat):
        """计算特征的梯度幅度"""
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=feat.dtype, device=feat.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        
        feat_mean = feat.mean(dim=1, keepdim=True)
        grad_x = F.conv2d(feat_mean, sobel_x, padding=1)
        grad_y = F.conv2d(feat_mean, sobel_y, padding=1)
        
        return torch.sqrt(grad_x ** 2 + grad_y ** 2)
```

### 2.3 Transformer + MoE 高斯预测网络

**文件**: `lib/gaussian_transformer_moe.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class GaussianTransformerMoE(nn.Module):
    """
    Transformer + MoE 高斯参数预测网络
    
    架构特点：
    1. Window Self-Attention 处理局部上下文
    2. Cross-View Attention 跨视图一致性
    3. MoE MLP 层处理不同材质/纹理
    4. 多层高斯输出支持动态分配
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.hidden_dim = cfg.gs_transformer.hidden_dim
        self.num_layers = cfg.gs_transformer.num_layers
        self.num_heads = cfg.gs_transformer.num_heads
        self.window_size = cfg.gs_transformer.window_size
        self.num_gaussian_layers = cfg.gs_transformer.num_gaussian_layers  # 每像素高斯层数
        
        # MoE 配置
        self.num_experts = cfg.moe.num_experts
        self.top_k = cfg.moe.top_k
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Conv2d(cfg.da3.feat_dim + 1, self.hidden_dim, 3, padding=1),
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
            )
            for i in range(self.num_layers)
        ])
        
        # 输出头 - 多层高斯参数
        self.output_heads = nn.ModuleDict({
            'rotation': nn.Conv2d(self.hidden_dim, 4 * self.num_gaussian_layers, 1),
            'scale': nn.Conv2d(self.hidden_dim, 3 * self.num_gaussian_layers, 1),
            'opacity': nn.Conv2d(self.hidden_dim, 1 * self.num_gaussian_layers, 1),
            'depth_offset': nn.Conv2d(self.hidden_dim, 1 * self.num_gaussian_layers, 1),
            'texture_complexity': nn.Conv2d(self.hidden_dim, 1, 1),  # 纹理复杂度预测
        })
        
        self._init_output_heads()
        
    def _init_output_heads(self):
        """初始化输出头，使用较小的初始值"""
        for name, head in self.output_heads.items():
            if isinstance(head, nn.Conv2d):
                nn.init.zeros_(head.weight)
                if head.bias is not None:
                    nn.init.zeros_(head.bias)
    
    def forward(self, features, depth, features_other=None):
        """
        Args:
            features: [B, C, H, W] - 图像特征 (来自 DA3)
            depth: [B, 1, H, W] - 融合深度
            features_other: [B, C, H, W] - 另一视图的特征 (用于跨视图注意力)
        Returns:
            gaussian_params: dict - 高斯参数
            router_weights: [B, num_experts, H, W] - MoE 路由权重 (用于分析)
        """
        B, _, H, W = features.shape
        
        # 输入融合
        x = torch.cat([features, depth], dim=1)
        x = self.input_proj(x)
        
        # 添加位置编码
        pos = F.interpolate(self.pos_embed, size=(H, W), mode='bilinear')
        x = x + pos
        
        # Transformer 处理
        all_router_weights = []
        for i, block in enumerate(self.transformer_blocks):
            if block.use_cross_view and features_other is not None:
                x_other = torch.cat([features_other, depth], dim=1)
                x_other = self.input_proj(x_other) + pos
                x, router_weights = block(x, x_other)
            else:
                x, router_weights = block(x)
            
            if router_weights is not None:
                all_router_weights.append(router_weights)
        
        # 输出头
        rotation = self.output_heads['rotation'](x)  # [B, 4*L, H, W]
        scale = self.output_heads['scale'](x)  # [B, 3*L, H, W]
        opacity = self.output_heads['opacity'](x)  # [B, L, H, W]
        depth_offset = self.output_heads['depth_offset'](x)  # [B, L, H, W]
        texture_complexity = self.output_heads['texture_complexity'](x)  # [B, 1, H, W]
        
        # 应用激活函数
        rotation = F.normalize(rotation.reshape(B, 4, self.num_gaussian_layers, H, W), dim=1)
        scale = F.softplus(scale.reshape(B, 3, self.num_gaussian_layers, H, W))
        scale = torch.clamp(scale, max=self.cfg.gs_transformer.max_scale)
        opacity = torch.sigmoid(opacity.reshape(B, 1, self.num_gaussian_layers, H, W))
        depth_offset = torch.tanh(depth_offset.reshape(B, 1, self.num_gaussian_layers, H, W))
        depth_offset = depth_offset * self.cfg.gs_transformer.max_depth_offset
        texture_complexity = torch.sigmoid(texture_complexity)
        
        return {
            'rotation': rotation,
            'scale': scale,
            'opacity': opacity,
            'depth_offset': depth_offset,
            'texture_complexity': texture_complexity,
        }, all_router_weights


class TransformerBlockMoE(nn.Module):
    """带 MoE 的 Transformer 块"""
    def __init__(self, dim, num_heads, window_size, num_experts, top_k, use_cross_view=False):
        super().__init__()
        self.use_cross_view = use_cross_view
        self.window_size = window_size
        
        # Window Self-Attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = WindowAttention(dim, num_heads, window_size)
        
        # Cross-View Attention (可选)
        if use_cross_view:
            self.norm_cross = nn.LayerNorm(dim)
            self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MoE MLP
        self.norm2 = nn.LayerNorm(dim)
        self.moe_mlp = MoEMLP(
            dim=dim,
            hidden_dim=dim * 4,
            num_experts=num_experts,
            top_k=top_k
        )
        
    def forward(self, x, x_other=None):
        B, C, H, W = x.shape
        
        # Window Self-Attention
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_flat = x_flat + self.self_attn(self.norm1(x_flat), H, W)
        
        # Cross-View Attention
        if self.use_cross_view and x_other is not None:
            x_other_flat = rearrange(x_other, 'b c h w -> b (h w) c')
            x_cross, _ = self.cross_attn(
                self.norm_cross(x_flat),
                x_other_flat,
                x_other_flat
            )
            x_flat = x_flat + x_cross
        
        # MoE MLP
        x_flat, router_weights = self.moe_mlp(self.norm2(x_flat))
        
        x = rearrange(x_flat, 'b (h w) c -> b c h w', h=H, w=W)
        
        return x, router_weights


class WindowAttention(nn.Module):
    """窗口注意力，减少计算复杂度"""
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        
        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
    def forward(self, x, H, W):
        B, N, C = x.shape
        
        # 窗口划分
        x = x.reshape(B, H, W, C)
        
        # Padding to fit window size
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w
        
        # 划分窗口
        x = x.reshape(B, Hp // self.window_size, self.window_size,
                      Wp // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size ** 2, C)
        
        # 注意力计算
        qkv = self.qkv(x).reshape(-1, self.window_size ** 2, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size ** 2, C)
        x = self.proj(x)
        
        # 窗口还原
        x = x.reshape(B, Hp // self.window_size, Wp // self.window_size,
                      self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, C)
        
        # 移除 padding
        x = x[:, :H, :W, :].reshape(B, N, C)
        
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
    """
    def __init__(self, dim, hidden_dim, num_experts, top_k):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # 路由器
        self.router = nn.Linear(dim, num_experts)
        
        # 专家网络
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, dim),
            )
            for _ in range(num_experts)
        ])
        
        # 负载均衡损失系数
        self.load_balance_weight = 0.01
        
    def forward(self, x):
        """
        Args:
            x: [B, N, C] - 输入特征
        Returns:
            output: [B, N, C] - 输出特征
            router_weights: [B, N, num_experts] - 路由权重
        """
        B, N, C = x.shape
        
        # 计算路由分数
        router_logits = self.router(x)  # [B, N, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)
        
        # 选择 top-k 专家
        top_k_weights, top_k_indices = router_weights.topk(self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        # 专家计算
        output = torch.zeros_like(x)
        for k in range(self.top_k):
            expert_idx = top_k_indices[:, :, k]  # [B, N]
            expert_weight = top_k_weights[:, :, k:k+1]  # [B, N, 1]
            
            for e in range(self.num_experts):
                mask = (expert_idx == e)  # [B, N]
                if mask.any():
                    expert_input = x[mask]  # [num_selected, C]
                    expert_output = self.experts[e](expert_input)
                    output[mask] += expert_weight[mask].squeeze(-1).unsqueeze(-1) * expert_output
        
        return x + output, router_weights
```

### 2.4 动态高斯分配模块

**文件**: `lib/dynamic_gaussian_allocation.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicGaussianAllocation(nn.Module):
    """
    动态高斯分配模块
    
    核心思路：
    1. 每个像素预测 K 层高斯 (K=3 或 4)
    2. 通过 Opacity 阈值裁剪无效高斯
    3. 纹理复杂度指导高斯数量分配
    
    优势：
    - 复杂纹理区域：保留多层高斯，提升细节
    - 平滑区域：仅保留1层高斯，节省计算
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.num_layers = cfg.dynamic_gs.num_layers  # 最大高斯层数
        self.opacity_threshold = cfg.dynamic_gs.opacity_threshold  # 裁剪阈值
        self.use_complexity_guidance = cfg.dynamic_gs.use_complexity_guidance
        
        # 复杂度到高斯数量的映射
        if self.use_complexity_guidance:
            self.complexity_to_count = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, self.num_layers),
                nn.Softmax(dim=-1),  # 每层的保留概率
            )
    
    def forward(self, gaussian_params, texture_complexity=None):
        """
        Args:
            gaussian_params: dict - 高斯参数
                - rotation: [B, 4, L, H, W]
                - scale: [B, 3, L, H, W]
                - opacity: [B, 1, L, H, W]
                - depth_offset: [B, 1, L, H, W]
            texture_complexity: [B, 1, H, W] - 纹理复杂度 (可选)
        Returns:
            filtered_params: dict - 过滤后的高斯参数
            valid_mask: [B, L, H, W] - 有效高斯掩码
            stats: dict - 统计信息
        """
        B, _, L, H, W = gaussian_params['opacity'].shape
        opacity = gaussian_params['opacity'].squeeze(1)  # [B, L, H, W]
        
        # 方法1: 基于 Opacity 阈值裁剪
        valid_mask = opacity > self.opacity_threshold  # [B, L, H, W]
        
        # 方法2: 结合纹理复杂度指导
        if self.use_complexity_guidance and texture_complexity is not None:
            # 计算每层的保留概率
            complexity_flat = texture_complexity.flatten(2).permute(0, 2, 1)  # [B, HW, 1]
            layer_probs = self.complexity_to_count(complexity_flat)  # [B, HW, L]
            layer_probs = layer_probs.permute(0, 2, 1).reshape(B, L, H, W)
            
            # 按概率采样决定是否保留
            # 训练时：使用 Gumbel-Softmax 实现可微采样
            # 推理时：直接按阈值裁剪
            if self.training:
                keep_decision = self._gumbel_softmax_sample(layer_probs, opacity)
            else:
                # 复杂区域保留更多层
                complexity_mask = texture_complexity > self.cfg.dynamic_gs.complexity_threshold
                complexity_mask = complexity_mask.expand(-1, L, -1, -1)
                
                # 简单区域只保留第一层
                simple_mask = ~complexity_mask
                valid_mask = valid_mask.clone()
                valid_mask[:, 1:][simple_mask[:, 1:]] = False
        
        # 应用掩码到所有参数
        filtered_params = {}
        for key, value in gaussian_params.items():
            if key == 'texture_complexity':
                filtered_params[key] = value
                continue
            
            if value.dim() == 5:  # [B, C, L, H, W]
                # 将无效高斯的 opacity 设为 0
                if key == 'opacity':
                    filtered_params[key] = value * valid_mask.unsqueeze(1).float()
                else:
                    filtered_params[key] = value
            else:
                filtered_params[key] = value
        
        # 统计信息
        stats = {
            'total_gaussians': B * L * H * W,
            'valid_gaussians': valid_mask.sum().item(),
            'pruning_ratio': 1.0 - valid_mask.float().mean().item(),
            'avg_layers_per_pixel': valid_mask.float().sum(dim=1).mean().item(),
        }
        
        return filtered_params, valid_mask, stats
    
    def _gumbel_softmax_sample(self, layer_probs, opacity, temperature=1.0):
        """可微的 Gumbel-Softmax 采样"""
        # 结合 opacity 和 layer_probs
        combined = layer_probs * torch.sigmoid(opacity * 10)  # 放大 opacity 的影响
        
        # Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(combined) + 1e-8) + 1e-8)
        
        # Softmax with temperature
        y = F.softmax((torch.log(combined + 1e-8) + gumbel_noise) / temperature, dim=1)
        
        return y
    
    def flatten_gaussians(self, gaussian_params, valid_mask, depth, intrinsics, extrinsics):
        """
        将多层高斯展平为点云格式
        
        Args:
            gaussian_params: dict - 高斯参数
            valid_mask: [B, L, H, W] - 有效掩码
            depth: [B, 1, H, W] - 深度图
            intrinsics: [B, 3, 3] - 内参
            extrinsics: [B, 3, 4] - 外参
        Returns:
            flattened_params: dict - 展平后的参数
        """
        B, L, H, W = valid_mask.shape
        
        # 计算每层的深度
        depth_offset = gaussian_params['depth_offset']  # [B, 1, L, H, W]
        layer_depths = depth.unsqueeze(2) + depth_offset  # [B, 1, L, H, W]
        layer_depths = F.relu(layer_depths)  # 确保深度为正
        
        # 生成像素坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=depth.device),
            torch.arange(W, device=depth.device),
            indexing='ij'
        )
        x_coords = x_coords.float().unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        y_coords = y_coords.float().unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        
        # 反投影到 3D
        fx, fy = intrinsics[:, 0, 0], intrinsics[:, 1, 1]
        cx, cy = intrinsics[:, 0, 2], intrinsics[:, 1, 2]
        
        z = layer_depths.squeeze(1)  # [B, L, H, W]
        x = (x_coords - cx.view(B, 1, 1, 1)) * z / fx.view(B, 1, 1, 1)
        y = (y_coords - cy.view(B, 1, 1, 1)) * z / fy.view(B, 1, 1, 1)
        
        xyz = torch.stack([x, y, z], dim=-1)  # [B, L, H, W, 3]
        
        # 应用外参转换到世界坐标
        R = extrinsics[:, :3, :3]  # [B, 3, 3]
        t = extrinsics[:, :3, 3]   # [B, 3]
        xyz_world = torch.einsum('bij,blhwj->blhwi', R.transpose(1, 2), xyz - t.view(B, 1, 1, 1, 3))
        
        # 展平有效高斯
        valid_flat = valid_mask.flatten(1)  # [B, L*H*W]
        xyz_flat = xyz_world.flatten(1, 3)  # [B, L*H*W, 3]
        
        flattened = {
            'xyz': xyz_flat,
            'rotation': gaussian_params['rotation'].permute(0, 2, 3, 4, 1).flatten(1, 3),
            'scale': gaussian_params['scale'].permute(0, 2, 3, 4, 1).flatten(1, 3),
            'opacity': gaussian_params['opacity'].squeeze(1).flatten(1),
            'valid_mask': valid_flat,
        }
        
        return flattened
```

### 2.5 完整网络集成

**文件**: `lib/network_v2.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from lib.da3_encoder import DA3FeatureExtractor
from lib.depth_fusion import DepthFusionModule
from lib.gaussian_transformer_moe import GaussianTransformerMoE
from lib.dynamic_gaussian_allocation import DynamicGaussianAllocation
from lib.utils import depth2pc


class GPSGaussianPlusPlusModel(nn.Module):
    """
    GPS-Gaussian++ 改进版模型
    
    架构改进：
    1. Depth Anything V3 替换 RAFT (更强泛化性)
    2. 深度联合优化模块 (解决尺度问题)
    3. Transformer + MoE 高斯预测 (更强表达能力)
    4. 动态高斯分配 (纹理自适应)
    """
    def __init__(self, cfg, with_gs_render=True):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        
        # 1. Depth Anything V3 特征提取器 (共享)
        self.da3_encoder = DA3FeatureExtractor(cfg)
        
        # 2. 深度联合优化模块
        self.depth_fusion = DepthFusionModule(cfg)
        
        # 3. Transformer + MoE 高斯预测网络
        if with_gs_render:
            self.gaussian_predictor = GaussianTransformerMoE(cfg)
            
            # 4. 动态高斯分配
            self.dynamic_allocation = DynamicGaussianAllocation(cfg)
    
    def forward(self, data, is_train=True):
        """
        前向传播
        
        Args:
            data: dict - 输入数据
                - lmain/rmain: {img, mask, intr, extr}
            is_train: bool - 是否训练模式
        Returns:
            data: dict - 增强后的数据（包含预测结果）
            loss_dict: dict - 各项损失
        """
        bs = data['lmain']['img'].shape[0]
        
        # ===== 1. DA3 特征提取 =====
        with autocast(enabled=self.cfg.da3.mixed_precision):
            # 左视图
            depth_l, feat_l = self.da3_encoder(data['lmain']['img'])
            # 右视图
            depth_r, feat_r = self.da3_encoder(data['rmain']['img'])
        
        data['lmain']['depth_mono'] = depth_l
        data['rmain']['depth_mono'] = depth_r
        
        # 获取最高分辨率特征用于后续处理
        feat_key = f"feat_layer_{self.cfg.da3.export_feat_layers[-1]}"
        feat_l_main = feat_l[feat_key].squeeze(1).permute(0, 3, 1, 2)  # [B, C, H', W']
        feat_r_main = feat_r[feat_key].squeeze(1).permute(0, 3, 1, 2)
        
        # ===== 2. 深度联合优化 =====
        baseline = self._compute_baseline(data['lmain']['extr'], data['rmain']['extr'])
        
        depth_fused_l, conf_l = self.depth_fusion(
            depth_l, depth_r, feat_l_main, feat_r_main,
            data['lmain']['intr'], baseline
        )
        depth_fused_r, conf_r = self.depth_fusion(
            depth_r, depth_l, feat_r_main, feat_l_main,
            data['rmain']['intr'], baseline
        )
        
        data['lmain']['depth'] = depth_fused_l
        data['rmain']['depth'] = depth_fused_r
        data['lmain']['depth_conf'] = conf_l
        data['rmain']['depth_conf'] = conf_r
        
        if not self.with_gs_render:
            return data, {}, {}
        
        # ===== 3. 高斯参数预测 (Transformer + MoE) =====
        # 上采样特征到深度分辨率
        H, W = depth_fused_l.shape[-2:]
        feat_l_up = F.interpolate(feat_l_main, size=(H, W), mode='bilinear')
        feat_r_up = F.interpolate(feat_r_main, size=(H, W), mode='bilinear')
        
        gaussian_params_l, router_weights_l = self.gaussian_predictor(
            feat_l_up, depth_fused_l, feat_r_up
        )
        gaussian_params_r, router_weights_r = self.gaussian_predictor(
            feat_r_up, depth_fused_r, feat_l_up
        )
        
        # ===== 4. 动态高斯分配 =====
        filtered_l, valid_mask_l, stats_l = self.dynamic_allocation(
            gaussian_params_l, gaussian_params_l['texture_complexity']
        )
        filtered_r, valid_mask_r, stats_r = self.dynamic_allocation(
            gaussian_params_r, gaussian_params_r['texture_complexity']
        )
        
        # ===== 5. 展平高斯并计算 3D 坐标 =====
        flattened_l = self.dynamic_allocation.flatten_gaussians(
            filtered_l, valid_mask_l, depth_fused_l,
            data['lmain']['intr'], data['lmain']['extr']
        )
        flattened_r = self.dynamic_allocation.flatten_gaussians(
            filtered_r, valid_mask_r, depth_fused_r,
            data['rmain']['intr'], data['rmain']['extr']
        )
        
        # 存储结果
        data['lmain']['xyz'] = flattened_l['xyz']
        data['lmain']['rot'] = flattened_l['rotation']
        data['lmain']['scale'] = flattened_l['scale']
        data['lmain']['opacity'] = flattened_l['opacity']
        data['lmain']['pts_valid'] = flattened_l['valid_mask']
        
        data['rmain']['xyz'] = flattened_r['xyz']
        data['rmain']['rot'] = flattened_r['rotation']
        data['rmain']['scale'] = flattened_r['scale']
        data['rmain']['opacity'] = flattened_r['opacity']
        data['rmain']['pts_valid'] = flattened_r['valid_mask']
        
        # 统计信息
        metrics = {
            'pruning_ratio': (stats_l['pruning_ratio'] + stats_r['pruning_ratio']) / 2,
            'avg_layers': (stats_l['avg_layers_per_pixel'] + stats_r['avg_layers_per_pixel']) / 2,
        }
        
        return data, {}, metrics
    
    def _compute_baseline(self, extr_l, extr_r):
        """计算立体基线距离"""
        t_l = extr_l[:, :3, 3]  # [B, 3]
        t_r = extr_r[:, :3, 3]
        baseline = torch.norm(t_l - t_r, dim=1)  # [B]
        return baseline
```

## 3. 配置文件

**文件**: `config/stage_v2.yaml`

```yaml
name: 'gps_plus_v2'

lr: 0.0001
wdecay: 1e-5
batch_size: 1
num_steps: 150000

# Depth Anything V3 配置
da3:
  model_size: 'large'  # 'small', 'base', 'large', 'giant'
  export_feat_layers: [11, 15, 19, 23]
  feat_dim: 1024  # large 模型的特征维度
  freeze_backbone: false  # 是否冻结主干网络
  mixed_precision: true

# 深度融合模块配置
depth_fusion:
  feat_dim: 256
  num_heads: 8
  dropout: 0.1
  num_sparse_points: 128

# Transformer + MoE 高斯预测网络配置
gs_transformer:
  hidden_dim: 512
  num_layers: 6
  num_heads: 8
  window_size: 8
  num_gaussian_layers: 3  # 每像素最大高斯层数
  max_scale: 0.002
  max_depth_offset: 0.5

# MoE 配置
moe:
  num_experts: 8
  top_k: 2
  load_balance_weight: 0.01

# 动态高斯分配配置
dynamic_gs:
  num_layers: 3
  opacity_threshold: 0.01
  complexity_threshold: 0.3
  use_complexity_guidance: true

# 数据集配置
dataset:
  source_id: [0, 1]
  train_novel_id: [2, 3, 4, 5]
  val_novel_id: [2, 3]
  use_hr_img: false
  use_depth_init: false  # DA3 不需要深度初始化
  use_local_data: true
  local_data_root: '/PATH/TO/data'
  train_data_root: '/PATH/TO/data/train'
  val_data_root: '/PATH/TO/data/val'

# 损失函数配置
loss:
  l1_weight: 0.8
  ssim_weight: 0.2
  depth_smooth_weight: 0.1
  moe_balance_weight: 0.01

# 训练记录
record:
  loss_freq: 1000
  eval_freq: 3000
  save_freq: 10000
```

## 4. 实施路线图

### Phase 1: 基础架构搭建 (1-2周)
- [ ] 集成 Depth Anything V3
- [ ] 实现特征提取接口
- [ ] 基础深度融合模块

### Phase 2: 核心模块开发 (2-3周)
- [ ] Transformer + MoE 高斯预测网络
- [ ] 窗口注意力机制
- [ ] MoE 路由器和专家网络

### Phase 3: 动态分配机制 (1-2周)
- [ ] 多层高斯输出
- [ ] Opacity 裁剪
- [ ] 纹理复杂度指导

### Phase 4: 训练优化 (2-3周)
- [ ] 损失函数设计
- [ ] 训练策略调优
- [ ] 消融实验

### Phase 5: 评估与发布 (1周)
- [ ] 基准测试
- [ ] 文档完善
- [ ] 代码清理

## 5. 潜在风险与应对

### 风险1: DA3 尺度漂移
- **问题**: DA3 输出相对深度，立体重建需要绝对深度
- **应对**: 深度联合优化模块中的 Scale Regressor

### 风险2: MoE 负载不均衡
- **问题**: 部分专家被过度使用
- **应对**: 添加 Load Balancing Loss

### 风险3: 动态分配不稳定
- **问题**: 训练初期高斯数量波动大
- **应对**: 渐进式开启动态裁剪，前期使用固定层数

### 风险4: 显存占用过高
- **问题**: Transformer + 多层高斯 显存需求大
- **应对**: 梯度检查点、混合精度训练、窗口注意力
```
