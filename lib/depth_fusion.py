"""
深度联合优化模块

融合左右视图的深度图，解决DA3相对深度的尺度不一致问题

Features:
- Cross-View Attention: 增强左右视图的一致性
- 稀疏相关性计算: 选择高梯度点计算视差，用于尺度校正
- 尺度回归网络: 从稀疏视差计算全局scale和shift
- 深度融合头: 输出融合深度和置信度

设计思路:
DA3 输出的是相对深度（无绝对尺度），我们通过以下方式获得具有几何一致性的深度：
1. 从左右视图特征中提取高梯度（高置信度）区域
2. 在这些区域计算稀疏视差
3. 利用立体几何关系（disp = baseline * fx / depth）回归尺度系数
4. 融合左右视图深度，输出具有绝对尺度的深度图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# 是否使用梯度检查点 (全局开关)
USE_GRADIENT_CHECKPOINT = True


class DepthFusionModule(nn.Module):
    """
    深度联合优化模块
    
    融合左右视图深度，解决DA3相对深度的尺度问题
    
    Args:
        cfg: 配置对象
    """
    
    def __init__(self, cfg, da3_feat_dim: int = None):
        super().__init__()
        self.cfg = cfg
        
        # 从配置读取参数，使用合理默认值
        fusion_cfg = getattr(cfg, 'depth_fusion', None)
        if fusion_cfg is not None:
            self.feat_dim = getattr(fusion_cfg, 'feat_dim', 256)
            self.num_heads = getattr(fusion_cfg, 'num_heads', 8)
            self.dropout = getattr(fusion_cfg, 'dropout', 0.1)
            self.num_sparse_points = getattr(fusion_cfg, 'num_sparse_points', 128)
        else:
            self.feat_dim = 256
            self.num_heads = 8
            self.dropout = 0.1
            self.num_sparse_points = 128
        
        # DA3 特征维度 - 根据模型类型动态设置
        if da3_feat_dim is not None:
            self.da3_feat_dim = da3_feat_dim
        else:
            # 从配置推断
            da3_cfg = getattr(cfg, 'da3', None)
            if da3_cfg is not None:
                model_name = getattr(da3_cfg, 'model_name', 'DA3-LARGE')
                if 'LARGE' in model_name.upper() or 'GIANT' in model_name.upper():
                    self.da3_feat_dim = 1024
                elif 'BASE' in model_name.upper():
                    self.da3_feat_dim = 768
                else:
                    self.da3_feat_dim = 384  # SMALL
            else:
                self.da3_feat_dim = 1024  # 默认 LARGE
        
        logger.info(f"[DepthFusion] DA3 特征维度: {self.da3_feat_dim}")
        
        # 深度特征投影
        self.depth_proj = nn.Sequential(
            nn.Conv2d(1, self.feat_dim // 4, 3, padding=1),
            nn.BatchNorm2d(self.feat_dim // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim // 4, self.feat_dim, 3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
        )
        
        # 图像特征投影 (DA3 特征 -> 融合特征)
        self.feat_proj = nn.Sequential(
            nn.Conv2d(self.da3_feat_dim, self.feat_dim, 1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
        )
        
        # Cross-View Attention
        self.cross_attention = CrossViewAttention(
            dim=self.feat_dim,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # 稀疏相关性模块
        self.sparse_corr = SparseCorrelationModule(
            num_points=self.num_sparse_points,
            feat_dim=self.feat_dim
        )
        
        # 尺度回归网络
        self.scale_regressor = nn.Sequential(
            nn.Linear(self.num_sparse_points * 3, 256),  # disparity + confidence
            nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),  # scale, shift
        )
        
        # 深度融合头
        self.fusion_head = nn.Sequential(
            nn.Conv2d(self.feat_dim * 2 + 2, self.feat_dim, 3, padding=1),
            nn.BatchNorm2d(self.feat_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim, self.feat_dim // 2, 3, padding=1),
            nn.BatchNorm2d(self.feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.feat_dim // 2, 2, 1),  # depth + confidence
        )
        
        # 初始化
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _cross_attention_forward(self, combined_l: torch.Tensor, combined_r: torch.Tensor):
        """可被checkpoint包装的跨视图注意力前向"""
        return self.cross_attention(combined_l, combined_r)
    
    def forward(
        self, 
        depth_l: torch.Tensor, 
        depth_r: torch.Tensor, 
        feat_l: torch.Tensor, 
        feat_r: torch.Tensor,
        intrinsics: Optional[torch.Tensor] = None,
        baseline: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            depth_l: [B, 1, H, W] - 左视图深度 (相对深度)
            depth_r: [B, 1, H, W] - 右视图深度 (相对深度)
            feat_l: [B, C, H', W'] - 左视图 DA3 特征
            feat_r: [B, C, H', W'] - 右视图 DA3 特征
            intrinsics: [B, 3, 3] - 相机内参 (可选)
            baseline: [B] - 基线距离 (可选)
            
        Returns:
            depth_fused: [B, 1, H, W] - 融合后的深度
            confidence: [B, 1, H, W] - 置信度图
            aux_outputs: dict - 辅助输出 (用于loss计算)
        """
        B, _, H, W = depth_l.shape
        
        # 1. 深度特征投影
        depth_feat_l = self.depth_proj(depth_l)  # [B, feat_dim, H, W]
        depth_feat_r = self.depth_proj(depth_r)
        
        # 2. 图像特征格式转换 (DA3 输出 [B, H, W, C] -> [B, C, H, W])
        if feat_l.dim() == 4 and feat_l.shape[-1] == self.da3_feat_dim:
            feat_l = feat_l.permute(0, 3, 1, 2).contiguous()
        if feat_r.dim() == 4 and feat_r.shape[-1] == self.da3_feat_dim:
            feat_r = feat_r.permute(0, 3, 1, 2).contiguous()
        
        # 3. 图像特征投影并上采样到深度分辨率
        feat_l_proj = self.feat_proj(feat_l)  # [B, feat_dim, H', W']
        feat_r_proj = self.feat_proj(feat_r)
        
        # 上采样特征到深度分辨率
        if feat_l_proj.shape[-2:] != (H, W):
            feat_l_proj = F.interpolate(feat_l_proj, size=(H, W), mode='bilinear', align_corners=False)
            feat_r_proj = F.interpolate(feat_r_proj, size=(H, W), mode='bilinear', align_corners=False)
        
        # 3. 融合深度特征和图像特征
        combined_l = depth_feat_l + feat_l_proj
        combined_r = depth_feat_r + feat_r_proj
        
        # 4. Cross-View Attention (使用梯度检查点)
        if USE_GRADIENT_CHECKPOINT and self.training:
            enhanced_l, enhanced_r = checkpoint(
                self._cross_attention_forward, combined_l, combined_r,
                use_reentrant=False
            )
        else:
            enhanced_l, enhanced_r = self.cross_attention(combined_l, combined_r)
        
        # 5. 稀疏相关性计算和尺度回归
        sparse_output = self.sparse_corr(enhanced_l, enhanced_r, depth_l, depth_r)
        sparse_features = sparse_output['sparse_features']  # [B, num_points * 3]
        
        # 尺度回归
        scale_shift = self.scale_regressor(sparse_features)  # [B, 2]
        scale = scale_shift[:, 0:1].unsqueeze(-1).unsqueeze(-1) + 1.0  # 残差学习
        shift = scale_shift[:, 1:2].unsqueeze(-1).unsqueeze(-1)
        
        # 6. 应用尺度校正
        depth_l_metric = depth_l * scale + shift
        depth_r_metric = depth_r * scale + shift
        
        # 确保深度为正
        depth_l_metric = F.relu(depth_l_metric) + 1e-6
        depth_r_metric = F.relu(depth_r_metric) + 1e-6
        
        # 7. 深度融合
        fusion_input = torch.cat([
            enhanced_l, enhanced_r,
            depth_l_metric, depth_r_metric
        ], dim=1)
        
        fusion_output = self.fusion_head(fusion_input)  # [B, 2, H, W]
        
        # 分离深度和置信度
        depth_residual = fusion_output[:, 0:1]  # 深度残差
        confidence_logits = fusion_output[:, 1:2]  # 置信度logits
        
        # 最终深度 = 平均深度 + 残差
        depth_avg = (depth_l_metric + depth_r_metric) / 2
        depth_fused = depth_avg + depth_residual * 0.1  # 缩放残差
        depth_fused = F.relu(depth_fused) + 1e-6  # 确保正值
        
        # 置信度 (sigmoid)
        confidence = torch.sigmoid(confidence_logits)
        
        # 辅助输出
        aux_outputs = {
            'scale': scale.squeeze(-1).squeeze(-1),  # [B, 1]
            'shift': shift.squeeze(-1).squeeze(-1),
            'depth_l_metric': depth_l_metric,
            'depth_r_metric': depth_r_metric,
            'sparse_disparities': sparse_output['disparities'],
            'sparse_points': sparse_output['points'],
        }
        
        return depth_fused, confidence, aux_outputs
    
    def compute_consistency_loss(
        self, 
        depth_l_metric: torch.Tensor, 
        depth_r_metric: torch.Tensor,
        intrinsics: torch.Tensor,
        baseline: torch.Tensor
    ) -> torch.Tensor:
        """
        计算左右视图深度一致性损失
        
        使用立体几何约束: disp = baseline * fx / depth
        
        Args:
            depth_l_metric: [B, 1, H, W] - 左视图绝对深度
            depth_r_metric: [B, 1, H, W] - 右视图绝对深度  
            intrinsics: [B, 3, 3] - 相机内参
            baseline: [B] - 基线距离
            
        Returns:
            loss: 一致性损失
        """
        B, _, H, W = depth_l_metric.shape
        
        # 从内参获取焦距
        fx = intrinsics[:, 0, 0]  # [B]
        
        # 计算预期视差
        disp_from_depth_l = baseline.view(B, 1, 1, 1) * fx.view(B, 1, 1, 1) / (depth_l_metric + 1e-6)
        disp_from_depth_r = baseline.view(B, 1, 1, 1) * fx.view(B, 1, 1, 1) / (depth_r_metric + 1e-6)
        
        # 一致性损失: 左右视图的视差应该相近
        consistency_loss = F.l1_loss(disp_from_depth_l, disp_from_depth_r)
        
        return consistency_loss


class CrossViewAttention(nn.Module):
    """
    跨视图注意力模块 (内存优化版)
    
    使用下采样减少内存使用，适用于高分辨率输入
    """
    
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0, downsample_factor: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.downsample_factor = downsample_factor
        
        # Q, K, V 投影 (使用卷积，更高效)
        self.q_proj = nn.Conv2d(dim, dim, 1)
        self.k_proj = nn.Conv2d(dim, dim, 1)
        self.v_proj = nn.Conv2d(dim, dim, 1)
        self.out_proj = nn.Conv2d(dim, dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Layer norm
        self.norm1 = nn.GroupNorm(8, dim)
        self.norm2 = nn.GroupNorm(8, dim)
        
    def forward(
        self, 
        feat_l: torch.Tensor, 
        feat_r: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            feat_l: [B, C, H, W] - 左视图特征
            feat_r: [B, C, H, W] - 右视图特征
            
        Returns:
            out_l: [B, C, H, W] - 增强后的左视图特征
            out_r: [B, C, H, W] - 增强后的右视图特征
        """
        B, C, H, W = feat_l.shape
        
        # 保存原始特征用于残差连接
        feat_l_orig = feat_l
        feat_r_orig = feat_r
        
        # Layer norm
        feat_l = self.norm1(feat_l)
        feat_r = self.norm1(feat_r)
        
        # 下采样以减少内存 (关键优化)
        ds = self.downsample_factor
        if H > ds * 4 and W > ds * 4:
            feat_l_ds = F.avg_pool2d(feat_l, ds, ds)
            feat_r_ds = F.avg_pool2d(feat_r, ds, ds)
            Hs, Ws = feat_l_ds.shape[-2:]
        else:
            feat_l_ds = feat_l
            feat_r_ds = feat_r
            Hs, Ws = H, W
        
        # 计算 Q, K, V
        q_l = self.q_proj(feat_l_ds)  # [B, C, Hs, Ws]
        k_r = self.k_proj(feat_r_ds)
        v_r = self.v_proj(feat_r_ds)
        
        q_r = self.q_proj(feat_r_ds)
        k_l = self.k_proj(feat_l_ds)
        v_l = self.v_proj(feat_l_ds)
        
        # 重塑为注意力格式
        def reshape_for_attention(x, H, W):
            # [B, C, H, W] -> [B, num_heads, HW, head_dim]
            B, C, H, W = x.shape
            return x.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        
        q_l = reshape_for_attention(q_l, Hs, Ws)
        k_r = reshape_for_attention(k_r, Hs, Ws)
        v_r = reshape_for_attention(v_r, Hs, Ws)
        q_r = reshape_for_attention(q_r, Hs, Ws)
        k_l = reshape_for_attention(k_l, Hs, Ws)
        v_l = reshape_for_attention(v_l, Hs, Ws)
        
        # 注意力计算 (L -> R)
        attn_l2r = (q_l @ k_r.transpose(-2, -1)) * self.scale  # [B, heads, HsWs, HsWs]
        attn_l2r = attn_l2r.softmax(dim=-1)
        attn_l2r = self.dropout(attn_l2r)
        out_l = (attn_l2r @ v_r)  # [B, heads, HsWs, head_dim]
        
        # 注意力计算 (R -> L)
        attn_r2l = (q_r @ k_l.transpose(-2, -1)) * self.scale
        attn_r2l = attn_r2l.softmax(dim=-1)
        attn_r2l = self.dropout(attn_r2l)
        out_r = (attn_r2l @ v_l)
        
        # 重塑回空间格式
        out_l = out_l.permute(0, 1, 3, 2).contiguous().view(B, C, Hs, Ws)
        out_r = out_r.permute(0, 1, 3, 2).contiguous().view(B, C, Hs, Ws)
        
        # 投影
        out_l = self.out_proj(out_l)
        out_r = self.out_proj(out_r)
        
        # 上采样回原始分辨率
        if Hs != H or Ws != W:
            out_l = F.interpolate(out_l, size=(H, W), mode='bilinear', align_corners=False)
            out_r = F.interpolate(out_r, size=(H, W), mode='bilinear', align_corners=False)
        
        # 残差连接
        out_l = self.norm2(feat_l_orig + out_l)
        out_r = self.norm2(feat_r_orig + out_r)
        
        return out_l, out_r


class SparseCorrelationModule(nn.Module):
    """
    稀疏相关性模块
    
    选择高梯度点计算视差，用于尺度校正
    """
    
    def __init__(self, num_points: int = 128, feat_dim: int = 256):
        super().__init__()
        self.num_points = num_points
        self.feat_dim = feat_dim
        
        # 点特征聚合
        self.point_aggregator = nn.Sequential(
            nn.Linear(feat_dim + 2, 128),  # feat + coords
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        
    def forward(
        self, 
        feat_l: torch.Tensor, 
        feat_r: torch.Tensor,
        depth_l: torch.Tensor,
        depth_r: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算稀疏点的视差
        
        Args:
            feat_l: [B, C, H, W] - 左视图特征
            feat_r: [B, C, H, W] - 右视图特征
            depth_l: [B, 1, H, W] - 左视图深度
            depth_r: [B, 1, H, W] - 右视图深度
            
        Returns:
            dict: {
                'disparities': [B, num_points] - 稀疏视差,
                'points': [B, num_points, 2] - 采样点坐标,
                'sparse_features': [B, num_points * 3] - 用于尺度回归的特征
            }
        """
        B, C, H, W = feat_l.shape
        device = feat_l.device
        
        # 1. 计算特征梯度幅度（用于选择高置信度点）
        grad_magnitude = self._compute_gradient_magnitude(feat_l)  # [B, 1, H, W]
        
        # 2. 选择 top-k 高梯度点
        grad_flat = grad_magnitude.flatten(2)  # [B, 1, HW]
        _, indices = grad_flat.topk(self.num_points, dim=-1)  # [B, 1, num_points]
        indices = indices.squeeze(1)  # [B, num_points]
        
        # 转换为坐标
        y_coords = indices // W
        x_coords = indices % W
        points = torch.stack([x_coords, y_coords], dim=-1).float()  # [B, num_points, 2]
        
        # 3. 在这些点上计算视差
        disparities = []
        confidences = []
        
        for b in range(B):
            disp_b = []
            conf_b = []
            
            for i in range(self.num_points):
                y, x = y_coords[b, i].item(), x_coords[b, i].item()
                
                # 获取左图特征向量
                feat_l_point = feat_l[b, :, int(y), int(x)]  # [C]
                
                # 在右图的同一行搜索最佳匹配 (立体约束: 水平搜索)
                # 搜索范围: 当前位置往左 (视差为正)
                search_range = min(int(x), W // 4)  # 限制搜索范围
                
                if search_range > 0:
                    # 获取右图搜索区域的特征
                    x_start = max(0, int(x) - search_range)
                    feat_r_row = feat_r[b, :, int(y), x_start:int(x)+1]  # [C, search_range+1]
                    
                    # 计算相关性
                    corr = (feat_r_row.T @ feat_l_point)  # [search_range+1]
                    
                    # 找最佳匹配
                    best_idx = corr.argmax()
                    best_x = x_start + best_idx.item()
                    disp = x - best_x
                    conf = corr[best_idx] / (corr.norm() + 1e-8)
                else:
                    disp = 0.0
                    conf = 0.0
                
                disp_b.append(disp)
                conf_b.append(conf)
            
            disparities.append(torch.tensor(disp_b, device=device))
            confidences.append(torch.tensor(conf_b, device=device))
        
        disparities = torch.stack(disparities)  # [B, num_points]
        confidences = torch.stack(confidences)  # [B, num_points]
        
        # 4. 获取对应点的深度值
        depth_values = []
        for b in range(B):
            depth_b = []
            for i in range(self.num_points):
                y, x = y_coords[b, i].item(), x_coords[b, i].item()
                depth_b.append(depth_l[b, 0, int(y), int(x)])
            depth_values.append(torch.stack(depth_b))
        depth_values = torch.stack(depth_values)  # [B, num_points]
        
        # 5. 组合稀疏特征 (用于尺度回归)
        sparse_features = torch.cat([
            disparities,  # [B, num_points]
            confidences,  # [B, num_points]
            depth_values,  # [B, num_points]
        ], dim=-1)  # [B, num_points * 3]
        
        return {
            'disparities': disparities,
            'confidences': confidences,
            'points': points,
            'sparse_features': sparse_features,
        }
    
    def _compute_gradient_magnitude(self, feat: torch.Tensor) -> torch.Tensor:
        """计算特征的梯度幅度"""
        # 使用平均特征
        feat_mean = feat.mean(dim=1, keepdim=True)  # [B, 1, H, W]
        
        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=feat.dtype, device=feat.device).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        
        grad_x = F.conv2d(feat_mean, sobel_x, padding=1)
        grad_y = F.conv2d(feat_mean, sobel_y, padding=1)
        
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return grad_magnitude


def create_depth_fusion_module(cfg) -> DepthFusionModule:
    """
    创建深度融合模块的工厂函数
    
    Args:
        cfg: 配置对象
        
    Returns:
        DepthFusionModule 实例
    """
    return DepthFusionModule(cfg)
