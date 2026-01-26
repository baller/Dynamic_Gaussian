"""
高斯参数回归网络

支持两种模式:
1. 单流模式 (GSRegresser): 原始的高斯参数回归
2. 双流模式 (DualStreamGSRegresser): MoE架构，背景和人体分别预测
"""

import torch
from torch import nn
import torch.nn.functional as F
from core.extractor import UnetExtractor, ResidualBlock


class GSRegresser(nn.Module):
    """
    单流高斯参数回归器（原始版本）
    """
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.rgb_dims = cfg.raft.encoder_dims
        self.depth_dims = cfg.gsnet.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)

        self.decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )

        self.decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )

        self.decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.out_conv = nn.Conv2d(48+rgb_dim+1, 32, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)

        self.rot_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
        
        self.scale_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Softplus(beta=1)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, img, depth, img_feat):
        img_feat1, img_feat2, img_feat3 = img_feat
        depth_feat1, depth_feat2, depth_feat3 = self.depth_encoder(depth)

        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)  # 96*2
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)  # 48*2
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)  # 32*2

        up3 = self.decoder3(feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))

        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)

        # scale head
        scale_out = torch.clamp_max(self.scale_head(out), 0.002) 
        # scale_out = torch.concat([scale_out, scale_out, scale_out], dim=1)

        # opacity head
        opacity_out = self.opacity_head(out)

        # rot head
        rot_out = self.rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)

        dep_out = self.depth_head(out)*0.5

        return rot_out, scale_out, opacity_out, dep_out


class ExpertModule(nn.Module):
    """
    单个专家模块
    
    包含解码器和输出头，用于预测高斯参数。
    """
    
    def __init__(self, rgb_dims, depth_dims, decoder_dims, head_dim, rgb_dim=3, norm_fn='group'):
        super().__init__()
        
        # 解码器
        self.decoder3 = nn.Sequential(
            ResidualBlock(rgb_dims[2]+depth_dims[2], decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(decoder_dims[2], decoder_dims[2], norm_fn=norm_fn)
        )
        self.decoder2 = nn.Sequential(
            ResidualBlock(rgb_dims[1]+depth_dims[1]+decoder_dims[2], decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(decoder_dims[1], decoder_dims[1], norm_fn=norm_fn)
        )
        self.decoder1 = nn.Sequential(
            ResidualBlock(rgb_dims[0]+depth_dims[0]+decoder_dims[1], decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(decoder_dims[0], decoder_dims[0], norm_fn=norm_fn)
        )
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        
        # 输出头
        self.out_conv = nn.Conv2d(decoder_dims[0]+rgb_dim+1, head_dim, kernel_size=3, padding=1)
        self.out_relu = nn.ReLU(inplace=True)
        
        self.rot_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 4, kernel_size=1),
        )
        self.scale_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 3, kernel_size=1),
            nn.Softplus(beta=1)
        )
        self.opacity_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.depth_head = nn.Sequential(
            nn.Conv2d(head_dim, head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_dim, 1, kernel_size=1),
            nn.Tanh()
        )
    
    def forward(self, img_feat, depth_feat, img, depth):
        """
        前向传播
        
        Args:
            img_feat: 图像特征元组 (feat1, feat2, feat3)
            depth_feat: 深度特征元组
            img: 原始图像 [B, 3, H, W]
            depth: 深度图 [B, 1, H, W]
            
        Returns:
            rot, scale, opacity, depth_residual
        """
        img_feat1, img_feat2, img_feat3 = img_feat
        depth_feat1, depth_feat2, depth_feat3 = depth_feat
        
        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)
        
        up3 = self.decoder3(feat3)
        up3 = self.up(up3)
        up2 = self.decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = self.decoder1(torch.cat([up2, feat1], dim=1))
        
        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = self.out_conv(out)
        out = self.out_relu(out)
        
        # 预测参数
        scale_out = torch.clamp_max(self.scale_head(out), 0.002)
        opacity_out = self.opacity_head(out)
        rot_out = self.rot_head(out)
        rot_out = F.normalize(rot_out, dim=1)
        dep_out = self.depth_head(out) * 0.5
        
        return rot_out, scale_out, opacity_out, dep_out


class MoEGSRegresser(nn.Module):
    """
    MoE高斯参数回归器 - 支持任意数量专家 + 共享专家
    
    架构:
    - 共享专家: 所有输入都会经过，输出与路由专家融合
    - 路由专家: 根据路由权重加权融合
    - 总专家数 = num_experts (路由) + 1 (共享，如果启用)
    
    Args:
        cfg: 配置对象
        num_experts: 路由专家数量（不包括共享专家）
        use_shared_expert: 是否使用共享专家
        rgb_dim: RGB通道数
        depth_dim: 深度通道数
        norm_fn: 归一化函数类型
    """
    
    def __init__(self, cfg, num_experts=2, use_shared_expert=True, 
                 rgb_dim=3, depth_dim=1, norm_fn='group'):
        super().__init__()
        self.cfg = cfg
        self.num_experts = num_experts  # 路由专家数量
        self.use_shared_expert = use_shared_expert
        self.total_experts = num_experts + 1 if use_shared_expert else num_experts
        
        self.rgb_dims = cfg.raft.encoder_dims
        self.depth_dims = cfg.gsnet.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim
        
        # 共享深度编码器
        self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)
        
        # 共享专家
        if use_shared_expert:
            self.shared_expert = ExpertModule(
                self.rgb_dims, self.depth_dims, self.decoder_dims, 
                self.head_dim, rgb_dim, norm_fn
            )
        
        # 路由专家列表
        self.routed_experts = nn.ModuleList([
            ExpertModule(
                self.rgb_dims, self.depth_dims, self.decoder_dims,
                self.head_dim, rgb_dim, norm_fn
            ) for _ in range(num_experts)
        ])
        
        # 专家名称（用于可视化）
        self.expert_names = [f'expert_{i}' for i in range(num_experts)]
        if use_shared_expert:
            self.expert_names.append('shared')
    
    def forward(self, img, depth, img_feat, router_weights=None, expert_info=None):
        """
        前向传播
        
        Args:
            img: 输入图像 [B, 3, H, W]
            depth: 深度图 [B, 1, H, W]
            img_feat: 图像特征元组 (feat1, feat2, feat3)
            router_weights: 路由权重 [B, num_experts(+1 if shared), H, W]
            expert_info: 专家信息字典
            
        Returns:
            expert_params_list: 各专家参数列表
            fused_params: 融合后的参数字典
        """
        # 编码深度特征
        depth_feat = self.depth_encoder(depth)
        
        expert_params_list = []
        
        # 计算路由专家输出
        for i, expert in enumerate(self.routed_experts):
            if router_weights is not None:
                # 获取该专家的权重
                weight_i = router_weights[:, i:i+1]
                
                # 对特征应用路由权重
                img_feat1, img_feat2, img_feat3 = img_feat
                weight_1 = F.interpolate(weight_i, size=img_feat1.shape[2:], mode='bilinear', align_corners=False)
                weight_2 = F.interpolate(weight_i, size=img_feat2.shape[2:], mode='bilinear', align_corners=False)
                weight_3 = F.interpolate(weight_i, size=img_feat3.shape[2:], mode='bilinear', align_corners=False)
                
                weighted_img_feat = (
                    img_feat1 * weight_1,
                    img_feat2 * weight_2,
                    img_feat3 * weight_3
                )
            else:
                weighted_img_feat = img_feat
            
            rot, scale, opacity, dep = expert(weighted_img_feat, depth_feat, img, depth)
            expert_params_list.append({
                'rot_maps': rot,
                'scale_maps': scale,
                'opacity_maps': opacity,
                'depth_residual': dep,
                'name': self.expert_names[i]
            })
        
        # 计算共享专家输出
        if self.use_shared_expert:
            rot, scale, opacity, dep = self.shared_expert(img_feat, depth_feat, img, depth)
            shared_params = {
                'rot_maps': rot,
                'scale_maps': scale,
                'opacity_maps': opacity,
                'depth_residual': dep,
                'name': 'shared'
            }
            expert_params_list.append(shared_params)
        
        # 融合所有专家输出
        fused_params = self._fuse_experts(expert_params_list, router_weights, expert_info)
        
        return expert_params_list, fused_params
    
    def _fuse_experts(self, expert_params_list, router_weights, expert_info):
        """
        融合专家输出
        
        Args:
            expert_params_list: 专家参数列表
            router_weights: 路由权重 [B, total_experts, H, W]
            expert_info: 专家信息
            
        Returns:
            fused_params: 融合后的参数字典
        """
        if router_weights is None:
            # 无路由权重时，简单平均
            n = len(expert_params_list)
            fused_rot = sum(p['rot_maps'] for p in expert_params_list) / n
            fused_scale = sum(p['scale_maps'] for p in expert_params_list) / n
            fused_opacity = sum(p['opacity_maps'] for p in expert_params_list) / n
            fused_depth = sum(p['depth_residual'] for p in expert_params_list) / n
        else:
            # 使用路由权重加权融合
            fused_rot = None
            fused_scale = None
            fused_opacity = None
            fused_depth = None
            
            for i, params in enumerate(expert_params_list):
                weight = router_weights[:, i:i+1]  # [B, 1, H, W]
                
                if fused_rot is None:
                    fused_rot = params['rot_maps'] * weight
                    fused_scale = params['scale_maps'] * weight
                    fused_opacity = params['opacity_maps'] * weight
                    fused_depth = params['depth_residual'] * weight
                else:
                    fused_rot = fused_rot + params['rot_maps'] * weight
                    fused_scale = fused_scale + params['scale_maps'] * weight
                    fused_opacity = fused_opacity + params['opacity_maps'] * weight
                    fused_depth = fused_depth + params['depth_residual'] * weight
        
        # 归一化旋转
        fused_rot = F.normalize(fused_rot, dim=1)
        
        return {
            'rot_maps': fused_rot,
            'scale_maps': fused_scale,
            'opacity_maps': fused_opacity,
            'depth_residual': fused_depth
        }
    
    def forward_with_routing(self, img, depth, img_feat, router_weights, expert_info=None):
        """
        带路由的前向传播（MoE模式）
        
        Args:
            img: 输入图像 [B, 3, H, W]
            depth: 深度图 [B, 1, H, W]
            img_feat: 图像特征元组
            router_weights: 路由权重 [B, total_experts, H, W]
            expert_info: 专家信息字典
            
        Returns:
            expert_params_list: 各专家参数列表
            fused_params: 融合后的参数字典
        """
        return self.forward(img, depth, img_feat, router_weights, expert_info)


# 保留旧版本的DualStreamGSRegresser用于向后兼容
class DualStreamGSRegresser(MoEGSRegresser):
    """
    双流高斯参数回归器 - MoE架构（兼容旧版本）
    
    这是MoEGSRegresser的特化版本，固定为2个路由专家（背景+人体），无共享专家。
    
    Args:
        cfg: 配置对象
        rgb_dim: RGB通道数
        depth_dim: 深度通道数
        norm_fn: 归一化函数类型
        share_depth_encoder: 是否共享深度编码器（此参数已弃用）
    """
    
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group', share_depth_encoder=True):
        # 读取MoE配置
        moe_cfg = getattr(cfg, 'moe', None)
        if moe_cfg is not None:
            num_experts = getattr(moe_cfg, 'num_experts', 2)
            use_shared = getattr(moe_cfg, 'use_shared_expert', False)
        else:
            num_experts = 2
            use_shared = False
        
        super().__init__(
            cfg, 
            num_experts=num_experts, 
            use_shared_expert=use_shared,
            rgb_dim=rgb_dim, 
            depth_dim=depth_dim, 
            norm_fn=norm_fn
        )
        
        # 设置专家名称（兼容旧代码）
        if num_experts >= 2:
            self.expert_names[0] = 'bg'
            self.expert_names[1] = 'human'
    
    def forward_with_routing(self, img, depth, img_feat, router_weights, expert_info=None):
        """
        带路由的前向传播
        
        Returns:
            expert_params_list: 各专家参数列表（与MoEGSRegresser一致）
            fused_params: 融合后的参数字典
        """
        # 直接调用父类方法，返回一致的格式
        return super().forward_with_routing(
            img, depth, img_feat, router_weights, expert_info
        )


def create_gs_regresser(cfg, rgb_dim=3, depth_dim=1, norm_fn='group'):
    """
    创建高斯参数回归器
    
    Args:
        cfg: 配置对象
        rgb_dim: RGB通道数
        depth_dim: 深度通道数
        norm_fn: 归一化函数类型
        
    Returns:
        regresser: GSRegresser或DualStreamGSRegresser实例
    """
    moe_cfg = getattr(cfg, 'moe', None)
    
    if moe_cfg is not None and getattr(moe_cfg, 'enabled', False):
        share_encoder = getattr(moe_cfg, 'share_depth_encoder', True)
        return DualStreamGSRegresser(
            cfg, rgb_dim=rgb_dim, depth_dim=depth_dim,
            norm_fn=norm_fn, share_depth_encoder=share_encoder
        )
    else:
        return GSRegresser(cfg, rgb_dim=rgb_dim, depth_dim=depth_dim, norm_fn=norm_fn)
