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


class DualStreamGSRegresser(nn.Module):
    """
    双流高斯参数回归器 - MoE架构
    
    分别为背景和人体预测高斯参数，通过路由权重加权融合特征。
    
    Args:
        cfg: 配置对象
        rgb_dim: RGB通道数
        depth_dim: 深度通道数
        norm_fn: 归一化函数类型
        share_depth_encoder: 是否共享深度编码器
    """
    
    def __init__(self, cfg, rgb_dim=3, depth_dim=1, norm_fn='group', share_depth_encoder=True):
        super().__init__()
        self.cfg = cfg
        self.share_depth_encoder = share_depth_encoder
        
        self.rgb_dims = cfg.raft.encoder_dims
        self.depth_dims = cfg.gsnet.encoder_dims
        self.decoder_dims = cfg.gsnet.decoder_dims
        self.head_dim = cfg.gsnet.parm_head_dim
        
        # 共享深度编码器
        if share_depth_encoder:
            self.depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)
        else:
            self.bg_depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)
            self.human_depth_encoder = UnetExtractor(in_channel=depth_dim, encoder_dim=self.depth_dims)
        
        # 背景专家解码器
        self.bg_decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )
        self.bg_decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )
        self.bg_decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        
        # 人体专家解码器
        self.human_decoder3 = nn.Sequential(
            ResidualBlock(self.rgb_dims[2]+self.depth_dims[2], self.decoder_dims[2], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[2], self.decoder_dims[2], norm_fn=norm_fn)
        )
        self.human_decoder2 = nn.Sequential(
            ResidualBlock(self.rgb_dims[1]+self.depth_dims[1]+self.decoder_dims[2], self.decoder_dims[1], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[1], self.decoder_dims[1], norm_fn=norm_fn)
        )
        self.human_decoder1 = nn.Sequential(
            ResidualBlock(self.rgb_dims[0]+self.depth_dims[0]+self.decoder_dims[1], self.decoder_dims[0], norm_fn=norm_fn),
            ResidualBlock(self.decoder_dims[0], self.decoder_dims[0], norm_fn=norm_fn)
        )
        
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        
        # 背景输出头
        self.bg_out_conv = nn.Conv2d(48+rgb_dim+1, 32, kernel_size=3, padding=1)
        self.bg_out_relu = nn.ReLU(inplace=True)
        self.bg_rot_head = self._make_rot_head()
        self.bg_scale_head = self._make_scale_head()
        self.bg_opacity_head = self._make_opacity_head()
        self.bg_depth_head = self._make_depth_head()
        
        # 人体输出头
        self.human_out_conv = nn.Conv2d(48+rgb_dim+1, 32, kernel_size=3, padding=1)
        self.human_out_relu = nn.ReLU(inplace=True)
        self.human_rot_head = self._make_rot_head()
        self.human_scale_head = self._make_scale_head()
        self.human_opacity_head = self._make_opacity_head()
        self.human_depth_head = self._make_depth_head()
        
    def _make_rot_head(self):
        return nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 4, kernel_size=1),
        )
    
    def _make_scale_head(self):
        return nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 3, kernel_size=1),
            nn.Softplus(beta=1)
        )
    
    def _make_opacity_head(self):
        return nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
    
    def _make_depth_head(self):
        return nn.Sequential(
            nn.Conv2d(self.head_dim, self.head_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.head_dim, 1, kernel_size=1),
            nn.Tanh()
        )
    
    def _decode_stream(self, img_feat, depth_feat, img, depth, decoder3, decoder2, decoder1, out_conv, out_relu):
        """解码单个流"""
        img_feat1, img_feat2, img_feat3 = img_feat
        depth_feat1, depth_feat2, depth_feat3 = depth_feat
        
        feat3 = torch.concat([img_feat3, depth_feat3], dim=1)
        feat2 = torch.concat([img_feat2, depth_feat2], dim=1)
        feat1 = torch.concat([img_feat1, depth_feat1], dim=1)
        
        up3 = decoder3(feat3)
        up3 = self.up(up3)
        up2 = decoder2(torch.cat([up3, feat2], dim=1))
        up2 = self.up(up2)
        up1 = decoder1(torch.cat([up2, feat1], dim=1))
        
        up1 = self.up(up1)
        out = torch.cat([up1, img, depth], dim=1)
        out = out_conv(out)
        out = out_relu(out)
        
        return out
    
    def _predict_params(self, out, rot_head, scale_head, opacity_head, depth_head):
        """预测高斯参数"""
        scale_out = torch.clamp_max(scale_head(out), 0.002)
        opacity_out = opacity_head(out)
        rot_out = rot_head(out)
        rot_out = torch.nn.functional.normalize(rot_out, dim=1)
        dep_out = depth_head(out) * 0.5
        
        return rot_out, scale_out, opacity_out, dep_out
    
    def forward(self, img, depth, img_feat, router_weights=None):
        """
        前向传播
        
        Args:
            img: 输入图像 [B, 3, H, W]
            depth: 深度图 [B, 1, H, W]
            img_feat: 图像特征元组 (feat1, feat2, feat3)
            router_weights: 路由权重 [B, 2, H, W]，可选
                           channel 0: 背景权重
                           channel 1: 人体权重
                           
        Returns:
            如果router_weights为None，返回融合结果:
                rot_out, scale_out, opacity_out, dep_out
            否则返回分离结果:
                bg_params: (rot, scale, opacity, depth) 背景参数
                human_params: (rot, scale, opacity, depth) 人体参数
                router_weights: 路由权重
        """
        # 编码深度特征
        if self.share_depth_encoder:
            depth_feat = self.depth_encoder(depth)
            bg_depth_feat = depth_feat
            human_depth_feat = depth_feat
        else:
            bg_depth_feat = self.bg_depth_encoder(depth)
            human_depth_feat = self.human_depth_encoder(depth)
        
        # 如果有路由权重，对特征进行加权
        if router_weights is not None:
            # 下采样路由权重到各尺度
            img_feat1, img_feat2, img_feat3 = img_feat
            
            # 对各尺度特征应用路由权重
            bg_weight_1 = F.interpolate(router_weights[:, 0:1], size=img_feat1.shape[2:], mode='bilinear', align_corners=False)
            bg_weight_2 = F.interpolate(router_weights[:, 0:1], size=img_feat2.shape[2:], mode='bilinear', align_corners=False)
            bg_weight_3 = F.interpolate(router_weights[:, 0:1], size=img_feat3.shape[2:], mode='bilinear', align_corners=False)
            
            human_weight_1 = F.interpolate(router_weights[:, 1:2], size=img_feat1.shape[2:], mode='bilinear', align_corners=False)
            human_weight_2 = F.interpolate(router_weights[:, 1:2], size=img_feat2.shape[2:], mode='bilinear', align_corners=False)
            human_weight_3 = F.interpolate(router_weights[:, 1:2], size=img_feat3.shape[2:], mode='bilinear', align_corners=False)
            
            bg_img_feat = (img_feat1 * bg_weight_1, img_feat2 * bg_weight_2, img_feat3 * bg_weight_3)
            human_img_feat = (img_feat1 * human_weight_1, img_feat2 * human_weight_2, img_feat3 * human_weight_3)
        else:
            bg_img_feat = img_feat
            human_img_feat = img_feat
        
        # 背景流解码
        bg_out = self._decode_stream(
            bg_img_feat, bg_depth_feat, img, depth,
            self.bg_decoder3, self.bg_decoder2, self.bg_decoder1,
            self.bg_out_conv, self.bg_out_relu
        )
        bg_rot, bg_scale, bg_opacity, bg_depth = self._predict_params(
            bg_out, self.bg_rot_head, self.bg_scale_head, self.bg_opacity_head, self.bg_depth_head
        )
        
        # 人体流解码
        human_out = self._decode_stream(
            human_img_feat, human_depth_feat, img, depth,
            self.human_decoder3, self.human_decoder2, self.human_decoder1,
            self.human_out_conv, self.human_out_relu
        )
        human_rot, human_scale, human_opacity, human_depth = self._predict_params(
            human_out, self.human_rot_head, self.human_scale_head, self.human_opacity_head, self.human_depth_head
        )
        
        if router_weights is not None:
            # 返回分离的参数
            bg_params = (bg_rot, bg_scale, bg_opacity, bg_depth)
            human_params = (human_rot, human_scale, human_opacity, human_depth)
            return bg_params, human_params, router_weights
        else:
            # 简单平均融合（无路由权重时的备用方案）
            rot_out = (bg_rot + human_rot) / 2
            scale_out = (bg_scale + human_scale) / 2
            opacity_out = (bg_opacity + human_opacity) / 2
            dep_out = (bg_depth + human_depth) / 2
            return rot_out, scale_out, opacity_out, dep_out
    
    def forward_with_routing(self, img, depth, img_feat, router_weights):
        """
        带路由的前向传播（显式调用，用于MoE模式）
        
        Args:
            img: 输入图像 [B, 3, H, W]
            depth: 深度图 [B, 1, H, W]
            img_feat: 图像特征元组
            router_weights: 路由权重 [B, 2, H, W]
            
        Returns:
            bg_params: 背景高斯参数字典
            human_params: 人体高斯参数字典
            fused_params: 融合后的高斯参数字典
        """
        bg_params, human_params, _ = self.forward(img, depth, img_feat, router_weights)
        
        # 使用路由权重融合参数
        bg_weight = router_weights[:, 0:1]
        human_weight = router_weights[:, 1:2]
        
        bg_rot, bg_scale, bg_opacity, bg_depth = bg_params
        human_rot, human_scale, human_opacity, human_depth = human_params
        
        # 加权融合
        fused_rot = bg_rot * bg_weight + human_rot * human_weight
        fused_rot = F.normalize(fused_rot, dim=1)
        fused_scale = bg_scale * bg_weight + human_scale * human_weight
        fused_opacity = bg_opacity * bg_weight + human_opacity * human_weight
        fused_depth = bg_depth * bg_weight + human_depth * human_weight
        
        bg_params_dict = {
            'rot_maps': bg_rot,
            'scale_maps': bg_scale,
            'opacity_maps': bg_opacity,
            'depth_residual': bg_depth
        }
        
        human_params_dict = {
            'rot_maps': human_rot,
            'scale_maps': human_scale,
            'opacity_maps': human_opacity,
            'depth_residual': human_depth
        }
        
        fused_params_dict = {
            'rot_maps': fused_rot,
            'scale_maps': fused_scale,
            'opacity_maps': fused_opacity,
            'depth_residual': fused_depth
        }
        
        return bg_params_dict, human_params_dict, fused_params_dict


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
