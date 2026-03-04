"""
NovelViewDepthModel: Novel View Depth Fusion 完整管线

Phase 1: DA3 深度估计 (冻结, 左右 + 训练时 GT)
Phase 2: 简单 scale 对齐
Phase 3: forward_warp 深度 + DepthFusionNet
Phase 4: depth2pc + geometry_guided_sample
Phase 5: SingleViewGSTransformer 预测高斯参数

输入:  data['lmain'], data['rmain'], data['novel_view']
输出:  data['novel_view'] 中填充高斯属性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from lib.da3_depth import DA3DepthEstimator
from lib.depth_fusion import simple_scale_align, forward_warp_depth, DepthFusionNet
from lib.geometry_sample import geometry_guided_sample
from lib.gs_transformer_single import SingleViewGSTransformer
from lib.utils import depth2pc

logger = logging.getLogger(__name__)


class NovelViewDepthModel(nn.Module):
    """
    Novel View Depth Fusion + Gaussian Splatting 完整管线

    可训练模块: DepthFusionNet, SingleViewGSTransformer
    冻结模块: DA3DepthEstimator
    """

    def __init__(self, cfg, device='cuda'):
        super().__init__()
        self.cfg = cfg
        self.device_target = device

        # 配置读取
        self.inverse_depth_init = getattr(cfg.dataset, 'inverse_depth_init', 0.3)
        depth_valid_min = getattr(cfg.gsnet, 'depth_valid_min', 0.01)
        depth_valid_max = getattr(cfg.gsnet, 'depth_valid_max', 10.0)
        self.depth_valid_min = depth_valid_min
        self.depth_valid_max = depth_valid_max

        # Phase 1: DA3 深度估计 (冻结)
        self.da3_estimator = DA3DepthEstimator(cfg, device=device)

        # Phase 3: 深度融合网络 (可训练)
        self.depth_fusion_net = DepthFusionNet(cfg)

        # Phase 5: 单视图 GSTransformer (可训练)
        self.gs_transformer = SingleViewGSTransformer(
            cfg,
            in_channels=7,  # sampled_rgb_L(3) + sampled_rgb_R(3) + depth_novel(1)
            embed_dim=getattr(cfg.gsnet, 'transformer_embed_dim', 768),
            depth=getattr(cfg.gsnet, 'transformer_depth', 8),
            num_heads=getattr(cfg.gsnet, 'transformer_num_heads', 12),
            patch_size=getattr(cfg.gsnet, 'transformer_patch_size', 14),
            mlp_ratio=getattr(cfg.gsnet, 'transformer_mlp_ratio', 4.0),
            use_checkpoint=getattr(cfg.gsnet, 'transformer_use_checkpoint', False),
        )

    def _normalize_for_da3(self, image):
        """将图像归一化到 DA3 期望的 [0, 1] 范围"""
        img_min = image.min().item()
        if img_min < 0:
            image = (image + 1.0) / 2.0
        return image.clamp(0, 1)

    def forward(self, data, is_train=True):
        """
        完整管线前向传播

        Args:
            data: dict 包含 'lmain', 'rmain', 'novel_view'
            is_train: 是否训练模式

        Returns:
            data: 填充了高斯属性的字典
            depth_loss_info: dict 包含深度监督信息
        """
        bs = data['lmain']['img'].shape[0]
        _, _, H, W = data['lmain']['img'].shape

        # ====== Phase 1: DA3 深度估计 ======
        l_img_01 = self._normalize_for_da3(data['lmain']['img'])
        r_img_01 = self._normalize_for_da3(data['rmain']['img'])

        with torch.no_grad():
            da3_out_L = self.da3_estimator(l_img_01, return_features=False, return_entropy=False)
            da3_out_R = self.da3_estimator(r_img_01, return_features=False, return_entropy=False)

        rel_depth_L = da3_out_L['depth']  # [B, 1, H, W] 相对深度
        rel_depth_R = da3_out_R['depth']

        # 训练时: 对 GT novel image 也用 DA3 生成 pseudo GT depth
        depth_novel_GT_aligned = None
        if is_train and 'img' in data['novel_view']:
            nv_img_01 = self._normalize_for_da3(data['novel_view']['img'])
            with torch.no_grad():
                da3_out_GT = self.da3_estimator(nv_img_01, return_features=False, return_entropy=False)
            rel_depth_GT = da3_out_GT['depth']  # [B, 1, H, W]

        # ====== Phase 2: 简单 Scale 对齐 ======
        mask_L = data['lmain'].get('mask', None)
        mask_R = data['rmain'].get('mask', None)

        aligned_L, aligned_R, scale_L, scale_R = simple_scale_align(
            rel_depth_L, rel_depth_R,
            self.inverse_depth_init,
            mask_L=mask_L, mask_R=mask_R
        )

        # 对 GT 深度也做同样的 scale 对齐
        if is_train and 'img' in data['novel_view']:
            avg_scale = (scale_L + scale_R) / 2.0  # [B]
            depth_novel_GT_aligned = rel_depth_GT * avg_scale[:, None, None, None]

        # ====== Phase 3: Forward Warp + DepthFusionNet ======
        intr_L = data['lmain']['intr']
        extr_L = data['lmain']['extr']
        if extr_L.shape[1] == 4:
            extr_L = extr_L[:, :3, :]
        intr_R = data['rmain']['intr']
        extr_R = data['rmain']['extr']
        if extr_R.shape[1] == 4:
            extr_R = extr_R[:, :3, :]
        intr_novel = data['novel_view']['intr']
        extr_novel = data['novel_view']['extr']
        if extr_novel.shape[1] == 4:
            extr_novel = extr_novel[:, :3, :]

        warped_L, conf_L = forward_warp_depth(aligned_L, intr_L, extr_L, intr_novel, extr_novel)
        warped_R, conf_R = forward_warp_depth(aligned_R, intr_R, extr_R, intr_novel, extr_novel)

        depth_novel = self.depth_fusion_net(warped_L, conf_L, warped_R, conf_R)
        # depth_novel: [B, 1, H, W] dense metric-like 深度

        # ====== Phase 4: depth2pc + geometry_guided_sample ======
        # depth2pc 期望逆深度输入: inv_depth = 1.0 / Z
        inv_depth_novel = 1.0 / (depth_novel + 1e-8)
        xyz_novel = depth2pc(inv_depth_novel, extr_novel, intr_novel, return_2d=True)
        # xyz_novel: [B, 3, H, W] 世界坐标

        # 从左右视图采样 RGB (不采样 DA3 features, 节省显存)
        sampled_rgb_L, _, valid_L = geometry_guided_sample(
            xyz_novel, data['lmain']['img'], None, intr_L, extr_L, H, W
        )
        sampled_rgb_R, _, valid_R = geometry_guided_sample(
            xyz_novel, data['rmain']['img'], None, intr_R, extr_R, H, W
        )

        # 加权混合 source RGB 作为 base color ([-1,1] 范围)
        valid_sum = valid_L + valid_R + 1e-8
        base_color = (sampled_rgb_L * valid_L + sampled_rgb_R * valid_R) / valid_sum

        # ====== Phase 5: SingleViewGSTransformer ======
        depth_normalized = depth_novel / (depth_novel.max() + 1e-8)
        gs_input = torch.cat([sampled_rgb_L, sampled_rgb_R, depth_normalized], dim=1)

        rot_maps, scale_maps, opacity_maps, xyz_res, color_maps = self.gs_transformer(gs_input)

        # xyz = xyz_novel + xyz_res
        xyz_final = xyz_novel + xyz_res  # [B, 3, H, W]

        # ====== 填充 data['novel_view'] ======
        data['novel_view']['xyz'] = xyz_final.view(bs, 3, -1).permute(0, 2, 1)  # [B, H*W, 3]
        data['novel_view']['rot_maps'] = rot_maps
        data['novel_view']['scale_maps'] = scale_maps
        data['novel_view']['opacity_maps'] = opacity_maps
        data['novel_view']['color_maps'] = color_maps
        data['novel_view']['base_color'] = base_color

        # 深度有效性
        inv_depth_flat = inv_depth_novel.view(bs, -1)
        pts_valid = (inv_depth_flat > self.depth_valid_min) & (inv_depth_flat < self.depth_valid_max)
        data['novel_view']['pts_valid'] = pts_valid  # [B, H*W]

        # 正则化量
        data['novel_view']['xyz_res_regular'] = torch.mean(xyz_res.abs())
        data['novel_view']['scale_regular'] = torch.mean(scale_maps)

        # 深度监督 + 可视化信息
        depth_loss_info = {
            'depth_novel': depth_novel,
            'depth_novel_GT_aligned': depth_novel_GT_aligned,
            'warped_depth_L': warped_L,
            'warped_depth_R': warped_R,
            'conf_L': conf_L,
            'conf_R': conf_R,
            'sampled_rgb_L': sampled_rgb_L,
            'sampled_rgb_R': sampled_rgb_R,
        }

        return data, depth_loss_info

    def freeze_da3(self):
        for param in self.da3_estimator.parameters():
            param.requires_grad = False

    def get_trainable_params(self):
        """返回所有可训练参数 (排除冻结的 DA3)"""
        params = []
        params += list(self.depth_fusion_net.parameters())
        params += list(self.gs_transformer.parameters())
        return params
