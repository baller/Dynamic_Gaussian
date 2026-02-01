
import torch
import torch.nn.functional as F
from torch import nn
from core.raft_stereo_human import RAFTStereoHuman
from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser
from lib.loss import sequence_loss
from lib.utils import flow2depth, depth2pc
from lib.embedder import get_embedder
from lib.attention_module import LocalFeatureTransformer
from torch.cuda.amp import autocast as autocast

# Import DA3 depth estimator (正确的实现)
try:
    from lib.da3_depth import DA3DepthEstimator, DA3FeatureAdapter, create_da3_estimator
    DAV3_AVAILABLE = True
except ImportError:
    DAV3_AVAILABLE = False
    print("Warning: DA3 depth estimator not available")


class RtStereoHumanModel(nn.Module):
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        self.use_flow_init = self.cfg.dataset.use_depth_init
        self.train_iters = self.cfg.raft.train_iters
        self.val_iters = self.cfg.raft.val_iters

        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        
        self.loftr_coarse = LocalFeatureTransformer()
        
        self.raft_stereo = RAFTStereoHuman(self.cfg.raft)
        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)

    def forward(self, data, is_train=True):
        bs = data['lmain']['img'].shape[0]

        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)

        flow_init = torch.cat([data['lmain']['flow_init'], data['rmain']['flow_init']], dim=0) if self.use_flow_init else None

        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)

        (feat_c0, feat_c1) = img_feat[2].split(bs)
        
        mask_c0 = mask_c1 = None  

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_cs = torch.cat((feat_c0, feat_c1), 0)
        
        #img_feat[2] = feat_cs 
        img_feat = img_feat[0], img_feat[1], feat_cs
        
        if is_train:
            flow_predictions = self.raft_stereo(feat_cs, flow_init=flow_init, iters=self.train_iters)
            # flow_loss, metrics = sequence_loss(flow_predictions, flow, valid)
            flow_loss = None
            metrics = {}
            flow_pred_lmain, flow_pred_rmain = torch.split(flow_predictions[-1], [bs, bs])

            if not self.with_gs_render:
                data['lmain']['flow_pred'] = flow_pred_lmain.detach()
                data['rmain']['flow_pred'] = flow_pred_rmain.detach()
                return data, flow_loss, metrics

            data['lmain']['flow_pred'] = flow_pred_lmain
            data['rmain']['flow_pred'] = flow_pred_rmain
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics

        else:
            flow_up = self.raft_stereo(feat_cs, flow_init=flow_init, iters=self.val_iters, test_mode=True)
            flow_loss, metrics = None, None

            data['lmain']['flow_pred'] = flow_up[0]
            data['rmain']['flow_pred'] = flow_up[1]

            if not self.with_gs_render:
                return data, flow_loss, metrics
            data = self.flow2gsparms(image, img_feat, data, bs)

            return data, flow_loss, metrics

    def flow2gsparms(self, lr_img, lr_img_feat, data, bs):
        for view in ['lmain', 'rmain']:
            data[view]['depth'] = flow2depth(data[view])
            
        l_depth = data['lmain']['depth']  
        r_depth = data['rmain']['depth'] 
        lr_depth = torch.concat([l_depth, r_depth], dim=0)
        
        # regress gaussian parms
        rot_maps, scale_maps, opacity_maps, depth_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)
        l_resdepth, r_resdepth =  torch.split(depth_maps, [bs, bs])
        # depth input

        data['lmain']['depth'] += l_resdepth
        data['rmain']['depth'] += r_resdepth

        cut_m = 0
        for view in ['lmain', 'rmain']:
            data[view]['xyz'] = depth2pc(data[view]['depth'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)  # [B, S*S, 3]
    
            valid = data[view]['mask'][:,:1,:,:] > 0.5  # [B, 1, S, S]
            data[view]['pts_valid'] = valid.view(bs, -1)  # [B, S*S]



        data['novel_view']['scale_regular'] = torch.mean(scale_maps)

        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])

        return data


class DAV3StereoHumanModel(nn.Module):
    """
    Stereo human model using Depth Anything V3 for depth estimation.
    
    使用正确的 DA3 API (da3_depth.py):
    1. DA3 提供单目相对深度
    2. 基于基线的简单尺度对齐
    3. GSRegresser 回归高斯参数
    """
    
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        
        if not DAV3_AVAILABLE:
            raise ImportError("DA3 depth estimator is required but not available. "
                            "Please ensure Depth-Anything-3 is installed.")
        
        # DA3 深度估计器
        self.da3_estimator = DA3DepthEstimator(cfg)
        
        # Image encoder for GSRegresser compatibility
        self.img_encoder = UnetExtractor(
            in_channel=3, 
            encoder_dim=self.cfg.raft.encoder_dims
        )
        
        # Gaussian parameter regresser
        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)
    
    def _normalize_for_da3(self, image: torch.Tensor) -> torch.Tensor:
        """
        将输入图像归一化到 DA3 期望的 [0, 1] 范围
        """
        img_range = getattr(self.cfg.dataset, 'img_range', None)
        if img_range is not None and len(img_range) == 2:
            img_min, img_max = img_range
            image = (image - img_min) / (img_max - img_min)
        else:
            # 假设输入是 [-1, 1]
            img_min = image.min().item()
            if img_min < 0:
                image = (image + 1.0) / 2.0
        return image.clamp(0, 1)
    
    def _compute_baseline(self, extr_l: torch.Tensor, extr_r: torch.Tensor) -> torch.Tensor:
        """计算立体基线距离"""
        if extr_l.shape[1] == 4:
            t_l = extr_l[:, :3, 3]
            t_r = extr_r[:, :3, 3]
        else:
            t_l = extr_l[:, :, 3]
            t_r = extr_r[:, :, 3]
        return torch.norm(t_l - t_r, dim=1)
    
    def _align_depth(
        self, 
        depth: torch.Tensor, 
        mask: torch.Tensor,
        intrinsics: torch.Tensor,
        baseline: torch.Tensor,
        img_width: int
    ) -> torch.Tensor:
        """将相对深度对齐到绝对尺度"""
        B = depth.shape[0]
        device = depth.device
        eps = 1e-6
        
        # 获取深度对齐配置
        depth_align_cfg = getattr(self.cfg, 'depth_align', None)
        disp_min_ratio = getattr(depth_align_cfg, 'typical_disp_min_ratio', 0.01) if depth_align_cfg else 0.01
        disp_max_ratio = getattr(depth_align_cfg, 'typical_disp_max_ratio', 0.15) if depth_align_cfg else 0.15
        min_depth = getattr(depth_align_cfg, 'min_depth', 0.1) if depth_align_cfg else 0.1
        
        valid_mask = (mask[:, :1] > 0.5).float()
        aligned_depths = []
        
        for b in range(B):
            fx = intrinsics[b, 0, 0]
            bl = baseline[b]
            
            # 估计目标深度范围
            typical_disp_min = img_width * disp_min_ratio
            typical_disp_max = img_width * disp_max_ratio
            
            depth_far = bl * fx / (typical_disp_min + eps)
            depth_near = bl * fx / (typical_disp_max + eps)
            
            target_median = (depth_near + depth_far) / 2
            target_range = (depth_far - depth_near) / 2
            
            # 计算相对深度统计量
            valid_depth = depth[b] * valid_mask[b]
            valid_flat = valid_depth[valid_mask[b] > 0.5]
            
            if valid_flat.numel() > 0:
                median = valid_flat.median()
                mad = (valid_flat - median).abs().median() + eps
            else:
                median = torch.tensor(0.5, device=device)
                mad = torch.tensor(0.25, device=device)
            
            # 对齐
            depth_norm = (depth[b] - median) / mad
            depth_aligned = depth_norm * target_range * 0.5 + target_median
            aligned_depths.append(depth_aligned)
        
        result = torch.stack(aligned_depths, dim=0)
        return result.clamp(min=min_depth)
    
    def forward(self, data, is_train=True):
        """前向传播"""
        bs = data['lmain']['img'].shape[0]
        device = data['lmain']['img'].device
        _, _, H, W = data['lmain']['img'].shape
        
        # 合并左右视图图像
        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
        
        # 归一化到 [0, 1]
        image_da3 = self._normalize_for_da3(image)
        
        # DA3 推理
        da3_output = self.da3_estimator(image_da3, return_features=True, return_entropy=False)
        
        # 获取深度
        depth = da3_output['depth']  # [2B, 1, H, W]
        l_depth_rel, r_depth_rel = torch.split(depth, [bs, bs])
        
        # 存储相对深度
        data['lmain']['depth_relative'] = l_depth_rel
        data['rmain']['depth_relative'] = r_depth_rel
        
        # 计算基线
        baseline = self._compute_baseline(data['lmain']['extr'], data['rmain']['extr'])
        
        # 深度对齐
        l_depth = self._align_depth(
            l_depth_rel, data['lmain']['mask'],
            data['lmain']['intr'], baseline, W
        )
        r_depth = self._align_depth(
            r_depth_rel, data['rmain']['mask'],
            data['rmain']['intr'], baseline, W
        )
        
        data['lmain']['depth'] = l_depth
        data['rmain']['depth'] = r_depth
        
        # 创建伪 flow_pred (用于兼容性)
        data['lmain']['flow_pred'] = torch.zeros(bs, 2, H, W, device=device)
        data['rmain']['flow_pred'] = torch.zeros(bs, 2, H, W, device=device)
        
        depth_loss = None
        metrics = {}
        
        if not self.with_gs_render:
            return data, depth_loss, metrics
        
        # 生成多尺度特征
        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)
        
        # 预测高斯参数
        data = self.depth2gsparms(image, img_feat, data, bs)
        
        return data, depth_loss, metrics
    
    def depth2gsparms(self, lr_img, lr_img_feat, data, bs):
        """转换深度到高斯参数"""
        l_depth = data['lmain']['depth']
        r_depth = data['rmain']['depth']
        lr_depth = torch.cat([l_depth, r_depth], dim=0)
        
        # 预测高斯参数
        rot_maps, scale_maps, opacity_maps, depth_maps = self.gs_parm_regresser(
            lr_img, lr_depth, lr_img_feat
        )
        
        # 添加深度残差
        l_resdepth, r_resdepth = torch.split(depth_maps, [bs, bs])
        data['lmain']['depth'] = data['lmain']['depth'] + l_resdepth
        data['rmain']['depth'] = data['rmain']['depth'] + r_resdepth
        
        # 转换深度到点云
        for view in ['lmain', 'rmain']:
            data[view]['xyz'] = depth2pc(
                data[view]['depth'], 
                data[view]['extr'], 
                data[view]['intr']
            ).view(bs, -1, 3)
            
            valid = data[view]['mask'][:, :1, :, :] > 0.5
            data[view]['pts_valid'] = valid.view(bs, -1)
        
        # 存储高斯参数
        data['novel_view']['scale_regular'] = torch.mean(scale_maps)
        
        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])
        
        return data
    
    def freeze_da3(self):
        """冻结 DA3 参数"""
        for param in self.da3_estimator.parameters():
            param.requires_grad = False
    
    def unfreeze_da3(self):
        """解冻 DA3 参数"""
        for param in self.da3_estimator.parameters():
            param.requires_grad = True


def create_model(cfg, with_gs_render=False, use_dav3=False):
    """
    Factory function to create the appropriate model.
    
    Args:
        cfg: Configuration object
        with_gs_render: Whether to include Gaussian rendering
        use_dav3: Whether to use DAV3 instead of RAFT-Stereo
        
    Returns:
        Model instance (RtStereoHumanModel or DAV3StereoHumanModel)
    """
    if use_dav3:
        if not DAV3_AVAILABLE:
            raise ImportError("DAV3 requested but not available")
        return DAV3StereoHumanModel(cfg, with_gs_render=with_gs_render)
    else:
        return RtStereoHumanModel(cfg, with_gs_render=with_gs_render)

