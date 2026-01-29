
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

# Import DAV3 depth estimator
try:
    from lib.dav3_depth_estimator import DAV3DepthEstimator, DAV3Config, create_dav3_depth_estimator
    DAV3_AVAILABLE = True
except ImportError:
    DAV3_AVAILABLE = False
    print("Warning: DAV3 depth estimator not available")


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
    
    This model replaces RAFT-Stereo with DAV3-based depth estimation:
    1. DAV3 provides monocular relative depth for both views
    2. Sparse stereo matching computes anchor points with high-confidence disparity
    3. Scale and shift are estimated from anchor points
    4. Relative depth is converted to absolute depth
    5. GSRegresser regresses Gaussian Splatting parameters
    
    Architecture:
    - DAV3: Monocular depth estimation + feature extraction
    - Sparse Cost Volume: Efficient stereo matching at low resolution
    - Scale/Shift Estimator: Linear regression for depth alignment
    - GSRegresser: Gaussian parameter regression (same as original)
    """
    
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        
        if not DAV3_AVAILABLE:
            raise ImportError("DAV3 depth estimator is required but not available. "
                            "Please ensure Depth-Anything-3 is installed.")
        
        # DAV3 depth estimator (replaces RAFT-Stereo + UnetExtractor + LoFTR)
        self.dav3_depth = create_dav3_depth_estimator(cfg)
        
        # Image encoder for GSRegresser compatibility
        # Keep a lightweight encoder for generating multi-scale features
        self.img_encoder = UnetExtractor(
            in_channel=3, 
            encoder_dim=self.cfg.raft.encoder_dims
        )
        
        # Gaussian parameter regresser (same as original)
        if self.with_gs_render:
            self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)
    
    def forward(self, data, is_train=True):
        """
        Forward pass through the model.
        
        Args:
            data: Dictionary containing:
                - 'lmain': Left view data (img, intr, extr, mask, etc.)
                - 'rmain': Right view data
                - 'novel_view': Novel view rendering target
            is_train: Whether in training mode
            
        Returns:
            data: Updated with depth, xyz, and Gaussian parameters
            depth_loss: None (no explicit depth supervision)
            metrics: Empty dict (can be extended for monitoring)
        """
        bs = data['lmain']['img'].shape[0]
        
        # Step 1: Estimate depth using DAV3 with sparse stereo alignment
        data = self.dav3_depth(data, is_train=is_train)
        
        depth_loss = None
        metrics = {
            'scale': data['scale'].mean().item(),
            'shift': data['shift'].mean().item(),
            'num_anchors': data['sparse_stereo']['num_anchors'].float().mean().item()
        }
        
        if not self.with_gs_render:
            return data, depth_loss, metrics
        
        # Step 2: Generate multi-scale features for GSRegresser
        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
        
        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)
        
        # Step 3: Regress Gaussian parameters
        data = self.depth2gsparms(image, img_feat, data, bs)
        
        return data, depth_loss, metrics
    
    def depth2gsparms(self, lr_img, lr_img_feat, data, bs):
        """
        Convert depth to Gaussian splatting parameters.
        
        Similar to flow2gsparms but works with absolute depth directly.
        """
        l_depth = data['lmain']['depth']
        r_depth = data['rmain']['depth']
        lr_depth = torch.cat([l_depth, r_depth], dim=0)
        
        # Regress Gaussian parameters
        rot_maps, scale_maps, opacity_maps, depth_maps = self.gs_parm_regresser(
            lr_img, lr_depth, lr_img_feat
        )
        
        # Add depth residual
        l_resdepth, r_resdepth = torch.split(depth_maps, [bs, bs])
        data['lmain']['depth'] = data['lmain']['depth'] + l_resdepth
        data['rmain']['depth'] = data['rmain']['depth'] + r_resdepth
        
        # Convert depth to point cloud
        for view in ['lmain', 'rmain']:
            data[view]['xyz'] = depth2pc(
                data[view]['depth'], 
                data[view]['extr'], 
                data[view]['intr']
            ).view(bs, -1, 3)  # [B, S*S, 3]
            
            valid = data[view]['mask'][:, :1, :, :] > 0.5  # [B, 1, S, S]
            data[view]['pts_valid'] = valid.view(bs, -1)  # [B, S*S]
        
        # Store Gaussian parameters
        data['novel_view']['scale_regular'] = torch.mean(scale_maps)
        
        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])
        
        return data
    
    def freeze_dav3(self):
        """Freeze DAV3 parameters for fine-tuning only the GSRegresser."""
        for param in self.dav3_depth.parameters():
            param.requires_grad = False
    
    def unfreeze_dav3(self):
        """Unfreeze DAV3 parameters for end-to-end training."""
        for param in self.dav3_depth.parameters():
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

