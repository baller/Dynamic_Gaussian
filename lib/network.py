
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
import logging

logger = logging.getLogger(__name__)


class RtStereoHumanModel(nn.Module):
    def __init__(self, cfg, with_gs_render=False):
        super().__init__()
        self.cfg = cfg
        self.with_gs_render = with_gs_render
        self.use_flow_init = self.cfg.dataset.use_depth_init
        self.train_iters = self.cfg.raft.train_iters
        self.val_iters = self.cfg.raft.val_iters
        
        # 深度估计模式: 'raft' 或 'da3'
        self.depth_mode = getattr(cfg, 'depth_mode', 'raft')
        logger.info(f"[Network] 深度估计模式: {self.depth_mode}")
        
        if self.depth_mode == 'da3':
            # DA3 模式
            self._init_da3_mode()
        else:
            # RAFT 模式 (原始)
            self._init_raft_mode()
        
        if self.with_gs_render:
            # 选择高斯参数预测网络
            gs_predictor_type = getattr(cfg, 'gs_predictor', 'gsregresser')  # 'gsregresser' or 'transformer_moe'
            self.gs_predictor_type = gs_predictor_type
            
            if gs_predictor_type == 'transformer_moe':
                from lib.gaussian_transformer_moe import GaussianTransformerMoESimple
                self.gs_parm_regresser = GaussianTransformerMoESimple(self.cfg, rgb_dim=3, depth_dim=1)
                logger.info("[Network] 使用 Transformer+MoE 高斯预测网络")
            else:
                self.gs_parm_regresser = GSRegresser(self.cfg, rgb_dim=3, depth_dim=1)
                logger.info("[Network] 使用原始 GSRegresser 高斯预测网络")

    def _init_raft_mode(self):
        """初始化 RAFT 模式的模块"""
        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        self.loftr_coarse = LocalFeatureTransformer()
        self.raft_stereo = RAFTStereoHuman(self.cfg.raft)
        
    def _init_da3_mode(self):
        """初始化 DA3 模式的模块"""
        from lib.da3_depth import DA3DepthEstimator, DA3FeatureAdapter
        from lib.depth_fusion import DepthFusionModule
        
        # DA3 深度估计器 (延迟加载)
        self.da3_estimator = DA3DepthEstimator(self.cfg)
        
        # 特征适配器: 将 DA3 特征转换为与 GSRegresser 兼容的格式
        da3_cfg = getattr(self.cfg, 'da3', None)
        use_adapter = True
        if da3_cfg is not None:
            use_adapter = getattr(da3_cfg, 'use_feature_adapter', True)
        
        # 根据模型类型确定特征维度
        model_name = getattr(da3_cfg, 'model_name', 'depth-anything/DA3-LARGE') if da3_cfg else 'depth-anything/DA3-LARGE'
        if 'LARGE' in model_name.upper() or 'GIANT' in model_name.upper():
            self.da3_feat_dim = 1024  # ViT-L/G
        elif 'BASE' in model_name.upper():
            self.da3_feat_dim = 768   # ViT-B
        else:
            self.da3_feat_dim = 384   # ViT-S
        
        if use_adapter:
            num_layers = len(getattr(da3_cfg, 'export_feat_layers', [11, 15, 19, 23]))
            in_dims = [self.da3_feat_dim] * num_layers
            logger.info(f"[Network] DA3 特征维度: {in_dims}")
            
            # DA3 特征维度 -> RAFT encoder 维度
            self.feature_adapter = DA3FeatureAdapter(
                in_dims=in_dims,
                out_dims=self.cfg.raft.encoder_dims
            )
        else:
            self.feature_adapter = None
            
        # 简单的图像编码器用于生成与 RAFT 兼容的特征尺寸
        self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
        
        # 深度融合模块
        depth_fusion_cfg = getattr(self.cfg, 'depth_fusion', None)
        self.use_depth_fusion = depth_fusion_cfg is not None and getattr(depth_fusion_cfg, 'enabled', True)
        if self.use_depth_fusion:
            self.depth_fusion = DepthFusionModule(self.cfg, da3_feat_dim=self.da3_feat_dim)
            logger.info("[Network] 深度融合模块已启用")
        else:
            self.depth_fusion = None
            logger.info("[Network] 深度融合模块未启用")

    def forward(self, data, is_train=True):
        if self.depth_mode == 'da3':
            return self.forward_da3(data, is_train)
        else:
            return self.forward_raft(data, is_train)
    
    def forward_raft(self, data, is_train=True):
        """原始 RAFT 模式的前向传播"""
        bs = data['lmain']['img'].shape[0]

        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)

        flow_init = torch.cat([data['lmain']['flow_init'], data['rmain']['flow_init']], dim=0) if self.use_flow_init else None

        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)

        (feat_c0, feat_c1) = img_feat[2].split(bs)
        
        mask_c0 = mask_c1 = None  

        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)
        feat_cs = torch.cat((feat_c0, feat_c1), 0)
        
        img_feat = img_feat[0], img_feat[1], feat_cs
        
        if is_train:
            flow_predictions = self.raft_stereo(feat_cs, flow_init=flow_init, iters=self.train_iters)
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
    
    def forward_da3(self, data, is_train=True):
        """DA3 模式的前向传播"""
        bs = data['lmain']['img'].shape[0]
        
        # 合并左右视图图像
        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
        
        # 使用 DA3 估计深度和提取特征
        da3_output = self.da3_estimator(image, return_features=True, return_entropy=True)
        
        # 获取深度图
        depth = da3_output['depth']  # [2*B, 1, H, W]
        l_depth, r_depth = torch.split(depth, [bs, bs])
        
        # 存储原始单目深度
        data['lmain']['depth_mono'] = l_depth
        data['rmain']['depth_mono'] = r_depth
        
        # 创建伪 flow_pred (用于兼容性)
        H, W = l_depth.shape[-2:]
        data['lmain']['flow_pred'] = torch.zeros(bs, 2, H, W, device=l_depth.device)
        data['rmain']['flow_pred'] = torch.zeros(bs, 2, H, W, device=r_depth.device)
        
        # 存储熵图（纹理复杂度）
        if 'entropy' in da3_output:
            entropy = da3_output['entropy']
            l_entropy, r_entropy = torch.split(entropy, [bs, bs])
            data['lmain']['entropy'] = l_entropy
            data['rmain']['entropy'] = r_entropy
        
        # 获取 DA3 特征
        da3_features = da3_output.get('features', None)
        l_da3_feat = None
        r_da3_feat = None
        if da3_features is not None and len(da3_features) > 0:
            # 使用最后一层特征作为主特征
            main_feat = da3_features[-1]  # [2*B, C, H', W']
            l_da3_feat, r_da3_feat = torch.split(main_feat, [bs, bs])
        
        # 深度融合
        flow_loss = None
        metrics = {}
        fusion_aux = {}
        
        if self.use_depth_fusion and self.depth_fusion is not None and l_da3_feat is not None:
            # 计算基线距离
            baseline = self._compute_baseline(data['lmain']['extr'], data['rmain']['extr'])
            
            # 深度融合
            depth_fused_l, conf_l, aux_l = self.depth_fusion(
                l_depth, r_depth, l_da3_feat, r_da3_feat,
                intrinsics=data['lmain']['intr'],
                baseline=baseline
            )
            depth_fused_r, conf_r, aux_r = self.depth_fusion(
                r_depth, l_depth, r_da3_feat, l_da3_feat,
                intrinsics=data['rmain']['intr'],
                baseline=baseline
            )
            
            # 使用融合后的深度
            data['lmain']['depth'] = depth_fused_l
            data['rmain']['depth'] = depth_fused_r
            data['lmain']['depth_conf'] = conf_l
            data['rmain']['depth_conf'] = conf_r
            
            # 存储融合辅助输出 (用于loss计算)
            fusion_aux = {
                'scale_l': aux_l['scale'],
                'scale_r': aux_r['scale'],
                'shift_l': aux_l['shift'],
                'shift_r': aux_r['shift'],
                'depth_l_metric': aux_l['depth_l_metric'],
                'depth_r_metric': aux_r['depth_l_metric'],
            }
            data['fusion_aux'] = fusion_aux
            
            logger.debug(f"[DepthFusion] scale_l={aux_l['scale'].mean().item():.4f}, "
                        f"scale_r={aux_r['scale'].mean().item():.4f}")
        else:
            # 不使用深度融合，直接使用单目深度
            data['lmain']['depth'] = l_depth
            data['rmain']['depth'] = r_depth
        
        if not self.with_gs_render:
            return data, flow_loss, metrics
        
        # 准备特征用于高斯参数预测
        if da3_features is not None and self.feature_adapter is not None:
            # 获取 img_encoder 的输出尺寸作为目标
            with torch.no_grad():
                ref_feat = self.img_encoder(image)
            target_sizes = [f.shape[-2:] for f in ref_feat]
            
            # 分割左右特征并分别适配
            adapted_features = []
            for feat in da3_features:
                l_feat, r_feat = torch.split(feat, [bs, bs])
                adapted_features.append(torch.cat([l_feat, r_feat], dim=0))
            
            # 适配特征
            img_feat = self.feature_adapter(adapted_features, target_sizes)
            img_feat = tuple(img_feat)
        else:
            # 使用原始图像编码器
            with autocast(enabled=self.cfg.raft.mixed_precision):
                img_feat = self.img_encoder(image)
        
        # 预测高斯参数
        data = self.da3_to_gsparms(image, img_feat, data, bs)
        
        return data, flow_loss, metrics
    
    def _compute_baseline(self, extr_l: torch.Tensor, extr_r: torch.Tensor) -> torch.Tensor:
        """
        计算立体基线距离
        
        Args:
            extr_l: [B, 3, 4] 或 [B, 4, 4] - 左视图外参
            extr_r: [B, 3, 4] 或 [B, 4, 4] - 右视图外参
            
        Returns:
            baseline: [B] - 基线距离
        """
        # 提取平移向量
        if extr_l.shape[1] == 4:
            t_l = extr_l[:, :3, 3]  # [B, 3]
            t_r = extr_r[:, :3, 3]
        else:
            t_l = extr_l[:, :, 3]  # [B, 3]
            t_r = extr_r[:, :, 3]
        
        baseline = torch.norm(t_l - t_r, dim=1)  # [B]
        return baseline
    
    def da3_to_gsparms(self, lr_img, lr_img_feat, data, bs):
        """DA3 模式下的高斯参数预测"""
        l_depth = data['lmain']['depth']  
        r_depth = data['rmain']['depth'] 
        lr_depth = torch.concat([l_depth, r_depth], dim=0)
        
        # 预测高斯参数
        rot_maps, scale_maps, opacity_maps, depth_maps = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)
        l_resdepth, r_resdepth = torch.split(depth_maps, [bs, bs])
        
        # 添加深度残差
        data['lmain']['depth'] = data['lmain']['depth'] + l_resdepth
        data['rmain']['depth'] = data['rmain']['depth'] + r_resdepth

        for view in ['lmain', 'rmain']:
            data[view]['xyz'] = depth2pc(data[view]['depth'], data[view]['extr'], data[view]['intr']).view(bs, -1, 3)
            valid = data[view]['mask'][:,:1,:,:] > 0.5
            data[view]['pts_valid'] = valid.view(bs, -1)

        data['novel_view']['scale_regular'] = torch.mean(scale_maps)

        data['lmain']['rot_maps'], data['rmain']['rot_maps'] = torch.split(rot_maps, [bs, bs])
        data['lmain']['scale_maps'], data['rmain']['scale_maps'] = torch.split(scale_maps, [bs, bs])
        data['lmain']['opacity_maps'], data['rmain']['opacity_maps'] = torch.split(opacity_maps, [bs, bs])

        return data

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

