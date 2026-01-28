from typing import Tuple
import torch
import torch.nn.functional as F
from torch import nn
from core.extractor import UnetExtractor
from lib.gs_parm_network import GSRegresser
from lib.loss import sequence_loss
from lib.utils import flow2depth, depth2pc
from lib.embedder import get_embedder
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
        from core.raft_stereo_human import RAFTStereoHuman
        from lib.attention_module import LocalFeatureTransformer
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
        
        # 动态高斯分配模块
        from lib.dynamic_gaussian_allocation import DynamicGaussianAllocation
        dynamic_gs_cfg = getattr(self.cfg, 'dynamic_gs', None)
        self.use_dynamic_gs = dynamic_gs_cfg is not None and getattr(dynamic_gs_cfg, 'enabled', True)
        if self.use_dynamic_gs:
            self.dynamic_gaussian_allocation = DynamicGaussianAllocation(self.cfg)
            logger.info("[Network] 动态高斯分配模块已启用")
        else:
            self.dynamic_gaussian_allocation = None
            logger.info("[Network] 动态高斯分配模块未启用")

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
        image_da3 = self._normalize_for_da3(image)
        
        # 使用 DA3 估计深度和提取特征
        da3_output = self.da3_estimator(image_da3, return_features=True, return_entropy=True)
        
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
            
            # 使用融合后的深度（可选 warmup 混合）
            blend_iters = getattr(getattr(self.cfg, 'depth_fusion', None), 'blend_warmup_iters', 0)
            step = data.get('global_step', None)
            if step is not None and blend_iters > 0:
                alpha = min(1.0, float(step) / float(blend_iters))
            else:
                alpha = 1.0

            data['lmain']['depth_fused'] = depth_fused_l
            data['rmain']['depth_fused'] = depth_fused_r
            data['lmain']['depth'] = (1.0 - alpha) * l_depth + alpha * depth_fused_l
            data['rmain']['depth'] = (1.0 - alpha) * r_depth + alpha * depth_fused_r
            data['lmain']['depth_conf'] = conf_l
            data['rmain']['depth_conf'] = conf_r
            
            # 存储融合辅助输出 (用于loss计算)
            fusion_aux = {
                'scale_l': aux_l['scale'],
                'scale_r': aux_r['scale'],
                'shift_l': aux_l['shift'],
                'shift_r': aux_r['shift'],
                'depth_l_metric': aux_l['depth_l_metric'],
                'depth_r_metric': aux_r['depth_r_metric'],
            }
            if 'depth_wide' in aux_l:
                data['depth_wide'] = aux_l['depth_wide']
                fusion_aux['depth_wide_mode'] = aux_l.get('depth_wide_mode', 'concat')
            data['fusion_aux'] = fusion_aux
            
            logger.debug(f"[DepthFusion] scale_l={aux_l['scale'].mean().item():.4f}, "
                        f"scale_r={aux_r['scale'].mean().item():.4f}")
        else:
            # 不使用深度融合时，进行简单的深度对齐
            # DA3 输出相对深度，需要对齐左右视图的尺度
            l_depth_aligned, r_depth_aligned = self._align_stereo_depth(
                l_depth, r_depth, 
                data['lmain']['mask'], data['rmain']['mask'],
                data['lmain']['intr'], 
                self._compute_baseline(data['lmain']['extr'], data['rmain']['extr'])
            )
            data['lmain']['depth'] = l_depth_aligned
            data['rmain']['depth'] = r_depth_aligned
        
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

    def _normalize_for_da3(self, image: torch.Tensor) -> torch.Tensor:
        """
        将输入图像归一化到 DA3 期望的 [0, 1] 范围。
        默认使用 cfg.dataset.img_range 进行线性映射。
        """
        img_range = getattr(self.cfg.dataset, 'img_range', None)
        if img_range is not None and len(img_range) == 2:
            img_min, img_max = img_range
            image = (image - img_min) / (img_max - img_min)
        return image.clamp(0, 1)
    
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
    
    def _align_stereo_depth(
        self, 
        depth_l: torch.Tensor, 
        depth_r: torch.Tensor,
        mask_l: torch.Tensor,
        mask_r: torch.Tensor,
        intrinsics: torch.Tensor,
        baseline: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对齐左右视图的相对深度到统一尺度
        
        DA3 输出的是相对深度 (0-1 范围)，需要转换为具有几何意义的深度。
        这里使用简单的统计对齐 + 立体几何约束。
        
        Args:
            depth_l: [B, 1, H, W] - 左视图相对深度
            depth_r: [B, 1, H, W] - 右视图相对深度
            mask_l: [B, C, H, W] - 左视图有效掩码
            mask_r: [B, C, H, W] - 右视图有效掩码
            intrinsics: [B, 3, 3] - 相机内参
            baseline: [B] - 基线距离
            
        Returns:
            depth_l_aligned: [B, 1, H, W] - 对齐后的左深度
            depth_r_aligned: [B, 1, H, W] - 对齐后的右深度
        """
        B, _, H, W = depth_l.shape
        eps = 1e-6
        
        # 获取有效区域掩码
        valid_l = (mask_l[:, :1] > 0.5).float()
        valid_r = (mask_r[:, :1] > 0.5).float()
        
        # 1. 首先将相对深度转换为逆深度 (DA3 通常输出的是逆深度或相对逆深度)
        # 检查深度值范围来判断格式
        depth_l_valid = depth_l * valid_l
        depth_r_valid = depth_r * valid_r
        
        # 2. 统计对齐：让左右深度具有相同的中值和范围
        # 计算有效区域的统计量
        def compute_stats(depth, mask):
            valid_depth = depth[mask > 0.5]
            if valid_depth.numel() == 0:
                return torch.tensor(1.0, device=depth.device), torch.tensor(0.0, device=depth.device)
            median = valid_depth.median()
            # 使用 robust scale (MAD - median absolute deviation)
            mad = (valid_depth - median).abs().median()
            return median, mad + eps
        
        depth_align_cfg = getattr(self.cfg, 'depth_align', None)
        disp_min_ratio = getattr(depth_align_cfg, 'typical_disp_min_ratio', 0.01)
        disp_max_ratio = getattr(depth_align_cfg, 'typical_disp_max_ratio', 0.15)
        target_range_scale = getattr(depth_align_cfg, 'target_range_scale', 0.5)
        scale_clamp_min = getattr(depth_align_cfg, 'scale_clamp_min', 0.8)
        scale_clamp_max = getattr(depth_align_cfg, 'scale_clamp_max', 1.2)
        min_depth = getattr(depth_align_cfg, 'min_depth', 0.1)
        
        # 对每个 batch 分别处理
        depth_l_aligned = torch.zeros_like(depth_l)
        depth_r_aligned = torch.zeros_like(depth_r)
        
        for b in range(B):
            # 获取焦距和基线
            fx = intrinsics[b, 0, 0]
            bl = baseline[b]
            
            # 计算统计量
            median_l, mad_l = compute_stats(depth_l[b], valid_l[b])
            median_r, mad_r = compute_stats(depth_r[b], valid_r[b])
            
            # 目标尺度：基于典型人体深度 (约 1-5 米)
            # 使用立体几何关系估计合理的深度范围
            # disp = baseline * fx / depth  =>  depth = baseline * fx / disp
            # 假设典型视差范围为图像宽度的 1%-10%
            typical_disp_min = W * disp_min_ratio
            typical_disp_max = W * disp_max_ratio
            
            depth_far = bl * fx / (typical_disp_min + eps)
            depth_near = bl * fx / (typical_disp_max + eps)
            
            # 目标深度范围
            target_median = (depth_near + depth_far) / 2
            target_range = (depth_far - depth_near) / 2
            
            # 对齐左视图
            # normalized = (depth - median) / mad
            # aligned = normalized * target_range + target_median
            depth_l_norm = (depth_l[b] - median_l) / (mad_l + eps)
            depth_l_aligned[b] = depth_l_norm * target_range * target_range_scale + target_median
            
            # 对齐右视图 (使用相同的目标尺度)
            depth_r_norm = (depth_r[b] - median_r) / (mad_r + eps)
            depth_r_aligned[b] = depth_r_norm * target_range * target_range_scale + target_median
        
        # 确保深度为正
        depth_l_aligned = depth_l_aligned.clamp(min=min_depth)
        depth_r_aligned = depth_r_aligned.clamp(min=min_depth)
        
        # 3. 额外的一致性约束：让左右视图在重叠区域具有相似的深度分布
        # 使用简单的平均来进一步对齐
        combined_mask = valid_l * valid_r
        if combined_mask.sum() > 0:
            mean_l = (depth_l_aligned * combined_mask).sum() / (combined_mask.sum() + eps)
            mean_r = (depth_r_aligned * combined_mask).sum() / (combined_mask.sum() + eps)
            
            # 微调使两者均值一致
            global_mean = (mean_l + mean_r) / 2
            scale_l = global_mean / (mean_l + eps)
            scale_r = global_mean / (mean_r + eps)
            
            # 限制调整幅度
            scale_l = scale_l.clamp(scale_clamp_min, scale_clamp_max)
            scale_r = scale_r.clamp(scale_clamp_min, scale_clamp_max)
            
            depth_l_aligned = depth_l_aligned * scale_l
            depth_r_aligned = depth_r_aligned * scale_r
        
        return depth_l_aligned, depth_r_aligned
    
    def da3_to_gsparms(self, lr_img, lr_img_feat, data, bs):
        """
        DA3 模式下的多层高斯参数预测
        
        核心改进：
        1. 输出多层高斯参数 [B, C, L, H, W]
        2. 使用 DynamicGaussianAllocation 进行动态裁剪
        3. 基于多层深度计算点云
        """
        l_depth = data['lmain']['depth']  # [B, 1, H, W]
        r_depth = data['rmain']['depth']  # [B, 1, H, W]
        lr_depth = torch.concat([l_depth, r_depth], dim=0)  # [2B, 1, H, W]
        
        # 预测多层高斯参数
        gs_output = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)
        
        # 分离左右视图的参数
        # gs_output 是 dict: {rotation, scale, opacity, depth_offset, texture_complexity, load_balance_loss}
        l_gs_params, r_gs_params = self._split_gs_params(gs_output, bs)
        
        # 存储 MoE 负载均衡损失
        data['moe_load_balance_loss'] = gs_output.get('load_balance_loss', torch.tensor(0.0, device=l_depth.device))
        
        # 对每个视图分别处理
        step = data.get('global_step', None)
        for view, gs_params, base_depth in [('lmain', l_gs_params, l_depth), ('rmain', r_gs_params, r_depth)]:
            # 应用动态高斯分配
            if self.use_dynamic_gs and self.dynamic_gaussian_allocation is not None:
                filtered_params, valid_mask, alloc_stats = self.dynamic_gaussian_allocation(
                    gs_params, 
                    gs_params.get('texture_complexity'),
                    step=step
                )
                data[view]['alloc_stats'] = alloc_stats
            else:
                filtered_params = gs_params
                # 仅基于 opacity 阈值创建有效掩码
                opacity_threshold = getattr(
                    getattr(self.cfg, 'dynamic_gs', None), 
                    'opacity_threshold', 0.01
                )
                valid_mask = (gs_params['opacity'].squeeze(1) > opacity_threshold)  # [B, L, H, W]
            
            # 计算多层点云
            xyz, pts_valid = self._compute_multilayer_pointcloud(
                filtered_params, valid_mask, base_depth,
                data[view]['intr'], data[view]['extr'], data[view]['mask']
            )
            
            # 存储结果
            data[view]['xyz'] = xyz  # [B, N, 3] 其中 N = L * H * W
            data[view]['pts_valid'] = pts_valid  # [B, N]
            
            # 存储高斯参数 (展平为渲染格式)
            B, _, L, H, W = filtered_params['rotation'].shape
            N = L * H * W
            
            # [B, C, L, H, W] -> [B, L, H, W, C] -> [B, N, C]
            data[view]['rot_maps'] = filtered_params['rotation'].permute(0, 2, 3, 4, 1).reshape(B, N, 4)
            data[view]['scale_maps'] = filtered_params['scale'].permute(0, 2, 3, 4, 1).reshape(B, N, 3)
            data[view]['opacity_maps'] = filtered_params['opacity'].squeeze(1).reshape(B, N)  # [B, N]
            
            # 也存储原始多层格式 (用于可视化)
            data[view]['multilayer_params'] = filtered_params
            data[view]['valid_mask'] = valid_mask
            data[view]['texture_complexity'] = gs_params.get('texture_complexity')
        
        # 计算 scale 正则项
        l_scale = l_gs_params['scale']
        r_scale = r_gs_params['scale']
        data['novel_view']['scale_regular'] = (l_scale.mean() + r_scale.mean()) / 2
        
        return data
    
    def _split_gs_params(self, gs_output: dict, bs: int) -> Tuple[dict, dict]:
        """将合并的高斯参数分离为左右视图"""
        l_params = {}
        r_params = {}
        
        for key, value in gs_output.items():
            if key == 'load_balance_loss':
                continue  # 跳过损失值
            
            if isinstance(value, torch.Tensor):
                if value.dim() >= 4:  # 有空间维度的参数
                    l_params[key], r_params[key] = torch.split(value, [bs, bs], dim=0)
                else:
                    l_params[key] = value
                    r_params[key] = value
            else:
                l_params[key] = value
                r_params[key] = value
        
        return l_params, r_params
    
    def _compute_multilayer_pointcloud(
        self,
        gs_params: dict,
        valid_mask: torch.Tensor,
        base_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算多层点云坐标
        
        Args:
            gs_params: 高斯参数 dict
                - depth_offset: [B, 1, L, H, W]
            valid_mask: [B, L, H, W] 有效高斯掩码
            base_depth: [B, 1, H, W] 基础深度
            intrinsics: [B, 3, 3] 相机内参
            extrinsics: [B, 3, 4] 或 [B, 4, 4] 相机外参
            mask: [B, C, H, W] 前景掩码
            
        Returns:
            xyz: [B, N, 3] 世界坐标点云 (N = L * H * W)
            pts_valid: [B, N] 有效点掩码
        """
        depth_offset = gs_params['depth_offset']  # [B, 1, L, H, W]
        B, _, L, H, W = depth_offset.shape
        device = depth_offset.device
        
        # 计算每层深度 = 基础深度 + 深度偏移
        base_depth_expanded = base_depth.unsqueeze(2)  # [B, 1, 1, H, W]
        layer_depths = base_depth_expanded + depth_offset  # [B, 1, L, H, W]
        layer_depths = F.relu(layer_depths) + 1e-6  # 确保深度为正
        layer_depths = layer_depths.squeeze(1)  # [B, L, H, W]
        
        # 生成像素坐标网格
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )
        x_coords = x_coords.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)  # [B, L, H, W]
        y_coords = y_coords.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        
        # 从内参获取焦距和主点
        fx = intrinsics[:, 0, 0].view(B, 1, 1, 1)
        fy = intrinsics[:, 1, 1].view(B, 1, 1, 1)
        cx = intrinsics[:, 0, 2].view(B, 1, 1, 1)
        cy = intrinsics[:, 1, 2].view(B, 1, 1, 1)
        
        # 反投影到相机坐标系: P_camera = K^{-1} @ [u, v, 1]^T * depth
        x_cam = (x_coords - cx) * layer_depths / fx
        y_cam = (y_coords - cy) * layer_depths / fy
        z_cam = layer_depths
        
        xyz_camera = torch.stack([x_cam, y_cam, z_cam], dim=-1)  # [B, L, H, W, 3]
        
        # 转换到世界坐标系
        if extrinsics.shape[1] == 4:
            R = extrinsics[:, :3, :3]  # [B, 3, 3]
            t = extrinsics[:, :3, 3]   # [B, 3]
        else:
            R = extrinsics[:, :, :3]
            t = extrinsics[:, :, 3]
        
        # 假设外参是 world_to_camera，需要求逆
        # P_world = R^T @ (P_camera - t) 不对，应该是:
        # 如果 P_camera = R @ P_world + t, 则 P_world = R^T @ (P_camera - t)
        # 但实际上需要根据具体定义来确定
        # 这里使用与 depth2pc 相同的变换方式
        R_inv = R.transpose(1, 2)  # [B, 3, 3]
        
        # 重塑: [B, L, H, W, 3] -> [B, N, 3]
        N = L * H * W
        xyz_camera_flat = xyz_camera.reshape(B, N, 3)
        
        # 变换到世界坐标
        # P_world = R^T @ P_camera - R^T @ t
        xyz_world = torch.bmm(xyz_camera_flat, R_inv.transpose(1, 2))
        t_transformed = torch.bmm(t.unsqueeze(1), R_inv.transpose(1, 2))  # [B, 1, 3]
        xyz_world = xyz_world - t_transformed
        
        # 计算有效掩码: valid_mask AND foreground_mask
        fg_mask = (mask[:, :1] > 0.5).float()  # [B, 1, H, W]
        fg_mask_expanded = fg_mask.unsqueeze(2).expand(-1, -1, L, -1, -1)  # [B, 1, L, H, W]
        fg_mask_squeezed = fg_mask_expanded.squeeze(1)  # [B, L, H, W]
        
        combined_valid = valid_mask.float() * fg_mask_squeezed  # [B, L, H, W]
        pts_valid = combined_valid.reshape(B, N) > 0.5  # [B, N]
        
        return xyz_world, pts_valid

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

