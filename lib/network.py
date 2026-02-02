
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
            # 根据配置选择 GSRegresser (CNN) 或 GSTransformer
            if getattr(self.cfg.gsnet, 'use_transformer', False):
                from lib.gs_transformer import GSTransformer
                self.gs_parm_regresser = GSTransformer(
                    self.cfg,
                    embed_dim=getattr(self.cfg.gsnet, 'transformer_embed_dim', 768),
                    depth=getattr(self.cfg.gsnet, 'transformer_depth', 16),
                    num_heads=getattr(self.cfg.gsnet, 'transformer_num_heads', 12),
                    patch_size=getattr(self.cfg.gsnet, 'transformer_patch_size', 14),
                    mlp_ratio=getattr(self.cfg.gsnet, 'transformer_mlp_ratio', 4.0),
                    use_checkpoint=getattr(self.cfg.gsnet, 'transformer_use_checkpoint', False),
                )
            else:
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
        """转换 flow 到高斯参数，使用 xyz 残差调整点云位置"""
        for view in ['lmain', 'rmain']:
            data[view]['depth'] = flow2depth(data[view])
            
        l_depth = data['lmain']['depth']  
        r_depth = data['rmain']['depth'] 
        lr_depth = torch.concat([l_depth, r_depth], dim=0)
        
        # 1. 先计算初始点云 (从原始深度，返回 [B, 3, H, W] 格式)
        l_xyz_init = depth2pc(
            data['lmain']['depth'], 
            data['lmain']['extr'], 
            data['lmain']['intr'],
            return_2d=True
        )  # [B, 3, H, W]
        r_xyz_init = depth2pc(
            data['rmain']['depth'], 
            data['rmain']['extr'], 
            data['rmain']['intr'],
            return_2d=True
        )  # [B, 3, H, W]
        
        # 2. 预测高斯参数和 xyz 残差
        rot_maps, scale_maps, opacity_maps, xyz_res = self.gs_parm_regresser(lr_img, lr_depth, lr_img_feat)
        
        # 3. 分割 xyz 残差
        l_xyz_res, r_xyz_res = torch.split(xyz_res, [bs, bs])
        
        # 4. 加上 xyz 残差
        l_xyz = l_xyz_init + l_xyz_res  # [B, 3, H, W]
        r_xyz = r_xyz_init + r_xyz_res  # [B, 3, H, W]
        
        # 5. 转换为 [B, H*W, 3] 格式
        data['lmain']['xyz'] = l_xyz.view(bs, 3, -1).permute(0, 2, 1)  # [B, H*W, 3]
        data['rmain']['xyz'] = r_xyz.view(bs, 3, -1).permute(0, 2, 1)  # [B, H*W, 3]
        
        # 6. 基于深度有效性设置 pts_valid
        for view in ['lmain', 'rmain']:
            # 倒数深度 > 0.01 表示真实深度 < 100m（有效范围）
            # 倒数深度 < 10 表示真实深度 > 0.1m（避免过近的点）
            depth_valid = (data[view]['depth'][:, :1, :, :] > 0.01) & \
                         (data[view]['depth'][:, :1, :, :] < 10.0)
            data[view]['pts_valid'] = depth_valid.view(bs, -1)  # [B, S*S]

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
            # 根据配置选择 GSRegresser (CNN) 或 GSTransformer
            if getattr(self.cfg.gsnet, 'use_transformer', False):
                from lib.gs_transformer import GSTransformer
                self.gs_parm_regresser = GSTransformer(
                    self.cfg,
                    embed_dim=getattr(self.cfg.gsnet, 'transformer_embed_dim', 768),
                    depth=getattr(self.cfg.gsnet, 'transformer_depth', 16),
                    num_heads=getattr(self.cfg.gsnet, 'transformer_num_heads', 12),
                    patch_size=getattr(self.cfg.gsnet, 'transformer_patch_size', 14),
                    mlp_ratio=getattr(self.cfg.gsnet, 'transformer_mlp_ratio', 4.0),
                    use_checkpoint=getattr(self.cfg.gsnet, 'transformer_use_checkpoint', False),
                )
            else:
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
    
    def _compute_gradient_magnitude(self, img: torch.Tensor) -> torch.Tensor:
        """
        计算图像梯度幅值，用于选取高梯度点
        
        Args:
            img: [B, C, H, W] 图像
            
        Returns:
            gradient: [B, 1, H, W] 梯度幅值
        """
        # 转换为灰度图
        if img.shape[1] == 3:
            gray = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        else:
            gray = img[:, :1]
        
        # Sobel 算子
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                               dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
        
        grad_x = F.conv2d(gray, sobel_x, padding=1)
        grad_y = F.conv2d(gray, sobel_y, padding=1)
        
        gradient = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
        return gradient
    
    def _select_keypoints(
        self, 
        gradient: torch.Tensor, 
        mask: torch.Tensor, 
        num_points: int = 500,
        min_distance: int = 8
    ) -> torch.Tensor:
        """
        选取高梯度点作为匹配关键点
        
        Args:
            gradient: [B, 1, H, W] 梯度幅值
            mask: [B, 1, H, W] 有效区域掩码
            num_points: 选取的点数
            min_distance: 点之间的最小距离
            
        Returns:
            keypoints: [B, N, 2] 关键点坐标 (y, x)
        """
        B, _, H, W = gradient.shape
        device = gradient.device
        
        # 应用掩码
        gradient_masked = gradient * (mask > 0.5).float()
        
        keypoints_list = []
        
        for b in range(B):
            grad_b = gradient_masked[b, 0]  # [H, W]
            
            # 使用非极大值抑制选取关键点
            # 简化实现：直接取 topk
            grad_flat = grad_b.view(-1)
            num_valid = (grad_flat > 0).sum().item()
            k = min(num_points * 4, num_valid)  # 先取更多点，后面过滤
            
            if k > 0:
                _, indices = torch.topk(grad_flat, k)
                y_coords = indices // W
                x_coords = indices % W
                
                # 简单的距离过滤
                selected_points = []
                for i in range(len(indices)):
                    y, x = y_coords[i].item(), x_coords[i].item()
                    
                    # 检查与已选点的距离
                    is_valid = True
                    for py, px in selected_points:
                        if abs(y - py) < min_distance and abs(x - px) < min_distance:
                            is_valid = False
                            break
                    
                    if is_valid:
                        selected_points.append((y, x))
                        if len(selected_points) >= num_points:
                            break
                
                if len(selected_points) > 0:
                    kpts = torch.tensor(selected_points, device=device, dtype=torch.long)
                else:
                    # 如果没有找到点，使用均匀采样
                    step = max(H // 20, 1)
                    y_range = torch.arange(step, H - step, step, device=device)
                    x_range = torch.arange(step, W - step, step, device=device)
                    yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
                    kpts = torch.stack([yy.flatten(), xx.flatten()], dim=1)[:num_points]
            else:
                # 均匀采样
                step = max(H // 20, 1)
                y_range = torch.arange(step, H - step, step, device=device)
                x_range = torch.arange(step, W - step, step, device=device)
                yy, xx = torch.meshgrid(y_range, x_range, indexing='ij')
                kpts = torch.stack([yy.flatten(), xx.flatten()], dim=1)[:num_points]
            
            keypoints_list.append(kpts)
        
        # 填充到相同长度
        max_len = max(kp.shape[0] for kp in keypoints_list)
        padded_keypoints = []
        for kpts in keypoints_list:
            if kpts.shape[0] < max_len:
                padding = torch.zeros(max_len - kpts.shape[0], 2, device=device, dtype=torch.long)
                kpts = torch.cat([kpts, padding], dim=0)
            padded_keypoints.append(kpts)
        
        return torch.stack(padded_keypoints, dim=0)  # [B, N, 2]
    
    def _patch_matching_sad(
        self,
        img_left: torch.Tensor,
        img_right: torch.Tensor,
        keypoints: torch.Tensor,
        patch_size: int = 11,
        max_disparity: int = 128
    ) -> tuple:
        """
        使用 SAD (Sum of Absolute Differences) 进行极线块匹配
        
        Args:
            img_left: [B, C, H, W] 左图
            img_right: [B, C, H, W] 右图
            keypoints: [B, N, 2] 关键点坐标 (y, x)
            patch_size: 匹配窗口大小
            max_disparity: 最大视差搜索范围
            
        Returns:
            disparities: [B, N] 视差值
            confidences: [B, N] 匹配置信度
        """
        B, C, H, W = img_left.shape
        N = keypoints.shape[1]
        device = img_left.device
        half_patch = patch_size // 2
        
        # 转换为灰度图
        if C == 3:
            left_gray = 0.299 * img_left[:, 0:1] + 0.587 * img_left[:, 1:2] + 0.114 * img_left[:, 2:3]
            right_gray = 0.299 * img_right[:, 0:1] + 0.587 * img_right[:, 1:2] + 0.114 * img_right[:, 2:3]
        else:
            left_gray = img_left[:, :1]
            right_gray = img_right[:, :1]
        
        disparities = torch.zeros(B, N, device=device)
        confidences = torch.zeros(B, N, device=device)
        
        for b in range(B):
            for i in range(N):
                y, x = keypoints[b, i, 0].item(), keypoints[b, i, 1].item()
                
                # 边界检查
                if (y < half_patch or y >= H - half_patch or 
                    x < half_patch or x >= W - half_patch):
                    continue
                
                # 提取左图 patch
                left_patch = left_gray[b, 0, 
                                       y - half_patch:y + half_patch + 1,
                                       x - half_patch:x + half_patch + 1]
                
                # 在右图同一行搜索
                min_sad = float('inf')
                second_min_sad = float('inf')
                best_d = 0
                
                search_start = max(half_patch, x - max_disparity)
                search_end = x  # 视差为正，右图在左侧
                
                for d in range(0, min(max_disparity, x - half_patch)):
                    rx = x - d
                    if rx < half_patch:
                        break
                    
                    right_patch = right_gray[b, 0,
                                            y - half_patch:y + half_patch + 1,
                                            rx - half_patch:rx + half_patch + 1]
                    
                    sad = (left_patch - right_patch).abs().sum().item()
                    
                    if sad < min_sad:
                        second_min_sad = min_sad
                        min_sad = sad
                        best_d = d
                    elif sad < second_min_sad:
                        second_min_sad = sad
                
                disparities[b, i] = best_d
                
                # 计算置信度：使用 peak ratio
                if second_min_sad > 0 and min_sad > 0:
                    ratio = second_min_sad / (min_sad + 1e-6)
                    confidences[b, i] = min(ratio - 1.0, 1.0)  # ratio > 1 表示好的匹配
                else:
                    confidences[b, i] = 0.0
        
        return disparities, confidences
    
    def _ransac_fit_scale_shift(
        self,
        depth_rel: torch.Tensor,
        depth_metric: torch.Tensor,
        confidence: torch.Tensor,
        num_iterations: int = 100,
        inlier_threshold: float = 0.1
    ) -> tuple:
        """
        使用 RANSAC 鲁棒地估计 scale 和 shift
        
        D_metric = scale * D_rel + shift
        
        Args:
            depth_rel: [N] 相对深度值
            depth_metric: [N] 度量深度值 (从立体匹配计算)
            confidence: [N] 匹配置信度
            num_iterations: RANSAC 迭代次数
            inlier_threshold: 内点阈值
            
        Returns:
            scale: 尺度因子
            shift: 偏移量
        """
        device = depth_rel.device
        
        # 过滤有效点
        valid_mask = (confidence > 0.3) & (depth_metric > 0.1) & (depth_metric < 100)
        valid_indices = torch.where(valid_mask)[0]
        
        if len(valid_indices) < 10:
            # 点太少，使用简单的中值对齐
            if len(valid_indices) > 0:
                rel_valid = depth_rel[valid_indices]
                metric_valid = depth_metric[valid_indices]
                scale = (metric_valid.median() / (rel_valid.median() + 1e-6)).clamp(0.1, 100)
                shift = torch.tensor(0.0, device=device)
            else:
                scale = torch.tensor(1.0, device=device)
                shift = torch.tensor(0.0, device=device)
            return scale, shift
        
        rel_valid = depth_rel[valid_indices]
        metric_valid = depth_metric[valid_indices]
        n_points = len(valid_indices)
        
        best_scale = torch.tensor(1.0, device=device)
        best_shift = torch.tensor(0.0, device=device)
        best_inliers = 0
        
        for _ in range(num_iterations):
            # 随机选择 2 个点
            idx = torch.randperm(n_points, device=device)[:2]
            
            r1, r2 = rel_valid[idx[0]], rel_valid[idx[1]]
            m1, m2 = metric_valid[idx[0]], metric_valid[idx[1]]
            
            # 求解 scale 和 shift
            # m1 = s * r1 + t
            # m2 = s * r2 + t
            # => s = (m1 - m2) / (r1 - r2)
            # => t = m1 - s * r1
            
            dr = r1 - r2
            if abs(dr) < 1e-6:
                continue
            
            s = (m1 - m2) / dr
            t = m1 - s * r1
            
            # 约束 scale 在合理范围
            if s < 0.1 or s > 100:
                continue
            
            # 计算内点数
            predicted = s * rel_valid + t
            errors = (predicted - metric_valid).abs()
            threshold = inlier_threshold * metric_valid.abs().clamp(min=0.5)
            inliers = (errors < threshold).sum().item()
            
            if inliers > best_inliers:
                best_inliers = inliers
                best_scale = s
                best_shift = t
        
        # 使用所有内点重新拟合
        predicted = best_scale * rel_valid + best_shift
        errors = (predicted - metric_valid).abs()
        threshold = inlier_threshold * metric_valid.abs().clamp(min=0.5)
        inlier_mask = errors < threshold
        
        if inlier_mask.sum() > 2:
            rel_inliers = rel_valid[inlier_mask]
            metric_inliers = metric_valid[inlier_mask]
            
            # 最小二乘拟合
            # [r, 1] @ [s, t]^T = m
            A = torch.stack([rel_inliers, torch.ones_like(rel_inliers)], dim=1)
            b = metric_inliers
            
            # 正规方程: (A^T A)^{-1} A^T b
            AtA = A.T @ A
            Atb = A.T @ b
            
            try:
                solution = torch.linalg.solve(AtA, Atb)
                best_scale = solution[0].clamp(0.1, 100)
                best_shift = solution[1].clamp(-10, 10)
            except:
                pass  # 保持 RANSAC 结果
        
        return best_scale, best_shift
    
    def _align_depth_stereo(
        self,
        img_left: torch.Tensor,
        img_right: torch.Tensor,
        depth_left: torch.Tensor,
        depth_right: torch.Tensor,
        mask_left: torch.Tensor,
        mask_right: torch.Tensor,
        intrinsics: torch.Tensor,
        baseline: torch.Tensor
    ) -> tuple:
        """
        使用块匹配极线搜索对齐深度
        
        Args:
            img_left: [B, 3, H, W] 左图
            img_right: [B, 3, H, W] 右图
            depth_left: [B, 1, H, W] 左视图相对深度
            depth_right: [B, 1, H, W] 右视图相对深度
            mask_left: [B, C, H, W] 左视图掩码
            mask_right: [B, C, H, W] 右视图掩码
            intrinsics: [B, 3, 3] 相机内参
            baseline: [B] 基线距离
            
        Returns:
            depth_left_aligned: [B, 1, H, W] 对齐后的左深度
            depth_right_aligned: [B, 1, H, W] 对齐后的右深度
            scale: [B] 尺度因子
            shift: [B] 偏移量
        """
        B, _, H, W = img_left.shape
        device = img_left.device
        eps = 1e-6
        
        # 获取配置
        depth_align_cfg = getattr(self.cfg, 'depth_align', None)
        num_keypoints = getattr(depth_align_cfg, 'num_keypoints', 500) if depth_align_cfg else 500
        patch_size = getattr(depth_align_cfg, 'patch_size', 11) if depth_align_cfg else 11
        max_disparity = getattr(depth_align_cfg, 'max_disparity', 128) if depth_align_cfg else 128
        min_depth = getattr(depth_align_cfg, 'min_depth', 0.1) if depth_align_cfg else 0.1
        max_depth = getattr(depth_align_cfg, 'max_depth', 100.0) if depth_align_cfg else 100.0
        
        # 计算梯度
        gradient = self._compute_gradient_magnitude(img_left)
        
        # 选取关键点
        keypoints = self._select_keypoints(gradient, mask_left[:, :1], num_keypoints)
        
        # 块匹配
        disparities, confidences = self._patch_matching_sad(
            img_left, img_right, keypoints, patch_size, max_disparity
        )
        
        scales = []
        shifts = []
        aligned_left = []
        aligned_right = []
        
        for b in range(B):
            fx = intrinsics[b, 0, 0]
            bl = baseline[b]
            
            # 获取关键点处的相对深度
            kpts = keypoints[b]  # [N, 2]
            depth_rel_at_kpts = depth_left[b, 0, kpts[:, 0], kpts[:, 1]]  # [N]
            
            # 从视差计算度量深度
            disp = disparities[b]  # [N]
            depth_metric = bl * fx / (disp + eps)  # D = f * B / d
            
            # 过滤无效视差
            valid_disp = disp > 1  # 至少 1 像素视差
            depth_metric = torch.where(valid_disp, depth_metric, torch.zeros_like(depth_metric))
            
            # RANSAC 拟合 scale 和 shift
            scale_b, shift_b = self._ransac_fit_scale_shift(
                depth_rel_at_kpts, depth_metric, confidences[b]
            )
            
            scales.append(scale_b)
            shifts.append(shift_b)
            
            # 应用对齐 (左右视图使用相同的 scale/shift)
            depth_l_aligned = scale_b * depth_left[b] + shift_b
            depth_r_aligned = scale_b * depth_right[b] + shift_b
            
            aligned_left.append(depth_l_aligned)
            aligned_right.append(depth_r_aligned)
        
        depth_left_aligned = torch.stack(aligned_left, dim=0).clamp(min=min_depth, max=max_depth)
        depth_right_aligned = torch.stack(aligned_right, dim=0).clamp(min=min_depth, max=max_depth)
        scale = torch.stack(scales, dim=0)
        shift = torch.stack(shifts, dim=0)
        
        return depth_left_aligned, depth_right_aligned, scale, shift
    
    def forward(self, data, is_train=True):
        """前向传播
        
        支持两种模式:
        1. Metric 模型 (da3metric-large): 直接输出绝对深度，不需要对齐
        2. 相对深度模型 (da3-large): 需要使用块匹配进行深度对齐
        """
        bs = data['lmain']['img'].shape[0]
        device = data['lmain']['img'].device
        _, _, H, W = data['lmain']['img'].shape
        
        # 合并左右视图图像
        image = torch.cat([data['lmain']['img'], data['rmain']['img']], dim=0)
        
        # 归一化到 [0, 1]
        image_da3 = self._normalize_for_da3(image)
        
        # 合并内参用于 metric 模型
        intrinsics = torch.cat([data['lmain']['intr'], data['rmain']['intr']], dim=0)
        
        # DA3 推理 (传入内参用于 metric 模型)
        da3_output = self.da3_estimator(
            image_da3, 
            intrinsics=intrinsics,
            return_features=True, 
            return_entropy=False
        )
        
        # 获取深度
        depth = da3_output['depth']  # [2B, 1, H, W]
        l_depth_raw, r_depth_raw = torch.split(depth, [bs, bs])
        
        # 检查是否为 metric 模型
        is_metric = da3_output.get('is_metric', False)
        
        # 存储原始深度
        data['lmain']['depth_raw'] = l_depth_raw
        data['rmain']['depth_raw'] = r_depth_raw
        
        if is_metric:
            # Metric 模型：DA3 输出绝对深度 (米)
            # 但 depth2pc 期望倒数深度 (1/Z)，需要转换
            
            # 限制深度范围 (避免除零和过大值)
            depth_align_cfg = getattr(self.cfg, 'depth_align', None)
            min_depth = getattr(depth_align_cfg, 'min_depth', 0.1) if depth_align_cfg else 0.1
            max_depth = getattr(depth_align_cfg, 'max_depth', 100.0) if depth_align_cfg else 100.0
            
            l_depth_metric = l_depth_raw.clamp(min=min_depth, max=max_depth)
            r_depth_metric = r_depth_raw.clamp(min=min_depth, max=max_depth)
            
            # 转换为倒数深度: inverse_depth = 1 / depth
            # inverse_depth 范围: [1/max_depth, 1/min_depth] = [0.01, 10]
            l_depth = 1.0 / l_depth_metric
            r_depth = 1.0 / r_depth_metric
            
            data['lmain']['depth'] = l_depth
            data['rmain']['depth'] = r_depth
            data['lmain']['depth_metric'] = l_depth_metric  # 保存原始 metric 深度
            data['rmain']['depth_metric'] = r_depth_metric
            data['depth_scale'] = torch.ones(bs, device=device)
            data['depth_shift'] = torch.zeros(bs, device=device)
            
            metrics = {
                'is_metric': 1.0,
                'depth_min': l_depth_metric.min().item(),
                'depth_max': l_depth_metric.max().item(),
            }
        else:
            # 相对深度模型：使用块匹配进行深度对齐
            baseline = self._compute_baseline(data['lmain']['extr'], data['rmain']['extr'])
            
            l_img_01 = self._normalize_for_da3(data['lmain']['img'])
            r_img_01 = self._normalize_for_da3(data['rmain']['img'])
            
            # _align_depth_stereo 返回的是度量深度 (米)
            l_depth_metric, r_depth_metric, scale, shift = self._align_depth_stereo(
                l_img_01, r_img_01,
                l_depth_raw, r_depth_raw,
                data['lmain']['mask'], data['rmain']['mask'],
                data['lmain']['intr'], baseline
            )
            
            # 转换为倒数深度: inverse_depth = 1 / depth
            # depth2pc 期望倒数深度
            l_depth = 1.0 / (l_depth_metric + 1e-8)
            r_depth = 1.0 / (r_depth_metric + 1e-8)
            
            data['lmain']['depth'] = l_depth
            data['rmain']['depth'] = r_depth
            data['lmain']['depth_metric'] = l_depth_metric  # 保存原始 metric 深度
            data['rmain']['depth_metric'] = r_depth_metric
            data['depth_scale'] = scale
            data['depth_shift'] = shift
            
            metrics = {
                'depth_scale': scale.mean().item(),
                'depth_shift': shift.mean().item(),
                'is_metric': 0.0,
                'depth_min': l_depth_metric.min().item(),
                'depth_max': l_depth_metric.max().item(),
            }
        
        # 创建伪 flow_pred (用于兼容性)
        data['lmain']['flow_pred'] = torch.zeros(bs, 2, H, W, device=device)
        data['rmain']['flow_pred'] = torch.zeros(bs, 2, H, W, device=device)
        
        depth_loss = None
        
        if not self.with_gs_render:
            return data, depth_loss, metrics
        
        # 生成多尺度特征
        with autocast(enabled=self.cfg.raft.mixed_precision):
            img_feat = self.img_encoder(image)
        
        # 预测高斯参数
        data = self.depth2gsparms(image, img_feat, data, bs)
        
        return data, depth_loss, metrics
    
    def depth2gsparms(self, lr_img, lr_img_feat, data, bs):
        """转换深度到高斯参数，使用 xyz 残差调整点云位置"""
        l_depth = data['lmain']['depth']
        r_depth = data['rmain']['depth']
        lr_depth = torch.cat([l_depth, r_depth], dim=0)
        
        # 1. 先计算初始点云 (从原始深度，返回 [B, 3, H, W] 格式)
        l_xyz_init = depth2pc(
            data['lmain']['depth'], 
            data['lmain']['extr'], 
            data['lmain']['intr'],
            return_2d=True
        )  # [B, 3, H, W]
        r_xyz_init = depth2pc(
            data['rmain']['depth'], 
            data['rmain']['extr'], 
            data['rmain']['intr'],
            return_2d=True
        )  # [B, 3, H, W]
        
        # 2. 预测高斯参数和 xyz 残差
        rot_maps, scale_maps, opacity_maps, xyz_res = self.gs_parm_regresser(
            lr_img, lr_depth, lr_img_feat
        )
        
        # 3. 分割 xyz 残差
        l_xyz_res, r_xyz_res = torch.split(xyz_res, [bs, bs])
        
        # 4. 加上 xyz 残差
        l_xyz = l_xyz_init + l_xyz_res  # [B, 3, H, W]
        r_xyz = r_xyz_init + r_xyz_res  # [B, 3, H, W]
        
        # 5. 转换为 [B, H*W, 3] 格式
        data['lmain']['xyz'] = l_xyz.view(bs, 3, -1).permute(0, 2, 1)  # [B, H*W, 3]
        data['rmain']['xyz'] = r_xyz.view(bs, 3, -1).permute(0, 2, 1)  # [B, H*W, 3]
        
        # 6. 基于深度有效性设置 pts_valid
        for view in ['lmain', 'rmain']:
            # 倒数深度 > 0.01 表示真实深度 < 100m（有效范围）
            # 倒数深度 < 10 表示真实深度 > 0.1m（避免过近的点）
            depth_valid = (data[view]['depth'][:, :1, :, :] > 0.01) & \
                         (data[view]['depth'][:, :1, :, :] < 10.0)
            data[view]['pts_valid'] = depth_valid.view(bs, -1)
        
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

