"""
Depth Anything V3 (DAV3) based depth estimator with sparse stereo matching.

This module replaces RAFT-Stereo with DAV3 for monocular depth estimation,
and uses sparse stereo matching to compute global scale and shift for
converting relative depth to absolute depth.

Key steps:
1. Feature extraction and downsampling
2. Sparse Cost Volume construction
3. Winner-Takes-All with confidence filtering
4. Scale and Shift estimation via least squares
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import torchvision.transforms as T

# Add Depth-Anything-3 to path
sys.path.insert(0, '/home/user_3/3DGS/Depth-Anything-3/src')

# ImageNet normalization constants for DA3
# DA3 expects ImageNet-normalized inputs
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])

try:
    from depth_anything_3.api import DepthAnything3
except ImportError:
    print("Warning: Depth-Anything-3 not found. Please install it first.")
    DepthAnything3 = None


@dataclass
class DAV3Config:
    """Configuration for DAV3 depth estimator."""
    # Model configuration
    model_name: str = "da3-base"
    export_feat_layers: Tuple[int, ...] = (11,)  # Use last layer features
    
    # Feature downsampling configuration
    feature_downsample_size: int = 64  # Downsample features to this resolution
    
    # Cost volume configuration
    max_disparity: int = 32  # Maximum disparity search range (at downsampled resolution)
    
    # Confidence filtering configuration
    confidence_threshold: float = 0.7  # Minimum confidence to keep anchor point
    peak_ratio_threshold: float = 1.3  # Peak should be this much higher than second best
    min_anchor_points: int = 50  # Minimum number of anchor points required
    max_anchor_points: int = 500  # Maximum number of anchor points to use
    
    # Scale/Shift network configuration
    scale_shift_hidden_dim: int = 64  # Hidden dimension for scale-shift network
    
    # Analytical fallback configuration (for comparison/debugging)
    use_ransac: bool = True  # Use RANSAC for analytical fallback
    ransac_iterations: int = 100
    ransac_threshold: float = 0.1  # Inlier threshold (relative to depth range)


class SparseStereoCostVolume(nn.Module):
    """
    Compute sparse correlation cost volume between left and right features.
    
    For each point in the left feature map, search along the epipolar line
    (horizontal direction) in the right feature map.
    """
    
    def __init__(self, max_disparity: int = 32):
        super().__init__()
        self.max_disparity = max_disparity
    
    def forward(
        self, 
        feat_left: torch.Tensor, 
        feat_right: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute cost volume.
        
        Args:
            feat_left: (B, C, H, W) - Left view features (normalized)
            feat_right: (B, C, H, W) - Right view features (normalized)
            
        Returns:
            cost_volume: (B, D, H, W) - Cost volume where D is disparity range
        """
        B, C, H, W = feat_left.shape
        D = self.max_disparity
        
        # Normalize features
        feat_left = F.normalize(feat_left, p=2, dim=1)
        feat_right = F.normalize(feat_right, p=2, dim=1)
        
        # Initialize cost volume
        cost_volume = torch.zeros(B, D, H, W, device=feat_left.device, dtype=feat_left.dtype)
        
        # Compute correlation for each disparity level
        for d in range(D):
            if d == 0:
                # No shift
                cost_volume[:, d, :, :] = (feat_left * feat_right).sum(dim=1)
            else:
                # Shift right features by d pixels to the right
                # feat_left at (u, v) matches feat_right at (u - d, v)
                cost_volume[:, d, :, d:] = (
                    feat_left[:, :, :, d:] * feat_right[:, :, :, :-d]
                ).sum(dim=1)
                # Set invalid positions (where right feature is out of bounds) to -inf
                cost_volume[:, d, :, :d] = -float('inf')
        
        return cost_volume


class WinnerTakesAllWithConfidence(nn.Module):
    """
    Apply Winner-Takes-All to cost volume and compute confidence scores.
    
    Confidence is based on:
    1. Peak value (high correlation = good match)
    2. Peak uniqueness (peak should be significantly higher than second best)
    """
    
    def __init__(
        self, 
        confidence_threshold: float = 0.7,
        peak_ratio_threshold: float = 1.3
    ):
        super().__init__()
        self.confidence_threshold = confidence_threshold
        self.peak_ratio_threshold = peak_ratio_threshold
    
    def forward(
        self, 
        cost_volume: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply WTA and compute confidence.
        
        Args:
            cost_volume: (B, D, H, W) - Cost volume
            
        Returns:
            disparity: (B, H, W) - Estimated disparity
            confidence: (B, H, W) - Confidence scores [0, 1]
            valid_mask: (B, H, W) - Boolean mask for high-confidence points
        """
        B, D, H, W = cost_volume.shape
        
        # Find best and second best matches
        sorted_costs, sorted_indices = torch.sort(cost_volume, dim=1, descending=True)
        best_cost = sorted_costs[:, 0, :, :]  # (B, H, W)
        second_best_cost = sorted_costs[:, 1, :, :]  # (B, H, W)
        disparity = sorted_indices[:, 0, :, :].float()  # (B, H, W)
        
        # Compute confidence based on:
        # 1. Absolute correlation value (normalized to [0, 1])
        peak_confidence = (best_cost + 1) / 2  # Cosine similarity is in [-1, 1]
        
        # 2. Peak uniqueness (ratio of best to second best)
        # Avoid division by zero
        second_best_cost_safe = torch.clamp(second_best_cost, min=1e-6)
        peak_ratio = best_cost / second_best_cost_safe
        uniqueness_confidence = torch.clamp((peak_ratio - 1) / (self.peak_ratio_threshold - 1), 0, 1)
        
        # Combined confidence
        confidence = peak_confidence * uniqueness_confidence
        
        # Valid mask: high confidence points
        valid_mask = (
            (confidence > self.confidence_threshold) & 
            (best_cost > 0) &  # Positive correlation
            (disparity > 0)  # Non-zero disparity (to avoid ambiguous matches at d=0)
        )
        
        return disparity, confidence, valid_mask


class ScaleShiftNetwork(nn.Module):
    """
    Neural network to predict scale and shift for depth alignment.
    
    This network learns to predict the affine transformation parameters
    (scale s, shift t) that convert relative depth to absolute depth:
    D_abs = s * D_rel + t
    
    Architecture:
    - Encoder: Process mono depth, stereo hints, and features
    - Global pooling: Aggregate spatial information
    - MLP head: Predict scale and shift
    
    Inputs:
    - Relative depth map from DAV3
    - Sparse stereo depth hints (from disparity)
    - Confidence map from stereo matching
    - Optional: image features
    """
    
    def __init__(
        self,
        in_channels: int = 4,  # mono_depth + stereo_depth + confidence + valid_mask
        hidden_dim: int = 64,
        num_layers: int = 3
    ):
        super().__init__()
        
        # Convolutional encoder to process spatial information
        layers = []
        current_dim = in_channels
        for i in range(num_layers):
            out_dim = hidden_dim * (2 ** i)
            layers.extend([
                nn.Conv2d(current_dim, out_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_dim),
                nn.ReLU(inplace=True)
            ])
            current_dim = out_dim
        
        self.encoder = nn.Sequential(*layers)
        self.final_dim = current_dim
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # MLP head to predict scale and shift
        self.scale_head = nn.Sequential(
            nn.Linear(self.final_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Softplus()  # Ensure scale > 0
        )
        
        self.shift_head = nn.Sequential(
            nn.Linear(self.final_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # Initialize scale head to output ~1.0 initially
        # Last linear layer bias
        nn.init.constant_(self.scale_head[-2].bias, 0.0)
        # Initialize shift head to output ~0.0 initially
        nn.init.constant_(self.shift_head[-1].bias, 0.0)
    
    def forward(
        self,
        depth_relative: torch.Tensor,
        depth_stereo: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scale and shift.
        
        Args:
            depth_relative: (B, 1, H, W) - Relative depth from DAV3
            depth_stereo: (B, 1, H, W) - Sparse stereo depth hints
            valid_mask: (B, 1, H, W) - Mask for valid stereo points
            confidence: (B, 1, H, W) - Confidence of stereo matches
            
        Returns:
            scale: (B,) - Predicted scale factor
            shift: (B,) - Predicted shift factor
        """
        B = depth_relative.shape[0]
        
        # Normalize inputs for better training stability
        # Normalize mono depth by its median
        depth_rel_norm = self._normalize_depth(depth_relative)
        
        # Normalize stereo depth hints (sparse, use valid mask)
        depth_stereo_norm = self._normalize_depth(depth_stereo, valid_mask)
        
        # Concatenate inputs: [mono_depth, stereo_depth, confidence, valid_mask]
        x = torch.cat([
            depth_rel_norm,
            depth_stereo_norm * valid_mask.float(),  # Mask out invalid stereo
            confidence,
            valid_mask.float()
        ], dim=1)  # (B, 4, H, W)
        
        # Encode
        features = self.encoder(x)  # (B, C, H', W')
        
        # Global pooling
        pooled = self.global_pool(features).view(B, -1)  # (B, C)
        
        # Predict scale and shift
        scale = self.scale_head(pooled).squeeze(-1)  # (B,)
        shift = self.shift_head(pooled).squeeze(-1)  # (B,)
        
        # Denormalize: account for the normalization we applied to inputs
        # The network predicts normalized scale/shift, we need to convert back
        # scale_actual = scale_pred * (stereo_scale / mono_scale)
        # shift_actual = shift_pred * stereo_scale
        
        return scale, shift
    
    def _normalize_depth(
        self, 
        depth: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Normalize depth map for network input."""
        B = depth.shape[0]
        depth_norm = depth.clone()
        
        for b in range(B):
            d = depth[b, 0]
            if mask is not None:
                m = mask[b, 0] > 0.5
                if m.sum() > 0:
                    valid_d = d[m]
                    median = torch.median(valid_d)
                else:
                    median = torch.median(d[d > 0]) if (d > 0).sum() > 0 else torch.tensor(1.0, device=d.device)
            else:
                median = torch.median(d[d > 0]) if (d > 0).sum() > 0 else torch.tensor(1.0, device=d.device)
            
            depth_norm[b] = depth[b] / (median + 1e-8)
        
        return depth_norm


class ScaleShiftNetworkV2(nn.Module):
    """
    Enhanced scale-shift prediction network with attention mechanism.
    
    This version uses cross-attention between mono depth features and 
    sparse stereo hints to better leverage the correlation information.
    """
    
    def __init__(
        self,
        feat_dim: int = 256,
        hidden_dim: int = 128,
        num_heads: int = 4
    ):
        super().__init__()
        
        # Depth encoder (shared for mono and stereo)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Confidence/mask encoder
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, stride=2, padding=1),  # confidence + valid_mask
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, hidden_dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        
        # Global context aggregation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Scale prediction (positive value)
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        # Shift prediction
        self.shift_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        depth_relative: torch.Tensor,
        depth_stereo: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict scale and shift using attention-based fusion.
        """
        B = depth_relative.shape[0]
        
        # Encode mono depth
        mono_feat = self.depth_encoder(depth_relative)  # (B, C, H', W')
        
        # Encode stereo depth (masked)
        stereo_masked = depth_stereo * valid_mask.float()
        stereo_feat = self.depth_encoder(stereo_masked)  # (B, C, H', W')
        
        # Encode confidence and mask
        mask_input = torch.cat([confidence, valid_mask.float()], dim=1)
        mask_feat = self.mask_encoder(mask_input)  # (B, C, H', W')
        
        # Fuse features
        fused = torch.cat([mono_feat, stereo_feat, mask_feat], dim=1)
        fused = self.fusion(fused)  # (B, C, H', W')
        
        # Global pooling
        global_feat = self.global_pool(fused).view(B, -1)  # (B, C)
        
        # Predict scale and shift
        scale_raw = self.scale_head(global_feat).squeeze(-1)  # (B,)
        shift_raw = self.shift_head(global_feat).squeeze(-1)  # (B,)
        
        # Ensure scale is positive (use exp for smooth gradient)
        # Center around 1.0: scale = exp(raw) where raw initialized near 0
        scale = torch.exp(scale_raw)
        shift = shift_raw
        
        return scale, shift


class ScaleShiftEstimator(nn.Module):
    """
    Wrapper that provides both learned and analytical scale-shift estimation.
    
    Can switch between:
    - 'network': Use neural network prediction
    - 'analytical': Use RANSAC/least squares (original method)
    """
    
    def __init__(
        self,
        method: str = 'network',  # 'network' or 'analytical'
        hidden_dim: int = 64,
        use_ransac: bool = True,
        ransac_iterations: int = 100,
        ransac_threshold: float = 0.1,
        min_points: int = 50,
        max_points: int = 500
    ):
        super().__init__()
        self.method = method
        self.min_points = min_points
        self.max_points = max_points
        
        if method == 'network':
            # Use learned network
            self.network = ScaleShiftNetworkV2(hidden_dim=hidden_dim)
        else:
            # Analytical method parameters
            self.use_ransac = use_ransac
            self.ransac_iterations = ransac_iterations
            self.ransac_threshold = ransac_threshold
    
    def forward(
        self,
        depth_relative: torch.Tensor,
        depth_stereo: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate scale and shift.
        """
        if self.method == 'network':
            return self.network(depth_relative, depth_stereo, valid_mask, confidence)
        else:
            return self._analytical_estimation(depth_relative, depth_stereo, valid_mask, confidence)
    
    def _analytical_estimation(
        self,
        depth_relative: torch.Tensor,
        depth_stereo: torch.Tensor,
        valid_mask: torch.Tensor,
        confidence: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Analytical RANSAC/least-squares estimation (original method)."""
        B = depth_relative.shape[0]
        device = depth_relative.device
        dtype = depth_relative.dtype
        
        scale = torch.ones(B, device=device, dtype=dtype)
        shift = torch.zeros(B, device=device, dtype=dtype)
        
        for b in range(B):
            mask_b = valid_mask[b, 0]
            if mask_b.sum() < self.min_points:
                continue
            
            d_rel = depth_relative[b, 0][mask_b]
            d_stereo = depth_stereo[b, 0][mask_b]
            conf = confidence[b, 0][mask_b]
            
            valid_depth = (d_rel > 0) & (d_stereo > 0) & torch.isfinite(d_rel) & torch.isfinite(d_stereo)
            if valid_depth.sum() < self.min_points:
                continue
            
            d_rel = d_rel[valid_depth]
            d_stereo = d_stereo[valid_depth]
            conf = conf[valid_depth]
            
            if len(d_rel) > self.max_points:
                _, top_indices = torch.topk(conf, self.max_points)
                d_rel = d_rel[top_indices]
                d_stereo = d_stereo[top_indices]
                conf = conf[top_indices]
            
            if self.use_ransac:
                s, t = self._ransac_fit(d_rel, d_stereo, conf)
            else:
                s, t = self._weighted_least_squares(d_rel, d_stereo, conf)
            
            scale[b] = s
            shift[b] = t
        
        return scale, shift
    
    def _weighted_least_squares(self, d_rel, d_stereo, weights):
        w = weights / (weights.sum() + 1e-8)
        w_d_rel_sq = (w * d_rel * d_rel).sum()
        w_d_rel = (w * d_rel).sum()
        w_sum = w.sum()
        w_d_rel_d_stereo = (w * d_rel * d_stereo).sum()
        w_d_stereo = (w * d_stereo).sum()
        
        A = torch.stack([
            torch.stack([w_d_rel_sq, w_d_rel]),
            torch.stack([w_d_rel, w_sum])
        ])
        b = torch.stack([w_d_rel_d_stereo, w_d_stereo])
        
        try:
            solution = torch.linalg.solve(A, b)
            scale, shift = solution[0], solution[1]
            if scale <= 0 or not torch.isfinite(scale):
                ratio = d_stereo / (d_rel + 1e-8)
                scale = torch.median(ratio)
                shift = torch.tensor(0.0, device=d_rel.device, dtype=d_rel.dtype)
        except:
            ratio = d_stereo / (d_rel + 1e-8)
            scale = torch.median(ratio)
            shift = torch.tensor(0.0, device=d_rel.device, dtype=d_rel.dtype)
        
        return scale, shift
    
    def _ransac_fit(self, d_rel, d_stereo, weights):
        N = len(d_rel)
        best_inliers = 0
        best_scale = torch.tensor(1.0, device=d_rel.device, dtype=d_rel.dtype)
        best_shift = torch.tensor(0.0, device=d_rel.device, dtype=d_rel.dtype)
        
        depth_range = d_stereo.max() - d_stereo.min()
        if depth_range < 1e-6:
            return best_scale, best_shift
        threshold = self.ransac_threshold * depth_range
        
        for _ in range(self.ransac_iterations):
            indices = torch.randperm(N, device=d_rel.device)[:2]
            d_r, d_s = d_rel[indices], d_stereo[indices]
            
            if abs(d_r[0] - d_r[1]) < 1e-6:
                continue
            
            scale = (d_s[0] - d_s[1]) / (d_r[0] - d_r[1] + 1e-8)
            shift = d_s[0] - scale * d_r[0]
            
            if scale <= 0:
                continue
            
            residuals = torch.abs(scale * d_rel + shift - d_stereo)
            inliers = residuals < threshold
            num_inliers = inliers.sum().item()
            
            if num_inliers > best_inliers:
                best_inliers = num_inliers
                if num_inliers >= 2:
                    best_scale, best_shift = self._weighted_least_squares(
                        d_rel[inliers], d_stereo[inliers], weights[inliers]
                    )
        
        return best_scale, best_shift


class DAV3DepthEstimator(nn.Module):
    """
    Main depth estimator that integrates DAV3 with sparse stereo matching.
    
    Pipeline:
    1. Extract monocular depth and features from DAV3
    2. Downsample features for efficiency
    3. Compute sparse cost volume between left and right features
    4. Apply WTA with confidence filtering to get anchor points
    5. Estimate scale and shift using anchor points
    6. Convert relative depth to absolute depth
    """
    
    # ViT patch size for DAV3
    PATCH_SIZE = 14
    
    def __init__(self, cfg: DAV3Config):
        super().__init__()
        self.cfg = cfg
        
        # Load DAV3 model
        if DepthAnything3 is None:
            raise ImportError("Depth-Anything-3 is required but not found.")
        
        self.dav3 = DepthAnything3(model_name=cfg.model_name)
        
        # Freeze DAV3 weights (we don't train DAV3, only use it for feature extraction)
        for param in self.dav3.parameters():
            param.requires_grad = False
        self.dav3.eval()
        
        # Determine feature dimension based on model
        # Auxiliary features from export_feat_layers are NOT concatenated
        # They have the raw embed_dim from the ViT backbone
        if 'base' in cfg.model_name or 'small' in cfg.model_name:
            self.feat_dim = 768  # ViT-B embed_dim
        elif 'large' in cfg.model_name:
            self.feat_dim = 1024  # ViT-L embed_dim
        elif 'giant' in cfg.model_name:
            self.feat_dim = 1536  # ViT-G embed_dim
        else:
            self.feat_dim = 768  # default to base
        
        print(f"[DAV3] Model: {cfg.model_name}, Feature dim: {self.feat_dim}")
        
        # Sparse stereo components
        self.cost_volume = SparseStereoCostVolume(max_disparity=cfg.max_disparity)
        self.wta = WinnerTakesAllWithConfidence(
            confidence_threshold=cfg.confidence_threshold,
            peak_ratio_threshold=cfg.peak_ratio_threshold
        )
        
        # Scale-shift estimation network (learnable)
        self.scale_shift_estimator = ScaleShiftEstimator(
            method='network',  # Use learned network instead of analytical
            hidden_dim=cfg.scale_shift_hidden_dim,
            use_ransac=cfg.use_ransac,
            ransac_iterations=cfg.ransac_iterations,
            ransac_threshold=cfg.ransac_threshold,
            min_points=cfg.min_anchor_points,
            max_points=cfg.max_anchor_points
        )
        
        # Feature projection layer (learnable)
        # Project high-dim features to lower dim for efficiency
        self.feat_proj = nn.Conv2d(self.feat_dim, 256, kernel_size=1, bias=False)
    
    def forward(
        self, 
        data: Dict, 
        is_train: bool = True
    ) -> Dict:
        """
        Estimate absolute depth for stereo pair.
        
        Args:
            data: Dictionary containing:
                - 'lmain': {'img': (B, 3, H, W), 'intr': (B, 3, 3), ...}
                - 'rmain': {'img': (B, 3, H, W), 'intr': (B, 3, 3), ...}
            is_train: Whether in training mode
            
        Returns:
            data: Updated dictionary with depth estimates
        """
        bs = data['lmain']['img'].shape[0]
        device = data['lmain']['img'].device
        
        # Get images
        img_left = data['lmain']['img']  # (B, 3, H, W)
        img_right = data['rmain']['img']  # (B, 3, H, W)
        
        # Get camera parameters for stereo depth computation
        # Baseline and focal length are needed for disparity -> depth conversion
        focal_length = data['lmain']['intr'][:, 0, 0]  # (B,)
        baseline = self._compute_baseline(data)  # (B,)
        
        # Step 1: Get monocular depth and features from DAV3
        # DAV3 is frozen, so we use torch.no_grad() for efficiency
        depth_left, feat_left = self._extract_depth_and_features(img_left)
        depth_right, feat_right = self._extract_depth_and_features(img_right)
        
        # Clone tensors to make them normal tensors that can be used with autograd
        # This is necessary because DAV3 runs in inference mode
        depth_left = depth_left.clone()
        depth_right = depth_right.clone()
        feat_left = feat_left.clone()
        feat_right = feat_right.clone()
        
        # Step 2: Downsample features (this involves learnable projection)
        feat_left_ds = self._downsample_features(feat_left)  # (B, C, H_ds, W_ds)
        feat_right_ds = self._downsample_features(feat_right)
        
        # Also downsample depth for scale/shift estimation (no gradient needed)
        H_ds, W_ds = feat_left_ds.shape[-2:]
        with torch.no_grad():
            depth_left_ds = F.interpolate(depth_left, size=(H_ds, W_ds), mode='bilinear', align_corners=False)
        
        # Step 3: Compute cost volume (no gradient needed for stereo matching)
        with torch.no_grad():
            # Detach features for cost volume computation
            cost_volume = self.cost_volume(feat_left_ds.detach(), feat_right_ds.detach())  # (B, D, H_ds, W_ds)
            
            # Step 4: WTA with confidence
            disparity_ds, confidence, valid_mask = self.wta(cost_volume)  # (B, H_ds, W_ds)
            
            # Step 5: Convert disparity to depth at anchor points
            # Scale disparity back to original resolution
            scale_factor = img_left.shape[-1] / W_ds
            disparity_scaled = disparity_ds * scale_factor
            
            # depth = focal_length * baseline / disparity
            depth_stereo_ds = self._disparity_to_depth(
                disparity_scaled, focal_length, baseline
            ).unsqueeze(1)  # (B, 1, H_ds, W_ds)
        
        # Step 6: Estimate scale and shift using learned network
        # This part has gradients - the network learns to predict s and t
        scale, shift = self.scale_shift_estimator(
            depth_left_ds,  # Keep gradient for network input
            depth_stereo_ds.detach(),  # Stereo depth is just a hint
            valid_mask.unsqueeze(1).detach(),
            confidence.unsqueeze(1).detach()
        )
        
        # Step 7: Convert relative depth to absolute depth
        # D_abs = s * D_mono + t
        # Gradients flow through scale and shift (learned), enabling end-to-end training
        scale_view = scale.view(bs, 1, 1, 1)
        shift_view = shift.view(bs, 1, 1, 1)
        
        # depth_left is detached from DAV3, but scale/shift have gradients
        depth_abs_left = scale_view * depth_left.detach() + shift_view
        depth_abs_right = scale_view * depth_right.detach() + shift_view
        
        # Ensure depth is positive
        depth_abs_left = torch.clamp(depth_abs_left, min=0.01)
        depth_abs_right = torch.clamp(depth_abs_right, min=0.01)
        
        # Store results
        data['lmain']['depth'] = depth_abs_left
        data['rmain']['depth'] = depth_abs_right
        data['lmain']['depth_mono'] = depth_left.detach()  # Store mono depth for reference
        data['rmain']['depth_mono'] = depth_right.detach()
        data['lmain']['dav3_features'] = feat_left.detach()  # Store features for visualization
        data['rmain']['dav3_features'] = feat_right.detach()
        data['scale'] = scale.detach()
        data['shift'] = shift.detach()
        
        # Store sparse stereo info for debugging/visualization
        data['sparse_stereo'] = {
            'disparity': disparity_ds.detach(),
            'confidence': confidence.detach(),
            'valid_mask': valid_mask.detach(),
            'num_anchors': valid_mask.sum(dim=(-2, -1)).detach()
        }
        
        return data
    
    def _extract_depth_and_features(
        self, 
        img: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract monocular depth and intermediate features from DAV3.
        
        Args:
            img: (B, 3, H, W) - Input image (in [-1, 1] range from human_loader)
            
        Returns:
            depth: (B, 1, H, W) - Relative depth (same size as input)
            features: (B, C, H_feat, W_feat) - Intermediate features
        """
        B, _, H, W = img.shape
        device = img.device
        
        # Debug: Print input image range (only once)
        if not hasattr(self, '_input_debug_printed'):
            img_min, img_max = img.min().item(), img.max().item()
            print(f"[DAV3 Debug] Input image range: min={img_min:.4f}, max={img_max:.4f}")
            print(f"[DAV3 Debug] Input image shape: {img.shape}")
            self._input_debug_printed = True
        
        # DAV3 ViT requires input size to be multiple of patch_size (14)
        H_pad = (self.PATCH_SIZE - H % self.PATCH_SIZE) % self.PATCH_SIZE
        W_pad = (self.PATCH_SIZE - W % self.PATCH_SIZE) % self.PATCH_SIZE
        H_new = H + H_pad
        W_new = W + W_pad
        
        # Resize image to be multiple of patch size
        if H_pad > 0 or W_pad > 0:
            img_resized = F.interpolate(img, size=(H_new, W_new), mode='bilinear', align_corners=False)
        else:
            img_resized = img
        
        # CRITICAL: Apply proper normalization before passing to DAV3
        # Detect input range and convert to [0, 1] first
        img_min = img_resized.min().item()
        img_max = img_resized.max().item()
        
        if img_min >= -1.1 and img_max <= 1.1 and img_min < 0:
            # Input is in [-1, 1] range (from human_loader: 2 * (img / 255) - 1)
            # Convert to [0, 1]: img_01 = (img + 1) / 2
            img_01 = (img_resized + 1.0) / 2.0
        elif img_min >= 0 and img_max <= 1.1:
            # Input is already in [0, 1] range
            img_01 = img_resized
        elif img_max > 1.1:
            # Input might be in [0, 255] range
            img_01 = img_resized / 255.0
        else:
            # Unknown range, assume [-1, 1] and convert
            img_01 = (img_resized + 1.0) / 2.0
        
        # Clamp to [0, 1] to be safe
        img_01 = torch.clamp(img_01, 0.0, 1.0)
        
        # Now apply ImageNet normalization
        # DAV3 expects ImageNet-normalized inputs (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # Without this normalization, the depth output will be incorrect (sparse/dotted pattern)
        mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(device=device, dtype=img_01.dtype)
        std = IMAGENET_STD.view(1, 3, 1, 1).to(device=device, dtype=img_01.dtype)
        img_normalized = (img_01 - mean) / std
        
        # Debug: Print normalized image range (only once)
        if not hasattr(self, '_norm_debug_printed'):
            norm_min, norm_max = img_normalized.min().item(), img_normalized.max().item()
            print(f"[DAV3 Debug] After ImageNet normalization: min={norm_min:.4f}, max={norm_max:.4f}")
            self._norm_debug_printed = True
        
        # Prepare input for DAV3 (expects (B, N, 3, H, W) where N is number of views)
        img_dav3 = img_normalized.unsqueeze(1)  # (B, 1, 3, H_new, W_new)
        
        # Forward pass through DAV3 with feature export
        # DAV3 is already in eval mode with frozen weights
        self.dav3.eval()
        with torch.no_grad():
            output = self.dav3.forward(
                img_dav3,
                export_feat_layers=list(self.cfg.export_feat_layers)
            )
        
        # Get depth
        depth = output['depth']  # (B, N, H, W) - should be full resolution from DualDPT
        
        # Debug: Print shape to verify resolution
        if not hasattr(self, '_debug_printed'):
            print(f"[DAV3 Debug] Output depth shape: {depth.shape}")
            print(f"[DAV3 Debug] Input image shape: (B, 3, {H}, {W})")
            print(f"[DAV3 Debug] Depth value range: min={depth.min().item():.4f}, max={depth.max().item():.4f}")
            self._debug_printed = True
        
        if depth.dim() == 4:
            depth = depth[:, 0]  # Take first view: (B, H', W')
        
        # Resize depth to input resolution (should be minimal change if DualDPT works correctly)
        depth = depth.unsqueeze(1)  # (B, 1, H', W')
        if depth.shape[-2:] != (H, W):
            depth = F.interpolate(depth.float(), size=(H, W), mode='bilinear', align_corners=False)
        else:
            depth = depth.float()
        
        # Get features from auxiliary outputs
        feat_key = f"feat_layer_{self.cfg.export_feat_layers[-1]}"
        features = output['aux'][feat_key]  # (B, N, H_patch, W_patch, embed_dim)
        features = features[:, 0]  # Take first view: (B, H_patch, W_patch, embed_dim)
        features = features.permute(0, 3, 1, 2).contiguous()  # (B, embed_dim, H_patch, W_patch)
        features = features.float()  # Ensure float32
        
        return depth, features
    
    def _downsample_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Downsample features to target resolution.
        
        Args:
            features: (B, C, H, W) - Input features
            
        Returns:
            features_ds: (B, C', H_ds, W_ds) - Downsampled features
        """
        target_size = self.cfg.feature_downsample_size
        
        # First project to lower dimension (learnable)
        features = self.feat_proj(features)
        
        # Then resize to target resolution
        features_ds = F.interpolate(
            features, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        return features_ds
    
    def _compute_baseline(self, data: Dict) -> torch.Tensor:
        """
        Compute baseline from camera extrinsics.
        
        Args:
            data: Dictionary with camera parameters
            
        Returns:
            baseline: (B,) - Baseline distance
        """
        # Get camera positions
        extr_left = data['lmain']['extr']  # (B, 3, 4)
        extr_right = data['rmain']['extr']  # (B, 3, 4)
        
        # Camera position is -R^T @ t
        R_left = extr_left[:, :3, :3]  # (B, 3, 3)
        t_left = extr_left[:, :3, 3:]  # (B, 3, 1)
        R_right = extr_right[:, :3, :3]
        t_right = extr_right[:, :3, 3:]
        
        pos_left = -torch.bmm(R_left.transpose(1, 2), t_left).squeeze(-1)  # (B, 3)
        pos_right = -torch.bmm(R_right.transpose(1, 2), t_right).squeeze(-1)  # (B, 3)
        
        # Baseline is the distance between camera centers
        baseline = torch.norm(pos_right - pos_left, dim=-1)  # (B,)
        
        # Ensure positive baseline
        baseline = torch.clamp(baseline, min=0.01)
        
        return baseline
    
    def _disparity_to_depth(
        self, 
        disparity: torch.Tensor, 
        focal_length: torch.Tensor,
        baseline: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert disparity to depth: depth = focal_length * baseline / disparity
        
        Args:
            disparity: (B, H, W) - Disparity map
            focal_length: (B,) - Focal length
            baseline: (B,) - Baseline
            
        Returns:
            depth: (B, H, W) - Depth map
        """
        B = disparity.shape[0]
        
        # Avoid division by zero
        disparity_safe = torch.clamp(disparity, min=0.1)
        
        # depth = f * B / d
        fb = focal_length * baseline  # (B,)
        depth = fb.view(B, 1, 1) / disparity_safe
        
        # Clamp depth to reasonable range
        depth = torch.clamp(depth, min=0.1, max=100.0)
        
        return depth


def create_dav3_depth_estimator(cfg) -> DAV3DepthEstimator:
    """
    Factory function to create DAV3 depth estimator from config.
    
    Args:
        cfg: Configuration object with dav3 section
        
    Returns:
        DAV3DepthEstimator instance
    """
    dav3_cfg = DAV3Config(
        model_name=getattr(cfg.dav3, 'model_name', 'da3-base'),
        export_feat_layers=tuple(getattr(cfg.dav3, 'export_feat_layers', [11])),
        feature_downsample_size=getattr(cfg.dav3, 'feature_downsample_size', 64),
        max_disparity=getattr(cfg.dav3, 'max_disparity', 32),
        confidence_threshold=getattr(cfg.dav3, 'confidence_threshold', 0.7),
        peak_ratio_threshold=getattr(cfg.dav3, 'peak_ratio_threshold', 1.3),
        min_anchor_points=getattr(cfg.dav3, 'min_anchor_points', 50),
        max_anchor_points=getattr(cfg.dav3, 'max_anchor_points', 500),
        scale_shift_hidden_dim=getattr(cfg.dav3, 'scale_shift_hidden_dim', 64),
        use_ransac=getattr(cfg.dav3, 'use_ransac', True),
        ransac_iterations=getattr(cfg.dav3, 'ransac_iterations', 100),
        ransac_threshold=getattr(cfg.dav3, 'ransac_threshold', 0.1),
    )
    
    return DAV3DepthEstimator(dav3_cfg)
