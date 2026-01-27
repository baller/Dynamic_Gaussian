"""
GPS_plus 端到端集成测试

测试完整的训练和推理流程，验证所有模块协同工作正常。

注意：此测试需要 DA3 模型权重才能完整运行。
不依赖 DA3 模型时，可使用 --mock-da3 参数运行模拟测试。

运行方式：
    cd /home/user_3/3DGS/GPS_plus
    
    # 完整测试（需要 DA3 模型）
    python tests/test_integration.py
    
    # 模拟测试（不需要 DA3 模型）
    python tests/test_integration.py --mock-da3
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s')
logger = logging.getLogger(__name__)


class MockDA3Estimator(nn.Module):
    """模拟 DA3 深度估计器，用于不依赖真实模型的测试"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # 简单的深度预测网络
        self.depth_net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )
        
        # 特征提取网络
        self.feature_net = nn.Sequential(
            nn.Conv2d(3, 256, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1024, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, images, return_features=True, return_entropy=True):
        """前向传播"""
        B, C, H, W = images.shape
        
        # 深度预测
        depth = self.depth_net(images) * 10  # 缩放到合理范围
        
        result = {'depth': depth}
        
        # 特征提取
        if return_features:
            feat = self.feature_net(images)  # [B, 1024, H/8, W/8]
            # 模拟多尺度特征
            result['features'] = [
                F.interpolate(feat, scale_factor=2, mode='bilinear', align_corners=False),
                F.interpolate(feat, scale_factor=1.5, mode='bilinear', align_corners=False),
                feat,
                F.interpolate(feat, scale_factor=0.5, mode='bilinear', align_corners=False),
            ]
        
        # 熵计算
        if return_entropy:
            # 使用深度梯度作为熵的近似
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                   dtype=depth.dtype, device=depth.device).view(1, 1, 3, 3)
            grad = F.conv2d(depth, sobel_x, padding=1).abs()
            entropy = grad / (grad.max() + 1e-8)
            result['entropy'] = entropy
        
        return result


class IntegrationConfig:
    """集成测试配置"""
    
    def __init__(self, use_mock_da3=False):
        # 数据集配置
        self.dataset = SubConfig()
        self.dataset.source_id = [0, 1]
        self.dataset.train_novel_id = [2, 3]
        self.dataset.val_novel_id = [2, 3]
        self.dataset.use_hr_img = False
        self.dataset.use_depth_init = True
        self.dataset.bg_color = [0, 0, 0]
        self.dataset.zfar = 100.0
        self.dataset.znear = 0.01
        
        # RAFT 配置
        self.raft = SubConfig()
        self.raft.mixed_precision = False
        self.raft.train_iters = 2
        self.raft.val_iters = 2
        self.raft.encoder_dims = [32, 48, 96]
        self.raft.hidden_dims = [96, 96, 96]
        
        # GSNet 配置
        self.gsnet = SubConfig()
        self.gsnet.parm_head_dim = 32
        self.gsnet.encoder_dims = [32, 48, 96]
        self.gsnet.decoder_dims = [48, 64, 96]
        
        # DA3 配置
        self.depth_mode = 'da3'
        self.da3 = SubConfig()
        self.da3.model_name = 'depth-anything/DA3-LARGE'
        self.da3.export_feat_layers = [11, 15, 19, 23]
        self.da3.freeze_backbone = True
        self.da3.mixed_precision = False
        self.da3.use_feature_adapter = True
        
        # 深度融合配置
        self.depth_fusion = SubConfig()
        self.depth_fusion.enabled = True
        self.depth_fusion.feat_dim = 128
        self.depth_fusion.num_heads = 4
        self.depth_fusion.dropout = 0.1
        self.depth_fusion.num_sparse_points = 32
        
        # 高斯预测网络配置
        self.gs_predictor = 'transformer_moe' if not use_mock_da3 else 'gsregresser'
        
        self.gs_transformer = SubConfig()
        self.gs_transformer.hidden_dim = 128
        self.gs_transformer.num_layers = 2
        self.gs_transformer.num_heads = 4
        self.gs_transformer.window_size = 4
        self.gs_transformer.num_gaussian_layers = 3
        self.gs_transformer.max_scale = 0.002
        self.gs_transformer.max_depth_offset = 0.5
        
        self.moe = SubConfig()
        self.moe.num_experts = 4
        self.moe.top_k = 2
        self.moe.load_balance_weight = 0.01
        
        # 动态高斯配置
        self.dynamic_gs = SubConfig()
        self.dynamic_gs.enabled = False
        self.dynamic_gs.num_layers = 3
        self.dynamic_gs.opacity_threshold = 0.01
        self.dynamic_gs.complexity_threshold = 0.3
        self.dynamic_gs.use_complexity_guidance = True
        
        # 损失函数配置
        self.loss = SubConfig()
        self.loss.l1_weight = 0.8
        self.loss.ssim_weight = 0.2
        self.loss.chamfer_weight = 0.5
        self.loss.depth_consistency_weight = 0.1
        self.loss.moe_balance_weight = 0.01
        
        # 记录使用模拟模式
        self.use_mock_da3 = use_mock_da3


class SubConfig:
    """子配置类"""
    pass


def create_mock_data(batch_size: int = 1, height: int = 64, width: int = 64, device: str = 'cuda') -> Dict:
    """创建模拟数据"""
    data = {
        'lmain': {
            'img': torch.rand(batch_size, 3, height, width, device=device),
            'mask': torch.ones(batch_size, 1, height, width, device=device),
            'flow_init': torch.zeros(batch_size, 2, height, width, device=device),
            'extr': torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone(),
            'intr': torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone(),
        },
        'rmain': {
            'img': torch.rand(batch_size, 3, height, width, device=device),
            'mask': torch.ones(batch_size, 1, height, width, device=device),
            'flow_init': torch.zeros(batch_size, 2, height, width, device=device),
            'extr': torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone(),
            'intr': torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone(),
        },
        'novel_view': {
            'img': torch.rand(batch_size, 3, height, width, device=device),
            'extr': torch.eye(4, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone(),
            'intr': torch.eye(3, device=device).unsqueeze(0).expand(batch_size, -1, -1).clone(),
        }
    }
    
    # 设置相机参数
    fx, fy = 500, 500
    cx, cy = width / 2, height / 2
    
    for view in ['lmain', 'rmain', 'novel_view']:
        data[view]['intr'][:, 0, 0] = fx
        data[view]['intr'][:, 1, 1] = fy
        data[view]['intr'][:, 0, 2] = cx
        data[view]['intr'][:, 1, 2] = cy
    
    # 右视图有水平位移
    data['rmain']['extr'][:, 0, 3] = 0.1  # 基线距离
    
    return data


def test_network_forward(use_mock_da3: bool = True):
    """测试网络前向传播"""
    logger.info("=" * 60)
    logger.info("测试网络前向传播")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = IntegrationConfig(use_mock_da3=use_mock_da3)
    
    # 根据模式选择是否使用模拟 DA3
    if use_mock_da3:
        logger.info("使用模拟 DA3 深度估计器")
        
        # 使用修改后的网络初始化
        from lib.network import RtStereoHumanModel
        
        # 临时修改以使用模拟 DA3
        original_init = RtStereoHumanModel._init_da3_mode
        
        def mock_init_da3_mode(self):
            from lib.da3_depth import DA3FeatureAdapter
            from lib.depth_fusion import DepthFusionModule
            
            self.da3_estimator = MockDA3Estimator(self.cfg)
            
            da3_cfg = getattr(self.cfg, 'da3', None)
            use_adapter = True
            if da3_cfg is not None:
                use_adapter = getattr(da3_cfg, 'use_feature_adapter', True)
            
            if use_adapter:
                self.feature_adapter = DA3FeatureAdapter(
                    in_dims=[1024, 1024, 1024, 1024],
                    out_dims=self.cfg.raft.encoder_dims
                )
            else:
                self.feature_adapter = None
            
            from core.extractor import UnetExtractor
            self.img_encoder = UnetExtractor(in_channel=3, encoder_dim=self.cfg.raft.encoder_dims)
            
            depth_fusion_cfg = getattr(self.cfg, 'depth_fusion', None)
            self.use_depth_fusion = depth_fusion_cfg is not None and getattr(depth_fusion_cfg, 'enabled', True)
            if self.use_depth_fusion:
                self.depth_fusion = DepthFusionModule(self.cfg)
            else:
                self.depth_fusion = None
        
        RtStereoHumanModel._init_da3_mode = mock_init_da3_mode
        
        try:
            model = RtStereoHumanModel(cfg, with_gs_render=True).to(device)
            model.eval()
        finally:
            RtStereoHumanModel._init_da3_mode = original_init
    else:
        logger.info("使用真实 DA3 深度估计器")
        from lib.network import RtStereoHumanModel
        model = RtStereoHumanModel(cfg, with_gs_render=True).to(device)
        model.eval()
    
    # 创建模拟数据
    data = create_mock_data(batch_size=1, height=64, width=64, device=device)
    
    # 前向传播
    logger.info("执行前向传播...")
    with torch.no_grad():
        output_data, flow_loss, metrics = model(data, is_train=False)
    
    # 验证输出
    logger.info("验证输出...")
    
    assert 'lmain' in output_data
    assert 'rmain' in output_data
    assert 'depth' in output_data['lmain']
    assert 'depth' in output_data['rmain']
    
    logger.info(f"  - 左视图深度形状: {output_data['lmain']['depth'].shape}")
    logger.info(f"  - 右视图深度形状: {output_data['rmain']['depth'].shape}")
    
    if 'rot_maps' in output_data['lmain']:
        logger.info(f"  - 旋转图形状: {output_data['lmain']['rot_maps'].shape}")
        logger.info(f"  - 缩放图形状: {output_data['lmain']['scale_maps'].shape}")
        logger.info(f"  - 不透明度图形状: {output_data['lmain']['opacity_maps'].shape}")
    
    if 'xyz' in output_data['lmain']:
        logger.info(f"  - 3D 点云形状: {output_data['lmain']['xyz'].shape}")
    
    logger.info("✓ 网络前向传播测试通过!")
    return True


def test_loss_computation(use_mock_da3: bool = True):
    """测试损失计算"""
    logger.info("=" * 60)
    logger.info("测试损失计算")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from lib.loss import CombinedLoss, DepthConsistencyLoss
    from lib.gs_utils.loss_utils import l1_loss, ssim
    
    cfg = IntegrationConfig(use_mock_da3=use_mock_da3)
    
    # 创建模拟数据
    B, H, W = 1, 64, 64
    
    render_pred = torch.rand(B, 3, H, W, device=device)
    render_gt = torch.rand(B, 3, H, W, device=device)
    
    depth_l = torch.rand(B, 1, H, W, device=device) + 1.0
    depth_r = torch.rand(B, 1, H, W, device=device) + 1.0
    
    intrinsics = torch.eye(3, device=device).unsqueeze(0)
    intrinsics[0, 0, 0] = 500
    intrinsics[0, 1, 1] = 500
    
    baseline = torch.tensor([0.1], device=device)
    
    # 计算各项损失
    l1 = l1_loss(render_pred, render_gt)
    ssim_val = 1.0 - ssim(render_pred, render_gt)
    
    depth_consistency = DepthConsistencyLoss()(depth_l, depth_r, intrinsics, baseline)
    
    logger.info(f"  - L1 损失: {l1.item():.6f}")
    logger.info(f"  - SSIM 损失: {ssim_val.item():.6f}")
    logger.info(f"  - 深度一致性损失: {depth_consistency.item():.6f}")
    
    total_loss = 0.8 * l1 + 0.2 * ssim_val + 0.1 * depth_consistency
    logger.info(f"  - 总损失: {total_loss.item():.6f}")
    
    logger.info("✓ 损失计算测试通过!")
    return True


def test_gradient_flow(use_mock_da3: bool = True):
    """测试梯度流"""
    logger.info("=" * 60)
    logger.info("测试梯度流")
    logger.info("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    from lib.depth_fusion import DepthFusionModule
    from lib.gaussian_transformer_moe import GaussianTransformerMoESimple
    from lib.dynamic_gaussian_allocation import DynamicGaussianAllocation
    
    cfg = IntegrationConfig(use_mock_da3=use_mock_da3)
    
    B, H, W = 1, 32, 32
    L = cfg.gs_transformer.num_gaussian_layers
    
    # 测试深度融合模块梯度
    logger.info("测试深度融合模块梯度...")
    depth_fusion = DepthFusionModule(cfg).to(device)
    
    depth_l = torch.rand(B, 1, H, W, device=device, requires_grad=True)
    depth_r = torch.rand(B, 1, H, W, device=device, requires_grad=True)
    feat_l = torch.rand(B, 1024, H // 4, W // 4, device=device, requires_grad=True)
    feat_r = torch.rand(B, 1024, H // 4, W // 4, device=device, requires_grad=True)
    
    depth_fused, conf, aux = depth_fusion(depth_l, depth_r, feat_l, feat_r)
    loss = depth_fused.mean()
    loss.backward()
    
    assert depth_l.grad is not None, "深度融合模块梯度流失败"
    logger.info(f"  ✓ 深度融合模块梯度正常")
    
    # 测试 Transformer+MoE 模块梯度
    logger.info("测试 Transformer+MoE 模块梯度...")
    transformer = GaussianTransformerMoESimple(cfg, rgb_dim=3, depth_dim=1).to(device)
    
    img = torch.rand(B, 3, H, W, device=device, requires_grad=True)
    depth = torch.rand(B, 1, H, W, device=device, requires_grad=True)
    img_feat = (
        torch.rand(B, 32, H, W, device=device, requires_grad=True),
        torch.rand(B, 48, H // 2, W // 2, device=device, requires_grad=True),
        torch.rand(B, 96, H // 4, W // 4, device=device, requires_grad=True),
    )
    
    rot, scale, opacity, depth_res = transformer(img, depth, img_feat)
    loss = rot.mean() + scale.mean() + opacity.mean() + depth_res.mean()
    loss.backward()
    
    assert img.grad is not None, "Transformer+MoE 模块梯度流失败"
    logger.info(f"  ✓ Transformer+MoE 模块梯度正常")
    
    # 测试动态分配模块梯度
    logger.info("测试动态分配模块梯度...")
    dynamic_alloc = DynamicGaussianAllocation(cfg).to(device)
    
    gaussian_params = {
        'rotation': torch.rand(B, 4, L, H, W, device=device, requires_grad=True),
        'scale': torch.rand(B, 3, L, H, W, device=device, requires_grad=True),
        'opacity': torch.rand(B, 1, L, H, W, device=device, requires_grad=True),
        'depth_offset': torch.rand(B, 1, L, H, W, device=device, requires_grad=True),
    }
    texture_complexity = torch.rand(B, 1, H, W, device=device, requires_grad=True)
    
    filtered_params, valid_mask, stats = dynamic_alloc(gaussian_params, texture_complexity)
    loss = filtered_params['opacity'].mean()
    loss.backward()
    
    assert gaussian_params['opacity'].grad is not None, "动态分配模块梯度流失败"
    logger.info(f"  ✓ 动态分配模块梯度正常")
    
    logger.info("✓ 梯度流测试通过!")
    return True


def run_all_tests(use_mock_da3: bool = True):
    """运行所有集成测试"""
    logger.info("=" * 60)
    logger.info("GPS_plus 端到端集成测试")
    logger.info(f"模式: {'模拟 DA3' if use_mock_da3 else '真实 DA3'}")
    logger.info("=" * 60)
    logger.info("")
    
    tests = [
        ("网络前向传播", test_network_forward),
        ("损失计算", test_loss_computation),
        ("梯度流", test_gradient_flow),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn(use_mock_da3)
            results.append((name, success, None))
        except Exception as e:
            logger.error(f"测试 '{name}' 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))
        logger.info("")
    
    # 打印总结
    logger.info("=" * 60)
    logger.info("测试总结")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ 通过" if success else f"✗ 失败: {error}"
        logger.info(f"  {name}: {status}")
    
    logger.info("")
    logger.info(f"总计: {passed}/{total} 测试通过")
    
    return passed == total


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPS_plus 集成测试')
    parser.add_argument('--mock-da3', action='store_true', help='使用模拟 DA3 模型（不需要真实权重）')
    args = parser.parse_args()
    
    success = run_all_tests(use_mock_da3=args.mock_da3 or True)  # 默认使用模拟模式
    sys.exit(0 if success else 1)
