"""
GPS_plus 架构改进模块单元测试

测试内容：
1. DA3 深度估计器
2. 深度融合模块
3. Transformer+MoE 高斯预测网络
4. 动态高斯分配模块
5. 损失函数

运行方式：
    cd /home/user_3/3DGS/GPS_plus
    python -m pytest tests/test_modules.py -v
    
或单独测试（无需 DA3 模型）：
    python tests/test_modules.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import unittest
from typing import Dict


class MockConfig:
    """模拟配置对象"""
    
    def __init__(self):
        # RAFT 配置
        self.raft = MockSubConfig()
        self.raft.encoder_dims = [32, 48, 96]
        self.raft.hidden_dims = [96, 96, 96]
        self.raft.mixed_precision = False
        
        # GSNet 配置
        self.gsnet = MockSubConfig()
        self.gsnet.parm_head_dim = 32
        self.gsnet.encoder_dims = [32, 48, 96]
        self.gsnet.decoder_dims = [48, 64, 96]
        
        # 深度融合配置
        self.depth_fusion = MockSubConfig()
        self.depth_fusion.enabled = True
        self.depth_fusion.feat_dim = 128
        self.depth_fusion.num_heads = 4
        self.depth_fusion.dropout = 0.1
        self.depth_fusion.num_sparse_points = 64
        
        # Transformer+MoE 配置
        self.gs_transformer = MockSubConfig()
        self.gs_transformer.hidden_dim = 128
        self.gs_transformer.num_layers = 2
        self.gs_transformer.num_heads = 4
        self.gs_transformer.window_size = 4
        self.gs_transformer.num_gaussian_layers = 3
        self.gs_transformer.max_scale = 0.002
        self.gs_transformer.max_depth_offset = 0.5
        
        # MoE 配置
        self.moe = MockSubConfig()
        self.moe.num_experts = 4
        self.moe.top_k = 2
        self.moe.load_balance_weight = 0.01
        
        # 动态高斯配置
        self.dynamic_gs = MockSubConfig()
        self.dynamic_gs.enabled = True
        self.dynamic_gs.num_layers = 3
        self.dynamic_gs.opacity_threshold = 0.01
        self.dynamic_gs.complexity_threshold = 0.3
        self.dynamic_gs.use_complexity_guidance = True
        self.dynamic_gs.soft_pruning = True
        
        # 损失函数配置
        self.loss = MockSubConfig()
        self.loss.l1_weight = 0.8
        self.loss.ssim_weight = 0.2
        self.loss.chamfer_weight = 0.5
        self.loss.depth_consistency_weight = 0.1
        self.loss.moe_balance_weight = 0.01
        self.loss.allocation_entropy_weight = 0.001


class MockSubConfig:
    """模拟子配置对象"""
    pass


class TestDepthFusion(unittest.TestCase):
    """深度融合模块测试"""
    
    def setUp(self):
        self.cfg = MockConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_depth_fusion_forward(self):
        """测试深度融合前向传播"""
        from lib.depth_fusion import DepthFusionModule
        
        module = DepthFusionModule(self.cfg).to(self.device)
        
        B, H, W = 2, 32, 32
        C = 1024  # DA3 特征维度
        
        depth_l = torch.rand(B, 1, H, W, device=self.device)
        depth_r = torch.rand(B, 1, H, W, device=self.device)
        feat_l = torch.rand(B, C, H // 4, W // 4, device=self.device)
        feat_r = torch.rand(B, C, H // 4, W // 4, device=self.device)
        intrinsics = torch.eye(3, device=self.device).unsqueeze(0).expand(B, -1, -1)
        baseline = torch.tensor([0.1] * B, device=self.device)
        
        depth_fused, confidence, aux = module(
            depth_l, depth_r, feat_l, feat_r, intrinsics, baseline
        )
        
        # 检查输出形状
        self.assertEqual(depth_fused.shape, (B, 1, H, W))
        self.assertEqual(confidence.shape, (B, 1, H, W))
        self.assertIn('scale', aux)
        self.assertIn('shift', aux)
        
        print(f"✓ 深度融合前向传播测试通过")
        print(f"  - 输入深度形状: {depth_l.shape}")
        print(f"  - 输出深度形状: {depth_fused.shape}")
        print(f"  - Scale 均值: {aux['scale'].mean().item():.4f}")
    
    def test_cross_view_attention(self):
        """测试跨视图注意力"""
        from lib.depth_fusion import CrossViewAttention
        
        B, C, H, W = 2, 128, 16, 16
        
        attn = CrossViewAttention(dim=C, num_heads=4).to(self.device)
        
        feat_l = torch.rand(B, C, H, W, device=self.device)
        feat_r = torch.rand(B, C, H, W, device=self.device)
        
        out_l, out_r = attn(feat_l, feat_r)
        
        self.assertEqual(out_l.shape, feat_l.shape)
        self.assertEqual(out_r.shape, feat_r.shape)
        
        print(f"✓ 跨视图注意力测试通过")


class TestTransformerMoE(unittest.TestCase):
    """Transformer+MoE 模块测试"""
    
    def setUp(self):
        self.cfg = MockConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_window_attention(self):
        """测试窗口注意力"""
        from lib.gaussian_transformer_moe import WindowAttention
        
        B, N, C = 2, 64, 128
        H, W = 8, 8
        
        attn = WindowAttention(dim=C, num_heads=4, window_size=4).to(self.device)
        
        x = torch.rand(B, N, C, device=self.device)
        out = attn(x, H, W)
        
        self.assertEqual(out.shape, x.shape)
        print(f"✓ 窗口注意力测试通过")
    
    def test_moe_mlp(self):
        """测试 MoE MLP"""
        from lib.gaussian_transformer_moe import MoEMLP
        
        B, N, C = 2, 64, 128
        
        moe = MoEMLP(
            dim=C, 
            hidden_dim=C * 4, 
            num_experts=4, 
            top_k=2
        ).to(self.device)
        
        x = torch.rand(B, N, C, device=self.device)
        out, router_weights, lb_loss = moe(x)
        
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(router_weights.shape, (B, N, 4))  # num_experts=4
        self.assertIsInstance(lb_loss.item(), float)
        
        print(f"✓ MoE MLP 测试通过")
        print(f"  - 路由权重形状: {router_weights.shape}")
        print(f"  - 负载均衡损失: {lb_loss.item():.6f}")
    
    def test_transformer_block_moe(self):
        """测试 Transformer 块"""
        from lib.gaussian_transformer_moe import TransformerBlockMoE
        
        B, C, H, W = 2, 128, 16, 16
        
        block = TransformerBlockMoE(
            dim=C,
            num_heads=4,
            window_size=4,
            num_experts=4,
            top_k=2,
            use_cross_view=True
        ).to(self.device)
        
        x = torch.rand(B, C, H, W, device=self.device)
        x_other = torch.rand(B, C, H, W, device=self.device)
        
        out, router_weights, lb_loss = block(x, x_other)
        
        self.assertEqual(out.shape, x.shape)
        print(f"✓ Transformer 块测试通过")
    
    def test_gaussian_transformer_moe_simple(self):
        """测试简化版 Transformer+MoE"""
        from lib.gaussian_transformer_moe import GaussianTransformerMoESimple
        
        B, H, W = 2, 32, 32
        
        model = GaussianTransformerMoESimple(self.cfg, rgb_dim=3, depth_dim=1).to(self.device)
        
        img = torch.rand(B, 3, H, W, device=self.device)
        depth = torch.rand(B, 1, H, W, device=self.device)
        
        # 模拟 img_feat (3 个尺度)
        img_feat = (
            torch.rand(B, 32, H, W, device=self.device),
            torch.rand(B, 48, H // 2, W // 2, device=self.device),
            torch.rand(B, 96, H // 4, W // 4, device=self.device),
        )
        
        rot, scale, opacity, depth_res = model(img, depth, img_feat)
        
        self.assertEqual(rot.shape, (B, 4, H, W))
        self.assertEqual(scale.shape, (B, 3, H, W))
        self.assertEqual(opacity.shape, (B, 1, H, W))
        self.assertEqual(depth_res.shape, (B, 1, H, W))
        
        print(f"✓ GaussianTransformerMoESimple 测试通过")
        print(f"  - 旋转输出形状: {rot.shape}")
        print(f"  - 缩放输出形状: {scale.shape}")


class TestDynamicAllocation(unittest.TestCase):
    """动态高斯分配模块测试"""
    
    def setUp(self):
        self.cfg = MockConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_dynamic_allocation_forward(self):
        """测试动态分配前向传播"""
        from lib.dynamic_gaussian_allocation import DynamicGaussianAllocation
        
        module = DynamicGaussianAllocation(self.cfg).to(self.device)
        
        B, L, H, W = 2, 3, 16, 16
        
        gaussian_params = {
            'rotation': torch.rand(B, 4, L, H, W, device=self.device),
            'scale': torch.rand(B, 3, L, H, W, device=self.device),
            'opacity': torch.rand(B, 1, L, H, W, device=self.device),
            'depth_offset': torch.rand(B, 1, L, H, W, device=self.device),
        }
        texture_complexity = torch.rand(B, 1, H, W, device=self.device)
        
        filtered_params, valid_mask, stats = module(gaussian_params, texture_complexity)
        
        self.assertEqual(valid_mask.shape, (B, L, H, W))
        self.assertIn('pruning_ratio', stats)
        self.assertIn('avg_layers_per_pixel', stats)
        
        print(f"✓ 动态分配前向传播测试通过")
        print(f"  - 裁剪率: {stats['pruning_ratio']:.4f}")
        print(f"  - 平均每像素层数: {stats['avg_layers_per_pixel']:.4f}")
    
    def test_flatten_gaussians(self):
        """测试高斯展平"""
        from lib.dynamic_gaussian_allocation import DynamicGaussianAllocation
        
        module = DynamicGaussianAllocation(self.cfg).to(self.device)
        
        B, L, H, W = 2, 3, 16, 16
        
        gaussian_params = {
            'rotation': torch.rand(B, 4, L, H, W, device=self.device),
            'scale': torch.rand(B, 3, L, H, W, device=self.device),
            'opacity': torch.rand(B, 1, L, H, W, device=self.device),
            'depth_offset': torch.rand(B, 1, L, H, W, device=self.device) * 0.1,
        }
        valid_mask = torch.ones(B, L, H, W, device=self.device, dtype=torch.bool)
        depth = torch.rand(B, 1, H, W, device=self.device) + 1.0  # 确保正深度
        
        intrinsics = torch.eye(3, device=self.device).unsqueeze(0).expand(B, -1, -1).clone()
        intrinsics[:, 0, 0] = 500  # fx
        intrinsics[:, 1, 1] = 500  # fy
        intrinsics[:, 0, 2] = W / 2  # cx
        intrinsics[:, 1, 2] = H / 2  # cy
        
        extrinsics = torch.eye(4, device=self.device).unsqueeze(0).expand(B, -1, -1).clone()
        extrinsics = extrinsics[:, :3, :]  # [B, 3, 4]
        
        flattened = module.flatten_gaussians(
            gaussian_params, valid_mask, depth, intrinsics, extrinsics
        )
        
        N = L * H * W
        self.assertEqual(flattened['xyz'].shape, (B, N, 3))
        self.assertEqual(flattened['rotation'].shape, (B, N, 4))
        self.assertEqual(flattened['scale'].shape, (B, N, 3))
        self.assertEqual(flattened['opacity'].shape, (B, N))
        
        print(f"✓ 高斯展平测试通过")
        print(f"  - 展平后点数: {N}")
        print(f"  - XYZ 形状: {flattened['xyz'].shape}")


class TestLossFunctions(unittest.TestCase):
    """损失函数测试"""
    
    def setUp(self):
        self.cfg = MockConfig()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def test_depth_consistency_loss(self):
        """测试深度一致性损失"""
        from lib.loss import DepthConsistencyLoss
        
        loss_fn = DepthConsistencyLoss().to(self.device)
        
        B, H, W = 2, 32, 32
        
        depth_l = torch.rand(B, 1, H, W, device=self.device) + 1.0
        depth_r = depth_l + torch.randn_like(depth_l) * 0.1  # 加少量噪声
        
        intrinsics = torch.eye(3, device=self.device).unsqueeze(0).expand(B, -1, -1).clone()
        intrinsics[:, 0, 0] = 500  # fx
        
        baseline = torch.tensor([0.1] * B, device=self.device)
        
        loss = loss_fn(depth_l, depth_r, intrinsics, baseline)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)
        
        print(f"✓ 深度一致性损失测试通过")
        print(f"  - 损失值: {loss.item():.6f}")
    
    def test_moe_balance_loss(self):
        """测试 MoE 负载均衡损失"""
        from lib.loss import MoEBalanceLoss
        
        loss_fn = MoEBalanceLoss(num_experts=4).to(self.device)
        
        B, N = 2, 64
        
        # 均匀分布
        router_weights_uniform = torch.ones(B, N, 4, device=self.device) / 4
        loss_uniform = loss_fn(router_weights_uniform)
        
        # 不均匀分布
        router_weights_skewed = torch.zeros(B, N, 4, device=self.device)
        router_weights_skewed[:, :, 0] = 1.0  # 所有都选第一个专家
        loss_skewed = loss_fn(router_weights_skewed)
        
        print(f"✓ MoE 负载均衡损失测试通过")
        print(f"  - 均匀分布损失: {loss_uniform.item():.6f}")
        print(f"  - 不均匀分布损失: {loss_skewed.item():.6f}")
    
    def test_edge_aware_smooth_loss(self):
        """测试边缘感知平滑损失"""
        from lib.loss import EdgeAwareDepthSmoothLoss
        
        loss_fn = EdgeAwareDepthSmoothLoss().to(self.device)
        
        B, H, W = 2, 32, 32
        
        depth = torch.rand(B, 1, H, W, device=self.device)
        image = torch.rand(B, 3, H, W, device=self.device)
        
        loss = loss_fn(depth, image)
        
        self.assertIsInstance(loss.item(), float)
        self.assertGreaterEqual(loss.item(), 0)
        
        print(f"✓ 边缘感知平滑损失测试通过")
        print(f"  - 损失值: {loss.item():.6f}")


def run_tests():
    """运行所有测试"""
    print("=" * 60)
    print("GPS_plus 架构改进模块单元测试")
    print("=" * 60)
    print()
    
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加测试类
    suite.addTests(loader.loadTestsFromTestCase(TestDepthFusion))
    suite.addTests(loader.loadTestsFromTestCase(TestTransformerMoE))
    suite.addTests(loader.loadTestsFromTestCase(TestDynamicAllocation))
    suite.addTests(loader.loadTestsFromTestCase(TestLossFunctions))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 打印总结
    print()
    print("=" * 60)
    print(f"测试完成: {result.testsRun} 个测试")
    print(f"成功: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print("=" * 60)
    
    return result


if __name__ == '__main__':
    run_tests()
