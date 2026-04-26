from yacs.config import CfgNode as CN


class ConfigStereoHuman:
    def __init__(self):
        self.cfg = CN()
        self.cfg.name = ''
        self.cfg.stage1_ckpt = None
        self.cfg.restore_ckpt = None
        self.cfg.lr = 0.0
        self.cfg.wdecay = 0.0
        self.cfg.batch_size = 0
        self.cfg.num_steps = 0
        
        # 深度估计模式: 'raft' 或 'da3'
        self.cfg.depth_mode = 'raft'

        self.cfg.dataset = CN()
        self.cfg.dataset.source_id = None
        self.cfg.dataset.train_novel_id = None
        self.cfg.dataset.val_novel_id = None
        self.cfg.dataset.use_hr_img = None
        self.cfg.dataset.use_depth_init = None
        self.cfg.dataset.use_local_data = None
        self.cfg.dataset.local_data_root = ''
        self.cfg.dataset.train_data_root = ''
        self.cfg.dataset.val_data_root = ''
        # gaussian render settings
        self.cfg.dataset.bg_color = [0, 0, 0]
        self.cfg.dataset.zfar = 100.0
        self.cfg.dataset.znear = 0.01
        self.cfg.dataset.trans = [0.0, 0.0, 0.0]
        self.cfg.dataset.scale = 1.0
        self.cfg.dataset.inverse_depth_init = 1.0
        # ── 多数据集混合训练 ──────────────────────────────────────────────────
        # 训练数据根目录列表（与 multi_formats 一一对应）
        # 示例: multi_train_roots: ["/…/enerf_outdoor/actor1_4/train", "/…/processed_data/train"]
        self.cfg.dataset.multi_train_roots   = ()   # tuple[str]
        self.cfg.dataset.multi_val_roots     = ()   # tuple[str]，可为空则各 train root 对应 val 目录
        # 每个 root 的数据格式："mini"（cameras.json）或 "legacy"（0_1.json + npy）
        self.cfg.dataset.multi_formats       = ()   # tuple[str]，默认全为 "mini"
        # 目标输出分辨率 (H, W)；None 表示不 resize
        self.cfg.dataset.target_hw           = None
        # __len__ 放大倍数
        self.cfg.dataset.train_boost         = 5
        self.cfg.dataset.val_boost           = 2
        # 随机选输入视角的相机间隔约束（mini 格式）
        self.cfg.dataset.min_cam_gap         = 2
        self.cfg.dataset.max_cam_gap         = 12
        # legacy 格式的 novel view id 列表
        self.cfg.dataset.legacy_novel_ids    = (2, 3, 4, 5)

        self.cfg.raft = CN()
        self.cfg.raft.mixed_precision = None
        self.cfg.raft.train_iters = 0
        self.cfg.raft.val_iters = 0
        self.cfg.raft.corr_implementation = 'reg'  # 'reg_cuda'
        self.cfg.raft.corr_levels = 4
        self.cfg.raft.corr_radius = 4
        self.cfg.raft.n_downsample = 3  # feature down sample rate, 3 means x8 down sample
        self.cfg.raft.context_norm = 'batch'  
        self.cfg.raft.n_gru_layers = 3  # down sample based on x8 down sample, 3 means x8 x16 x32 in gru
        self.cfg.raft.slow_fast_gru = False  # not valid if n_gru_layers==1
        self.cfg.raft.encoder_dims = [64, 96, 128]
        self.cfg.raft.hidden_dims = [128, 128, 128]
        
        # DA3配置
        self.cfg.da3 = CN()
        self.cfg.da3.model_name = 'depth-anything/DA3-LARGE'
        self.cfg.da3.mode = 'single'  # 'single' 或 'dual'
        self.cfg.da3.use_loftr = True
        self.cfg.da3.finetune = False
        self.cfg.da3.mixed_precision = False
        self.cfg.da3.scale_factor = 1.0
        self.cfg.da3.use_metric = False

        # Fast-FoundationStereo 配置
        self.cfg.ffs = CN()
        self.cfg.ffs.ffs_root = ''           # FFS 仓库根目录路径
        self.cfg.ffs.model_path = ''         # FFS 模型权重路径 (.pth)
        self.cfg.ffs.valid_iters = 8         # GRU 迭代次数 (4=快, 8=准)
        self.cfg.ffs.max_disp = 320          # 最大视差搜索范围（需 > |Tf_x|/Z_min）
        self.cfg.ffs.use_hiera = False       # 是否使用层级推理
        self.cfg.ffs.use_loftr = True        # 是否用 LoFTR 增强图像特征
        self.cfg.ffs.finetune = False        # 是否微调 FFS 模型

        self.cfg.gsnet = CN()
        self.cfg.gsnet.use_pe = None
        self.cfg.gsnet.use_depth_net = None
        self.cfg.gsnet.use_warped_depth = None
        self.cfg.gsnet.encoder_dims = None
        self.cfg.gsnet.decoder_dims = None
        self.cfg.gsnet.parm_head_dim = None

        # PAGSplat 专属配置
        self.cfg.pagsplat = CN()
        # DA3 模型: HF repo id 或本地 checkpoint 路径
        self.cfg.pagsplat.da3_checkpoint = 'depth-anything/DA3-LARGE'
        # DINOv2 backbone embed_dim (ViT-B=768, ViT-L=1024, ViT-G=1536)
        self.cfg.pagsplat.embed_dim = 768
        # 内部统一特征通道数
        self.cfg.pagsplat.feat_channels = 256
        # 特征图相对原图的下采样倍数
        self.cfg.pagsplat.feat_stride = 4
        # 从 DINOv2 哪一层提取中间特征 (ViT-B 共12层)
        self.cfg.pagsplat.feat_layer = 8
        # ScaleAlignmentMLP 隐藏层维度
        self.cfg.pagsplat.mlp_hidden = 256
        # GaussianDecoder 编/解码器通道配置
        self.cfg.pagsplat.enc_dims = [128, 256, 512]
        self.cfg.pagsplat.dec_dims = [128, 256, 512]
        # 预测头共享特征通道数
        self.cfg.pagsplat.head_ch = 32
        # 高斯缩放上限 (与 GPS+ 一致)
        self.cfg.pagsplat.scale_max = 0.002
        # 混合精度训练
        self.cfg.pagsplat.mixed_precision = False
        # 辅助损失权重
        self.cfg.pagsplat.loss_smooth  = 0.001  # 边缘感知深度平滑
        self.cfg.pagsplat.loss_warp    = 0.010  # 扭曲一致性 (无效区域高 opacity 惩罚)
        # P0: 几何先验损失 (G3Splat 风格)
        self.cfg.pagsplat.loss_scale   = 0.05   # Scale 各向同性惩罚 (抑制蚯蚓状浮块)
        self.cfg.pagsplat.loss_normal  = 0.01   # 方向一致性 (Gaussian 短轴 ≈ 表面法线)
        # P1: 点云几何一致性
        self.cfg.pagsplat.loss_chamfer  = 0.50  # Chamfer Distance (左右点云一致性，与 GPS+ 同权)
        self.cfg.pagsplat.chamfer_samples = 5000  # Chamfer 每次采样点数
        # 渲染时不透明度裁剪阈值
        self.cfg.pagsplat.min_opacity  = 0.01
        # 迭代深度精化 (DepthGRUCell，参考 Splat-SAP Translation)
        self.cfg.pagsplat.gru_iters     = 3    # GRU 迭代次数 (0=不使用 GRU)
        self.cfg.pagsplat.gru_hidden_ch = 64   # GRU 隐藏状态通道数

        # StereoGS 配置
        self.cfg.stereo_gs = CN()
        self.cfg.stereo_gs.adapt_dims = [96, 96, 128]
        self.cfg.stereo_gs.use_context = True
        self.cfg.stereo_gs.use_gru = True
        self.cfg.stereo_gs.fusion_mode = 'occlusion_aware'
        self.cfg.stereo_gs.confidence_mode = 'learned'
        self.cfg.stereo_gs.head_dim = 64
        self.cfg.stereo_gs.confidence_alpha = 0.3
        self.cfg.stereo_gs.confidence_beta = 0.3
        self.cfg.stereo_gs.max_scale = 0.003
        self.cfg.stereo_gs.cags_depth_residual_bound = 0.5
        self.cfg.stereo_gs.sr_mode = 'convex'
        self.cfg.stereo_gs.use_post_refine = True
        self.cfg.stereo_gs.refine_hidden = 32
        self.cfg.stereo_gs.refine_layers = 3
        self.cfg.stereo_gs.use_cags = False
        self.cfg.stereo_gs.cags_split_mode = 'learned'
        self.cfg.stereo_gs.cags_k_max = 4
        self.cfg.stereo_gs.cags_split_hidden = 32
        self.cfg.stereo_gs.cags_max_pos_offset = 0.002
        self.cfg.stereo_gs.cags_sparsity_weight = 0.01
        self.cfg.stereo_gs.cags_split_threshold = 0.1
        self.cfg.stereo_gs.chamfer_weight = 0.0
        self.cfg.stereo_gs.chamfer_n_samples = 10000
        self.cfg.stereo_gs.warp_padding_mode = 'zeros'

        # ── W-CVCT-GS 配置项 ──
        self.cfg.wcvct = CN()
        self.cfg.wcvct.enable = False
        self.cfg.wcvct.fdsg = CN()
        self.cfg.wcvct.fdsg.wavelet_levels = 3
        self.cfg.wcvct.fdsg.wavelet_type = 'haar'
        self.cfg.wcvct.fdsg.band_weights = [0.3, 0.2, 0.1]
        self.cfg.wcvct.fdsg.ll_weight = 0.4
        self.cfg.wcvct.fdsg.lambda_band = 0.3
        self.cfg.wcvct.fdsg.lambda_disentangle = 0.5
        self.cfg.wcvct.fdsg.lambda_disentangle_warmup = 0.1
        self.cfg.wcvct.fdsg.lambda_disentangle_warmup_steps = 5000
        self.cfg.wcvct.fdsg.lambda_active = 0.05
        self.cfg.wcvct.fdsg.log_compress_k = 10.0
        self.cfg.wcvct.fdsg.use_per_level_render = False
        self.cfg.wcvct.cvct = CN()
        self.cfg.wcvct.cvct.residual_bound = 0.05
        self.cfg.wcvct.cvct.visibility_hidden = 32
        self.cfg.wcvct.cvct.residual_hidden = 16
        self.cfg.wcvct.cvct.lambda_cycle = 0.2
        self.cfg.wcvct.cvct.lambda_omega_align = 0.1
        self.cfg.wcvct.cvct.lambda_omega_entropy = 0.01
        self.cfg.wcvct.schedule = CN()
        self.cfg.wcvct.schedule.phase1_end = 5000
        self.cfg.wcvct.schedule.phase2_end = 30000
        self.cfg.wcvct.override_cags_sparsity = True

        self.cfg.record = CN()
        self.cfg.record.ckpt_path = None
        self.cfg.record.show_path = None
        self.cfg.record.logs_path = None
        self.cfg.record.file_path = None
        self.cfg.record.save_iter = 500
        self.cfg.record.loss_freq = 0
        self.cfg.record.eval_freq = 0

    def get_cfg(self):
        return self.cfg.clone()
    
    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()


if __name__ == '__main__':
    cc = ConfigStereoHuman()
    cc.load("./stage.yaml")
    print(cc.cfg)
