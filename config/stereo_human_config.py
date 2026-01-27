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
        # gsussian render settings
        self.cfg.dataset.bg_color = [0, 0, 0]
        self.cfg.dataset.zfar = 100.0
        self.cfg.dataset.znear = 0.01
        self.cfg.dataset.trans = [0.0, 0.0, 0.0]
        self.cfg.dataset.scale = 1.0
        self.cfg.dataset.inverse_depth_init = 1.0

        self.cfg.raft = CN()
        self.cfg.raft.mixed_precision = None
        self.cfg.raft.train_iters = 0
        self.cfg.raft.val_iters = 0
        self.cfg.raft.corr_implementation = 'reg' # 'reg_cuda'
        self.cfg.raft.corr_levels = 4
        self.cfg.raft.corr_radius = 4
        self.cfg.raft.n_downsample = 3  # feature down sample rate, 3 means x8 down sample
        self.cfg.raft.context_norm = 'group'  
        self.cfg.raft.n_gru_layers = 1  # down sample based on x8 down sample, 3 means x8 x16 x32 in gru, set 1 for fast
        self.cfg.raft.slow_fast_gru = None  # not valid if n_gru_layers==1
        self.cfg.raft.encoder_dims = [64, 96, 128]
        self.cfg.raft.hidden_dims = [128]*3

        self.cfg.gsnet = CN()
        self.cfg.gsnet.use_pe = None
        self.cfg.gsnet.use_depth_net = None
        self.cfg.gsnet.use_warped_depth = None
        self.cfg.gsnet.encoder_dims = None
        self.cfg.gsnet.decoder_dims = None
        self.cfg.gsnet.parm_head_dim = None

        # 深度估计模式配置
        self.cfg.depth_mode = 'raft'  # 'raft' 或 'da3'
        
        # DA3 深度估计配置
        self.cfg.da3 = CN()
        self.cfg.da3.model_name = 'depth-anything/DA3-LARGE'
        self.cfg.da3.export_feat_layers = [11, 15, 19, 23]
        self.cfg.da3.freeze_backbone = True
        self.cfg.da3.mixed_precision = False
        self.cfg.da3.use_feature_adapter = True  # 是否使用特征适配器

        # 深度融合配置
        self.cfg.depth_fusion = CN()
        self.cfg.depth_fusion.enabled = True
        self.cfg.depth_fusion.feat_dim = 256
        self.cfg.depth_fusion.num_heads = 8
        self.cfg.depth_fusion.dropout = 0.1
        self.cfg.depth_fusion.num_sparse_points = 128

        # 高斯预测网络类型
        self.cfg.gs_predictor = 'gsregresser'  # 'gsregresser' or 'transformer_moe'
        
        # Transformer+MoE 高斯预测网络配置
        self.cfg.gs_transformer = CN()
        self.cfg.gs_transformer.hidden_dim = 256
        self.cfg.gs_transformer.num_layers = 4
        self.cfg.gs_transformer.num_heads = 8
        self.cfg.gs_transformer.window_size = 8
        self.cfg.gs_transformer.num_gaussian_layers = 3
        self.cfg.gs_transformer.max_scale = 0.002
        self.cfg.gs_transformer.max_depth_offset = 0.5
        self.cfg.gs_transformer.downsample_factor = 4  # 下采样因子，节省显存
        
        # MoE 配置
        self.cfg.moe = CN()
        self.cfg.moe.num_experts = 8
        self.cfg.moe.top_k = 2
        self.cfg.moe.load_balance_weight = 0.01

        # 动态高斯分配配置
        self.cfg.dynamic_gs = CN()
        self.cfg.dynamic_gs.enabled = False
        self.cfg.dynamic_gs.num_layers = 3
        self.cfg.dynamic_gs.opacity_threshold = 0.01
        self.cfg.dynamic_gs.complexity_threshold = 0.3
        self.cfg.dynamic_gs.use_complexity_guidance = True
        self.cfg.dynamic_gs.soft_pruning = True

        # 损失函数配置
        self.cfg.loss = CN()
        self.cfg.loss.l1_weight = 0.8
        self.cfg.loss.ssim_weight = 0.2
        self.cfg.loss.chamfer_weight = 0.5
        self.cfg.loss.depth_consistency_weight = 0.1
        self.cfg.loss.moe_balance_weight = 0.01
        self.cfg.loss.allocation_entropy_weight = 0.001

        self.cfg.record = CN()
        self.cfg.record.ckpt_path = None
        self.cfg.record.show_path = None
        self.cfg.record.logs_path = None
        self.cfg.record.file_path = None
        self.cfg.record.save_iter = 5000
        self.cfg.record.loss_freq = 0
        self.cfg.record.eval_freq = 0

        # 可视化配置
        self.cfg.visualization = CN()
        self.cfg.visualization.enabled = True
        self.cfg.visualization.vis_freq = 100
        self.cfg.visualization.save_depth = True
        self.cfg.visualization.save_gaussian = True
        self.cfg.visualization.save_novel_view = True
        self.cfg.visualization.save_opacity = True
        self.cfg.visualization.save_features = False
        self.cfg.visualization.colormap = 'turbo'
        self.cfg.visualization.max_images = 4

    def get_cfg(self):
        return self.cfg.clone()
    
    def load(self, config_file):
        self.cfg.defrost()
        self.cfg.merge_from_file(config_file)
        self.cfg.freeze()


if __name__ == '__main__':
    cc = ConfigStereoHuman()
    cc.load("./raft_stereo_human.yaml")
    print(cc.cfg)

