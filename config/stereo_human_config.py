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
        
        # MoE配置（动静分离）
        self.cfg.moe = CN()
        self.cfg.moe.enabled = False  # 是否启用MoE模式
        self.cfg.moe.num_experts = 2  # 专家数量（背景+人体）
        self.cfg.moe.router_type = 'basic'  # 路由器类型: 'basic', 'multiscale', 'depth_aware'
        self.cfg.moe.router_channels = 64  # 路由器隐藏层通道数
        self.cfg.moe.soft_routing = True  # 是否使用软路由
        self.cfg.moe.share_depth_encoder = True  # 是否共享深度编码器
        
        # MoE高斯分配配置
        self.cfg.moe.allocation = CN()
        self.cfg.moe.allocation.total_gaussians = 1048576  # 总高斯数量 (1024*1024)
        self.cfg.moe.allocation.min_ratio = 0.1  # 最小分配比例
        self.cfg.moe.allocation.learnable = True  # 是否使用可学习分配
        
        # MoE背景缓存配置
        self.cfg.moe.bg_cache = CN()
        self.cfg.moe.bg_cache.enabled = True  # 是否启用背景缓存
        self.cfg.moe.bg_cache.update_threshold = 25.0  # PSNR阈值，低于此值更新缓存
        self.cfg.moe.bg_cache.min_update_interval = 1  # 最小更新间隔（帧数）
        self.cfg.moe.bg_cache.quality_metric = 'psnr'  # 质量评估指标
        self.cfg.moe.bg_cache.momentum = 0.0  # 缓存更新动量
        
        # MoE损失配置
        self.cfg.moe.loss = CN()
        self.cfg.moe.loss.sparsity_weight = 0.1  # 稀疏性损失权重
        self.cfg.moe.loss.temporal_weight = 0.05  # 时序一致性损失权重
        self.cfg.moe.loss.balance_weight = 0.01  # 分配平衡损失权重
        self.cfg.moe.loss.separation_weight = 0.01  # 分离一致性损失权重
        self.cfg.moe.loss.target_bg_ratio = 0.3  # 目标背景比例
        
        # MoE训练配置
        self.cfg.moe.training = CN()
        self.cfg.moe.training.freeze_router_epochs = 5  # 冻结路由器的epoch数
        self.cfg.moe.training.progressive = True  # 是否使用渐进式训练

        self.cfg.gsnet = CN()
        self.cfg.gsnet.use_pe = None
        self.cfg.gsnet.use_depth_net = None
        self.cfg.gsnet.use_warped_depth = None
        self.cfg.gsnet.encoder_dims = None
        self.cfg.gsnet.decoder_dims = None
        self.cfg.gsnet.parm_head_dim = None

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
