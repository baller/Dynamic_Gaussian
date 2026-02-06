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

        # Depth Anything V3 configuration
        self.cfg.da3 = CN()
        self.cfg.da3.model_name = 'depth-anything/DA3METRIC-LARGE'  # Metric 模型直接输出绝对深度
        self.cfg.da3.export_feat_layers = [11, 15, 19, 23]
        self.cfg.da3.freeze_backbone = True
        self.cfg.da3.mixed_precision = False
        self.cfg.da3.local_files_only = False
        self.cfg.da3.metric_scale_factor = 300.0   # focal scaling factor for metric model
        
        # Depth alignment configuration (块匹配极线搜索)
        self.cfg.depth_align = CN()
        self.cfg.depth_align.num_keypoints = 500
        self.cfg.depth_align.patch_size = 11
        self.cfg.depth_align.max_disparity = 128
        self.cfg.depth_align.min_depth = 0.1
        self.cfg.depth_align.max_depth = 100.0

        self.cfg.gsnet = CN()
        self.cfg.gsnet.use_pe = None
        self.cfg.gsnet.use_depth_net = None
        self.cfg.gsnet.use_warped_depth = None
        self.cfg.gsnet.encoder_dims = None
        self.cfg.gsnet.decoder_dims = None
        self.cfg.gsnet.parm_head_dim = None
        # GSTransformer 配置
        self.cfg.gsnet.use_transformer = False  # 是否使用 Transformer 替代 CNN
        self.cfg.gsnet.transformer_embed_dim = 768
        self.cfg.gsnet.transformer_depth = 16
        self.cfg.gsnet.transformer_num_heads = 12
        self.cfg.gsnet.transformer_patch_size = 14
        self.cfg.gsnet.transformer_mlp_ratio = 4.0
        self.cfg.gsnet.transformer_use_checkpoint = False  # Gradient Checkpointing 节省显存
        # GSTransformer / GSRegresser 参数约束
        self.cfg.gsnet.xyz_res_scale = 0.01       # Tanh 后的缩放因子
        self.cfg.gsnet.scale_max = 0.002           # scale head 最大值
        self.cfg.gsnet.depth_valid_min = 0.01      # 倒数深度有效下界
        self.cfg.gsnet.depth_valid_max = 10.0      # 倒数深度有效上界
        # DA3 特征融合
        self.cfg.gsnet.da3_feat_dim = 1024         # DA3 DINOv2 特征维度
        self.cfg.gsnet.use_da3_features = True     # 是否使用 DA3 特征融合

        # Loss 权重配置
        self.cfg.loss = CN()
        self.cfg.loss.l1_weight = 0.8
        self.cfg.loss.ssim_weight = 0.2
        self.cfg.loss.chamfer_weight = 2.0
        self.cfg.loss.xyz_res_weight = 10.0
        self.cfg.loss.chamfer_sample_num = 100000

        # 深度细化模块配置
        self.cfg.depth_refine = CN()
        self.cfg.depth_refine.enabled = True
        self.cfg.depth_refine.channels = 64
        self.cfg.depth_refine.num_blocks = 3
        self.cfg.depth_refine.residual_scale = 0.1

        self.cfg.record = CN()
        self.cfg.record.ckpt_path = None
        self.cfg.record.show_path = None
        self.cfg.record.logs_path = None
        self.cfg.record.file_path = None
        self.cfg.record.save_iter = [20000, 30000, 40000]
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
    cc.load("./raft_stereo_human.yaml")
    print(cc.cfg)

