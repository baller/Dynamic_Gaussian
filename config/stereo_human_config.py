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
        self.cfg.pagsplat.loss_smooth = 0.001   # 边缘感知深度平滑
        self.cfg.pagsplat.loss_warp   = 0.010   # 扭曲一致性 (无效区域高 opacity 惩罚)
        self.cfg.pagsplat.loss_unc    = 0.010   # 不确定性稀疏正则
        # 渲染时不透明度裁剪阈值
        self.cfg.pagsplat.min_opacity = 0.01

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
