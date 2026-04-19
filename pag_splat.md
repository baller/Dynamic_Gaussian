# PAG-Splat 当前方法流程梳理

本文档描述的是当前仓库中 `GPS_plus` 分支下已经回退后的 `pag-splat` 实际实现，而不是最早设计中的 `warping + GRU + SH` 版本。当前代码保留了 `train_pag.py + pag_stage.yaml` 这套入口，但模型内部已经改成：

`DA3 单目先验 + ScaleAlignmentMLP 尺度对齐 + GPS+ 风格 GSRegresser + 原始 RGB Gaussian 渲染`

对应核心文件：

- `pag_splat/model.py`
- `train_pag.py`
- `run_interpolation_pag.py`
- `pag_splat/render.py`

---

## 1. 方法整体目标

当前版本的目标不是继续保留原 PAG-Splat 中“几何扭曲增强、高斯法线约束、SH 颜色建模”的那条复杂链路，而是：

1. 保留 `DA3` 作为单目深度和单目特征的强先验。
2. 保留 `ScaleAlignmentMLP`，把单目相对深度对齐成带物理尺度的 metric depth。
3. 去掉几何扭曲、GRU 迭代深度优化、法线恢复、SH 颜色建模。
4. 回退到原 GPS+ 的思路，用 `GSRegresser` 直接从图像与深度预测高斯参数。
5. 渲染阶段直接复用原 `lib.GaussianRender.pts2render`。

因此，当前版本可以理解为一种“带 DA3 尺度先验的 GPS+”。

---

## 2. 和旧版 PAG-Splat 的关键区别

当前实现已经移除或不再参与执行的部分包括：

- `SingleSurfaceWarping`
- `DepthGRUCell`
- `GaussianDecoder`
- `SH` 颜色参数 `sh_dc / sh_rest / color_residual`
- 几何扭曲相关损失
- 法线恢复与方向一致性损失
- 各向同性约束损失
- 基于覆盖区域的局部渲染损失

当前保留下来的核心约束只有两类：

- 图像重建损失：`0.8 * L1 + 0.2 * (1 - SSIM)`
- 轻量几何正则：
  - `scale_regular`
  - `3D chamfer`

所以当前方法的训练目标明显更简洁，也更接近原始 GPS+。

---

## 3. 当前模型的输入输出结构

整个训练和推理过程仍然沿用 GPS+ 原始的数据字典风格，主要分成三个部分：

- `data["lmain"]`
- `data["rmain"]`
- `data["novel_view"]`

其中：

- `lmain` 和 `rmain` 表示左右两个输入视图。
- `novel_view` 表示待渲染的新视角监督或新视角相机参数。

### 3.1 `lmain` / `rmain` 的常见输入字段

训练时一般会包含：

- `img`：输入图像，形状通常是 `(B, 3, H, W)`
- `mask`：前景或人体有效区域掩码
- `intr`：相机内参
- `ref_intr`：参考内参
- `extr`：相机外参

### 3.2 当前模型前向后新增的关键字段

在 `pag_splat/model.py` 的 `PAGSplat.forward()` 中，左右视图都会新增：

- `xyz`：由最终 metric depth 反投影得到的世界坐标点云，形状 `(B, H*W, 3)`
- `rot_maps`：每像素高斯旋转四元数图
- `scale_maps`：每像素高斯尺度图
- `opacity_maps`：每像素高斯不透明度图
- `pts_valid`：有效点布尔掩码，形状 `(B, H*W)`

此外，整个 `data` 还会新增：

- `metric_depth_l`
- `metric_depth_r`
- `final_depth_l`
- `final_depth_r`
- `log_scale_l`
- `log_scale_r`
- `data["novel_view"]["scale_regular"]`

这里要注意：

- `metric_depth_*` 是尺度对齐后得到的度量深度。
- `final_depth_*` 是在 `metric_depth` 基础上加上 `GSRegresser` 预测的 `depth_res` 后得到的最终深度。
- `scale_regular` 现在只是一个非常简单的尺度正则项，本质上取的是 `scale_maps` 的平均值。

---

## 4. 当前模型前向流程

当前模型定义在 `pag_splat/model.py` 中，核心类是 `PAGSplat`。

整体前向可以拆成 6 个阶段。

### 4.1 阶段一：读取左右输入图像和相机参数

模型首先从数据字典中取出：

- 左图 `img1`
- 右图 `img2`
- 左右相机内参 `intr1 / intr2`
- 左右相机外参 `extr1 / extr2`

这一步没有做几何扭曲，也没有构造双视图代价体，而是直接把左右图分别送进单目先验分支。

### 4.2 阶段二：DA3 提取单目特征和相对深度

`self.prior_extractor = MonoPriorExtractor(...)`

这个模块的职责是：

1. 调用冻结的 `Depth Anything 3` 网络。
2. 对每张图提取：
   - 单目相对深度 `d_rel`
   - 单目特征 `f_mono`

它的输出为：

- `f_mono1, f_mono2`
- `d_rel1, d_rel2`

其中：

- `d_rel` 仍然是单目网络意义下的相对深度，不保证具备真实物理尺度。
- `f_mono` 是供后续尺度对齐使用的语义特征。

### 4.3 阶段三：尺度对齐，把相对深度变成 metric depth

`self.scale_align = ScaleAlignmentMLP(...)`

当前实现不会直接使用 `d_rel` 去做高斯生成，而是先做尺度对齐：

1. 结合两侧相机内参和外参。
2. 结合左右单目特征。
3. 由 `ScaleAlignmentMLP` 预测尺度相关参数。
4. 把 `d_rel` 转成 `d_metric`。

这一步是当前版本保留下来的核心价值之一。它等价于给原本没有真实尺度的单目深度加上跨视图和相机条件约束，使深度更接近真实米制空间。

模型中左右视图分别调用一次 `_run_scale_align(...)`，最终得到：

- `d_metric1, log_s1`
- `d_metric2, log_s2`

### 4.4 阶段四：图像编码与高斯参数回归

在回退后的方案中，不再使用 warping 后的融合特征，而是回到原 GPS+ 风格的 `GSRegresser`。

这里包含两个子模块：

- `self.img_encoder = UnetExtractor(...)`
- `self.gs_parm_regresser = GSRegresser(...)`

具体做法是：

1. 把左右图像在 batch 维拼起来：
   - `lr_img = cat([img1, img2], dim=0)`
2. 把左右 metric depth 在 batch 维拼起来：
   - `lr_depth = cat([d_metric1, d_metric2], dim=0)`
3. 用 `img_encoder` 提取图像多尺度特征。
4. 用 `GSRegresser` 联合输入：
   - 图像 `lr_img`
   - metric depth `lr_depth`
   - 图像特征 `lr_img_feat`

`GSRegresser` 会输出四类结果：

- `rot_maps`
- `scale_maps`
- `opacity_maps`
- `depth_res`

这完全符合原 GPS+ 的“预测若干 Gaussian 参数 + 一个深度残差”的思路。

### 4.5 阶段五：最终深度生成

当前版本的最终深度不是直接使用 `d_metric`，而是：

`d_final = softplus(d_metric + depth_res) + 1e-3`

这个设计有两个目的：

1. 允许网络在尺度对齐的深度基础上再做局部修正。
2. 通过 `softplus` 保证最终深度为正值，避免出现非法深度。

于是左右视图分别得到：

- `d_final1`
- `d_final2`

### 4.6 阶段六：从深度反投影得到 3D 点云

当前实现使用 `pag_splat/model.py` 中的 `depth_to_pointcloud()`，语义是“把 metric depth 作为 z 值进行相机反投影”，不是旧版 GPS+ 中 inverse-depth 的约定。

流程是：

1. 使用内参把像素坐标转成相机坐标。
2. 使用外参把点从相机坐标变换到世界坐标。
3. 展平得到 `(B, H*W, 3)` 的世界点云。

最终：

- `xyz1 = depth_to_pointcloud(d_final1, intr1, extr1)`
- `xyz2 = depth_to_pointcloud(d_final2, intr2, extr2)`

然后模型把：

- `xyz`
- `rot_maps`
- `scale_maps`
- `opacity_maps`
- `pts_valid`

分别写回到 `lmain` 和 `rmain`。

---

## 5. 渲染流程

### 5.1 当前渲染已经完全回退到 GPS+ 原渲染器

`pag_splat/render.py` 现在只是一个兼容层：

- `move_data_to_cuda(data)`：负责把字典里的 tensor 搬到 GPU
- `pts2render`：直接从 `lib.GaussianRender` 导入

这意味着当前没有专门的 PAG-Splat 渲染逻辑。渲染完全依赖原 GPS+ 的 `pts2render` 约定。

### 5.2 渲染时真正使用的内容

`pts2render` 所需要的核心输入来自左右视图：

- `xyz`
- `rot_maps`
- `scale_maps`
- `opacity_maps`
- `pts_valid`
- 输入图像本身的 RGB

这里有一个关键变化：

当前颜色直接来自输入图像 RGB，不再使用 SH 系数去建模颜色，也不再维护：

- `sh_dc`
- `sh_rest`
- `color_residual`

所以当前方法的颜色表达能力比早期 PAG-Splat 设想更简单，但也更稳定直接。

### 5.3 左右点如何参与 novel view 渲染

渲染器会把左右两张输入图产生的像素级高斯合并后，从 `novel_view` 给出的相机视角进行渲染，输出：

- `data["novel_view"]["img_pred"]`

训练时，这个结果会和 `data["novel_view"]["img"]` 做监督。
推理时，这个结果会直接保存为自由视角插值图像。

---

## 6. 训练流程

训练入口是 `train_pag.py`，主类是 `PAGSplatTrainer`。

整体训练逻辑可以概括为：

`取 batch -> 模型前向 -> 渲染 novel view -> 计算重建损失和几何正则 -> 反向传播 -> 记录日志 -> 定期验证和保存`

下面按步骤展开。

### 6.1 初始化阶段

`PAGSplatTrainer.__init__()` 主要完成以下工作：

1. 读取配置。
2. 调用 `build_pag_splat(...)` 构建模型。
3. 构建训练集和验证集。
4. 构建优化器和学习率调度器。
5. 根据配置加载 checkpoint。

### 6.2 模型构建

模型由 `build_pag_splat(...)` 创建。这个函数目前保持了原来的入口形式，但内部语义已经变了：

1. 加载 DA3 权重。
2. 自动检测 `embed_dim`。
3. 读取配置中的：
   - `raft.encoder_dims`
   - `gsnet.encoder_dims`
   - `gsnet.decoder_dims`
   - `gsnet.parm_head_dim`
4. 用这些配置构造 `GSRegresser` 需要的最小配置对象。
5. 返回新的简化版 `PAGSplat`。

### 6.3 数据集构建

训练时支持两种模式：

1. 多数据集模式：
   - 当 `cfg.dataset.multi_train_roots` 非空时，使用 `build_pag_dataset(...)`
2. legacy 模式：
   - 否则退回 `StereoHumanDataset`

因此，当前 `train_pag.py` 在入口层面仍兼容 PAG 多数据集训练和 GPS+ 旧数据集训练。

### 6.4 训练单步的执行顺序

在 `train()` 中，每一次迭代的顺序如下。

#### 第一步：取一个 batch

`data = self.fetch_data("train")`

随后通过 `move_data_to_cuda(...)` 将其移动到 GPU。

#### 第二步：模型前向

`data = self.model(data, is_train=True)`

这一步会完成：

- DA3 单目先验提取
- metric depth 尺度对齐
- GS 参数预测
- 最终深度生成
- 点云反投影

#### 第三步：渲染 novel view

`data = pts2render(data, bg_color=bg)`

渲染完成后可以得到：

- `render_novel = data["novel_view"]["img_pred"]`
- `gt_novel = data["novel_view"]["img"]`

#### 第四步：计算主重建损失

当前主损失是：

`loss_recon = 0.8 * L1 + 0.2 * (1 - SSIM)`

具体为：

- `Ll1 = l1_loss(render_novel, gt_novel)`
- `Lssim = 1.0 - ssim(render_novel, gt_novel)`

这和原 GPS+ 的图像重建风格是一致的，且当前是对整张 novel view 直接计算，不再只对某些覆盖区域或扭曲有效区域计算。

#### 第五步：加入轻量几何正则

当前保留两项几何约束。

##### 1. `scale_regular`

直接来自：

- `data["novel_view"]["scale_regular"]`

当前实现里它是所有 `scale_maps` 的平均值，用来抑制高斯尺度无控制膨胀。

最终加入：

`loss += pag_cfg.loss_scale * Lscale`

##### 2. `3D chamfer`

由 `chamfer_distance_loss(...)` 计算左右视图点云之间的双向倒角距离。

输入包括：

- `data["lmain"]["xyz"]`
- `data["rmain"]["xyz"]`
- `data["lmain"]["pts_valid"]`
- `data["rmain"]["pts_valid"]`

也就是说：

- 只在有效点上计算。
- 只使用 3D 空间位置，不再使用颜色拼接成 6D chamfer。

最终加入：

`loss += pag_cfg.loss_chamfer * Lchamfer`

#### 第六步：反向传播与优化

训练中使用：

- `torch.autocast(...)` 控制混合精度
- `GradScaler` 进行缩放
- `clip_grad_norm_` 做梯度裁剪
- `AdamW` 做参数更新
- `OneCycleLR` 调度学习率

因此训练稳定性方面，当前仍保留了比较完整的工程保护。

### 6.5 当前已经删除的训练损失

以下损失虽然在历史设计或配置注释中仍可见，但当前训练主路径中已不参与计算：

- `loss_smooth`
- `loss_warp`
- `loss_normal`
- 各向同性损失
- 基于法线恢复的方向一致性损失
- 基于 SH / 颜色特征的 6D chamfer

所以当前训练目标其实是一个非常清晰的组合：

`图像重建 + scale_regular + 3D chamfer`

---

## 7. 验证流程

验证逻辑也在 `train_pag.py` 中，由 `run_eval()` 执行。

流程与训练类似，但更简单：

1. 取验证 batch。
2. `self.model(data, is_train=False)` 做前向。
3. `pts2render(...)` 渲染 novel view。
4. 使用整张图计算：
   - `PSNR`
   - `SSIM`
5. 对每个数据集保存第一张样本的可视化结果。

当前验证不会再额外依赖扭曲掩码或覆盖区域掩码。

### 7.1 验证时保存的可视化内容

当前可视化重点围绕以下几类内容：

- 渲染结果与 GT 对比
- 差值图
- `metric_depth_l/r`
- `final_depth_l/r`
- `pts_valid`
- `opacity_maps`

因此，虽然模型已经删除了法线恢复和 warping，可视化仍然足够帮助观察：

- 深度是否稳定
- 高斯尺度和透明度是否异常
- 左右点云是否存在明显不一致

---

## 8. 推理与自由视角插值流程

推理入口是 `run_interpolation_pag.py`。

虽然名字仍叫 `pag`，但当前推理本质上也是“简化版 PAG + GPS+ 渲染器”。

### 8.1 模型加载

推理开始时会：

1. 根据配置构建简化版 `PAGSplat`。
2. 从 checkpoint 中读取 `network` 权重。
3. 严格加载：
   - `model.load_state_dict(..., strict=True)`

这意味着当前 checkpoint 加载策略是明确的：

- 只接受和当前简化模型结构严格匹配的权重。
- 不再尝试兼容旧版 PAG-Splat checkpoint。

### 8.2 相机插值

推理脚本会先生成 novel view 的相机轨迹：

- `mini` 模式走 `extr_interpolate_mini(...)`
- 普通模式走 `extr_interpolate(...)`

然后构造一个渲染辅助对象，用于逐帧返回：

- 左右输入视图
- 当前 novel view 的相机参数

### 8.3 每一帧的执行流程

每一帧自由视角渲染时的流程是：

1. 生成当前视角对应的数据字典。
2. 调用 `model(data, is_train=False)` 生成左右视图的高斯参数和点云。
3. 调用 `pts2render(data, bg_color=...)` 渲染 novel view。
4. 将 `data["novel_view"]["img_pred"]` 保存为图片。
5. 额外把 `metric_depth_l` 可视化后保存。

最终所有帧会合成为插值视频。

---

## 9. 当前配置项哪些有效，哪些只是保留兼容

`pag_stage.yaml` 中仍然保留了不少旧版 PAG-Splat 字段，但当前代码里并不是全部生效。

### 9.1 当前仍然有效的关键配置

#### 与图像编码和 GS 参数回归有关

- `raft.encoder_dims`
- `gsnet.encoder_dims`
- `gsnet.decoder_dims`
- `gsnet.parm_head_dim`

#### 与 DA3 和尺度对齐有关

- `pagsplat.da3_checkpoint`
- `pagsplat.feat_channels`
- `pagsplat.feat_stride`
- `pagsplat.feat_layer`
- `pagsplat.mlp_hidden`

#### 与训练损失有关

- `pagsplat.mixed_precision`
- `pagsplat.loss_scale`
- `pagsplat.loss_chamfer`
- `pagsplat.chamfer_samples`

### 9.2 当前保留但已基本失效或未参与主路径的字段

- `pagsplat.enc_dims`
- `pagsplat.dec_dims`
- `pagsplat.head_ch`
- `pagsplat.loss_smooth`
- `pagsplat.loss_warp`
- `pagsplat.loss_normal`
- `pagsplat.min_opacity`
- `pagsplat.gru_iters`
- `pagsplat.gru_hidden_ch`

其中：

- `gru_iters` 和 `gru_hidden_ch` 只是为了兼容原入口参数还留在 `build_pag_splat(...)` 里，但内部已经 `del` 掉，不再参与模型构建。
- 注释里提到的 `GaussianDecoder`、`DepthGRUCell` 也已经不是当前方法的一部分。

因此，阅读配置时必须以当前代码执行路径为准，不能只看注释。

---

## 10. 当前方法的优点与局限

### 10.1 优点

1. 结构明显更简单。
2. 训练目标更直接，更接近原 GPS+，更容易排查问题。
3. DA3 提供的单目深度与特征先验仍然保留。
4. `ScaleAlignmentMLP` 让深度进入带尺度的空间，有助于提升跨视图几何一致性。
5. 彻底去掉 warping、GRU、法线恢复后，训练链路更短，调试更容易。

### 10.2 局限

1. 失去了原 PAG-Splat 设想中的几何扭曲增强能力。
2. 不再利用法线约束来规范高斯朝向。
3. 不再用 SH 颜色建模，表达能力比复杂版本更弱。
4. `scale_regular` 当前实现较简单，只是均值约束，几何先验强度有限。
5. 最终性能更依赖 `DA3 + scale_align + GSRegresser` 这条主链本身是否足够稳定。

---

## 11. 一句话总结当前方法

当前仓库中的 `pag-splat`，本质上已经不是“原始设计意义下的 PAG-Splat”，而是：

`用 DA3 提供单目深度与特征先验，用 ScaleAlignmentMLP 把深度对齐到 metric 空间，再用 GPS+ 风格 GSRegresser 回归每像素 Gaussian 参数，最后通过原始 RGB Gaussian 渲染器生成 novel view。`

如果要继续分析训练效果或定位问题，建议后续重点盯以下三段：

1. `DA3 -> metric depth` 是否稳定。
2. `GSRegresser` 对 `rot/scale/opacity/depth_res` 的预测是否合理。
3. `pts2render` 输出的 novel view 与左右点云几何是否一致。
