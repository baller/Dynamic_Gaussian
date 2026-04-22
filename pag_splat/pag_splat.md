# PAG-Splat 当前实现说明与近期改动总结

本文档描述的是当前 `GPS_plus` 仓库里真实在执行的 `pag-splat` 方法，不再是早期设想中的 `warping + GRU + SH` 版本。当前实现已经进入新的阶段：

`DA3 单目先验 + scale_align 尺度对齐 + GPS+ coarse Gaussian 回归 + DA3 小波引导的分层 child Gaussian 分裂 + 2K 直接栅格化`

对应核心文件：

- [pag_splat/model.py](/data/sifang/GPS_plus/pag_splat/model.py)
- [lib/pag_multi_loader.py](/data/sifang/GPS_plus/lib/pag_multi_loader.py)
- [lib/GaussianRender.py](/data/sifang/GPS_plus/lib/GaussianRender.py)
- [train_pag.py](/data/sifang/GPS_plus/train_pag.py)
- [run_interpolation_pag.py](/data/sifang/GPS_plus/run_interpolation_pag.py)
- [pag_splat/pag_stage.yaml](/data/sifang/GPS_plus/pag_splat/pag_stage.yaml)

---

## 1. 当前方法相对旧版 PAG-Splat 的变化

当前实现已经不再走最早的 PAG-Splat 链路。下面这些模块或思想已经被移除，或者不再参与主执行路径：

- `SingleSurfaceWarping`
- `DepthGRUCell`
- `GaussianDecoder`
- 法线恢复与法线一致性约束
- SH 颜色建模 `sh_dc / sh_rest / color_residual`
- warping / smooth / normal / scale isotropy 这类旧几何损失

同时，也已经不再是单纯的“DA3 + scale_align + GPS+ coarse GS”。

最近新引入的核心变化是：

1. 在 coarse 分支之上增加了 `DA3 feature` 的 `1-level Haar DWT` 小波分解。
2. 用小波高频特征指导 `split proposal`，决定哪些 coarse 像素需要细化。
3. 对被选中的 coarse 像素生成最多 `K=4` 个 `child Gaussians`。
4. 渲染时不做单独 2D 超分，而是在 `2K rasterization` 里直接把 `coarse + child` 一起渲染。
5. 训练损失除了原有重建和轻量几何项外，又增加了：
   - wavelet render loss
   - split sparsity loss
   - child consistency loss
   - DA3 高频先验引导损失

所以当前版本已经可以概括为：

`DA3 负责深度与结构先验，GPS+ coarse GS 负责主体几何，DA3 小波高频负责局部 Gaussian 密度自适应细化。`

---

## 2. 最近这轮具体改了什么

这部分只总结最近新增的改动，便于后续读代码时快速定位。

### 2.1 模型结构改动

在 [pag_splat/model.py](/data/sifang/GPS_plus/pag_splat/model.py) 中新增了分层细化分支：

- `WaveletFeatureDecomposer`
- `SplitProposalHead`
- `HierarchicalRefinementHead`

模型现在分两层高斯：

1. `coarse Gaussian`
   - 仍然保持 `1 pixel -> 1 Gaussian`
   - 来源是 `DA3 -> scale_align -> GSRegresser`
2. `child Gaussian`
   - 只在被 `split_score` 选中的 coarse 像素上生成
   - 每个 coarse 像素最多生成 `split_k_max = 4` 个 child

每个 child 当前会预测：

- `delta_uv`
- `delta_depth`
- `delta_scale_ratio`
- `delta_rot`
- `delta_opacity`
- `delta_color_residual`
- `mixture weight`

child 的最终参数通过“父高斯参数 + 局部残差”的方式得到，而不是完全独立从零回归。

### 2.2 渲染链路改动

在 [lib/GaussianRender.py](/data/sifang/GPS_plus/lib/GaussianRender.py) 中，`pts2render()` 现在会：

1. 先收集原本的 coarse Gaussian：
   - `xyz`
   - `rgb`
   - `rot`
   - `scale`
   - `opacity`
2. 再检查是否存在 child 分支输出：
   - `child_xyz`
   - `child_rgb`
   - `child_rot`
   - `child_scale`
   - `child_opacity`
   - `child_valid`
3. 把 coarse 和 child 合并成同一批高斯后，一次性送入 rasterizer

因此当前方法语义已经明确是：

`1K coarse 回归，2K 直接渲染`

而不是：

`1K 渲染后再做 2D 超分`

### 2.3 数据加载与监督改动

在 [lib/pag_multi_loader.py](/data/sifang/GPS_plus/lib/pag_multi_loader.py) 中，数据加载已经扩展为同时支持：

- `img`：coarse 输入图，通常是 `1024x1024`
- `img_hr`：高分辨率输入图，通常是 `2048x2048`
- `novel_view.img`：2K GT

并新增了：

- `dataset.render_hw`
- `dataset.raw_data_root`

其中最重要的变化有两个。

#### 变化一：legacy 数据现在支持真 2K novel GT

对 legacy processed 数据，`novel_view.img` 会优先从 raw source 读图，再根据 `export_report.json` 里的 crop 信息裁剪并 resize 到 `render_hw`，不再只是简单把 processed 图上采样。

#### 变化二：legacy source 现在支持真 2K rectified `img_hr`

这是最近专门补进去的能力。

对 legacy source 视图 `0/1`，loader 现在会：

1. 从 raw session 读取：
   - `sparse/0/cameras.bin`
   - `sparse/0/images.bin`
2. 解析 raw camera intrinsics / distortion / extrinsics
3. 根据 `export_report.json` 中的 `source_pair`
4. 在运行时调用 `cv2.stereoRectify`
5. 对 raw source 图像做高分辨率极线矫正
6. 再按 crop 配置裁成 `2048x2048`

这样得到的 `lmain.img_hr / rmain.img_hr` 已经不再是 processed 图的简单双线性上采样，而是真正来自 raw source 的高分辨率 rectified 图。

### 2.4 训练可视化改动

在 [train_pag.py](/data/sifang/GPS_plus/train_pag.py) 中，最近新增了专门面向分裂行为诊断的可视化。

新增辅助函数：

- `build_split_debug_maps(view)`

新增输出图：

- `cmp_split_l.jpg`
- `cmp_split_r.jpg`

每张图会展示三个面板：

- `split_score`
- `child_density`
- `wavelet_high`

此外，原来的 `cmp_geom.jpg` 也扩成了：

- `PtsValid`
- `Opacity_L`
- `SplitScore_L`

这样当前训练时已经能比较直观地排查：

- 哪些区域被提议分裂
- child 是否在全图泛滥
- 小波高频先验与 split 行为是否匹配

---

## 3. 当前方法的整体目标

当前方法的目标不是尽量复杂化几何链路，而是用更直接的方式提升复杂纹理、文字、弱纹理区域的渲染质量：

1. 用 `DA3 + scale_align` 保持 coarse 几何稳定。
2. 用 `GSRegresser` 继续做 GPS+ 风格的 coarse Gaussian 参数回归。
3. 用 `DA3 feature` 的小波高频部分判断哪里需要更高的 Gaussian 密度。
4. 在这些区域局部生成 child Gaussians，打破“一像素一个 Gaussian”的固定范式。
5. 最终在 2K 新视角上直接渲染出更细的文字、纹理和边界。

因此，当前方法最重要的创新点不是“再换一个更强的深度 backbone”，而是：

`利用 DA3 高频特征去控制 Gaussian 的局部密度分配。`

---

## 4. 当前数据字典结构

训练和推理时仍然沿用 `lmain / rmain / novel_view` 三段结构。

### 4.1 `lmain` / `rmain` 输入字段

常见字段包括：

- `img`：coarse 输入图，`(B, 3, H, W)`，归一化到 `[-1, 1]`
- `img_hr`：高分辨率输入图，通常是 `2048x2048`
- `intr`：coarse 分辨率对应的内参
- `intr_hr`：高分辨率图对应的内参
- `extr`：外参

### 4.2 `novel_view` 输入字段

常见字段包括：

- `img`：2K GT
- `intr`
- `extr`
- `width / height`
- `world_view_transform`
- `full_proj_transform`
- `camera_center`

`novel_view` 的相机矩阵现在以 `render_hw` 为准，而不是默认继承 `target_hw`。

---

## 5. 当前模型前向流程

当前 `PAGSplat.forward()` 可以拆成 8 个阶段。

### 5.1 输入读取

从 `data["lmain"] / data["rmain"]` 取出：

- `img`
- `intr`
- `extr`

并读取 warmup 相关控制量：

- `data["_refine_warmup_alpha"]`

### 5.2 DA3 单目先验

调用 `MonoPriorExtractor`，得到：

- `f_mono1`
- `f_mono2`
- `d_rel1`
- `d_rel2`

这里：

- `d_rel` 是 DA3 的相对深度
- `f_mono` 是后续尺度对齐和小波分裂都会使用的单目特征

### 5.3 尺度对齐

调用 `ScaleAlignmentMLP`，结合：

- 左右相机参数
- 左右 DA3 特征
- 左右相对深度

得到：

- `d_metric1, log_s1`
- `d_metric2, log_s2`

这一步负责把单目相对深度拉到 metric 空间。

### 5.4 coarse GS 参数回归

调用：

- `UnetExtractor`
- `GSRegresser`

输入为：

- 左右图像
- `d_metric`
- 图像特征

输出为：

- `rot_maps`
- `scale_maps`
- `opacity_maps`
- `depth_res`

随后得到 coarse 最终深度：

- `d_final = softplus(d_metric + depth_res) + 1e-3`

### 5.5 coarse 点云反投影

调用 `depth_to_pointcloud()`，把 `d_final` 反投影成世界坐标点云：

- `xyz`

因此 coarse 分支仍然输出 GPS+ 风格的基本字段：

- `xyz`
- `rot_maps`
- `scale_maps`
- `opacity_maps`
- `pts_valid`

### 5.6 DA3 特征小波分解

这是当前版本新增的关键环节。

在 `WaveletFeatureDecomposer` 中，对 `f_mono` 做 `1-level Haar DWT`，得到：

- `F_low = LL`
- `F_high = concat(LH, HL, HH)`

这里：

- `LL` 更偏结构和低频语义
- `LH / HL / HH` 更偏方向性高频细节

### 5.7 split proposal

把下列信息拼起来送入 `SplitProposalHead`：

- `F_low`
- `F_high`
- coarse RGB feat
- `d_final`
- `opacity_map`

输出：

- `split_score_maps`
- `child_weight_logits`
- `orientation_hint_maps`

另外，还会从 `F_high` 构造：

- `split_prior_maps`

它本质上是高频能量图，作为训练前期的辅助先验监督。

### 5.8 child refinement

对通过 proposal 的 coarse 像素，`HierarchicalRefinementHead` 会预测最多 `K=4` 个 child 的残差参数。

训练时：

- 用 `topk_ratio` 在有效像素中选一部分 coarse 父像素进入 child 分支

推理时：

- 用 `split_score_thresh`
- `child_weight_thresh`

来裁掉无效 child

最终每个视图会新增：

- `child_weight_maps`
- `child_delta_uv_maps`
- `child_depth_res_maps`
- `child_scale_ratio_maps`
- `child_rot_delta_maps`
- `child_color_res_maps`
- `child_xyz`
- `child_rgb`
- `child_rot`
- `child_scale`
- `child_opacity`
- `child_valid`

这些字段供训练损失和 2K 渲染一起使用。

---

## 6. 当前渲染流程

### 6.1 coarse 与 child 合并渲染

`pts2render()` 现在不再只消费 coarse GS，而是：

1. 收集左右视图 coarse 高斯
2. 收集左右视图 child 高斯
3. 按 `valid` 过滤
4. 合并为统一高斯集合
5. 一次性渲染到 `novel_view`

输出仍然是：

- `data["novel_view"]["img_pred"]`

### 6.2 颜色表达

当前颜色仍然是 RGB 直接建模，不重新引入 SH。

具体是：

- coarse 颜色来自父像素 RGB
- child 颜色来自 `parent_rgb + color_residual`

这意味着当前方法的复杂度主要增加在 Gaussian 密度与局部细化上，而不是颜色球谐展开上。

---

## 7. 当前训练流程

训练入口是 [train_pag.py](/data/sifang/GPS_plus/train_pag.py)，整体流程是：

`取 batch -> 模型前向 -> 2K 渲染 -> 计算损失 -> 反向传播 -> 日志与可视化 -> 定期验证`

### 7.1 训练集与验证集

当前统一走 `build_pag_dataset(...)`。

支持：

- `mini` 多相机场景
- `legacy` processed 场景

当 `multi_train_roots` 为空时，会自动退回 legacy 模式。

### 7.2 主损失

当前主重建损失是：

`0.8 * L1 + 0.2 * (1 - SSIM)`

而且是直接对 `2K img_pred` 与 `2K GT` 计算。

### 7.3 保留的 coarse 几何正则

仍然保留：

- `scale_regular`
- `3D chamfer`

其中 `3D chamfer` 仍然只基于左右视图有效点的 3D 位置计算。

### 7.4 新增的细化层损失

最近新增四项与分层分裂相关的损失。

#### 1. `wavelet_render_loss`

对最终渲染图和 GT 做 `1-level Haar DWT`，并分别比较：

- `LL`
- `LH`
- `HL`
- `HH`

其中高频子带权重更大，目的是更直接地约束文字、纹理和边缘细节。

#### 2. `split_sparsity_loss`

约束：

- `split_score`
- `child_weight`

不要无脑全图激活。

当前形式是：

- `score L1`
- `mixture entropy`

的组合。

#### 3. `child_consistency_loss`

约束 child 的残差不要发散，当前会惩罚：

- `delta_uv`
- `delta_depth`
- `delta_scale`
- `color_residual`

#### 4. `da3_split_prior_loss`

训练前期用 `split_prior_maps` 去引导 `split_score_maps`，帮助模型更快学会把分裂集中到 DA3 高频更强的位置。

### 7.5 warmup 策略

当前 refinement 分支不是从第一步就全强度参与，而是通过：

- `refine_warmup_steps`
- `da3_wavelet_prior_warmup_steps`

做渐进式启用。

这样可以减少训练初期 child 分支抢占 coarse 分支的问题。

---

## 8. 当前验证与可视化

验证时除了常规的：

- `cmp_render.jpg`
- `cmp_psnr.jpg`
- `cmp_color.jpg`
- `cmp_depth_l.jpg`
- `cmp_depth_r.jpg`
- `cmp_pts.jpg`

现在还会重点输出细化层诊断图：

- `cmp_geom.jpg`
- `cmp_split_l.jpg`
- `cmp_split_r.jpg`

### 8.1 `cmp_geom.jpg`

当前面板为：

- `PtsValid`
- `Opacity_L`
- `SplitScore_L`

### 8.2 `cmp_split_l.jpg / cmp_split_r.jpg`

当前面板为：

- `SplitScore`
- `ChildDensity`
- `WaveletHigh`

用途分别是：

- `SplitScore`：看 proposal 是否学到了复杂区域优先细化
- `ChildDensity`：看 child 是否在少量局部激活，而不是全图铺满
- `WaveletHigh`：看 DA3 高频先验和实际 split 行为是否一致

---

## 9. 推理与自由视角插值

[run_interpolation_pag.py](/data/sifang/GPS_plus/run_interpolation_pag.py) 现在已经同步支持：

- `render_hw`
- child Gaussian 渲染
- 新的 `split_k_max / threshold` 配置

推理时每一帧会：

1. 构造当前视角数据字典
2. 前向得到 coarse + child GS
3. 在 2K 相机下直接渲染 `novel_view.img_pred`
4. 保存结果图

因此，推理阶段和训练阶段在方法语义上是完全一致的，没有额外接一个 2D 超分尾部。

---

## 10. 当前配置中真正生效的关键项

在 [pag_splat/pag_stage.yaml](/data/sifang/GPS_plus/pag_splat/pag_stage.yaml) 与 [config/stereo_human_config.py](/data/sifang/GPS_plus/config/stereo_human_config.py) 中，当前真正影响执行的关键项包括：

### 10.1 数据相关

- `dataset.target_hw`
- `dataset.render_hw`
- `dataset.raw_data_root`
- `dataset.multi_train_roots`
- `dataset.multi_val_roots`
- `dataset.multi_formats`

### 10.2 child 分裂相关

- `pagsplat.split_k_max`
- `pagsplat.split_score_thresh`
- `pagsplat.child_weight_thresh`
- `pagsplat.split_topk_ratio`

### 10.3 损失相关

- `pagsplat.loss_wavelet`
- `pagsplat.loss_split_sparse`
- `pagsplat.loss_child_consistency`
- `pagsplat.loss_da3_prior`
- `pagsplat.wavelet_high_weight`
- `pagsplat.wavelet_low_weight`

### 10.4 warmup 相关

- `pagsplat.refine_warmup_steps`
- `pagsplat.da3_wavelet_prior_warmup_steps`

---

## 11. 当前版本的优势与局限

### 11.1 优势

1. coarse 几何仍然建立在 `DA3 + scale_align` 上，稳定性比纯纹理驱动细化更好。
2. child 分裂只在局部激活，方法上比“全图统一提 2K 高斯密度”更节省。
3. 通过小波高频先验，当前方案更容易针对：
   - 文字
   - 条纹
   - 服饰纹理
   - 弱纹理边界
   做定向增强。
4. legacy 数据已经支持真 2K source rectification，不再只能依赖上采样 HR 输入。

### 11.2 局限

1. 目前仍是 `1-level Haar DWT`，还没有做更深层的小波金字塔。
2. child 分裂虽然已经是自适应的，但训练时本质上还是 `soft mixture + top-k parent selection`，不是完全离散的分裂数学习。
3. 颜色建模仍然是 RGB residual，没有重新引入更强的 SH 或 view-dependent appearance。
4. 当前 shell 环境里没有直接做完整 CUDA 训练 smoke，因此端到端效果仍要以实际训练结果为准。

---

## 12. 一句话概括当前方法

当前 `pag-splat` 已经从“DA3 引导的简化版 GPS+”进一步演化成：

`用 DA3 提供深度与结构先验，再用 DA3 特征的小波高频去驱动局部 Gaussian 分裂，从而在 2K 新视角中直接渲染更高频的细节。`
