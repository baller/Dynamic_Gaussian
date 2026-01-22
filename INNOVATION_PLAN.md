# GPS_plus 创新改进计划

## 项目目标
基于GPS_plus项目，集成Depth-Anything-3 (DA3)替换原有的RAFT-Stereo深度估计模块，以获得更鲁棒的深度预测能力。

## 改进方案

### 阶段1: DA3深度估计模块集成 ✅ 已完成
**目标**: 用DA3替换RAFT-Stereo进行深度估计

**技术路线**:
1. 创建DA3深度估计包装器 `DA3DepthEstimator`
2. 修改 `RtStereoHumanModel` 支持DA3模式
3. 适配DA3输出格式到GPS_plus数据流
4. 修改配置文件支持DA3参数

**实施步骤**:
- [x] 分析GPS_plus和DA3代码架构
- [x] 创建 `lib/da3_depth.py` DA3深度估计模块
- [x] 修改 `lib/network.py` 集成DA3
- [x] 更新 `config/stage.yaml` 添加DA3配置
- [x] 更新 `config/stereo_human_config.py` 添加DA3配置定义
- [x] 更新 `train.py` 支持DA3模式
- [x] 更新 `test.py` 支持DA3模式
- [x] 更新 `run_interpolation.py` 支持DA3模式
- [x] 测试训练流程
- [x] 测试推理流程
- [x] 创建结果可视化工具 `make_video.py`

### 阶段2: 特征融合优化 (计划中)
**目标**: 利用DA3的DinoV2特征增强高斯参数预测

### 阶段3: 单目扩展 (计划中)
**目标**: 支持单张图像输入进行3DGS重建

---

## 实施记录

### 2026-01-20 - 阶段1全部完成 + 测试功能增强
- 完成训练和推理测试
- 修复checkpoint加载问题
- 添加结果视频生成工具
- **新增**: 动态视角测试功能
- **新增**: 多视角测试功能
- **更新**: `make_video.py` 支持动态视角和多视角视频生成

### 2026-01-19 - 阶段1代码实现

#### Bug修复记录
1. **导入路径修复**: DA3模块位于 `src/` 子目录，已更新路径为 `DA3_SRC_PATH`
2. **API调用修复**: DA3的底层网络通过 `self.model.model` 访问，而非 `self.net`
3. **图像预处理修复**: 添加ImageNet标准化（DA3使用ImageNet预训练权重）
4. **Patch尺寸修复**: DA3使用patch_size=14的ViT，输入必须是14的倍数。添加 `_pad_to_patch_size()` 方法进行反射填充

#### 1. 创建DA3深度估计模块 ✅
- **文件**: `lib/da3_depth.py`
- **功能**: 
  - `DA3DepthEstimator`: 单目深度估计器
  - `DA3DepthEstimatorDual`: 双目深度估计器
  - 深度尺度对齐功能
  - 伪光流生成（用于兼容原有流程）

#### 2. 修改网络架构 ✅
- **文件**: `lib/network.py`
- **变更**: 
  - 添加 `depth_mode` 参数支持 ('raft' 或 'da3')
  - 新增 `_forward_da3()` 方法处理DA3深度估计
  - 新增 `depth2gsparms()` 方法从DA3深度计算高斯参数
  - 保持与原有RAFT模式的完全兼容

#### 3. 更新配置系统 ✅
- **文件**: `config/stage.yaml`
- **变更**: 
  - 添加 `depth_mode` 全局配置
  - 添加完整的 `da3` 配置块
  - 支持模型选择、微调、混合精度等选项

- **文件**: `config/stereo_human_config.py`
- **变更**: 
  - 添加 `depth_mode` 配置项
  - 添加 `da3` 配置节点定义

#### 4. 更新训练脚本 ✅
- **文件**: `train.py`
- **变更**: 
  - 添加深度模式识别和日志
  - 修改BN冻结逻辑支持DA3
  - 实验名称中添加深度模式标识

#### 5. 更新测试脚本 ✅
- **文件**: `test.py`
- **变更**: 
  - 添加深度模式支持
  - 修改BN冻结逻辑

#### 6. 更新插值渲染脚本 ✅
- **文件**: `run_interpolation.py`
- **变更**: 
  - 添加深度模式支持
  - 修改BN冻结逻辑

#### 7. 测试checkpoint加载修复 ✅
- **问题**: DA3模式下加载checkpoint报错 `Unexpected key(s) in state_dict`
- **原因**: DA3模型从HuggingFace独立加载权重，与GPS_plus checkpoint冲突
- **解决**: `test.py` 中使用 `strict=False` 加载checkpoint

#### 8. 创建结果可视化工具 ✅
- **文件**: `make_video.py`
- **功能**: 
  - 将测试输出图片合成视频
  - 支持多种模式：render/depth/side_by_side/all
  - 自动按帧序号排序
  - 支持自定义帧率

#### 9. 动态视角测试功能 ✅
- **文件**: `test.py`
- **新增方法**:
  - `val_dynamic()`: 动态视角测试，在多个视角之间进行插值渲染
  - `val_all_views()`: 多视角测试，分别渲染所有指定视角
  - `_interpolate_camera()`: 相机外参插值（使用SLERP进行旋转插值）
- **命令行参数**:
  - `--dynamic`: 启用动态视角测试
  - `--all-views`: 启用多视角测试
  - `--views`: 指定视角列表（如 "0,1,2,3"）
  - `--interp`: 动态视角插值帧数（默认10）
  - `--ckpt`: 指定checkpoint路径

#### 10. 更新视频生成工具 ✅
- **文件**: `make_video.py`
- **新增参数**:
  - `--type`: 输入类型 (standard/dynamic/allviews)
  - `--views`: 多视角模式下要处理的视角列表
- **新增函数**:
  - `find_dynamic_images()`: 查找动态视角图片
  - `find_allviews_images()`: 查找多视角图片
  - `create_allviews_video()`: 创建多视角循环视频

#### 11. 添加可用视角自动检测 ✅
- **文件**: `test.py`
- **新增方法**: `_detect_available_views()` 
- **功能**:
  - 自动扫描数据集参数目录检测哪些视角可用
  - 避免因指定不存在的视角导致的FileNotFoundError
  - 支持 `--views auto` 参数自动检测
- **修改**: `val_all_views()` 和 `val_dynamic()` 现在会过滤不可用的视角

---

## 测试结果

### 训练测试
- **数据集**: StereoHuman
- **序列**: s1a6
- **配置**: DA3-LARGE, 单目模式
- **结果**: 训练正常收敛

### 推理测试
- **测试序列**: s1a6
- **目标视角**: 2
- **输出目录**: `experiments/gps_plus_da3/show_s1a6_process/`
- **生成视频**: 
  - `s1a6_result_render.mp4` - 渲染结果
  - `s1a6_result_depth.mp4` - 深度图
  - `s1a6_result_combined.mp4` - 并排对比

---

## 使用说明

### 切换深度估计模式

在 `config/stage.yaml` 中修改 `depth_mode` 参数：

```yaml
# 使用RAFT-Stereo（原始模式）
depth_mode: 'raft'

# 使用DA3深度估计
depth_mode: 'da3'
```

### DA3配置选项

```yaml
da3:
  # 模型名称
  model_name: 'depth-anything/DA3-LARGE'
  
  # 模式: 'single' 单目, 'dual' 双目
  mode: 'single'
  
  # 是否使用LoFTR特征匹配
  use_loftr: True
  
  # 是否微调DA3
  finetune: False
  
  # 混合精度
  mixed_precision: False
```

### 可用的DA3模型

| 模型名称 | 参数量 | 推荐用途 |
|---------|--------|---------|
| `depth-anything/DA3-SMALL` | 0.08B | 快速推理 |
| `depth-anything/DA3-BASE` | 0.12B | 平衡 |
| `depth-anything/DA3-LARGE` | 0.35B | 高质量 |
| `depth-anything/DA3-GIANT` | 1.15B | 最高质量 |
| `depth-anything/DA3NESTED-GIANT-LARGE` | 1.40B | 度量深度 |

### 测试模式

#### 单视角测试（默认）
```bash
# 指定单一目标视角
python test.py -i s1a6 -v 2
```

#### 多视角测试
```bash
# 自动检测可用视角并测试（推荐）
python test.py -i s1a6 --all-views

# 指定特定视角（会自动过滤不存在的视角）
python test.py -i s1a6 --all-views --views 2,3
```

#### 动态视角测试
```bash
# 自动检测可用视角进行动态测试（推荐）
python test.py -i s1a6 --dynamic

# 自定义插值帧数
python test.py -i s1a6 --dynamic --interp 15

# 指定视角范围（会自动过滤不存在的视角）
python test.py -i s1a6 --dynamic --views 2,3 --interp 20

# 指定checkpoint
python test.py -i s1a6 --dynamic --ckpt path/to/checkpoint.pth
```

### 生成结果视频

使用 `make_video.py` 将测试结果转换为视频：

#### 标准测试结果
```bash
# 仅渲染结果视频
python make_video.py -i experiments/gps_plus_da3/show_s1a6_process -o output.mp4 --mode render

# 仅深度图视频
python make_video.py -i experiments/gps_plus_da3/show_s1a6_process -o output.mp4 --mode depth

# 并排显示（渲染+深度）
python make_video.py -i experiments/gps_plus_da3/show_s1a6_process -o output.mp4 --mode side_by_side

# 生成所有类型视频
python make_video.py -i experiments/gps_plus_da3/show_s1a6_process -o output.mp4 --mode all
```

#### 动态视角测试结果
```bash
# 动态视角渲染视频
python make_video.py -i experiments/gps_plus_da3/show_s1a6_process_dynamic -o output.mp4 --type dynamic --mode render

# 动态视角所有视频
python make_video.py -i experiments/gps_plus_da3/show_s1a6_process_dynamic -o output.mp4 --type dynamic --mode all
```

#### 多视角测试结果
```bash
# 多视角循环视频
python make_video.py -i experiments/gps_plus_da3/show_s1a6_process_allviews -o output.mp4 --type allviews --mode render

# 指定视角顺序
python make_video.py -i experiments/gps_plus_da3/show_s1a6_process_allviews -o output.mp4 --type allviews --views 0,2,1,3
```

#### 通用选项
```bash
# 调整帧率（默认15fps）
python make_video.py -i <input_dir> -o output.mp4 --fps 30
```

---

## 依赖要求

### DA3依赖
```bash
# 基础依赖
pip install xformers torch>=2 torchvision

# 可选：gsplat渲染器
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git@0b4dddf04cb687367602c01196913cde6a743d70
```

### 项目结构要求
确保Depth-Anything-3项目位于正确位置：
```
3DGS/
├── GPS_plus/          # 当前项目
├── Depth-Anything-3/  # DA3项目
└── ...
```

---

## 架构对比

### RAFT-Stereo模式（原始）
```
立体图像对 → UNet特征 → LoFTR交叉注意力 → RAFT-Stereo视差 
    → 深度 → GSRegresser → 高斯参数 → 3DGS渲染
```

### DA3模式（创新）
```
立体图像对 → UNet特征 → DA3深度估计 → 深度尺度对齐
    → (可选LoFTR) → GSRegresser → 高斯参数 → 3DGS渲染
```

---

## 注意事项

1. **GPU显存**: DA3模型较大，建议使用16GB+显存的GPU
2. **深度尺度**: DA3输出相对深度，已实现自动尺度对齐
3. **兼容性**: 保留原有RAFT-Stereo模式，可随时切换对比
4. **首次运行**: DA3模型会自动从HuggingFace下载，可能需要网络代理

---

## 后续计划

### 阶段2: 特征融合优化
- 提取DA3的DinoV2特征
- 设计特征融合模块
- 增强高斯参数预测精度

### 阶段3: 单目扩展
- 移除双视图要求
- 支持单图像输入
- 借鉴SHARP的多层高斯表示
