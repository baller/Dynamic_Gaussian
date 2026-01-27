# GPS_plus Loss 计算文档

本文档详细说明项目中所有损失函数的计算方式。

## 1. 总损失公式

```
Total Loss = Render Loss + MoE Loss
```

其中：
- **Render Loss**: 渲染重建损失（主要损失）
- **MoE Loss**: MoE相关辅助损失（仅在启用MoE模式时计算）

---

## 2. 渲染损失 (Render Loss)

### 2.1 公式

```
Render Loss = 0.8 * L1_Loss + 0.2 * SSIM_Loss
```

**注意**: PSNR **不作为训练损失**，仅作为**评估指标**使用（在验证和测试阶段）。

### 2.2 L1 Loss

**位置**: `lib/gs_utils/loss_utils.py`

**公式**:
```python
L1_Loss = mean(|render_novel - gt_novel|)
```

**实现**:
```python
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()
```

**说明**:
- 计算渲染图像与真实图像之间的平均绝对误差
- 输入: `render_novel` [B, C, H, W], `gt_novel` [B, C, H, W]
- 输出: 标量

### 2.3 SSIM Loss

**位置**: `lib/gs_utils/loss_utils.py`

**公式**:
```python
SSIM_Loss = 1.0 - SSIM(render_novel, gt_novel)
```

**SSIM 计算**:
```
SSIM = (2*μ₁μ₂ + C₁)(2*σ₁₂ + C₂) / ((μ₁² + μ₂² + C₁)(σ₁² + σ₂² + C₂))

其中:
- μ₁, μ₂: 图像均值（使用高斯窗口卷积）
- σ₁², σ₂²: 图像方差
- σ₁₂: 协方差
- C₁ = 0.01², C₂ = 0.03²
- 窗口大小: 11x11
- 高斯核标准差: 1.5
```

**实现**:
```python
def ssim(img1, img2, window_size=11, size_average=True):
    # 使用高斯窗口计算局部统计量
    # 返回 SSIM 值 [0, 1]
    return ssim_map.mean()

Lssim = 1.0 - ssim(render_novel, gt_novel)
```

**说明**:
- SSIM 衡量结构相似性，范围 [0, 1]
- 值越大表示越相似
- Loss = 1 - SSIM，值越小越好

### 2.4 为什么没有使用 PSNR 作为损失？

**PSNR 的用途**:
- ✅ **评估指标**: 在验证和测试阶段计算，用于评估模型性能
- ❌ **不作为训练损失**: 训练时使用 L1 + SSIM 损失

**为什么不使用 PSNR 作为损失函数？**

1. **PSNR 与 MSE 的关系**:
   ```
   PSNR = 20 * log10(1 / sqrt(MSE))
   ```
   - PSNR 本质上是 MSE 的对数变换
   - MSE 对异常值敏感，梯度特性不如 L1 好

2. **L1 vs MSE/PSNR**:
   - **L1 损失**: 对异常值更鲁棒，梯度稳定
   - **MSE/PSNR**: 对异常值敏感，可能导致训练不稳定

3. **实际效果**:
   - L1 + SSIM 的组合在实践中表现更好
   - SSIM 提供结构信息，L1 提供像素级精度
   - 两者结合比单独使用 PSNR 更有效

4. **PSNR 作为评估指标的优势**:
   - 与论文中的评估标准一致
   - 易于理解和比较
   - 在验证时计算，不影响训练过程

**PSNR 计算位置**:
- 验证阶段: `train_accelerate.py` - `run_eval()` 方法
- 测试阶段: `test.py` - 测试循环中
- 实现: `lib/gs_utils/image_utils.py` - `psnr()` 函数

---

## 3. MoE 损失 (MoE Loss)

**位置**: `lib/loss.py` - `MoELoss` 类

**总公式**:
```
MoE_Loss = w_sparsity * L_sparsity 
         + w_temporal * L_temporal
         + w_balance * L_balance
         + w_separation * L_separation
```

**默认权重** (来自 `config/stage.yaml`):
- `w_sparsity = 0.1`
- `w_temporal = 0.05`
- `w_balance = 0.01`
- `w_separation = 0.01`

### 3.1 稀疏性损失 (Sparsity Loss)

**目的**: 鼓励路由权重接近0或1，实现清晰的背景/人体分离

**公式**:
```python
L_sparsity = mean(Entropy(router_weights))

其中:
Entropy = -Σ(p * log(p))
p = router_weights [B, num_experts, H, W]
```

**实现**:
```python
def sparsity_loss(self, router_weights):
    # 数值稳定: clamp到 [1e-8, 1-1e-8]
    router_weights = router_weights.clamp(min=1e-8, max=1.0 - 1e-8)
    
    # 计算熵
    entropy = -router_weights * torch.log(router_weights)
    entropy = entropy.sum(dim=1)  # [B, H, W]
    
    return entropy.mean()
```

**说明**:
- 熵越低表示路由权重越稀疏（更接近0或1）
- 鼓励每个像素明确分配给某个专家

### 3.2 时序一致性损失 (Temporal Consistency Loss)

**目的**: 鼓励背景区域的路由权重在时间上保持稳定

**公式**:
```python
L_temporal = mean(|router_weights_t - router_weights_{t-1}|) 
            仅在共同背景区域计算

其中:
- bg_mask_t = (router_weights_t[:, 0] > 0.5)
- bg_mask_{t-1} = (router_weights_{t-1}[:, 0] > 0.5)
- common_bg_mask = bg_mask_t * bg_mask_{t-1}
```

**实现**:
```python
def temporal_consistency_loss(self, router_weights, prev_router_weights):
    if prev_router_weights is None:
        return 0.0
    
    # 背景mask
    bg_mask = (router_weights[:, 0:1] > 0.5).float()
    prev_bg_mask = (prev_router_weights[:, 0:1] > 0.5).float()
    common_bg_mask = bg_mask * prev_bg_mask
    
    # 计算背景区域的L1差异
    diff = (router_weights[:, 0:1] - prev_router_weights[:, 0:1]).abs()
    return (diff * common_bg_mask).sum() / (common_bg_mask.sum() + 1e-8)
```

**说明**:
- 只在两帧都判定为背景的区域计算
- 鼓励静态背景的路由权重稳定

### 3.3 分配平衡损失 (Balance Loss)

**目的**: 防止高斯分配过于极端（全部分配给背景或人体）

**公式**:
```python
L_balance = mean(|bg_ratio - target_bg_ratio|)

其中:
- bg_ratio = allocation_ratio[:, 0]  # 背景分配比例
- target_bg_ratio = 0.3 (默认)
```

**实现**:
```python
def balance_loss(self, allocation_ratio):
    if allocation_ratio is None:
        return 0.0
    
    bg_ratio = allocation_ratio[:, 0]
    return (bg_ratio - self.target_bg_ratio).abs().mean()
```

**说明**:
- 鼓励背景比例接近目标值（默认30%）
- 防止分配退化

### 3.4 分离一致性损失 (Separation Loss)

**目的**: 鼓励背景和人体专家预测不同的高斯参数

**公式**:
```python
L_separation = (L_scale + L_opacity + L_confidence) / 3

其中:
L_scale = 1 / (1 + scale_diff * 1000)
L_opacity = 1 / (1 + opacity_diff * 10)
L_confidence = 1 - mean_opacity
```

**实现**:
```python
def separation_loss(self, bg_params, human_params, router_weights):
    total_loss = 0.0
    count = 0
    
    # 1. 尺度差异
    scale_diff = |bg_scale - human_scale|.mean()
    L_scale = 1.0 / (1.0 + scale_diff * 1000)
    
    # 2. 不透明度差异
    opacity_diff = |bg_opacity - human_opacity|.mean()
    L_opacity = 1.0 / (1.0 + opacity_diff * 10)
    
    # 3. 置信度（鼓励高不透明度）
    avg_opacity = (bg_opacity.mean() + human_opacity.mean()) / 2
    L_confidence = 1.0 - avg_opacity
    
    return (L_scale + L_opacity + L_confidence) / 3
```

**说明**:
- 鼓励两个专家预测不同的参数（差异越大，损失越小）
- 同时鼓励高不透明度（提高渲染质量）

---

## 4. 光流损失 (Flow Loss) - 仅RAFT模式

**位置**: `lib/loss.py` - `sequence_loss` 函数

**用途**: 仅在 `depth_mode='raft'` 时使用，用于训练RAFT-Stereo深度估计

**公式**:
```python
L_flow = Σ(i_weight * |flow_pred[i] - flow_gt|)

其中:
- i_weight = (loss_gamma^(15/(n-1)))^(n-i-1)
- loss_gamma = 0.9 (默认)
- n = len(flow_preds) (RAFT迭代次数)
```

**实现**:
```python
def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9):
    n_predictions = len(flow_preds)
    flow_loss = 0.0
    
    for i in range(n_predictions):
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * i_loss[valid.bool()].mean()
    
    return flow_loss, metrics
```

**说明**:
- 对RAFT的多步预测进行加权
- 后期预测权重更高
- 仅在有效区域计算

---

## 5. 损失计算流程

### 5.1 训练循环中的损失计算

**位置**: `train_accelerate.py` - `train()` 方法

**流程**:
```python
# 1. 渲染
render_novel = data['novel_view']['img_pred']
gt_novel = data['novel_view']['img']

# 2. 计算渲染损失
Ll1 = l1_loss(render_novel, gt_novel)
Lssim = 1.0 - ssim(render_novel, gt_novel)
render_loss = 0.8 * Ll1 + 0.2 * Lssim

# 3. 计算MoE损失（如果启用）
if use_moe:
    moe_loss, moe_loss_dict = moe_loss_fn(data, prev_data)
    total_loss = render_loss + moe_loss
else:
    total_loss = render_loss

# 4. 反向传播
loss.backward()
```

### 5.2 MoE损失计算细节

**位置**: `lib/loss.py` - `MoELoss.forward()`

**流程**:
```python
# 1. 收集路由权重
router_weights = cat([data['lmain']['router_weights'], 
                      data['rmain']['router_weights']])

# 2. 稀疏性损失
sparsity = sparsity_loss(router_weights)
total_moe += 0.1 * sparsity

# 3. 时序一致性损失
if prev_data is not None:
    temporal = temporal_consistency_loss(router_weights, prev_router)
    total_moe += 0.05 * temporal

# 4. 分配平衡损失
if 'allocation_ratio' in data:
    balance = balance_loss(data['allocation_ratio'])
    total_moe += 0.01 * balance

# 5. 分离一致性损失
if 'bg_params' in data and 'human_params' in data:
    separation = separation_loss(bg_params, human_params, router_weights)
    total_moe += 0.01 * separation
```

---

## 6. 配置参数

### 6.1 渲染损失权重

**位置**: `train_accelerate.py` (硬编码)

```python
loss = 0.8 * Ll1 + 0.2 * Lssim
```

### 6.2 MoE损失权重

**位置**: `config/stage.yaml`

```yaml
moe:
  loss:
    sparsity_weight: 0.1      # 稀疏性损失权重
    temporal_weight: 0.05     # 时序一致性损失权重
    balance_weight: 0.01      # 分配平衡损失权重
    separation_weight: 0.01    # 分离一致性损失权重
    target_bg_ratio: 0.3      # 目标背景比例
```

---

## 7. 数值稳定性

所有损失函数都包含数值稳定性保护：

1. **稀疏性损失**: `clamp(min=1e-8, max=1-1e-8)` 防止 log(0)
2. **分离损失**: `clamp(min=1e-8)` 防止除零
3. **NaN检查**: 所有损失函数都有 `torch.isnan()` 检查，返回0.0

---

## 8. 总结

### 8.1 训练损失

| 损失类型 | 权重 | 目的 | 计算位置 |
|---------|------|------|---------|
| L1 Loss | 0.8 | 像素级重建精度 | `loss_utils.py` |
| SSIM Loss | 0.2 | 结构相似性 | `loss_utils.py` |
| Sparsity Loss | 0.1 | 路由清晰度 | `loss.py` |
| Temporal Loss | 0.05 | 时序稳定性 | `loss.py` |
| Balance Loss | 0.01 | 分配平衡 | `loss.py` |
| Separation Loss | 0.01 | 专家差异化 | `loss.py` |

**总损失**:
```
Total = 0.8*L1 + 0.2*SSIM + 0.1*Sparsity + 0.05*Temporal + 0.01*Balance + 0.01*Separation
```

### 8.2 评估指标（不作为损失）

| 指标类型 | 用途 | 计算位置 |
|---------|------|---------|
| PSNR | 验证/测试评估 | `image_utils.py` |
| SSIM | 验证/测试评估 | `loss_utils.py` |
| LPIPS | 测试评估（如果启用） | `test.py` |

**注意**: 
- PSNR 仅在验证和测试阶段计算，用于评估模型性能
- PSNR 不作为训练损失，因为 L1 + SSIM 的组合效果更好
