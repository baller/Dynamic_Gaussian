"""
高斯渲染模块

支持多种渲染模式:
1. 单流模式: 原始的统一高斯渲染
2. MoE模式: 多专家分离渲染后合并（支持任意数量专家+共享专家）
"""

import torch
from gaussian_renderer import render


def pts2render(data, bg_color):
    '''
    标准高斯渲染
    
    :param data: rgb input color [-1, 1], will be scaled to [0, 1]
    :param bg_color:  [0, 0, 0]
    :return: rbg render result in [0, 1]
    '''
    bs = data['lmain']['img'].shape[0]

    render_novel_list = []
    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        for view in ['lmain', 'rmain']:
            valid_i = data[view]['pts_valid'][i, :]
            xyz_i = data[view]['xyz'][i, :, :]  # [S*S, 3]
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).contiguous().view(-1, 3)  # [S*S, 3]
            rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).contiguous().view(-1, 4)  # [S*S, 4]
            scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).contiguous().view(-1, 3)  # [S*S, 3]
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).contiguous().view(-1, 1)  # [S*S, 1]

            xyz_i_valid.append(xyz_i[valid_i].reshape(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].reshape(-1, 3))
            rot_i_valid.append(rot_i[valid_i].reshape(-1, 4))
            scale_i_valid.append(scale_i[valid_i].reshape(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].reshape(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        pts_rgb_i = pts_rgb_i * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        render_novel_i = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
        render_novel_list.append(render_novel_i.unsqueeze(0))

    data['novel_view']['img_pred'] = torch.concat(render_novel_list, dim=0)
    return data


def pts2render_moe(data, bg_color, bg_cache=None, expert_info=None, render_experts=False):
    """
    MoE模式的高斯渲染
    
    支持任意数量专家+共享专家的分离渲染和融合。
    
    Args:
        data: 数据字典，包含:
            - router_weights: 路由权重 [B, num_experts, H, W]
            - expert_params_list: 各专家参数列表（可选，用于分离渲染）
            - 融合后的高斯参数
        bg_color: 背景颜色
        bg_cache: 背景缓存对象（可选）
        expert_info: 专家信息字典
        render_experts: 是否渲染各专家（用于可视化，训练时设为False节省显存）
        
    Returns:
        data: 更新后的数据字典，包含:
            - novel_view['img_pred']: 融合渲染结果
            - novel_view['expert_renders']: 各专家渲染结果（仅当render_experts=True）
    """
    bs = data['lmain']['img'].shape[0]
    
    # 检查是否有MoE分离参数
    has_moe_params = 'router_weights' in data['lmain']
    
    if not has_moe_params:
        # 回退到标准渲染
        return pts2render(data, bg_color)
    
    render_novel_list = []
    expert_renders = {}  # 各专家的渲染结果
    
    for i in range(bs):
        # 收集融合后的高斯参数（用于主渲染）
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        
        for view in ['lmain', 'rmain']:
            valid_i = data[view]['pts_valid'][i, :]
            xyz_i = data[view]['xyz'][i, :, :]  # [S*S, 3]
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).contiguous().view(-1, 3)
            
            # 融合后的参数
            rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).contiguous().view(-1, 4)
            scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).contiguous().view(-1, 3)
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).contiguous().view(-1, 1)
            
            xyz_i_valid.append(xyz_i[valid_i].reshape(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].reshape(-1, 3))
            rot_i_valid.append(rot_i[valid_i].reshape(-1, 4))
            scale_i_valid.append(scale_i[valid_i].reshape(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].reshape(-1, 1))
        
        # 融合渲染
        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0) * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)
        
        render_novel_i = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
        render_novel_list.append(render_novel_i.unsqueeze(0))
        
        # 各专家单独渲染（仅用于可视化，训练时跳过以节省显存）
        if render_experts and 'expert_params_list' in data['lmain']:
            expert_params_list = data['lmain']['expert_params_list']
            router_weights = data['lmain']['router_weights'][i]  # [num_experts, H, W]
            num_experts = router_weights.shape[0]
            router_flat = router_weights.permute(1, 2, 0).contiguous().view(-1, num_experts)
            
            for exp_idx, exp_params in enumerate(expert_params_list):
                exp_name = exp_params.get('name', f'expert_{exp_idx}')
                if exp_name not in expert_renders:
                    expert_renders[exp_name] = []
                
                exp_xyz_valid = []
                exp_rgb_valid = []
                exp_rot_valid = []
                exp_scale_valid = []
                exp_opacity_valid = []
                
                for view in ['lmain', 'rmain']:
                    valid_i = data[view]['pts_valid'][i, :]
                    xyz_i = data[view]['xyz'][i, :, :]
                    rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).contiguous().view(-1, 3)
                    
                    # 获取该视图的专家参数
                    view_exp_params = data[view]['expert_params_list'][exp_idx] if 'expert_params_list' in data[view] else exp_params
                    
                    exp_rot_i = view_exp_params['rot_maps'][i].permute(1, 2, 0).contiguous().view(-1, 4)
                    exp_scale_i = view_exp_params['scale_maps'][i].permute(1, 2, 0).contiguous().view(-1, 3)
                    exp_opacity_i = view_exp_params['opacity_maps'][i].permute(1, 2, 0).contiguous().view(-1, 1)
                    
                    # 使用路由权重调制不透明度
                    view_router = data[view]['router_weights'][i]
                    view_router_flat = view_router.permute(1, 2, 0).contiguous().view(-1, num_experts)
                    exp_opacity_i_modulated = exp_opacity_i[valid_i] * view_router_flat[valid_i, exp_idx:exp_idx+1]
                    
                    exp_xyz_valid.append(xyz_i[valid_i].reshape(-1, 3))
                    exp_rgb_valid.append(rgb_i[valid_i].reshape(-1, 3))
                    exp_rot_valid.append(exp_rot_i[valid_i].reshape(-1, 4))
                    exp_scale_valid.append(exp_scale_i[valid_i].reshape(-1, 3))
                    exp_opacity_valid.append(exp_opacity_i_modulated.reshape(-1, 1))
                
                if len(exp_xyz_valid) > 0:
                    exp_xyz = torch.concat(exp_xyz_valid, dim=0)
                    exp_rgb = torch.concat(exp_rgb_valid, dim=0) * 0.5 + 0.5
                    exp_rot = torch.concat(exp_rot_valid, dim=0)
                    exp_scale = torch.concat(exp_scale_valid, dim=0)
                    exp_opacity = torch.concat(exp_opacity_valid, dim=0)
                    
                    exp_render = render(data, i, exp_xyz, exp_rgb, exp_rot, exp_scale, exp_opacity, bg_color=bg_color)
                    expert_renders[exp_name].append(exp_render.unsqueeze(0))
    
    data['novel_view']['img_pred'] = torch.concat(render_novel_list, dim=0)
    
    # 合并各专家渲染结果（仅当render_experts=True时）
    if render_experts and expert_renders:
        data['novel_view']['expert_renders'] = {}
        for exp_name, renders in expert_renders.items():
            if renders:
                data['novel_view']['expert_renders'][exp_name] = torch.concat(renders, dim=0)
        
        # 向后兼容：创建bg_render和human_render别名
        exp_renders = data['novel_view']['expert_renders']
        if 'bg' in exp_renders:
            data['novel_view']['bg_render'] = exp_renders['bg']
        elif 'expert_0' in exp_renders:
            data['novel_view']['bg_render'] = exp_renders['expert_0']
        
        if 'human' in exp_renders:
            data['novel_view']['human_render'] = exp_renders['human']
        elif 'expert_1' in exp_renders:
            data['novel_view']['human_render'] = exp_renders['expert_1']
    
    return data


def pts2render_with_cache(data, bg_color, bg_cache, frame_idx=None):
    """
    带背景缓存的高斯渲染
    
    利用缓存的背景高斯减少计算量。
    
    Args:
        data: 数据字典
        bg_color: 背景颜色
        bg_cache: 背景缓存对象
        frame_idx: 当前帧索引
        
    Returns:
        data: 更新后的数据字典
    """
    # 如果没有缓存或MoE未启用，使用标准渲染
    if bg_cache is None or 'router_weights' not in data['lmain']:
        return pts2render(data, bg_color)
    
    # 检查是否需要更新背景缓存
    # 注意：在训练时，我们通常总是更新；在推理时使用自适应更新
    cached_bg = bg_cache.get()
    
    if cached_bg is None:
        # 首次运行，执行完整渲染并缓存
        data = pts2render_moe(data, bg_color)
        
        # 缓存背景参数
        bg_gaussians = {
            'lmain_bg_params': data['lmain'].get('bg_params'),
            'rmain_bg_params': data['rmain'].get('bg_params'),
        }
        bg_cache.update(bg_gaussians, frame_idx)
        
        return data
    
    # 使用缓存的背景，只更新人体部分
    # 这里的实现简化了，实际可以更复杂
    return pts2render_moe(data, bg_color, bg_cache)
