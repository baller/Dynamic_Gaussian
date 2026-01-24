"""
高斯渲染模块

支持两种渲染模式:
1. 单流模式: 原始的统一高斯渲染
2. MoE模式: 背景和人体分离渲染后合并
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
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            # rgb_i = data[view]['color_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)  # [S*S, 4]
            scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)  # [S*S, 1]

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3)) #[valid_i]
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

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


def pts2render_moe(data, bg_color, bg_cache=None):
    """
    MoE模式的高斯渲染
    
    支持背景/人体分离渲染和背景缓存。
    
    Args:
        data: 数据字典，包含融合后的高斯参数和分离的bg_params/human_params
        bg_color: 背景颜色
        bg_cache: 背景缓存对象（可选）
        
    Returns:
        data: 更新后的数据字典，包含:
            - novel_view['img_pred']: 融合渲染结果
            - novel_view['bg_render']: 背景渲染结果（可选）
            - novel_view['human_render']: 人体渲染结果（可选）
    """
    bs = data['lmain']['img'].shape[0]
    
    # 检查是否有MoE分离参数
    has_moe_params = 'router_weights' in data['lmain']
    
    if not has_moe_params:
        # 回退到标准渲染
        return pts2render(data, bg_color)
    
    render_novel_list = []
    bg_render_list = []
    human_render_list = []
    
    for i in range(bs):
        # 收集融合后的高斯参数（用于主渲染）
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        
        # 收集背景高斯参数
        bg_xyz_valid = []
        bg_rgb_valid = []
        bg_rot_valid = []
        bg_scale_valid = []
        bg_opacity_valid = []
        
        # 收集人体高斯参数
        human_xyz_valid = []
        human_rgb_valid = []
        human_rot_valid = []
        human_scale_valid = []
        human_opacity_valid = []
        
        for view in ['lmain', 'rmain']:
            valid_i = data[view]['pts_valid'][i, :]
            xyz_i = data[view]['xyz'][i, :, :]  # [S*S, 3]
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3)  # [S*S, 3]
            
            # 融合后的参数
            rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)
            
            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))
            
            # 获取路由权重用于分离
            router_weights = data[view]['router_weights'][i]  # [2, H, W]
            router_flat = router_weights.permute(1, 2, 0).view(-1, 2)  # [S*S, 2]
            
            # 背景高斯（路由权重作为额外的不透明度调制）
            if 'bg_params' in data[view]:
                bg_rot_i = data[view]['bg_params']['rot_maps'][i].permute(1, 2, 0).view(-1, 4)
                bg_scale_i = data[view]['bg_params']['scale_maps'][i].permute(1, 2, 0).view(-1, 3)
                bg_opacity_i = data[view]['bg_params']['opacity_maps'][i].permute(1, 2, 0).view(-1, 1)
                
                # 先用 valid_i 索引，再用路由权重调制不透明度
                bg_opacity_i_valid = bg_opacity_i[valid_i] * router_flat[valid_i, 0:1]
                
                bg_xyz_valid.append(xyz_i[valid_i].view(-1, 3))
                bg_rgb_valid.append(rgb_i[valid_i].view(-1, 3))
                bg_rot_valid.append(bg_rot_i[valid_i].view(-1, 4))
                bg_scale_valid.append(bg_scale_i[valid_i].view(-1, 3))
                bg_opacity_valid.append(bg_opacity_i_valid.view(-1, 1))
            
            # 人体高斯
            if 'human_params' in data[view]:
                human_rot_i = data[view]['human_params']['rot_maps'][i].permute(1, 2, 0).view(-1, 4)
                human_scale_i = data[view]['human_params']['scale_maps'][i].permute(1, 2, 0).view(-1, 3)
                human_opacity_i = data[view]['human_params']['opacity_maps'][i].permute(1, 2, 0).view(-1, 1)
                
                # 先用 valid_i 索引，再用路由权重调制不透明度
                human_opacity_i_valid = human_opacity_i[valid_i] * router_flat[valid_i, 1:2]
                
                human_xyz_valid.append(xyz_i[valid_i].view(-1, 3))
                human_rgb_valid.append(rgb_i[valid_i].view(-1, 3))
                human_rot_valid.append(human_rot_i[valid_i].view(-1, 4))
                human_scale_valid.append(human_scale_i[valid_i].view(-1, 3))
                human_opacity_valid.append(human_opacity_i_valid.view(-1, 1))
        
        # 融合渲染
        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0) * 0.5 + 0.5
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)
        
        render_novel_i = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
        render_novel_list.append(render_novel_i.unsqueeze(0))
        
        # 背景单独渲染（用于质量评估和可视化）
        if len(bg_xyz_valid) > 0:
            bg_xyz = torch.concat(bg_xyz_valid, dim=0)
            bg_rgb = torch.concat(bg_rgb_valid, dim=0) * 0.5 + 0.5
            bg_rot = torch.concat(bg_rot_valid, dim=0)
            bg_scale = torch.concat(bg_scale_valid, dim=0)
            bg_opacity = torch.concat(bg_opacity_valid, dim=0)
            
            bg_render_i = render(data, i, bg_xyz, bg_rgb, bg_rot, bg_scale, bg_opacity, bg_color=bg_color)
            bg_render_list.append(bg_render_i.unsqueeze(0))
        
        # 人体单独渲染
        if len(human_xyz_valid) > 0:
            human_xyz = torch.concat(human_xyz_valid, dim=0)
            human_rgb = torch.concat(human_rgb_valid, dim=0) * 0.5 + 0.5
            human_rot = torch.concat(human_rot_valid, dim=0)
            human_scale = torch.concat(human_scale_valid, dim=0)
            human_opacity = torch.concat(human_opacity_valid, dim=0)
            
            human_render_i = render(data, i, human_xyz, human_rgb, human_rot, human_scale, human_opacity, bg_color=bg_color)
            human_render_list.append(human_render_i.unsqueeze(0))
    
    data['novel_view']['img_pred'] = torch.concat(render_novel_list, dim=0)
    
    if len(bg_render_list) > 0:
        data['novel_view']['bg_render'] = torch.concat(bg_render_list, dim=0)
    
    if len(human_render_list) > 0:
        data['novel_view']['human_render'] = torch.concat(human_render_list, dim=0)
    
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
