"""
Novel View 单组高斯渲染

只处理 novel view 的一组高斯 (而非左右两组合并)。
使用 GSNet 预测的 color 替代原始 RGB。
"""

import torch
from gaussian_renderer import render


def novel_pts2render(data, bg_color):
    """
    将 novel view 的高斯参数渲染成图像

    与原 pts2render 的区别:
    - 只有一组高斯 (novel view), 无需合并左右
    - 颜色来自 GSNet 预测的 color_maps, 而非原始 RGB

    Args:
        data: dict, 包含 'novel_view' 键:
            - xyz:          [B, H*W, 3] 或 [B, 3, H, W]
            - color_maps:   [B, 3, H, W] (Tanh 输出, [-1, 1])
            - rot_maps:     [B, 4, H, W]
            - scale_maps:   [B, 3, H, W]
            - opacity_maps: [B, 1, H, W]
            - pts_valid:    [B, H*W] bool mask
            + standard novel_view camera params
        bg_color: list [r, g, b], e.g. [0, 0, 0]

    Returns:
        data: dict, 新增 data['novel_view']['img_pred'] [B, 3, H, W]
    """
    nv = data['novel_view']
    B = nv['rot_maps'].shape[0]
    _, _, H, W = nv['rot_maps'].shape

    render_list = []
    for i in range(B):
        valid_i = nv['pts_valid'][i, :]  # [H*W]

        # xyz: 如果是 [B, 3, H, W] 格式，先转为 [H*W, 3]
        xyz_i = nv['xyz'][i]
        if xyz_i.dim() == 3 and xyz_i.shape[0] == 3:
            xyz_i = xyz_i.permute(1, 2, 0).reshape(-1, 3)
        elif xyz_i.dim() == 2 and xyz_i.shape[1] == 3:
            pass  # already [H*W, 3]
        else:
            xyz_i = xyz_i.reshape(-1, 3)

        base_color_i = nv['base_color'][i].permute(1, 2, 0).reshape(-1, 3)
        color_res_i = nv['color_maps'][i].permute(1, 2, 0).reshape(-1, 3)
        rot_i = nv['rot_maps'][i].permute(1, 2, 0).reshape(-1, 4)
        scale_i = nv['scale_maps'][i].permute(1, 2, 0).reshape(-1, 3)
        opacity_i = nv['opacity_maps'][i].permute(1, 2, 0).reshape(-1, 1)

        xyz_valid = xyz_i[valid_i].view(-1, 3)
        final_color_i = base_color_i + color_res_i  # [-1,1] + small residual
        color_valid = (final_color_i[valid_i].view(-1, 3) * 0.5 + 0.5).clamp(0, 1)
        rot_valid = rot_i[valid_i].view(-1, 4)
        scale_valid = scale_i[valid_i].view(-1, 3)
        opacity_valid = opacity_i[valid_i].view(-1, 1)

        rendered_i = render(data, i, xyz_valid, color_valid,
                            rot_valid, scale_valid, opacity_valid,
                            bg_color=bg_color)
        render_list.append(rendered_i.unsqueeze(0))

    data['novel_view']['img_pred'] = torch.cat(render_list, dim=0)
    return data
