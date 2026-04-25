
import torch
from gaussian_renderer import render


def pts2render(data, bg_color):
    '''
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


def pts2render_cags(data, bg_color):
    """
    CAGS 渲染: 基础高斯 (spatial maps) + 子高斯 (flat tensors) → novel view。

    data[view] 需包含:
      基础高斯:  xyz (B,N,3), rot_maps (B,4,H,W), scale_maps, opacity_maps,
                 img (B,3,H,W), pts_valid (B,N)
      子高斯:    sub_xyz (B,N_sub,3), sub_rot (B,N_sub,4), sub_scale (B,N_sub,3),
                 sub_opacity (B,N_sub,1), sub_rgb (B,N_sub,3), sub_valid (B,N_sub)
    """
    bs = data['lmain']['img'].shape[0]

    render_novel_list = []
    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []

        for view in ['lmain', 'rmain']:
            # ── 基础高斯 (和 pts2render 完全相同) ──
            valid_i = data[view]['pts_valid'][i, :]
            xyz_i = data[view]['xyz'][i, :, :]
            rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
            scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
            opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

            # ── 子高斯 (flat tensors) ──
            if 'sub_xyz' in data[view]:
                s_valid = data[view]['sub_valid'][i, :]
                s_xyz = data[view]['sub_xyz'][i, :, :]
                s_rgb = data[view]['sub_rgb'][i, :, :]
                s_rot = data[view]['sub_rot'][i, :, :]
                s_scale = data[view]['sub_scale'][i, :, :]
                s_opacity = data[view]['sub_opacity'][i, :, :]

                xyz_i_valid.append(s_xyz[s_valid].view(-1, 3))
                rgb_i_valid.append(s_rgb[s_valid].view(-1, 3))
                rot_i_valid.append(s_rot[s_valid].view(-1, 4))
                scale_i_valid.append(s_scale[s_valid].view(-1, 3))
                opacity_i_valid.append(s_opacity[s_valid].view(-1, 1))

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


def pts2render_cags_per_level(data, bg_color, level: str = "all"):
    """逐级渲染 CAGS 场景: 仅渲染指定层级的高斯。

    用于 L_disentangle: 通过分别屏蔽各子高斯层级进行渲染, 从全量渲染中相减
    即可得到该层级的贡献图 ΔI_j。

    Args:
        data: 与 pts2render_cags 相同的数据字典。
        bg_color: 背景颜色 (list/tuple)。
        level: 取值之一:
          - 'all'           — 父高斯 + 所有子高斯 (等价于 pts2render_cags)
          - 'base_only'     — 仅父 (基础) 高斯
          - 'drop_1' / 'drop_2' / 'drop_3'
                            — 父高斯 + 子高斯 (排除子层级 k=j)

    Returns:
        data, 其中 data['novel_view']['img_pred'] 已被设置, 与 pts2render_cags 一致。

    实现说明: 利用 data[view]['sub_*'] 在 flat 维度上按 k_sub 个 H*W 连续块组织的
    特性, 通过层级标签对 sub_valid 进行逐级门控。父 (基础) 高斯始终视为层级 0,
    第 j 个子高斯块对应层级 j。对于 flat 索引 n, 其层级为 (n // (H*W)) + 1。
    """
    bs = data['lmain']['img'].shape[0]

    valid_overrides = {}
    for view in ('lmain', 'rmain'):
        N_sub = data[view]['sub_valid'].shape[1]
        H, W = data[view]['rot_maps'].shape[-2:]
        per_level = N_sub // (H * W)  # = k_sub

        # 每个 flat 索引的层级标签 (B, N_sub)
        sub_levels = torch.arange(N_sub, device=data[view]['sub_valid'].device) // (H * W) + 1
        sub_levels = sub_levels.unsqueeze(0).expand(bs, -1)

        original_sub_valid = data[view]['sub_valid']
        original_pts_valid = data[view]['pts_valid']

        if level == 'all':
            new_sub_valid = original_sub_valid
            new_pts_valid = original_pts_valid
        elif level == 'base_only':
            new_sub_valid = torch.zeros_like(original_sub_valid)
            new_pts_valid = original_pts_valid
        elif level.startswith('drop_'):
            j = int(level.split('_', 1)[1])
            assert 1 <= j <= per_level
            new_sub_valid = original_sub_valid & (sub_levels != j)
            new_pts_valid = original_pts_valid
        else:
            raise ValueError(f"unknown level={level!r}")

        valid_overrides[view] = (original_sub_valid, original_pts_valid)
        data[view]['sub_valid'] = new_sub_valid
        data[view]['pts_valid'] = new_pts_valid

    try:
        data = pts2render_cags(data, bg_color)
    finally:
        # 还原原始 valid 掩码, 保证调用方看到的 data 字典除 img_pred 外不被修改
        for view, (orig_sub, orig_pts) in valid_overrides.items():
            data[view]['sub_valid'] = orig_sub
            data[view]['pts_valid'] = orig_pts

    return data
