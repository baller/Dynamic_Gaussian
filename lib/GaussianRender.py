
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
