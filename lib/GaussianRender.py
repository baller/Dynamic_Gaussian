
import torch
from gaussian_renderer import render


def pts2render(data, bg_color):
    '''
    支持两种数据格式:
    1. 原始格式 (单层): rot_maps [B, 4, H, W], xyz [B, H*W, 3]
    2. 多层格式: rot_maps [B, N, 4], xyz [B, N, 3], N = L * H * W
    
    :param data: rgb input color [-1, 1], will be scaled to [0, 1]
    :param bg_color:  [0, 0, 0]
    :return: rbg render result in [0, 1]
    '''
    bs = data['lmain']['img'].shape[0]
    
    # 检测数据格式 (通过 rot_maps 的维度)
    rot_maps_sample = data['lmain']['rot_maps']
    is_multilayer_format = (rot_maps_sample.dim() == 3)  # [B, N, 4] vs [B, 4, H, W]

    render_novel_list = []
    for i in range(bs):
        xyz_i_valid = []
        rgb_i_valid = []
        rot_i_valid = []
        scale_i_valid = []
        opacity_i_valid = []
        
        for view in ['lmain', 'rmain']:
            valid_i = data[view]['pts_valid'][i, :]  # [N] bool
            xyz_i = data[view]['xyz'][i, :, :]  # [N, 3]
            
            if is_multilayer_format:
                # 多层格式: 数据已经是 [B, N, C] 格式
                rot_i = data[view]['rot_maps'][i, :, :]  # [N, 4]
                scale_i = data[view]['scale_maps'][i, :, :]  # [N, 3]
                opacity_i = data[view]['opacity_maps'][i, :]  # [N]
                if opacity_i.dim() == 1:
                    opacity_i = opacity_i.unsqueeze(-1)  # [N, 1]
                
                # RGB: 需要复制以匹配多层点数
                # 如果有 L 层，每个像素对应 L 个点
                img_i = data[view]['img'][i, :, :, :].permute(1, 2, 0)  # [H, W, 3]
                H, W, _ = img_i.shape
                N = xyz_i.shape[0]
                L = N // (H * W)  # 推断层数
                
                # 扩展 RGB 到多层
                rgb_i = img_i.reshape(-1, 3)  # [H*W, 3]
                rgb_i = rgb_i.unsqueeze(0).expand(L, -1, -1).reshape(-1, 3)  # [L*H*W, 3]
            else:
                # 原始格式: 数据是 [B, C, H, W] 格式
                rgb_i = data[view]['img'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
                rot_i = data[view]['rot_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 4)
                scale_i = data[view]['scale_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 3)
                opacity_i = data[view]['opacity_maps'][i, :, :, :].permute(1, 2, 0).view(-1, 1)

            xyz_i_valid.append(xyz_i[valid_i].view(-1, 3))
            rgb_i_valid.append(rgb_i[valid_i].view(-1, 3))
            rot_i_valid.append(rot_i[valid_i].view(-1, 4))
            scale_i_valid.append(scale_i[valid_i].view(-1, 3))
            opacity_i_valid.append(opacity_i[valid_i].view(-1, 1))

        pts_xyz_i = torch.concat(xyz_i_valid, dim=0)
        pts_rgb_i = torch.concat(rgb_i_valid, dim=0)
        img_range = data.get('img_range', None)
        if img_range is not None and len(img_range) == 2:
            img_min, img_max = img_range
            pts_rgb_i = (pts_rgb_i - img_min) / (img_max - img_min)
            pts_rgb_i = pts_rgb_i.clamp(0, 1)
        rot_i = torch.concat(rot_i_valid, dim=0)
        scale_i = torch.concat(scale_i_valid, dim=0)
        opacity_i = torch.concat(opacity_i_valid, dim=0)

        render_novel_i = render(data, i, pts_xyz_i, pts_rgb_i, rot_i, scale_i, opacity_i, bg_color=bg_color)
        render_novel_list.append(render_novel_i.unsqueeze(0))

    data['novel_view']['img_pred'] = torch.concat(render_novel_list, dim=0)
    return data
