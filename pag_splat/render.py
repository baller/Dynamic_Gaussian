"""
PAG-Splat 渲染适配层

将 PAGSplat 输出格式适配到 diff_gaussian_rasterization，
接口与 GPS+ lib/GaussianRender.pts2render 语义兼容。

PAGSplat 输出格式 (来自 model.py):
  data[view]['xyz']:         (B, H*W, 3)  世界坐标
  data[view]['rot']:         (B, H*W, 4)  单位四元数
  data[view]['scale']:       (B, H*W, 3)  各向异性缩放
  data[view]['opacity']:     (B, H*W, 1)  不透明度
  data[view]['uncertainty']: (B, H*W, 1)  不确定性 (用于调制 opacity)
  data[view]['img']:         (B, 3, H, W) 原始 RGB [-1, 1] → 颜色来源

核心差异 (对比 GPS+ pts2render):
  1. 无 pts_valid 二值掩码，改用 uncertainty 软权重
  2. 有效不透明度 = opacity * (1 - uncertainty)
  3. 通过 min_opacity 阈值裁剪噪点
"""

from __future__ import annotations

import torch
from gaussian_renderer import render


def pag_pts2render(
    data: dict,
    bg_color: list[float] = [0.0, 0.0, 0.0],
    min_opacity: float = 0.01,
    use_mask: bool = True,
) -> dict:
    """
    PAGSplat 专用渲染函数。

    Args:
        data:        PAGSplat.forward() 返回的数据字典
        bg_color:    渲染背景颜色 [R, G, B]
        min_opacity: 低于此值的高斯点被剔除 (节省显存 & 加速渲染)
                     设为 0 则使用所有点
        use_mask:    是否使用 lmain/rmain 的 mask 过滤背景点

    Returns:
        data (原地新增 'novel_view'['img_pred'] 键)
    """
    bs = data["lmain"]["img"].shape[0]
    render_list = []

    for i in range(bs):
        pts_xyz_all, pts_shs_all, pts_rot_all, pts_scale_all, pts_opa_all = (
            [], [], [], [], []
        )

        for view in ["lmain", "rmain"]:
            v = data[view]
            HW = v["xyz"].shape[1]

            xyz_i   = v["xyz"][i]       # (HW, 3)
            rot_i   = v["rot"][i]       # (HW, 4)
            sc_i    = v["scale"][i]     # (HW, 3)
            opa_i   = v["opacity"][i]   # (HW, 1)
            sh_dc_i   = v["sh_dc"][i]   # (HW, 3)  DC 系数
            sh_rest_i = v["sh_rest"][i] # (HW, 9)  1阶 SH rest 系数

            # 构建 SH 张量：(HW, 4, 3)
            # shs[:, 0, :] = DC 系数 (来自输入像素，100% 锁定)
            # shs[:, 1:4, :] = 1阶 SH rest 系数
            shs_i = torch.cat([
                sh_dc_i.unsqueeze(1),                     # (HW, 1, 3)
                sh_rest_i.reshape(HW, 3, 3),              # (HW, 3, 3)
            ], dim=1)  # (HW, 4, 3)

            # --- 有效点筛选 ---
            keep = torch.ones(HW, dtype=torch.bool, device=xyz_i.device)
            if use_mask and "mask" in v:
                mask_i = v["mask"][i]
                if mask_i.shape[0] == 3:
                    mask_i = mask_i.mean(dim=0, keepdim=True)
                keep = keep & (mask_i.reshape(-1) > 0.5)
            if min_opacity > 0:
                keep = keep & (opa_i[:, 0] > min_opacity)

            pts_xyz_all.append(xyz_i[keep])
            pts_shs_all.append(shs_i[keep])
            pts_rot_all.append(rot_i[keep])
            pts_scale_all.append(sc_i[keep])
            pts_opa_all.append(opa_i[keep])

        # 合并两视图点云
        pts_xyz   = torch.cat(pts_xyz_all,   dim=0)  # (N_total, 3)
        pts_shs   = torch.cat(pts_shs_all,   dim=0)  # (N_total, 4, 3)
        pts_rot   = torch.cat(pts_rot_all,   dim=0)
        pts_scale = torch.cat(pts_scale_all, dim=0)
        pts_opa   = torch.cat(pts_opa_all,   dim=0)

        # 调用渲染函数（SH 模式：degree=1）
        rendered = render(
            data, i,
            pts_xyz, pts_shs, pts_rot, pts_scale, pts_opa,
            bg_color=bg_color,
            sh_degree=1,
        )  # (3, H_novel, W_novel)
        render_list.append(rendered.unsqueeze(0))

    data["novel_view"]["img_pred"] = torch.cat(render_list, dim=0)
    return data


def move_data_to_cuda(data: dict) -> dict:
    """
    将数据字典中的所有 Tensor 移至 CUDA。

    覆盖 lmain / rmain / novel_view 中的所有 Tensor。
    非 Tensor 类型 (int, float, str, list) 保持原样。
    """
    def _to_cuda(v):
        if isinstance(v, torch.Tensor):
            return v.cuda()
        elif isinstance(v, dict):
            return {k: _to_cuda(vv) for k, vv in v.items()}
        return v

    for key in ["lmain", "rmain", "novel_view"]:
        if key in data:
            data[key] = {k: _to_cuda(v) for k, v in data[key].items()}
    return data
