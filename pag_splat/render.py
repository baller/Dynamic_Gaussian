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
        pts_xyz_all, pts_rgb_all, pts_rot_all, pts_scale_all, pts_opa_all = (
            [], [], [], [], []
        )

        for view in ["lmain", "rmain"]:
            v = data[view]
            B, HW, _ = v["xyz"].shape
            H, W = v["img"].shape[-2], v["img"].shape[-1]

            xyz_i  = v["xyz"][i]      # (HW, 3)
            rot_i  = v["rot"][i]      # (HW, 4)
            sc_i   = v["scale"][i]    # (HW, 3)
            opa_i  = v["opacity"][i]  # (HW, 1)

            # uncertainty_head 已移除（避免梯度消失陷阱）
            # 可靠性控制由 opacity_head + valid_mask 直接承担
            eff_opa = opa_i  # (HW, 1)

            # RGB 颜色: 优先用解码器预测的融合颜色图，退回原始像素
            # color_map (B,3,H,W) [0,1] 来自 GaussianDecoder 的双视图颜色融合头
            if "color_map" in v:
                rgb_i = v["color_map"][i].permute(1, 2, 0).reshape(HW, 3)
            else:
                rgb_i = (
                    v["img"][i].permute(1, 2, 0).reshape(HW, 3) * 0.5 + 0.5
                )  # (HW, 3)

            # --- 有效点筛选 ---
            keep = torch.ones(HW, dtype=torch.bool, device=xyz_i.device)

            # (1) 前景掩码过滤 (若数据中有 mask)
            if use_mask and "mask" in v:
                mask_i = v["mask"][i]  # (3, H, W) 或 (1, H, W)
                if mask_i.shape[0] == 3:
                    mask_i = mask_i.mean(dim=0, keepdim=True)
                # 下采样到 H*W 展平
                fg = mask_i.reshape(-1) > 0.5
                keep = keep & fg

            # (2) 低不透明度裁剪
            if min_opacity > 0:
                keep = keep & (eff_opa[:, 0] > min_opacity)

            pts_xyz_all.append(xyz_i[keep])
            pts_rgb_all.append(rgb_i[keep])
            pts_rot_all.append(rot_i[keep])
            pts_scale_all.append(sc_i[keep])
            pts_opa_all.append(eff_opa[keep])

        # 合并两视图点云
        pts_xyz   = torch.cat(pts_xyz_all,   dim=0)  # (N_total, 3)
        pts_rgb   = torch.cat(pts_rgb_all,   dim=0)
        pts_rot   = torch.cat(pts_rot_all,   dim=0)  # (N_total, 4)
        pts_scale = torch.cat(pts_scale_all, dim=0)
        pts_opa   = torch.cat(pts_opa_all,   dim=0)  # (N_total, 1)

        # 调用 GPS+ 低层渲染函数
        rendered = render(
            data, i,
            pts_xyz, pts_rgb, pts_rot, pts_scale, pts_opa,
            bg_color=bg_color,
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
