import unittest
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from lib.pag_multi_loader import PAGLegacyDataset
from pag_splat.model import PAGSplat
from train_pag import build_split_debug_maps, compute_refine_schedule


class TestSplitDebugMaps(unittest.TestCase):
    def test_build_split_debug_maps_outputs_child_density_and_wavelet_prior(self):
        height, width, k_max = 4, 5, 3
        view = {
            "opacity_maps": torch.ones(2, 1, height, width),
            "split_score_maps": torch.linspace(0.0, 1.0, steps=2 * height * width).view(2, 1, height, width),
            "split_prior_maps": torch.full((2, 1, height, width), 0.75),
            "child_valid": torch.zeros(2, k_max * height * width, dtype=torch.bool),
        }
        view["child_valid"][0, : height * width] = True
        view["child_valid"][1, height * width : 2 * height * width] = True
        view["child_valid"][1, 2 * height * width :] = True

        debug = build_split_debug_maps(view)

        self.assertEqual(set(debug.keys()), {"split_score", "child_density", "wavelet_high"})
        self.assertEqual(tuple(debug["split_score"].shape), (2, 1, height, width))
        self.assertEqual(tuple(debug["child_density"].shape), (2, 1, height, width))
        self.assertEqual(tuple(debug["wavelet_high"].shape), (2, 1, height, width))
        self.assertTrue(torch.allclose(debug["wavelet_high"], view["split_prior_maps"]))
        self.assertTrue(torch.allclose(debug["child_density"][0], torch.ones(1, height, width)))
        self.assertTrue(torch.allclose(debug["child_density"][1], torch.full((1, height, width), 2.0)))


class TestRefineSchedule(unittest.TestCase):
    def test_compute_refine_schedule_matches_warmup_progress(self):
        warmup_alpha, prior_alpha = compute_refine_schedule(
            total_steps=50,
            refine_warmup_steps=5000,
            prior_warmup_steps=5000,
        )
        self.assertAlmostEqual(warmup_alpha, 0.01, places=6)
        self.assertAlmostEqual(prior_alpha, 0.99, places=6)

        warmup_alpha, prior_alpha = compute_refine_schedule(
            total_steps=6000,
            refine_warmup_steps=5000,
            prior_warmup_steps=5000,
        )
        self.assertAlmostEqual(warmup_alpha, 1.0, places=6)
        self.assertAlmostEqual(prior_alpha, 0.0, places=6)


class TestSplitSelection(unittest.TestCase):
    def test_eval_topk_override_keeps_sparse_selection_during_warmup(self):
        split_score = torch.tensor([[[[0.60, 0.55], [0.52, 0.51]]]], dtype=torch.float32)
        valid_map = torch.ones_like(split_score, dtype=torch.bool)

        mask = PAGSplat._select_split_mask(
            split_score=split_score,
            valid_map=valid_map,
            topk_ratio=0.25,
            thresh=0.20,
            training=False,
            force_topk=True,
        )

        self.assertEqual(int(mask.sum().item()), 1)
        self.assertTrue(bool(mask[0, 0, 0, 0].item()))


class TestLegacyRawRectifiedSource(unittest.TestCase):
    def _build_cfg(self):
        return SimpleNamespace(
            target_hw=(1024, 1024),
            render_hw=(2048, 2048),
            raw_data_root="/data/sifang/GPS_plus_data/hias_sifang_s1",
            znear=0.01,
            zfar=100.0,
            trans=[0.0, 0.0, 0.0],
            scale=1.0,
            train_boost=1,
            val_boost=1,
            min_cam_gap=1,
            max_cam_gap=12,
        )

    def test_rectified_source_hr_is_not_processed_upsample(self):
        ds = PAGLegacyDataset(
            data_root="/data/sifang/GPS_plus_data/hias_sifang_s1_colmap_processed_metric/train",
            cfg_dataset=self._build_cfg(),
            phase="train",
        )
        sample_name = "s1_0053"
        cams = ds._load_all_cameras(sample_name)

        hr_views = ds._load_source_pair_hr(sample_name, cams)

        self.assertIsNotNone(hr_views)
        self.assertIn(0, hr_views)
        self.assertIn(1, hr_views)

        img0_hr, intr0_hr = hr_views[0]
        img1_hr, intr1_hr = hr_views[1]
        self.assertEqual(img0_hr.shape[:2], (2048, 2048))
        self.assertEqual(img1_hr.shape[:2], (2048, 2048))
        self.assertEqual(intr0_hr.shape, (3, 3))
        self.assertEqual(intr1_hr.shape, (3, 3))

        img0_proc, intr0_proc, _ = ds._load_one(sample_name, 0, cams)
        img1_proc, intr1_proc, _ = ds._load_one(sample_name, 1, cams)
        img0_up = cv2.resize(img0_proc, (2048, 2048), interpolation=cv2.INTER_LINEAR)
        img1_up = cv2.resize(img1_proc, (2048, 2048), interpolation=cv2.INTER_LINEAR)

        self.assertGreater(np.abs(img0_hr.astype(np.float32) - img0_up.astype(np.float32)).mean(), 0.5)
        self.assertGreater(np.abs(img1_hr.astype(np.float32) - img1_up.astype(np.float32)).mean(), 0.5)

        intr0_up = intr0_proc.copy()
        intr0_up[0] *= 2.0
        intr0_up[1] *= 2.0
        intr1_up = intr1_proc.copy()
        intr1_up[0] *= 2.0
        intr1_up[1] *= 2.0
        self.assertGreater(np.abs(intr0_hr - intr0_up).mean(), 1e-3)
        self.assertGreater(np.abs(intr1_hr - intr1_up).mean(), 1e-3)

    def test_legacy_view_selection_keeps_source_0_1_and_novel_from_config(self):
        cfg = self._build_cfg()
        cfg.legacy_novel_ids = [2, 3]
        ds = PAGLegacyDataset(
            data_root="/data/sifang/GPS_plus_data/hias_sifang_s1_colmap_processed_metric/train",
            cfg_dataset=cfg,
            phase="train",
        )

        id_l, id_r, id_novel = ds._pick_legacy_views([0, 1, 2, 3, 4, 5])

        self.assertEqual(id_l, 0)
        self.assertEqual(id_r, 1)
        self.assertIn(id_novel, {2, 3})


if __name__ == "__main__":
    unittest.main()
