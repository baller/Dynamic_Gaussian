import torch

from lib.stereo_gs.loss_masks import (
    apply_loss_mask,
    build_border_ignore_mask,
    masked_l1_loss,
)


def test_border_ignore_mask_zeros_only_the_configured_image_border():
    image = torch.zeros(2, 3, 6, 8)

    mask = build_border_ignore_mask(image, border=2)

    assert mask.shape == (2, 1, 6, 8)
    assert mask.dtype == image.dtype
    assert mask.device == image.device
    assert torch.all(mask[:, :, :2, :] == 0)
    assert torch.all(mask[:, :, -2:, :] == 0)
    assert torch.all(mask[:, :, :, :2] == 0)
    assert torch.all(mask[:, :, :, -2:] == 0)
    assert torch.all(mask[:, :, 2:4, 2:6] == 1)


def test_masked_l1_loss_ignores_errors_in_the_image_border():
    pred = torch.zeros(1, 3, 4, 4)
    gt = torch.zeros_like(pred)
    pred[:, :, :, :] = 10.0
    pred[:, :, 1:3, 1:3] = 2.0
    mask = build_border_ignore_mask(pred, border=1)

    loss = masked_l1_loss(pred, gt, mask)

    assert torch.isclose(loss, torch.tensor(2.0))


def test_apply_loss_mask_replaces_ignored_pixels_with_gt_values():
    pred = torch.ones(1, 3, 4, 4)
    gt = torch.full_like(pred, 5.0)
    mask = build_border_ignore_mask(pred, border=1)

    composed = apply_loss_mask(pred, gt, mask)

    assert torch.all(composed[:, :, 1:3, 1:3] == 1.0)
    assert torch.all(composed[:, :, 0, :] == 5.0)
    assert torch.all(composed[:, :, :, 0] == 5.0)
