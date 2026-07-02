from __future__ import annotations

import torch

from raam_lm.compression import DynamicHourglassCompressor
from raam_lm.config import CompressionConfig


def test_anchor_selection_fixed_topk_per_block():
    cfg = CompressionConfig(block_size=4, anchors_per_block=1)
    comp = DynamicHourglassCompressor(8, cfg)
    x = torch.randn(3, 16, 8)
    out = comp(x)
    assert out.anchor_indices.shape == (3, 4, 1)
    assert torch.all(out.anchor_indices[:, :, 0] >= torch.tensor([0, 4, 8, 12]))
    assert torch.all(out.anchor_indices[:, :, 0] <= torch.tensor([3, 7, 11, 15]))

