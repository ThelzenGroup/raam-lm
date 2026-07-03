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


def test_uniform_anchor_selection_is_deterministic_per_block():
    cfg = CompressionConfig(block_size=8, anchors_per_block=4, anchor_selection="uniform")
    comp = DynamicHourglassCompressor(8, cfg)
    x = torch.randn(1, 16, 8)
    out = comp(x)

    assert out.anchor_indices.tolist() == [[[0, 2, 5, 7], [8, 10, 13, 15]]]
    assert out.metadata["anchor_selection"] == "uniform"


def test_anchor_selection_rejects_more_anchors_than_block_size():
    cfg = CompressionConfig(block_size=4, anchors_per_block=5)
    comp = DynamicHourglassCompressor(8, cfg)
    x = torch.randn(1, 8, 8)

    try:
        comp(x)
    except ValueError as exc:
        assert "anchors_per_block" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected too many anchors to raise")
