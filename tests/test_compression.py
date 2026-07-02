from __future__ import annotations

import torch

from raam_lm.compression import DynamicHourglassCompressor
from raam_lm.config import CompressionConfig


def test_dynamic_hourglass_static_shape_and_anchor_order():
    cfg = CompressionConfig(block_size=4, anchors_per_block=2, pooled_chunks_per_block=1)
    comp = DynamicHourglassCompressor(16, cfg)
    x = torch.randn(2, 12, 16)
    out = comp(x)
    assert out.compressed_x.shape == (2, 9, 16)
    assert out.anchor_indices.shape == (2, 3, 2)
    assert torch.all(out.anchor_indices[:, :, 1] >= out.anchor_indices[:, :, 0])
    assert out.token_to_block.tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
    assert torch.isfinite(out.recon_loss)


def test_delayed_chunks_use_completed_block_origin():
    cfg = CompressionConfig(block_size=4, anchors_per_block=1, pooled_chunks_per_block=1, delayed_chunk_context=True)
    comp = DynamicHourglassCompressor(8, cfg)
    out = comp(torch.randn(1, 8, 8))
    origins = out.compressed_causal_origin[0].tolist()
    assert origins[0] == 3
    assert origins[2] == 7

