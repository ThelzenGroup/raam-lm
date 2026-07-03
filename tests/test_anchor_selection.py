from __future__ import annotations

import torch
import pytest
from torch import nn

from raam_lm.compression import DynamicHourglassCompressor
from raam_lm.config import CompressionConfig


class FixedAnchorScorer(nn.Module):
    def __init__(self, scores: torch.Tensor):
        super().__init__()
        self.register_buffer("scores", scores)

    def forward(self, blocks: torch.Tensor) -> torch.Tensor:
        return self.scores.expand(blocks.shape[0], -1, -1).unsqueeze(-1)


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


def test_token_id_anchor_selection_uses_highest_ids_per_block():
    cfg = CompressionConfig(block_size=4, anchors_per_block=2, anchor_selection="token_id_topk")
    comp = DynamicHourglassCompressor(8, cfg)
    x = torch.randn(1, 8, 8)
    input_ids = torch.tensor([[10, 80, 20, 70, 5, 99, 6, 98]])
    out = comp(x, input_ids=input_ids)

    assert out.anchor_indices.tolist() == [[[1, 3], [5, 7]]]
    assert out.metadata["anchor_selection"] == "token_id_topk"


def test_token_id_anchor_selection_requires_input_ids():
    cfg = CompressionConfig(block_size=4, anchors_per_block=1, anchor_selection="token_id_topk")
    comp = DynamicHourglassCompressor(8, cfg)
    x = torch.randn(1, 8, 8)

    with pytest.raises(ValueError, match="input_ids"):
        comp(x)


def test_hybrid_anchor_selection_reserves_token_ids_and_fills_learned_scores():
    cfg = CompressionConfig(
        block_size=4,
        anchors_per_block=3,
        anchor_selection="hybrid_token_id_learned",
        token_id_anchor_count=1,
    )
    comp = DynamicHourglassCompressor(8, cfg)
    comp.anchor_scorer = FixedAnchorScorer(torch.tensor([[[100.0, 1.0, 90.0, 80.0]]]))
    x = torch.randn(1, 4, 8)
    input_ids = torch.tensor([[10, 80, 20, 70]])
    out = comp(x, input_ids=input_ids)

    assert out.anchor_indices.tolist() == [[[0, 1, 2]]]
    assert out.metadata["anchor_selection"] == "hybrid_token_id_learned"
    assert out.metadata["token_id_anchor_count"] == 1


def test_hybrid_anchor_selection_requires_input_ids_when_reserving_token_ids():
    cfg = CompressionConfig(
        block_size=4,
        anchors_per_block=2,
        anchor_selection="hybrid_token_id_learned",
        token_id_anchor_count=1,
    )
    comp = DynamicHourglassCompressor(8, cfg)
    x = torch.randn(1, 8, 8)

    with pytest.raises(ValueError, match="input_ids"):
        comp(x)


def test_hybrid_anchor_selection_rejects_too_many_token_id_anchors():
    cfg = CompressionConfig(
        block_size=4,
        anchors_per_block=2,
        anchor_selection="hybrid_token_id_learned",
        token_id_anchor_count=3,
    )
    comp = DynamicHourglassCompressor(8, cfg)
    x = torch.randn(1, 8, 8)
    input_ids = torch.arange(8).unsqueeze(0)

    with pytest.raises(ValueError, match="token_id_anchor_count"):
        comp(x, input_ids=input_ids)


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
