from __future__ import annotations

import torch

from raam_lm.config import MTPConfig
from raam_lm.losses import shifted_cross_entropy
from raam_lm.mtp import current_mtp_weights


def test_curriculum_mtp_weights():
    cfg = MTPConfig(enabled=True, start_step=5, horizon2_step=5, horizon3_step=10, horizon4_step=15, ramp_steps=10)
    assert current_mtp_weights(cfg, 4) == {}
    assert set(current_mtp_weights(cfg, 5)) == {2}
    assert set(current_mtp_weights(cfg, 10)) == {2, 3}
    assert set(current_mtp_weights(cfg, 15)) == {2, 3, 4}


def test_shifted_cross_entropy_alignment():
    logits = torch.full((1, 5, 7), -10.0)
    labels = torch.tensor([[1, 2, 3, 4, 5]])
    logits[0, 0, 3] = 10.0
    logits[0, 1, 4] = 10.0
    logits[0, 2, 5] = 10.0
    loss = shifted_cross_entropy(logits, labels, horizon=2)
    assert loss.item() < 1e-4

