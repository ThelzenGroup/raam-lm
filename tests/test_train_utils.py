from __future__ import annotations

import math

from raam_lm.train_utils import lr_for_step


def test_lr_for_step_supports_delayed_cosine_decay():
    base_lr = 5.0e-5
    min_lr = 1.0e-5

    assert math.isclose(lr_for_step(base_lr, 0, 500), 1.0e-7)
    assert lr_for_step(
        base_lr,
        799,
        500,
        total_steps=2200,
        cosine_decay=True,
        min_lr=min_lr,
        decay_start_step=800,
        decay_end_step=2200,
    ) == base_lr
    assert lr_for_step(
        base_lr,
        800,
        500,
        total_steps=2200,
        cosine_decay=True,
        min_lr=min_lr,
        decay_start_step=800,
        decay_end_step=2200,
    ) == base_lr
    assert lr_for_step(
        base_lr,
        2199,
        500,
        total_steps=2200,
        cosine_decay=True,
        min_lr=min_lr,
        decay_start_step=800,
        decay_end_step=2200,
    ) < 1.000001e-5


def test_lr_for_step_keeps_constant_after_warmup_without_decay():
    assert lr_for_step(5.0e-5, 1000, 500, total_steps=2200, cosine_decay=False) == 5.0e-5
