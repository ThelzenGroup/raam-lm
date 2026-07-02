from __future__ import annotations

from raam_lm.train_utils import run_training
from tests.test_shapes import tiny_config


def test_tiny_train_step_completes(tmp_path):
    cfg = tiny_config("raam")
    cfg.train.output_dir = str(tmp_path)
    result = run_training(cfg, steps=2, device_override="cpu", log_path=tmp_path / "train.jsonl")
    assert result["last_metrics"]["train_loss"] > 0
    assert (tmp_path / "train.jsonl").exists()

