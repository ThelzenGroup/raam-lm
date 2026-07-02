from __future__ import annotations

import json

from raam_lm.flops import estimate_flops_per_token
from raam_lm.profiling import profile_training_step
from tests.test_shapes import tiny_config


def test_flops_estimate_positive():
    assert estimate_flops_per_token(tiny_config("raam")) > 0


def test_profile_manifest_keys(tmp_path):
    cfg = tiny_config("raam")
    cfg.train.output_dir = str(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = profile_training_step(cfg, config_path="inline", device_override="cpu", steps=1, output_path=path)
    loaded = json.loads(path.read_text())
    for key in [
        "param_count_total",
        "estimated_flops_per_token",
        "tokens_per_sec",
        "step_time_ms_mean",
        "step_time_ms_p95",
        "peak_memory_allocated_mb",
        "device",
        "dtype",
        "config_hash",
        "git_sha",
    ]:
        assert key in manifest
        assert key in loaded

