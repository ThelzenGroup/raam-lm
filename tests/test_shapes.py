from __future__ import annotations

import torch

from raam_lm.config import ModelConfig, CompressionConfig, MTPConfig, TrainConfig
from raam_lm.registry import build_model


def tiny_config(model_name: str = "raam") -> ModelConfig:
    return ModelConfig(
        model_name=model_name,
        vocab_size=64,
        max_seq_len=16,
        d_model=32,
        n_layers=4,
        n_heads=4,
        n_kv_heads=4,
        d_ff=64,
        mixer_backend="fallback",
        attention_island_layers=[2] if model_name == "raam" else [],
        compression=CompressionConfig(enabled=model_name == "raam", block_size=4, anchors_per_block=1, recon_loss_weight=0.01),
        mtp=MTPConfig(enabled=True, start_step=1, horizon2_step=1, horizon3_step=2, horizon4_step=3, ramp_steps=2),
        train=TrainConfig(batch_size=2, seq_len=16, steps=2),
    )


def test_forward_shape_raam():
    config = tiny_config("raam")
    model = build_model(config)
    x = torch.randint(0, config.vocab_size, (2, 16))
    out = model(x, labels=x, global_step=3)
    assert out["logits"].shape == (2, 16, config.vocab_size)
    assert torch.isfinite(out["loss"])
    assert "mixer_backend" in out["aux"]


def test_forward_shape_baselines():
    for name in ["transformer", "pure_mamba_like"]:
        config = tiny_config(name)
        config.compression.enabled = False
        model = build_model(config)
        x = torch.randint(0, config.vocab_size, (2, 16))
        out = model(x, labels=x, global_step=3)
        assert out["logits"].shape == (2, 16, config.vocab_size)
        assert torch.isfinite(out["loss"])

