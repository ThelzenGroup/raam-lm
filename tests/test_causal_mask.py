from __future__ import annotations

import torch

from raam_lm.registry import build_model
from tests.test_shapes import tiny_config


def assert_causal(model, vocab_size: int, seq_len: int, block_size: int) -> None:
    model.eval()
    torch.manual_seed(123)
    x = torch.randint(0, vocab_size, (2, seq_len))
    cutoffs = [1, block_size - 1, block_size, block_size + 1, seq_len // 2]
    with torch.no_grad():
        for cutoff in cutoffs:
            cutoff = max(1, min(cutoff, seq_len - 1))
            y = x.clone()
            y[:, cutoff:] = (y[:, cutoff:] + 17) % vocab_size
            a = model(x, global_step=5)["logits"][:, :cutoff]
            b = model(y, global_step=5)["logits"][:, :cutoff]
            max_diff = (a - b).abs().max().item()
            assert max_diff < 1e-4, f"future leakage at cutoff={cutoff}, diff={max_diff}"


def test_causal_future_perturbation_all_models():
    for name in ["transformer", "pure_mamba_like", "raam"]:
        config = tiny_config(name)
        if name != "raam":
            config.compression.enabled = False
        model = build_model(config)
        assert_causal(model, config.vocab_size, config.train.seq_len, config.compression.block_size)

