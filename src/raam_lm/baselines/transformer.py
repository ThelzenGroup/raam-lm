"""Dense Transformer baseline."""

from __future__ import annotations

import torch
from torch import nn

from raam_lm.attention import TransformerBlock
from raam_lm.config import ModelConfig
from raam_lm.layers import RMSNorm, causal_positions, init_weights
from raam_lm.losses import compute_lm_losses
from raam_lm.mtp import MTPHeads


class DenseTransformerForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.n_heads,
                    config.d_ff,
                    n_kv_heads=config.n_kv_heads,
                    dropout=config.dropout,
                    rope_base=config.rope_base,
                )
                for _ in range(config.n_layers)
            ]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_heads = MTPHeads(config.d_model, config.vocab_size, config.mtp) if config.mtp.enabled else None
        self.apply(init_weights)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        global_step: int = 0,
        loss_mask: torch.Tensor | None = None,
    ) -> dict[str, object]:
        x = self.tok_embeddings(input_ids)
        origins = causal_positions(input_ids.shape[0], input_ids.shape[1], input_ids.device)
        for block in self.blocks:
            x = block(x, origins=origins)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        output = compute_lm_losses(
            logits,
            labels,
            hidden,
            self.lm_head,
            self.mtp_heads,
            self.config.mtp,
            global_step,
            loss_mask=loss_mask,
        )
        return {
            **output,
            "logits": logits,
            "hidden_states": hidden,
            "aux": {
                "model_name": "transformer",
                "mixer_backend": "none",
                "compression_ratio": 1.0,
                "mean_anchor_score": 0.0,
                "anchor_token_fraction": 0.0,
            },
        }
