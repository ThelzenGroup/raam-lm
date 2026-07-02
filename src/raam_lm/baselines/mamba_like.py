"""Pure fallback recurrent/mixer baseline."""

from __future__ import annotations

import torch
from torch import nn

from raam_lm.config import ModelConfig
from raam_lm.layers import RMSNorm, init_weights
from raam_lm.losses import compute_lm_losses
from raam_lm.mixer import MixerBlock
from raam_lm.mtp import MTPHeads


class PureMambaLikeForCausalLM(nn.Module):
    """Cheap mixer-only baseline. The fallback is not real Mamba."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                MixerBlock(
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    backend=config.mixer_backend,
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

    @property
    def mixer_backend(self) -> str:
        for block in self.blocks:
            return block.mixer_backend
        return "none"

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        global_step: int = 0,
    ) -> dict[str, object]:
        x = self.tok_embeddings(input_ids)
        for block in self.blocks:
            x = block(x)
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
        )
        return {
            **output,
            "logits": logits,
            "hidden_states": hidden,
            "aux": {
                "model_name": "pure_mamba_like",
                "mixer_backend": self.mixer_backend,
                "compression_ratio": 1.0,
                "mean_anchor_score": 0.0,
                "anchor_token_fraction": 0.0,
            },
        }

