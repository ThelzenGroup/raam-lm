"""Curriculum multi-token prediction heads."""

from __future__ import annotations

import torch
from torch import nn

from .config import MTPConfig


def current_mtp_weights(config: MTPConfig, global_step: int) -> dict[int, float]:
    if not config.enabled or global_step < config.start_step:
        return {}
    weights: dict[int, float] = {}
    ramp = min(1.0, max(0.0, (global_step - config.start_step + 1) / max(config.ramp_steps, 1)))
    horizon_steps = {
        2: config.horizon2_step,
        3: config.horizon3_step,
        4: config.horizon4_step,
    }
    for horizon in config.enabled_horizons:
        if horizon > config.max_horizon:
            continue
        if global_step >= horizon_steps.get(horizon, config.start_step):
            weights[horizon] = float(config.horizon_weights.get(horizon, 0.0)) * ramp
    return weights


class MTPHeads(nn.Module):
    def __init__(self, d_model: int, vocab_size: int, config: MTPConfig):
        super().__init__()
        self.config = config
        self.projections = nn.ModuleDict(
            {str(h): nn.Linear(d_model, d_model, bias=False) for h in config.enabled_horizons}
        )
        self.separate_heads = nn.ModuleDict()
        if not config.use_shared_lm_head:
            self.separate_heads = nn.ModuleDict(
                {str(h): nn.Linear(d_model, vocab_size, bias=False) for h in config.enabled_horizons}
            )

    def logits_for_horizon(
        self,
        hidden: torch.Tensor,
        horizon: int,
        shared_lm_head: nn.Module,
    ) -> torch.Tensor:
        context = hidden.detach() if self.config.detach_auxiliary_context else hidden
        projected = self.projections[str(horizon)](context)
        if self.config.use_shared_lm_head:
            return shared_lm_head(projected)
        return self.separate_heads[str(horizon)](projected)

