"""Loss helpers for next-token, reconstruction, and MTP objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import CompressionConfig, MTPConfig
from .mtp import MTPHeads, current_mtp_weights


def shifted_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    horizon: int = 1,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape[1] <= horizon:
        return logits.sum() * 0.0
    pred = logits[:, :-horizon, :].contiguous()
    target = labels[:, horizon:].contiguous()
    token_losses = F.cross_entropy(
        pred.view(-1, pred.shape[-1]),
        target.view(-1),
        reduction="none",
    )
    if loss_mask is None:
        return token_losses.mean()
    target_mask = loss_mask[:, horizon:].contiguous().view(-1).to(dtype=token_losses.dtype)
    denom = target_mask.sum()
    if denom <= 0:
        return token_losses.sum() * 0.0
    return (token_losses * target_mask).sum() / denom


def current_recon_weight(config: CompressionConfig, global_step: int) -> float:
    if not config.enabled or config.recon_loss_weight <= 0:
        return 0.0
    if global_step < config.recon_loss_start_step:
        return 0.0
    ramp = min(
        1.0,
        max(
            0.0,
            (global_step - config.recon_loss_start_step + 1)
            / max(config.recon_loss_ramp_steps, 1),
        ),
    )
    return float(config.recon_loss_weight) * ramp


def compute_lm_losses(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    mtp_heads: MTPHeads | None,
    mtp_config: MTPConfig,
    global_step: int,
    recon_loss: torch.Tensor | None = None,
    recon_weight: float = 0.0,
    loss_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | dict[int, torch.Tensor] | dict[int, float]]:
    zero = logits.sum() * 0.0
    if labels is None:
        next_loss = zero
        total = zero
    else:
        next_loss = shifted_cross_entropy(logits, labels, horizon=1, loss_mask=loss_mask)
        total = next_loss
    recon_value = recon_loss if recon_loss is not None else zero
    if labels is not None and recon_weight:
        total = total + float(recon_weight) * recon_value

    mtp_losses: dict[int, torch.Tensor] = {}
    mtp_weights = current_mtp_weights(mtp_config, global_step)
    if labels is not None and mtp_heads is not None and mtp_weights:
        for horizon, weight in mtp_weights.items():
            aux_logits = mtp_heads.logits_for_horizon(hidden, horizon, lm_head)
            mtp_loss = shifted_cross_entropy(aux_logits, labels, horizon=horizon, loss_mask=loss_mask)
            mtp_losses[horizon] = mtp_loss
            total = total + float(weight) * mtp_loss

    return {
        "loss": total,
        "next_token_loss": next_loss,
        "recon_loss": recon_value,
        "mtp_loss_by_horizon": mtp_losses,
        "mtp_weights": mtp_weights,
    }
