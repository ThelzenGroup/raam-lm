"""RAAM-LM decoder-only causal language model."""

from __future__ import annotations

import torch
from torch import nn

from .attention import CausalAttentionIsland
from .compression import (
    AnchorPreservedLocalGlobal,
    DynamicHourglassCompressor,
    sort_by_origin,
    unsort_sequence,
)
from .config import ModelConfig
from .layers import RMSNorm, causal_positions, init_weights
from .losses import compute_lm_losses, current_recon_weight
from .mixer import MixerBlock
from .mtp import MTPHeads


def _tie_or_init(model: nn.Module, tok_embeddings: nn.Embedding, lm_head: nn.Linear, tie: bool) -> None:
    model.apply(init_weights)
    if tie:
        lm_head.weight = tok_embeddings.weight


def _loss_output(
    logits: torch.Tensor,
    labels: torch.Tensor | None,
    hidden: torch.Tensor,
    lm_head: nn.Module,
    mtp_heads: MTPHeads | None,
    config: ModelConfig,
    global_step: int,
    recon_loss: torch.Tensor | None = None,
    recon_weight: float = 0.0,
    aux: dict[str, float | str | list[int]] | None = None,
) -> dict[str, object]:
    losses = compute_lm_losses(
        logits=logits,
        labels=labels,
        hidden=hidden,
        lm_head=lm_head,
        mtp_heads=mtp_heads,
        mtp_config=config.mtp,
        global_step=global_step,
        recon_loss=recon_loss,
        recon_weight=recon_weight,
    )
    output: dict[str, object] = {
        **losses,
        "logits": logits,
        "hidden_states": hidden,
        "aux": aux or {},
    }
    return output


class RAAMForCausalLM(nn.Module):
    """Reconstructive Anchor-Attention Mamba Language Model prototype."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        lower_count = max(1, config.n_layers // 3)
        final_count = 1
        global_count = max(1, config.n_layers - lower_count - final_count)
        self.lower_blocks = nn.ModuleList(
            [
                MixerBlock(
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    backend=config.mixer_backend if config.use_mamba_or_fallback_backbone else "fallback",
                )
                for _ in range(lower_count)
            ]
        )
        self.compressor = (
            DynamicHourglassCompressor(config.d_model, config.compression)
            if config.compression.enabled and config.use_dynamic_hourglass_compression
            else None
        )
        self.global_blocks = nn.ModuleList()
        attention_layers = set(config.attention_island_layers if config.use_attention_islands else [])
        for idx in range(global_count):
            layer_id = lower_count + idx
            if layer_id in attention_layers:
                self.global_blocks.append(
                    CausalAttentionIsland(
                        config.d_model,
                        config.n_heads,
                        config.d_ff,
                        n_kv_heads=config.n_kv_heads,
                        dropout=config.dropout,
                        rope_base=config.rope_base,
                    )
                )
            else:
                self.global_blocks.append(
                    MixerBlock(
                        config.d_model,
                        config.d_ff,
                        dropout=config.dropout,
                        backend=config.mixer_backend if config.use_mamba_or_fallback_backbone else "fallback",
                    )
                )
        self.expander = AnchorPreservedLocalGlobal(config.d_model)
        self.final_blocks = nn.ModuleList(
            [
                MixerBlock(
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    backend=config.mixer_backend if config.use_mamba_or_fallback_backbone else "fallback",
                )
                for _ in range(final_count)
            ]
        )
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.mtp_heads = (
            MTPHeads(config.d_model, config.vocab_size, config.mtp)
            if config.mtp.enabled and config.use_curriculum_mtp
            else None
        )
        _tie_or_init(self, self.tok_embeddings, self.lm_head, config.tie_embeddings)

    @property
    def mixer_backend(self) -> str:
        for block in list(self.lower_blocks) + list(self.global_blocks) + list(self.final_blocks):
            if hasattr(block, "mixer_backend"):
                return block.mixer_backend
        return "none"

    def _run_global(self, global_x: torch.Tensor, origins: torch.Tensor) -> torch.Tensor:
        sorted_x, sorted_origins, inverse = sort_by_origin(global_x, origins)
        for block in self.global_blocks:
            if isinstance(block, CausalAttentionIsland):
                sorted_x = block(sorted_x, origins=sorted_origins)
            else:
                sorted_x = block(sorted_x)
        return unsort_sequence(sorted_x, inverse)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        global_step: int = 0,
    ) -> dict[str, object]:
        x = self.tok_embeddings(input_ids)
        for block in self.lower_blocks:
            x = block(x)
        local_x = x
        bsz, seq_len, _ = x.shape
        recon_loss = None
        mean_anchor_score = torch.zeros((), device=x.device, dtype=x.dtype)
        anchor_fraction = 0.0

        if self.compressor is not None:
            comp = self.compressor(x, input_ids=input_ids)
            global_x = self._run_global(comp.compressed_x, comp.compressed_causal_origin)
            x = self.expander(local_x, global_x, comp.metadata)
            recon_loss = comp.recon_loss
            ratio = comp.compressed_x.shape[1] / max(seq_len, 1)
            mean_anchor_score = comp.mean_anchor_score
            anchor_fraction = (
                self.config.compression.anchors_per_block
                / max(self.config.compression.block_size, 1)
                if self.config.compression.enabled
                else 0.0
            )
        else:
            origins = causal_positions(bsz, seq_len, x.device)
            x = self._run_global(x, origins)
            ratio = 1.0

        for block in self.final_blocks:
            x = block(x)
        hidden = self.norm(x)
        logits = self.lm_head(hidden)
        recon_weight = current_recon_weight(self.config.compression, global_step)
        aux = {
            "model_name": "raam",
            "mixer_backend": self.mixer_backend,
            "compression_ratio": float(ratio),
            "mean_anchor_score": float(mean_anchor_score.detach().cpu()),
            "anchor_token_fraction": float(anchor_fraction),
            "attention_island_layers": list(self.config.attention_island_layers),
        }
        return _loss_output(
            logits,
            labels,
            hidden,
            self.lm_head,
            self.mtp_heads,
            self.config,
            global_step,
            recon_loss=recon_loss,
            recon_weight=recon_weight,
            aux=aux,
        )

