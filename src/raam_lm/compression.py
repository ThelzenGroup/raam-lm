"""Dynamic hourglass compression and causal-safe expansion."""

from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from .config import CompressionConfig
from .layers import RMSNorm


def dtype_mask_min(tensor: torch.Tensor) -> float:
    """Return a finite mask value representable by the tensor dtype."""

    return torch.finfo(tensor.dtype).min if tensor.dtype.is_floating_point else -1e9


@dataclass
class CompressionOutput:
    compressed_x: torch.Tensor
    compressed_positions: torch.Tensor
    compressed_causal_origin: torch.Tensor
    anchor_indices: torch.Tensor
    token_to_block: torch.Tensor
    metadata: dict[str, torch.Tensor | int | float]
    recon_loss: torch.Tensor
    mean_anchor_score: torch.Tensor


def batch_gather_sequence(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Gather [B, L, D] by [B, M] indices."""

    return torch.gather(x, 1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))


def sort_by_origin(
    x: torch.Tensor, origins: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sort a batch of sequences by causal origin for safe causal convolutions."""

    order = torch.argsort(origins, dim=-1, stable=True)
    sorted_x = batch_gather_sequence(x, order)
    sorted_origins = torch.gather(origins, 1, order)
    inverse = torch.empty_like(order)
    arange = torch.arange(order.shape[1], device=order.device).unsqueeze(0).expand_as(order)
    inverse.scatter_(1, order, arange)
    return sorted_x, sorted_origins, inverse


def unsort_sequence(x: torch.Tensor, inverse_order: torch.Tensor) -> torch.Tensor:
    return batch_gather_sequence(x, inverse_order)


class DynamicHourglassCompressor(nn.Module):
    """Fixed-shape learned top-k anchor compressor with delayed chunk origins."""

    def __init__(self, d_model: int, config: CompressionConfig):
        super().__init__()
        self.config = config
        hidden = max(config.anchor_score_hidden, d_model // 2)
        self.anchor_scorer = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        self.chunk_scorer = nn.Linear(d_model, config.pooled_chunks_per_block)
        self.recon_proj = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> CompressionOutput:
        bsz, seq_len, d_model = x.shape
        cfg = self.config
        block_size = cfg.block_size
        n_blocks = (seq_len + block_size - 1) // block_size
        padded_len = n_blocks * block_size
        pad_len = padded_len - seq_len
        if pad_len:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
        blocks = x_pad.view(bsz, n_blocks, block_size, d_model)
        valid = torch.arange(padded_len, device=x.device).view(1, n_blocks, block_size) < seq_len
        if attention_mask is not None:
            mask_pad = F.pad(attention_mask.to(torch.bool), (0, pad_len), value=False)
            valid = valid & mask_pad.view(bsz, n_blocks, block_size)
        else:
            valid = valid.expand(bsz, -1, -1)

        pooled = cfg.pooled_chunks_per_block
        chunk_logits = self.chunk_scorer(blocks)
        mask_min = dtype_mask_min(chunk_logits)
        chunk_logits = chunk_logits.masked_fill(~valid.unsqueeze(-1), mask_min)
        chunk_weights = torch.softmax(chunk_logits, dim=2)
        chunk_tokens = torch.einsum("bnsp,bnsd->bnpd", chunk_weights, blocks)

        anchor_scores = self.anchor_scorer(blocks).squeeze(-1)
        anchor_scores = anchor_scores.masked_fill(~valid, dtype_mask_min(anchor_scores))
        anchors = int(cfg.anchors_per_block)
        if anchors > block_size:
            raise ValueError("anchors_per_block must be <= block_size")
        if anchors > 0:
            valid_counts = valid.sum(dim=-1, keepdim=True).clamp_min(1)
            if cfg.anchor_selection == "learned_topk":
                _, local_idx = torch.topk(anchor_scores, k=anchors, dim=-1)
            elif cfg.anchor_selection == "uniform":
                base_idx = torch.linspace(
                    0,
                    block_size - 1,
                    steps=anchors,
                    device=x.device,
                    dtype=torch.float32,
                ).round().to(torch.long)
                local_idx = base_idx.view(1, 1, anchors).expand(bsz, n_blocks, anchors)
            else:
                raise ValueError("anchor_selection must be 'learned_topk' or 'uniform'")
            local_idx = torch.sort(local_idx, dim=-1).values
            local_idx = torch.minimum(local_idx, valid_counts - 1)
            anchor_valid_selected = torch.gather(valid, 2, local_idx)
            anchor_tokens = torch.gather(
                blocks,
                2,
                local_idx.unsqueeze(-1).expand(-1, -1, -1, d_model),
            )
            anchor_scores_selected = torch.gather(anchor_scores, 2, local_idx)
        else:
            local_idx = torch.empty(bsz, n_blocks, 0, device=x.device, dtype=torch.long)
            anchor_valid_selected = torch.empty(bsz, n_blocks, 0, device=x.device, dtype=torch.bool)
            anchor_tokens = torch.empty(bsz, n_blocks, 0, d_model, device=x.device, dtype=x.dtype)
            anchor_scores_selected = torch.empty(bsz, n_blocks, 0, device=x.device, dtype=x.dtype)

        block_starts = torch.arange(n_blocks, device=x.device, dtype=torch.long) * block_size
        anchor_indices = (block_starts.view(1, n_blocks, 1) + local_idx).clamp(max=max(seq_len - 1, 0))
        entries = pooled + anchors
        compressed_blocks = torch.cat([chunk_tokens, anchor_tokens], dim=2)
        compressed_x = compressed_blocks.reshape(bsz, n_blocks * entries, d_model)

        block_ends = (block_starts + block_size - 1).clamp(max=max(seq_len - 1, 0))
        chunk_origins = block_ends
        chunk_origins = chunk_origins.view(1, n_blocks, 1).expand(bsz, -1, pooled)
        anchor_origins = anchor_indices
        origins = torch.cat([chunk_origins, anchor_origins], dim=2).reshape(bsz, n_blocks * entries)

        chunk_positions = block_starts.view(1, n_blocks, 1).expand(bsz, -1, pooled)
        positions = torch.cat([chunk_positions, anchor_indices], dim=2).reshape(bsz, n_blocks * entries)
        token_to_block = torch.div(
            torch.arange(seq_len, device=x.device, dtype=torch.long),
            block_size,
            rounding_mode="floor",
        ).clamp(max=n_blocks - 1)

        recon_source = chunk_tokens.mean(dim=2)
        recon = self.recon_proj(recon_source).unsqueeze(2).expand(-1, -1, block_size, -1)
        target = blocks.detach() if cfg.stopgrad_recon_target else blocks
        recon_per_token = (recon - target).pow(2).mean(dim=-1)
        recon_loss = (recon_per_token * valid.float()).sum() / valid.float().sum().clamp_min(1.0)

        chunk_stream_indices = (
            torch.arange(n_blocks, device=x.device).view(n_blocks, 1) * entries
            + torch.arange(pooled, device=x.device).view(1, pooled)
        )
        anchor_stream_indices = (
            torch.arange(n_blocks, device=x.device).view(n_blocks, 1) * entries
            + pooled
            + torch.arange(anchors, device=x.device).view(1, anchors)
        )
        valid_anchor_scores = anchor_scores_selected.masked_select(anchor_valid_selected)
        mean_anchor_score = (
            valid_anchor_scores.mean()
            if valid_anchor_scores.numel() > 0
            else torch.zeros((), device=x.device, dtype=x.dtype)
        )
        metadata: dict[str, torch.Tensor | int | float] = {
            "n_blocks": n_blocks,
            "block_size": block_size,
            "entries_per_block": entries,
            "pooled_chunks_per_block": pooled,
            "anchors_per_block": anchors,
            "anchor_selection": cfg.anchor_selection,
            "anchor_indices": anchor_indices,
            "anchor_local_indices": local_idx,
            "token_to_block": token_to_block,
            "chunk_stream_indices": chunk_stream_indices,
            "anchor_stream_indices": anchor_stream_indices,
            "compressed_causal_origin": origins,
            "compressed_positions": positions,
            "seq_len": seq_len,
        }
        return CompressionOutput(
            compressed_x=compressed_x,
            compressed_positions=positions,
            compressed_causal_origin=origins,
            anchor_indices=anchor_indices,
            token_to_block=token_to_block,
            metadata=metadata,
            recon_loss=recon_loss,
            mean_anchor_score=mean_anchor_score,
        )


class AnchorPreservedLocalGlobal(nn.Module):
    """Causal expansion from delayed global stream back to token states.

    Current-block anchor routing is intentionally not scattered into current
    positions: dynamic top-k anchor selection is block-level and would leak if
    it changed logits inside the same block. Preserved anchors are still present
    in the global stream for later blocks and attention islands.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.chunk_proj = nn.Linear(d_model, d_model, bias=False)
        self.anchor_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        local_x: torch.Tensor,
        global_x: torch.Tensor,
        metadata: dict[str, torch.Tensor | int | float],
    ) -> torch.Tensor:
        bsz, seq_len, _ = local_x.shape
        chunk_indices = metadata["chunk_stream_indices"]
        if not isinstance(chunk_indices, torch.Tensor):
            raise TypeError("chunk_stream_indices missing from compression metadata")
        first_chunk_idx = chunk_indices[:, 0]
        chunk_states = global_x[:, first_chunk_idx, :]
        token_to_block = metadata["token_to_block"]
        if not isinstance(token_to_block, torch.Tensor):
            raise TypeError("token_to_block missing from compression metadata")
        prior_block = (token_to_block - 1).clamp_min(0)
        valid_prior = (token_to_block > 0).to(local_x.dtype).view(1, seq_len, 1)
        prior_chunk = chunk_states[:, prior_block, :] * valid_prior
        return local_x + self.chunk_proj(prior_chunk)
