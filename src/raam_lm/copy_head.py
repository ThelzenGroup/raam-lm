"""Causal pointer-copy logits for exact current-context token binding."""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import CopyHeadConfig


class CausalCopyHead(nn.Module):
    """Mix vocabulary logits with a causal distribution over previous input tokens."""

    def __init__(self, d_model: int, vocab_size: int, config: CopyHeadConfig):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        d_copy = int(config.d_copy or d_model)
        self.query = nn.Linear(d_model, d_copy, bias=False)
        self.key = nn.Linear(d_model, d_copy, bias=False)

    def _consistency_bias(self, input_ids: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor | None:
        recent_tokens = max(0, int(self.config.consistency_recent_tokens))
        source_window = max(0, int(self.config.consistency_source_window))
        strength = float(self.config.consistency_strength)
        if not strength or not recent_tokens or not source_window:
            return None

        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        bias = torch.zeros(bsz, seq_len, seq_len, device=input_ids.device, dtype=torch.float32)
        causal_sources = source_mask[0]
        comparison_weight = 0.0
        # Use the current token plus recent context to favor source positions whose
        # preceding neighborhood contains the same already-generated binding token.
        for recent_offset in range(recent_tokens):
            recent_weight = float(recent_tokens - recent_offset)
            recent_valid = positions >= recent_offset
            recent_pos = (positions - recent_offset).clamp_min(0)
            recent_ids = input_ids[:, recent_pos]
            for source_offset in range(-source_window, 1):
                source_pos = positions + source_offset
                source_valid = (source_pos >= 0) & (source_pos < seq_len)
                clipped_source_pos = source_pos.clamp(0, seq_len - 1)
                source_ids = input_ids[:, clipped_source_pos]
                pair_valid = (
                    recent_valid[:, None]
                    & source_valid[None, :]
                    & (source_pos[None, :] <= positions[:, None])
                    & causal_sources
                )
                matches = recent_ids[:, :, None] == source_ids[:, None, :]
                bias = bias + recent_weight * (matches & pair_valid.unsqueeze(0)).to(dtype=bias.dtype)
                comparison_weight += recent_weight
        return bias * (strength / max(comparison_weight, 1.0))

    def _binding_carry_probs_by_pos(
        self,
        input_ids: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor | None:
        recent_tokens = max(0, int(self.config.binding_carry_recent_tokens))
        source_window = max(0, int(self.config.binding_carry_source_window))
        strength = float(self.config.binding_carry_strength)
        if not strength or not recent_tokens or not source_window:
            return None

        bsz, seq_len = input_ids.shape
        min_source_gap = max(0, int(self.config.binding_carry_min_source_gap))
        max_anchor_occurrences = max(0, int(self.config.binding_carry_max_anchor_occurrences))
        positions = torch.arange(seq_len, device=input_ids.device)
        carry = torch.zeros(bsz, seq_len, seq_len, device=input_ids.device, dtype=torch.float32)
        causal_sources = source_mask[0]
        if min_source_gap:
            causal_sources = causal_sources & ((positions[:, None] - positions[None, :]) >= min_source_gap)
        visible_prefix = positions[None, :] <= positions[:, None]

        # Carry a rare, recently emitted binding token forward to source tokens
        # that follow the same token in the causal context row.
        for recent_offset in range(1, recent_tokens + 1):
            recent_weight = float(recent_tokens - recent_offset + 1)
            recent_valid = positions >= recent_offset
            recent_pos = (positions - recent_offset).clamp_min(0)
            recent_ids = input_ids[:, recent_pos]
            if max_anchor_occurrences:
                visible_matches = input_ids[:, None, :] == recent_ids[:, :, None]
                occurrence_counts = (visible_matches & visible_prefix.unsqueeze(0)).sum(dim=-1)
                recent_valid_by_batch = occurrence_counts <= max_anchor_occurrences
            else:
                recent_valid_by_batch = torch.ones(bsz, seq_len, device=input_ids.device, dtype=torch.bool)

            for source_distance in range(1, source_window + 1):
                anchor_pos = positions - source_distance
                anchor_valid = anchor_pos >= 0
                clipped_anchor_pos = anchor_pos.clamp(0, seq_len - 1)
                anchor_ids = input_ids[:, clipped_anchor_pos]
                pair_valid = recent_valid[:, None] & anchor_valid[None, :] & causal_sources
                matches = recent_ids[:, :, None] == anchor_ids[:, None, :]
                valid = pair_valid.unsqueeze(0) & recent_valid_by_batch[:, :, None]
                carry = carry + recent_weight * (matches & valid).to(dtype=carry.dtype)

        denom = carry.sum(dim=-1, keepdim=True)
        return torch.where(denom > 0, carry / denom.clamp_min(1e-12), torch.zeros_like(carry))

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        lm_logits: torch.Tensor,
    ) -> torch.Tensor:
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            bsz, seq_len, _ = hidden.shape
            hidden = hidden.float()
            lm_logits = lm_logits.float()
            query = self.query(hidden)
            key = self.key(hidden)
            scores = torch.matmul(query, key.transpose(-1, -2))
            scores = scores / math.sqrt(query.shape[-1])
            scores = scores / max(float(self.config.temperature), 1e-6)

            offset = 0 if self.config.include_current_token else 1
            causal_mask = torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool).tril(diagonal=-offset)
            source_mask = causal_mask.unsqueeze(0)
            consistency_bias = self._consistency_bias(input_ids, source_mask)
            if consistency_bias is not None:
                scores = scores + consistency_bias
            scores = scores.masked_fill(~source_mask, -1.0e9)
            copy_probs_by_pos = torch.softmax(scores, dim=-1)
            copy_probs_by_pos = copy_probs_by_pos * source_mask.to(dtype=copy_probs_by_pos.dtype)
            denom = copy_probs_by_pos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            copy_probs_by_pos = copy_probs_by_pos / denom
            copy_probs_by_vocab = torch.zeros(
                bsz,
                seq_len,
                self.vocab_size,
                device=hidden.device,
                dtype=torch.float32,
            )
            index = input_ids.unsqueeze(1).expand(bsz, seq_len, seq_len)
            copy_probs_by_vocab.scatter_add_(dim=-1, index=index, src=copy_probs_by_pos)
            copy_logits = torch.log(copy_probs_by_vocab.clamp_min(1e-12)) + float(self.config.logit_scale)
            carry_probs_by_pos = self._binding_carry_probs_by_pos(input_ids, source_mask)
            if carry_probs_by_pos is not None:
                carry_probs_by_vocab = torch.zeros_like(copy_probs_by_vocab)
                carry_probs_by_vocab.scatter_add_(dim=-1, index=index, src=carry_probs_by_pos)
                carry_logits = torch.log(carry_probs_by_vocab.clamp_min(1e-12)) + float(
                    self.config.binding_carry_strength
                )
                copy_logits = torch.logaddexp(copy_logits, carry_logits)
            return torch.logaddexp(lm_logits, copy_logits)
