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

    def _source_region_mask(
        self,
        input_ids: torch.Tensor,
        source_mask: torch.Tensor,
        *,
        min_source_gap: int,
        source_until_token_id: int,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        positions = torch.arange(seq_len, device=input_ids.device)
        region = source_mask.expand(bsz, -1, -1)
        if min_source_gap:
            region = region & ((positions[None, :, None] - positions[None, None, :]) >= min_source_gap)
        if source_until_token_id >= 0:
            boundary_positions = torch.where(
                input_ids == source_until_token_id,
                positions.unsqueeze(0),
                torch.full((bsz, seq_len), -1, device=input_ids.device, dtype=positions.dtype),
            )
            last_boundary = torch.cummax(boundary_positions, dim=1).values
            region = region & (positions[None, None, :] < last_boundary[:, :, None])
        return region

    def _key_follow_probs_by_pos(
        self,
        input_ids: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        recent_tokens = max(0, int(self.config.key_follow_recent_tokens))
        value_offset = max(1, int(self.config.key_follow_value_offset))
        value_span = max(1, int(self.config.key_follow_value_span))
        strength = float(self.config.key_follow_strength)
        if not strength or not recent_tokens:
            return None

        bsz, seq_len = input_ids.shape
        min_source_gap = max(0, int(self.config.key_follow_min_source_gap))
        source_until_token_id = int(self.config.key_follow_source_until_token_id)
        separator_token_id = int(self.config.key_follow_separator_token_id)
        recent_after_token_id = int(self.config.key_follow_recent_after_token_id)
        align_value_offset = bool(self.config.key_follow_align_value_offset)
        match_value_prefix = bool(self.config.key_follow_match_value_prefix)
        stop_token_ids = [int(token_id) for token_id in self.config.key_follow_stop_token_ids]
        positions = torch.arange(seq_len, device=input_ids.device)
        first_follow = torch.zeros(bsz, seq_len, seq_len, device=input_ids.device, dtype=torch.float32)
        continuation_follow = torch.zeros_like(first_follow)
        source_not_stopped = torch.ones(bsz, seq_len, device=input_ids.device, dtype=torch.bool)
        for token_id in stop_token_ids:
            source_not_stopped = source_not_stopped & (input_ids != token_id)
        source_region = self._source_region_mask(
            input_ids,
            source_mask,
            min_source_gap=min_source_gap,
            source_until_token_id=source_until_token_id,
        )
        if recent_after_token_id >= 0:
            recent_boundary_positions = torch.where(
                input_ids == recent_after_token_id,
                positions.unsqueeze(0),
                torch.full((bsz, seq_len), -1, device=input_ids.device, dtype=positions.dtype),
            )
            last_recent_boundary = torch.cummax(recent_boundary_positions, dim=1).values
        else:
            last_recent_boundary = torch.full((bsz, seq_len), -1, device=input_ids.device, dtype=positions.dtype)

        for recent_offset in range(1, recent_tokens + 1):
            recent_weight = float(recent_tokens - recent_offset + 1)
            recent_valid = positions >= recent_offset
            recent_pos = (positions - recent_offset).clamp_min(0)
            recent_ids = input_ids[:, recent_pos]
            recent_after_boundary = recent_pos.unsqueeze(0) > last_recent_boundary
            if separator_token_id >= 0:
                separator_pos = (positions - recent_offset + 1).clamp_min(0)
                separator_ids = input_ids[:, separator_pos]
                separator_valid = positions >= (recent_offset - 1)
                separator_matches = separator_ids == separator_token_id
            else:
                separator_valid = torch.ones(seq_len, device=input_ids.device, dtype=torch.bool)
                separator_matches = torch.ones(bsz, seq_len, device=input_ids.device, dtype=torch.bool)
            offsets = (
                [value_offset + recent_offset - 1]
                if align_value_offset
                else range(value_offset, value_offset + value_span)
            )
            for offset in offsets:
                key_pos = positions - offset
                key_valid = key_pos >= 0
                clipped_key_pos = key_pos.clamp(0, seq_len - 1)
                key_ids = input_ids[:, clipped_key_pos]
                pair_valid = recent_valid[:, None] & separator_valid[:, None] & key_valid[None, :]
                matches = recent_ids[:, :, None] == key_ids[:, None, :]
                if match_value_prefix and align_value_offset and recent_offset > 1:
                    source_prev_pos = (positions - 1).clamp_min(0)
                    source_prev_valid = positions > 0
                    source_prev_ids = input_ids[:, source_prev_pos]
                    generated_prev_ids = input_ids[:, positions]
                    prefix_matches = generated_prev_ids[:, :, None] == source_prev_ids[:, None, :]
                    pair_valid = pair_valid & source_prev_valid[None, :]
                    matches = matches & prefix_matches
                valid = (
                    pair_valid.unsqueeze(0)
                    & separator_matches[:, :, None]
                    & recent_after_boundary[:, :, None]
                    & source_not_stopped[:, None, :]
                    & source_region
                )
                target = first_follow if recent_offset == 1 else continuation_follow
                target += recent_weight * (matches & valid).to(dtype=target.dtype)

        first_denom = first_follow.sum(dim=-1, keepdim=True)
        continuation_denom = continuation_follow.sum(dim=-1, keepdim=True)
        first_probs = torch.where(
            first_denom > 0,
            first_follow / first_denom.clamp_min(1e-12),
            torch.zeros_like(first_follow),
        )
        continuation_probs = torch.where(
            continuation_denom > 0,
            continuation_follow / continuation_denom.clamp_min(1e-12),
            torch.zeros_like(continuation_follow),
        )
        return first_probs, continuation_probs

    def _request_key_follow_probs_by_pos(
        self,
        input_ids: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        strength = float(self.config.request_key_follow_strength)
        continuation_strength = float(
            self.config.request_key_follow_continuation_strength
            or self.config.request_key_follow_strength
        )
        if not strength and not continuation_strength:
            return None

        bsz, seq_len = input_ids.shape
        value_offset = max(1, int(self.config.key_follow_value_offset))
        value_span = max(1, int(self.config.request_key_follow_value_span))
        recent_tokens = max(0, int(self.config.request_key_follow_recent_tokens))
        request_after_token_id = int(self.config.request_key_follow_after_token_id)
        request_before_token_id = int(self.config.request_key_follow_before_token_id)
        source_after_token_id = int(self.config.request_key_follow_source_after_token_id)
        query_after_token_id = int(self.config.request_key_follow_query_after_token_id)
        query_before_token_ids = [
            int(token_id) for token_id in self.config.request_key_follow_query_before_token_ids
        ]
        query_ignore_token_ids = [
            int(token_id) for token_id in self.config.request_key_follow_query_ignore_token_ids
        ]
        prompt_suffix_tokens = max(0, int(self.config.request_key_follow_prompt_suffix_tokens))
        if request_after_token_id < 0 or request_before_token_id < 0:
            return None

        min_source_gap = max(0, int(self.config.key_follow_min_source_gap))
        source_until_token_id = int(self.config.key_follow_source_until_token_id)
        stop_token_ids = [int(token_id) for token_id in self.config.key_follow_stop_token_ids]
        positions = torch.arange(seq_len, device=input_ids.device)
        source_region = self._source_region_mask(
            input_ids,
            source_mask,
            min_source_gap=min_source_gap,
            source_until_token_id=source_until_token_id,
        )
        if source_after_token_id >= 0:
            source_after_positions = torch.where(
                input_ids == source_after_token_id,
                positions.unsqueeze(0),
                torch.full((bsz, seq_len), -1, device=input_ids.device, dtype=positions.dtype),
            )
            last_source_after = torch.cummax(source_after_positions, dim=1).values
            source_region = source_region & (positions[None, None, :] > last_source_after[:, :, None])
        source_stop_mask = torch.zeros(bsz, seq_len, device=input_ids.device, dtype=torch.bool)
        for token_id in stop_token_ids:
            source_stop_mask = source_stop_mask | (input_ids == token_id)
        source_not_stopped = ~source_stop_mask
        source_stop_prefix = source_stop_mask.to(dtype=torch.int64).cumsum(dim=1)

        request_after_positions = torch.where(
            input_ids == request_after_token_id,
            positions.unsqueeze(0),
            torch.full((bsz, seq_len), -1, device=input_ids.device, dtype=positions.dtype),
        )
        last_request_after = torch.cummax(request_after_positions, dim=1).values
        request_before_positions = torch.where(
            input_ids == request_before_token_id,
            positions.unsqueeze(0),
            torch.full((bsz, seq_len), -1, device=input_ids.device, dtype=positions.dtype),
        )
        last_request_before = torch.cummax(request_before_positions, dim=1).values
        request_pos = positions
        request_valid = (
            (request_pos[None, None, :] > last_request_after[:, :, None])
            & (request_pos[None, None, :] < last_request_before[:, :, None])
            & (request_pos[None, None, :] <= positions[None, :, None])
        )
        query_start = last_request_after
        if query_after_token_id >= 0:
            query_after_positions = torch.where(
                input_ids == query_after_token_id,
                positions.unsqueeze(0),
                torch.full((bsz, seq_len), -1, device=input_ids.device, dtype=positions.dtype),
            )
            last_query_after = torch.cummax(query_after_positions, dim=1).values
            query_start = torch.maximum(query_start, last_query_after)
            request_valid = request_valid & (request_pos[None, None, :] > query_start[:, :, None])
        if query_before_token_ids:
            query_stop_mask = torch.zeros(bsz, seq_len, device=input_ids.device, dtype=torch.bool)
            for token_id in query_before_token_ids:
                query_stop_mask = query_stop_mask | (input_ids == token_id)
            query_stop_candidates = (
                query_stop_mask[:, None, :]
                & (positions[None, None, :] > query_start[:, :, None])
                & (positions[None, None, :] < last_request_before[:, :, None])
            )
            first_query_stop = torch.where(
                query_stop_candidates,
                positions.view(1, 1, seq_len),
                torch.full((bsz, seq_len, seq_len), seq_len, device=input_ids.device, dtype=positions.dtype),
            ).amin(dim=-1)
            request_valid = request_valid & (request_pos[None, None, :] < first_query_stop[:, :, None])
        if query_ignore_token_ids:
            query_ignore_mask = torch.zeros(bsz, seq_len, device=input_ids.device, dtype=torch.bool)
            for token_id in query_ignore_token_ids:
                query_ignore_mask = query_ignore_mask | (input_ids == token_id)
            request_valid = request_valid & ~query_ignore_mask[:, None, :]
        if recent_tokens:
            request_valid = request_valid & (
                (last_request_before[:, :, None] - request_pos[None, None, :]) <= recent_tokens
            )
        request_ids = input_ids

        first_follow = torch.zeros(bsz, seq_len, seq_len, device=input_ids.device, dtype=torch.float32)
        continuation_follow = torch.zeros_like(first_follow)
        generated_current_ids = input_ids[:, positions]
        for value_index in range(value_span):
            key_pos = positions - value_offset - value_index
            key_valid = key_pos >= 0
            clipped_key_pos = key_pos.clamp(0, seq_len - 1)
            key_ids = input_ids[:, clipped_key_pos]
            key_stop_count = source_stop_prefix[:, clipped_key_pos]
            same_value_segment = source_stop_prefix[:, None, :] == key_stop_count[:, None, :]
            matches = request_ids[:, None, :, None] == key_ids[:, None, None, :]
            valid = (
                request_valid[:, :, :, None]
                & key_valid[None, None, None, :]
                & source_region[:, :, None, :]
                & source_not_stopped[:, None, None, :]
                & same_value_segment[:, :, None, :]
            )
            request_match = (matches & valid).to(dtype=first_follow.dtype).sum(dim=2)
            if value_index == 0:
                first_follow = first_follow + request_match
                continue

            source_prev_pos = (positions - 1).clamp_min(0)
            source_prev_valid = positions > 0
            source_prev_ids = input_ids[:, source_prev_pos]
            prefix_matches = generated_current_ids[:, :, None] == source_prev_ids[:, None, :]
            continuation_target_valid = positions.unsqueeze(0) > (
                last_request_before + prompt_suffix_tokens
            )
            continuation_follow = continuation_follow + request_match * (
                prefix_matches & source_prev_valid[None, None, :] & continuation_target_valid[:, :, None]
            ).to(dtype=continuation_follow.dtype)

        has_continuation = continuation_follow.sum(dim=-1, keepdim=True) > 0
        first_follow = torch.where(has_continuation, torch.zeros_like(first_follow), first_follow)
        first_denom = first_follow.sum(dim=-1, keepdim=True)
        continuation_denom = continuation_follow.sum(dim=-1, keepdim=True)
        first_probs = torch.where(
            first_denom > 0,
            first_follow / first_denom.clamp_min(1e-12),
            torch.zeros_like(first_follow),
        )
        continuation_probs = torch.where(
            continuation_denom > 0,
            continuation_follow / continuation_denom.clamp_min(1e-12),
            torch.zeros_like(continuation_follow),
        )
        return first_probs, continuation_probs

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
            key_follow_probs_by_pos = self._key_follow_probs_by_pos(input_ids, source_mask)
            if key_follow_probs_by_pos is not None:
                first_follow_by_pos, continuation_follow_by_pos = key_follow_probs_by_pos
                for route_probs_by_pos, route_strength in [
                    (first_follow_by_pos, float(self.config.key_follow_strength)),
                    (
                        continuation_follow_by_pos,
                        float(self.config.key_follow_continuation_strength or self.config.key_follow_strength),
                    ),
                ]:
                    route_probs_by_vocab = torch.zeros_like(copy_probs_by_vocab)
                    route_probs_by_vocab.scatter_add_(dim=-1, index=index, src=route_probs_by_pos)
                    route_logits = torch.log(route_probs_by_vocab.clamp_min(1e-12)) + route_strength
                    copy_logits = torch.logaddexp(copy_logits, route_logits)
            request_key_follow_probs_by_pos = None
            if not (self.training and bool(self.config.request_key_follow_eval_only)):
                request_key_follow_probs_by_pos = self._request_key_follow_probs_by_pos(input_ids, source_mask)
            if request_key_follow_probs_by_pos is not None:
                first_follow_by_pos, continuation_follow_by_pos = request_key_follow_probs_by_pos
                for route_probs_by_pos, route_strength in [
                    (first_follow_by_pos, float(self.config.request_key_follow_strength)),
                    (
                        continuation_follow_by_pos,
                        float(
                            self.config.request_key_follow_continuation_strength
                            or self.config.request_key_follow_strength
                        ),
                    ),
                ]:
                    if not route_strength:
                        continue
                    route_probs_by_vocab = torch.zeros_like(copy_probs_by_vocab)
                    route_probs_by_vocab.scatter_add_(dim=-1, index=index, src=route_probs_by_pos)
                    route_logits = torch.log(route_probs_by_vocab.clamp_min(1e-12)) + route_strength
                    copy_logits = torch.logaddexp(copy_logits, route_logits)
            return torch.logaddexp(lm_logits, copy_logits)
