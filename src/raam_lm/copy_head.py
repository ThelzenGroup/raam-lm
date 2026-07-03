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

    def forward(
        self,
        hidden: torch.Tensor,
        input_ids: torch.Tensor,
        lm_logits: torch.Tensor,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden.shape
        query = self.query(hidden)
        key = self.key(hidden)
        scores = torch.matmul(query, key.transpose(-1, -2))
        scores = scores / math.sqrt(query.shape[-1])
        scores = scores / max(float(self.config.temperature), 1e-6)

        offset = 0 if self.config.include_current_token else 1
        causal_mask = torch.ones(seq_len, seq_len, device=hidden.device, dtype=torch.bool).tril(diagonal=-offset)
        source_mask = causal_mask.unsqueeze(0)
        scores = scores.masked_fill(~source_mask, torch.finfo(scores.dtype).min)
        copy_probs_by_pos = torch.softmax(scores.float(), dim=-1).to(dtype=lm_logits.dtype)
        copy_probs_by_pos = copy_probs_by_pos * source_mask.to(dtype=copy_probs_by_pos.dtype)
        denom = copy_probs_by_pos.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        copy_probs_by_pos = copy_probs_by_pos / denom
        copy_probs_by_pos = copy_probs_by_pos.to(dtype=lm_logits.dtype)
        copy_probs_by_vocab = torch.zeros(
            bsz,
            seq_len,
            self.vocab_size,
            device=hidden.device,
            dtype=lm_logits.dtype,
        )
        index = input_ids.unsqueeze(1).expand(bsz, seq_len, seq_len)
        copy_probs_by_vocab.scatter_add_(dim=-1, index=index, src=copy_probs_by_pos)
        copy_logits = torch.log(copy_probs_by_vocab.clamp_min(1e-12)) + float(self.config.logit_scale)
        return torch.logaddexp(lm_logits, copy_logits)
