"""Causal attention layers and attention-island blocks."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLUFFN, apply_rope, causal_positions, origin_attention_mask


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        if n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be a multiple of n_kv_heads")
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.rope_base = rope_base

    def _shape_q(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

    def _shape_kv(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, origins: torch.Tensor | None = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        q = self._shape_q(self.q_proj(x))
        k = self._shape_kv(self.k_proj(x))
        v = self._shape_kv(self.v_proj(x))
        if origins is None:
            origins = causal_positions(bsz, seq_len, x.device)
        q, k = apply_rope(q, k, origins, self.rope_base)
        if self.n_kv_heads != self.n_heads:
            repeat = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(repeat, dim=1)
            v = v.repeat_interleave(repeat, dim=1)
        mask = origin_attention_mask(origins, origins, q.dtype)
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_kv_heads: int | None = None,
        dropout: float = 0.0,
        rope_base: float = 10000.0,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model=d_model,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, origins: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), origins=origins)
        x = x + self.ffn(self.norm2(x))
        return x


class CausalAttentionIsland(TransformerBlock):
    """An exact-attention block intended for sparse upper-layer placement."""

    pass

