"""Small neural network layers shared by RAAM-LM models."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * scale


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w12 = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w3 = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.dropout(value * F.silu(gate)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half : 2 * half]
    return torch.cat((-x2, x1), dim=-1)


def rope_cache(
    positions: torch.Tensor,
    head_dim: int,
    base: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head dimension")
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, device=positions.device, dtype=torch.float32) / half)
    )
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.cat((freqs.cos(), freqs.cos()), dim=-1).to(dtype)
    sin = torch.cat((freqs.sin(), freqs.sin()), dim=-1).to(dtype)
    return cos, sin


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    rope_base: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to [B, H, T, Dh] query/key tensors."""

    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(q.shape[0], -1)
    cos, sin = rope_cache(positions, q.shape[-1], rope_base, q.dtype)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def origin_attention_mask(
    query_origins: torch.Tensor,
    key_origins: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return an additive SDPA mask allowing only key_origin <= query_origin."""

    if query_origins.dim() == 1:
        query_origins = query_origins.unsqueeze(0)
    if key_origins.dim() == 1:
        key_origins = key_origins.unsqueeze(0)
    allowed = key_origins[:, None, :] <= query_origins[:, :, None]
    mask = torch.zeros(allowed.shape, device=query_origins.device, dtype=dtype)
    neg_inf = torch.finfo(dtype).min if dtype.is_floating_point else -1e9
    mask = mask.masked_fill(~allowed, neg_inf)
    return mask.unsqueeze(1)


def causal_positions(batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
    return torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)


def init_weights(module: nn.Module, std: float = 0.02) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=std)


def param_count(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())

