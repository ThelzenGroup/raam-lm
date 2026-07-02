"""Causal mixer backbones, including a clearly named non-Mamba fallback."""

from __future__ import annotations

import warnings
import torch
from torch import nn
import torch.nn.functional as F

from .layers import RMSNorm, SwiGLUFFN


class GatedDepthwiseConvMixer(nn.Module):
    """Shape-stable causal depthwise-convolution mixer.

    This is a fallback mixer, not a real Mamba implementation.
    """

    backend_name = "fallback_gated_conv"

    def __init__(self, d_model: int, expansion: int = 2, kernel_size: int = 5, dropout: float = 0.0):
        super().__init__()
        self.d_inner = d_model * expansion
        self.kernel_size = kernel_size
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=kernel_size,
            groups=self.d_inner,
            bias=True,
        )
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value_t = value.transpose(1, 2)
        value_t = F.pad(value_t, (self.kernel_size - 1, 0))
        mixed = self.conv(value_t).transpose(1, 2)
        return self.out_proj(self.dropout(mixed * F.silu(gate)))


class MambaMixer(nn.Module):
    """Optional mamba_ssm wrapper, used only when the dependency is installed."""

    backend_name = "mamba_ssm"

    def __init__(self, d_model: int, dropout: float = 0.0):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except Exception as exc:  # pragma: no cover - optional dependency path.
            raise RuntimeError("mamba_ssm is unavailable") from exc
        self.mamba = Mamba(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - optional.
        return self.dropout(self.mamba(x))


def make_mixer(d_model: int, backend: str = "auto", dropout: float = 0.0) -> nn.Module:
    if backend in {"auto", "mamba_ssm"}:
        try:
            return MambaMixer(d_model=d_model, dropout=dropout)
        except RuntimeError:
            if backend == "mamba_ssm":
                raise
            warnings.warn(
                "mamba_ssm unavailable; using fallback_gated_conv mixer_backend",
                RuntimeWarning,
                stacklevel=2,
            )
    return GatedDepthwiseConvMixer(d_model=d_model, dropout=dropout)


class MixerBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0, backend: str = "auto"):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.mixer = make_mixer(d_model, backend=backend, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)

    @property
    def mixer_backend(self) -> str:
        return getattr(self.mixer, "backend_name", "unknown")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

