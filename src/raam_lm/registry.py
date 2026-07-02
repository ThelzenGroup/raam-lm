"""Model registry for RAAM-LM and baselines."""

from __future__ import annotations

from .baselines import DenseTransformerForCausalLM, PureMambaLikeForCausalLM
from .config import ModelConfig
from .model import RAAMForCausalLM


def available_models() -> list[str]:
    return ["transformer", "pure_mamba_like", "raam"]


def build_model(config: ModelConfig):
    if config.model_name == "transformer":
        return DenseTransformerForCausalLM(config)
    if config.model_name == "pure_mamba_like":
        return PureMambaLikeForCausalLM(config)
    if config.model_name == "raam":
        return RAAMForCausalLM(config)
    raise ValueError(f"unknown model_name={config.model_name!r}; expected one of {available_models()}")

