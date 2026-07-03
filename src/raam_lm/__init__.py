"""RAAM-LM research prototype package."""

from .config import ModelConfig, load_config

__all__ = ["ModelConfig", "load_config", "build_model", "available_models"]


def available_models() -> list[str]:
    from .registry import available_models as _available_models

    return _available_models()


def build_model(config: ModelConfig):
    from .registry import build_model as _build_model

    return _build_model(config)
