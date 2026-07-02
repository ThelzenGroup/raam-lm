from __future__ import annotations

from raam_lm.registry import available_models, build_model
from tests.test_shapes import tiny_config


def test_registry_models_build():
    assert available_models() == ["transformer", "pure_mamba_like", "raam"]
    for name in available_models():
        model = build_model(tiny_config(name))
        assert model is not None

