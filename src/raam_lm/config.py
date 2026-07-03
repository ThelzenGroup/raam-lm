"""Typed configuration loading for RAAM-LM experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


@dataclass
class CompressionConfig:
    enabled: bool = True
    block_size: int = 8
    anchors_per_block: int = 2
    anchor_selection: str = "learned_topk"
    token_id_anchor_count: int = 0
    pooled_chunks_per_block: int = 1
    delayed_chunk_context: bool = True
    anchor_score_hidden: int = 64
    compression_dropout: float = 0.0
    recon_loss_weight: float = 0.05
    recon_loss_start_step: int = 5
    recon_loss_ramp_steps: int = 20
    stopgrad_recon_target: bool = True
    preserve_special_tokens: bool = False
    static_shapes: bool = True


@dataclass
class MTPConfig:
    enabled: bool = True
    max_horizon: int = 4
    enabled_horizons: list[int] = field(default_factory=lambda: [2, 3, 4])
    mtp_total_weight_final: float = 0.2
    horizon_weights: dict[int, float] = field(
        default_factory=lambda: {2: 0.10, 3: 0.06, 4: 0.04}
    )
    start_step: int = 5
    horizon2_step: int = 5
    horizon3_step: int = 10
    horizon4_step: int = 15
    ramp_steps: int = 10
    use_shared_lm_head: bool = True
    detach_auxiliary_context: bool = False


@dataclass
class CopyHeadConfig:
    enabled: bool = False
    d_copy: int | None = None
    logit_scale: float = 4.0
    temperature: float = 1.0
    include_current_token: bool = True


@dataclass
class TrainConfig:
    seed: int = 17
    batch_size: int = 4
    seq_len: int = 64
    steps: int = 30
    lr: float = 5e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 3
    cosine_decay: bool = False
    device: str = "auto"
    dtype: str = "auto"
    log_every: int = 5
    eval_every: int = 15
    save_every: int = 0
    output_dir: str = "runs/debug"


@dataclass
class EvalConfig:
    eval_batches: int = 2
    probe_lengths: list[int] = field(default_factory=lambda: [32, 64])
    long_context_lengths: list[int] = field(default_factory=lambda: [128])
    target_loss_threshold: float | None = None


@dataclass
class ModelConfig:
    model_name: str = "raam"
    vocab_size: int = 512
    max_seq_len: int = 128
    d_model: int = 64
    n_layers: int = 4
    n_heads: int = 4
    n_kv_heads: int | None = None
    d_ff: int = 192
    dropout: float = 0.0
    norm_type: str = "rmsnorm"
    rope_base: float = 10000.0
    tie_embeddings: bool = True
    dtype: str = "auto"
    block_layout: list[str] = field(default_factory=list)
    attention_island_layers: list[int] = field(default_factory=lambda: [1, 3])
    local_window: int = 32
    mixer_backend: str = "auto"
    use_flash_or_sdpa: bool = True
    use_gradient_checkpointing: bool = False
    use_mamba_or_fallback_backbone: bool = True
    use_dynamic_hourglass_compression: bool = True
    use_anchor_preserved_local_global: bool = True
    use_attention_islands: bool = True
    use_curriculum_mtp: bool = True
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)
    copy_head: CopyHeadConfig = field(default_factory=CopyHeadConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _coerce_horizon_weights(value: dict[Any, Any]) -> dict[int, float]:
    return {int(k): float(v) for k, v in value.items()}


def _update_dataclass(instance: Any, values: dict[str, Any]) -> Any:
    field_names = set(instance.__dataclass_fields__)
    for key, value in values.items():
        if key not in field_names:
            continue
        if key == "horizon_weights" and isinstance(value, dict):
            value = _coerce_horizon_weights(value)
        if key == "betas" and isinstance(value, list):
            value = tuple(value)
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ModelConfig:
    """Load a YAML config into nested dataclasses."""

    path = Path(path)
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - pyproject requires yaml.
        raise RuntimeError("pyyaml is required to load YAML configs") from exc

    raw = yaml.safe_load(path.read_text()) or {}
    compression = _update_dataclass(CompressionConfig(), raw.pop("compression", {}) or {})
    mtp = _update_dataclass(MTPConfig(), raw.pop("mtp", {}) or {})
    copy_head = _update_dataclass(CopyHeadConfig(), raw.pop("copy_head", {}) or {})
    train = _update_dataclass(TrainConfig(), raw.pop("train", {}) or {})
    eval_config = _update_dataclass(EvalConfig(), raw.pop("eval", {}) or {})
    config = _update_dataclass(ModelConfig(), raw)
    config.compression = compression
    config.mtp = mtp
    config.copy_head = copy_head
    config.train = train
    config.eval = eval_config

    if not config.use_dynamic_hourglass_compression:
        config.compression.enabled = False
    if not config.use_curriculum_mtp:
        config.mtp.enabled = False
    if not config.use_attention_islands:
        config.attention_island_layers = []
    if not config.use_anchor_preserved_local_global:
        config.compression.anchors_per_block = 0
    if config.n_kv_heads is None:
        config.n_kv_heads = config.n_heads
    return config


def to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {k: to_dict(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_dict(v) for v in value]
    return value


def config_hash(config: ModelConfig) -> str:
    payload = json.dumps(to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_config_dict(path: str | Path) -> dict[str, Any]:
    return to_dict(load_config(path))
