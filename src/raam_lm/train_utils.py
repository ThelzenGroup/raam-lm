"""Shared training helpers for smoke runs and ablation scripts."""

from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import torch

from .config import ModelConfig
from .data import GeneratedTinyDataset, dataset_identity
from .flops import count_non_embedding_parameters, count_parameters, estimate_flops_per_token
from .registry import build_model


def seed_all(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype: str, device: torch.device) -> torch.dtype:
    if dtype == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32
    if dtype in {"float16", "fp16"}:
        return torch.float16
    if dtype in {"bfloat16", "bf16"}:
        return torch.bfloat16
    return torch.float32


def maybe_autocast(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def lr_for_step(
    base_lr: float,
    step: int,
    warmup_steps: int,
    *,
    total_steps: int | None = None,
    cosine_decay: bool = False,
    min_lr: float = 0.0,
    decay_start_step: int = -1,
    decay_end_step: int = -1,
) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * min(1.0, float(step + 1) / warmup_steps)
    if not cosine_decay or total_steps is None:
        return base_lr

    start = warmup_steps if decay_start_step < 0 else max(warmup_steps, decay_start_step)
    if step < start:
        return base_lr
    end = decay_end_step if decay_end_step > start else total_steps
    span = max(1, end - start - 1)
    progress = min(1.0, max(0.0, float(step - start) / span))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _float(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def grad_norm(parameters) -> float:
    total = 0.0
    for param in parameters:
        if param.grad is None:
            continue
        total += float(param.grad.detach().pow(2).sum().cpu())
    return math.sqrt(total)


def build_optimizer(model: torch.nn.Module, config: ModelConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        betas=config.train.betas,
        weight_decay=config.train.weight_decay,
    )


def run_training(
    config: ModelConfig,
    steps: int | None = None,
    device_override: str | None = None,
    log_path: str | Path | None = None,
    print_logs: bool = True,
) -> dict[str, Any]:
    seed_all(config.train.seed)
    if steps is not None:
        config.train.steps = steps
    if device_override is not None:
        config.train.device = device_override
    device = resolve_device(config.train.device)
    dtype = resolve_dtype(config.train.dtype, device)
    model = build_model(config).to(device)
    dataset = GeneratedTinyDataset(config.vocab_size, seed=config.train.seed)
    optimizer = build_optimizer(model, config)
    output_dir = Path(config.train.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if log_path is None:
        log_path = output_dir / "train_log.jsonl"
    log_path = Path(log_path)
    if log_path.exists():
        log_path.unlink()
    est_flops = estimate_flops_per_token(config)
    last_metrics: dict[str, Any] = {}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    for step in range(config.train.steps):
        lr = lr_for_step(
            config.train.lr,
            step,
            config.train.warmup_steps,
            total_steps=config.train.steps,
            cosine_decay=config.train.cosine_decay,
            min_lr=config.train.min_lr,
            decay_start_step=config.train.lr_decay_start_step,
            decay_end_step=config.train.lr_decay_end_step,
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        batch = dataset.next_batch(config.train.batch_size, config.train.seq_len, device)
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with maybe_autocast(device, dtype):
            out = model(batch, labels=batch, global_step=step)
            loss = out["loss"]
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss.detach().cpu().item()}")
        loss.backward()
        pre_clip_norm = grad_norm(model.parameters())
        if config.train.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
        optimizer.step()
        elapsed = time.perf_counter() - start
        tokens = config.train.batch_size * config.train.seq_len
        aux = out.get("aux", {})
        mtp_losses = out.get("mtp_loss_by_horizon", {})
        mtp_weights = out.get("mtp_weights", {})
        metrics = {
            "global_step": step,
            "tokens_seen": tokens * (step + 1),
            "train_loss": _float(loss),
            "next_token_loss": _float(out["next_token_loss"]),
            "recon_loss": _float(out["recon_loss"]),
            "mtp_loss_h2": _float(mtp_losses.get(2, torch.zeros(()))),
            "mtp_loss_h3": _float(mtp_losses.get(3, torch.zeros(()))),
            "mtp_loss_h4": _float(mtp_losses.get(4, torch.zeros(()))),
            "learning_rate": lr,
            "grad_norm": pre_clip_norm,
            "tokens_per_sec": tokens / max(elapsed, 1e-9),
            "step_time_ms": elapsed * 1000.0,
            "peak_memory_allocated_mb": (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
            ),
            "estimated_flops_per_token": est_flops,
            "estimated_total_flops_seen": est_flops * tokens * (step + 1),
            "enabled_mtp_horizons": sorted(int(k) for k in mtp_weights.keys()),
            "compression_ratio": aux.get("compression_ratio", 1.0),
            "mean_anchor_score": aux.get("mean_anchor_score", 0.0),
            "anchor_token_fraction": aux.get("anchor_token_fraction", 0.0),
            "mixer_backend": aux.get("mixer_backend", "unknown"),
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "tokenizer_id": dataset_identity(dataset),
        }
        with log_path.open("a") as fh:
            fh.write(json.dumps(_jsonable(metrics), sort_keys=True) + "\n")
        last_metrics = metrics
        if print_logs and (step % max(config.train.log_every, 1) == 0 or step == config.train.steps - 1):
            print(json.dumps(_jsonable(metrics), sort_keys=True))

    return {
        "model": model,
        "last_metrics": _jsonable(last_metrics),
        "log_path": str(log_path),
        "param_count_total": count_parameters(model),
        "param_count_non_embedding": count_non_embedding_parameters(model),
        "estimated_flops_per_token": est_flops,
        "nan_status": False,
    }
