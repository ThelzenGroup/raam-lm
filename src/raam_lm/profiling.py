"""Profiling manifest generation."""

from __future__ import annotations

import json
from pathlib import Path
import statistics
import subprocess
import time
from typing import Any

import torch

from .config import ModelConfig, config_hash
from .data import GeneratedTinyDataset, dataset_identity
from .flops import count_non_embedding_parameters, count_parameters, estimate_flops_per_token
from .registry import build_model
from .train_utils import build_optimizer, maybe_autocast, resolve_device, resolve_dtype, seed_all


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return ordered[idx]


def profile_training_step(
    config: ModelConfig,
    config_path: str,
    device_override: str | None = None,
    steps: int = 5,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    seed_all(config.train.seed)
    if device_override is not None:
        config.train.device = device_override
    device = resolve_device(config.train.device)
    dtype = resolve_dtype(config.train.dtype, device)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    dataset = GeneratedTinyDataset(config.vocab_size, seed=config.train.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times: list[float] = []
    model.train()
    for step in range(max(1, steps)):
        batch = dataset.next_batch(config.train.batch_size, config.train.seq_len, device)
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with maybe_autocast(device, dtype):
            out = model(batch, labels=batch, global_step=step)
            loss = out["loss"]
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - start)

    tokens = config.train.batch_size * config.train.seq_len
    mean_time = statistics.mean(times)
    manifest = {
        "model_name": config.model_name,
        "config_path": config_path,
        "config_hash": config_hash(config),
        "param_count_total": count_parameters(model),
        "param_count_non_embedding": count_non_embedding_parameters(model),
        "estimated_flops_per_token": estimate_flops_per_token(config),
        "batch_size": config.train.batch_size,
        "seq_len": config.train.seq_len,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "tokens_per_sec": tokens / max(mean_time, 1e-9),
        "step_time_ms_mean": mean_time * 1000.0,
        "step_time_ms_p95": p95(times) * 1000.0,
        "peak_memory_allocated_mb": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
        ),
        "peak_memory_reserved_mb": (
            torch.cuda.max_memory_reserved(device) / (1024 * 1024) if device.type == "cuda" else 0.0
        ),
        "git_sha": git_sha(),
        "tokenizer_id": dataset_identity(dataset),
    }
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest

