#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raam_lm.agent_data import PackedTokenDataset
from raam_lm.config import config_hash, load_config, to_dict
from raam_lm.flops import count_non_embedding_parameters, count_parameters, estimate_flops_per_token
from raam_lm.registry import build_model
from raam_lm.tokenization import AgentCoderTokenizer
from raam_lm.train_utils import (
    build_optimizer,
    grad_norm,
    lr_for_step,
    maybe_autocast,
    resolve_device,
    resolve_dtype,
    seed_all,
)


def sync_run_dir(output_dir: Path, sync_root: Path | None) -> None:
    if sync_root is None:
        return
    output_resolved = output_dir.resolve()
    target = sync_root.expanduser().resolve() / output_dir.name
    if target == output_resolved or output_resolved in target.parents:
        raise ValueError(f"sync target {target} must not live inside output dir {output_resolved}")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(output_dir, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.tmp"))


def _float(value: Any) -> float:
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


def save_checkpoint(path: Path, model, optimizer, step: int, config, tokenizer_path: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": step,
            "config": to_dict(config),
            "config_hash": config_hash(config),
            "tokenizer_path": tokenizer_path,
        },
        path,
    )


def load_resume_checkpoint(path: str, model, optimizer, device) -> tuple[int, bool]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer_loaded = "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None
    if optimizer_loaded:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    start_step = int(checkpoint.get("step", -1)) + 1
    return max(0, start_step), optimizer_loaded


def evaluate(model, dataset, config, device, dtype, global_step: int) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    batches = max(1, config.eval.eval_batches)
    with torch.no_grad():
        for _ in range(batches):
            batch = dataset.next_batch(config.train.batch_size, config.train.seq_len, device)
            with maybe_autocast(device, dtype):
                out = model(batch, labels=batch, global_step=global_step)
            losses.append(_float(out["next_token_loss"]))
    model.train()
    return {"val_next_token_loss": sum(losses) / len(losses)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scratch train RAAM-AgentCoder from packed local data.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-bin", required=True)
    parser.add_argument("--val-bin", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--grad-accumulation-steps", type=int, default=None)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument(
        "--sync-dir",
        default=None,
        help="Optional mounted or external-sync directory that receives a copy of the run after saves.",
    )
    parser.add_argument("--sync-every", type=int, default=0, help="Also sync every N steps; 0 disables step sync.")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = AgentCoderTokenizer.load(args.tokenizer)
    config.vocab_size = tokenizer.vocab_size
    if args.steps is not None:
        config.train.steps = args.steps
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.seq_len is not None:
        config.train.seq_len = args.seq_len
    if args.grad_accumulation_steps is not None:
        config.train.gradient_accumulation_steps = args.grad_accumulation_steps
    if args.eval_batches is not None:
        config.eval.eval_batches = args.eval_batches
    if args.seed is not None:
        config.train.seed = args.seed
    if args.device is not None:
        config.train.device = args.device
    if args.output_dir is not None:
        config.train.output_dir = args.output_dir
    if args.eval_every is not None:
        config.train.eval_every = args.eval_every
    if args.save_every is not None:
        config.train.save_every = args.save_every

    output_dir = Path(config.train.output_dir)
    ckpt_dir = output_dir / "checkpoints"
    sync_root = Path(args.sync_dir) if args.sync_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, output_dir / "config.yaml")
    shutil.copyfile(args.tokenizer, output_dir / "tokenizer.json")
    sync_run_dir(output_dir, sync_root)

    seed_all(config.train.seed)
    device = resolve_device(config.train.device)
    dtype = resolve_dtype(config.train.dtype, device)
    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    start_step = 0
    resume_optimizer_loaded = False
    if args.resume:
        start_step, resume_optimizer_loaded = load_resume_checkpoint(args.resume, model, optimizer, device)

    train_data = PackedTokenDataset(args.train_bin, seed=config.train.seed)
    val_data = PackedTokenDataset(args.val_bin, seed=config.train.seed + 1)
    est_flops = estimate_flops_per_token(config)
    manifest = {
        "model_name": config.model_name,
        "config_path": args.config,
        "config_hash": config_hash(config),
        "tokenizer_path": args.tokenizer,
        "train_bin": args.train_bin,
        "val_bin": args.val_bin,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "param_count_total": count_parameters(model),
        "param_count_non_embedding": count_non_embedding_parameters(model),
        "estimated_flops_per_token": est_flops,
        "resume_from": args.resume,
        "resume_start_step": start_step,
        "resume_optimizer_loaded": resume_optimizer_loaded,
        "resume_mode": "optimizer" if resume_optimizer_loaded else ("model_only" if args.resume else "none"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sync_run_dir(output_dir, sync_root)
    log_path = output_dir / "train_log.jsonl"
    mode = "a" if args.resume else "w"
    grad_accum = max(1, int(getattr(config.train, "gradient_accumulation_steps", 1)))
    tokens_per_step = config.train.batch_size * config.train.seq_len * grad_accum
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    interrupted = False
    try:
        with log_path.open(mode) as log_fh:
            for step in range(start_step, config.train.steps):
                lr = lr_for_step(config.train.lr, step, config.train.warmup_steps)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                start = time.perf_counter()
                last_out = None
                for _ in range(grad_accum):
                    batch = train_data.next_batch(config.train.batch_size, config.train.seq_len, device)
                    with maybe_autocast(device, dtype):
                        out = model(batch, labels=batch, global_step=step)
                        loss = out["loss"] / grad_accum
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at step {step}: {loss.detach().cpu().item()}")
                    loss.backward()
                    last_out = out
                pre_clip_norm = grad_norm(model.parameters())
                if config.train.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
                optimizer.step()
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                assert last_out is not None
                aux = last_out.get("aux", {})
                mtp_losses = last_out.get("mtp_loss_by_horizon", {})
                metrics: dict[str, Any] = {
                    "global_step": step,
                    "tokens_seen": tokens_per_step * (step + 1),
                    "train_loss": _float(last_out["loss"]),
                    "next_token_loss": _float(last_out["next_token_loss"]),
                    "recon_loss": _float(last_out["recon_loss"]),
                    "mtp_loss_h2": _float(mtp_losses.get(2, torch.zeros(()))),
                    "mtp_loss_h3": _float(mtp_losses.get(3, torch.zeros(()))),
                    "mtp_loss_h4": _float(mtp_losses.get(4, torch.zeros(()))),
                    "learning_rate": lr,
                    "grad_norm": pre_clip_norm,
                    "tokens_per_sec": tokens_per_step / max(elapsed, 1e-9),
                    "step_time_ms": elapsed * 1000.0,
                    "peak_memory_allocated_mb": (
                        torch.cuda.max_memory_allocated(device) / (1024 * 1024) if device.type == "cuda" else 0.0
                    ),
                    "estimated_flops_per_token": est_flops,
                    "estimated_total_flops_seen": est_flops * tokens_per_step * (step + 1),
                    "compression_ratio": aux.get("compression_ratio", 1.0),
                    "mixer_backend": aux.get("mixer_backend", "unknown"),
                }
                if step % max(config.train.eval_every, 1) == 0 or step == config.train.steps - 1:
                    metrics.update(evaluate(model, val_data, config, device, dtype, step))
                log_fh.write(json.dumps(metrics, sort_keys=True) + "\n")
                log_fh.flush()
                print(json.dumps(metrics, sort_keys=True))
                if config.train.save_every and (
                    step % config.train.save_every == 0 or step == config.train.steps - 1
                ):
                    save_checkpoint(ckpt_dir / f"step_{step:06d}.pt", model, optimizer, step, config, args.tokenizer)
                    sync_run_dir(output_dir, sync_root)
                elif args.sync_every and step % args.sync_every == 0:
                    sync_run_dir(output_dir, sync_root)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        final_step = max(start_step - 1, min(config.train.steps - 1, step if "step" in locals() else start_step - 1))
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, final_step, config, args.tokenizer)
        sync_run_dir(output_dir, sync_root)
        print(f"saved_checkpoint path={ckpt_dir / 'last.pt'} step={final_step} interrupted={interrupted}")


if __name__ == "__main__":
    main()
