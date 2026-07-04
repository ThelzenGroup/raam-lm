#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raam_lm.agent_data import PackedTokenDataset
from raam_lm.config import config_hash, load_config, resolve_copy_head_token_ids, to_dict
from raam_lm.flops import count_non_embedding_parameters, count_parameters, estimate_flops_per_token
from raam_lm.mlops import (
    add_artifact_reference,
    append_metrics,
    finish_run as finish_mlops_run,
    stable_run_id,
    start_run as start_mlops_run,
)
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


def save_checkpoint(
    path: Path,
    model,
    optimizer,
    step: int,
    config,
    tokenizer_path: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "step": step,
        "config": to_dict(config),
        "config_hash": config_hash(config),
        "tokenizer_path": tokenizer_path,
    }
    if metadata:
        payload.update(metadata)
    torch.save(payload, path)


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
    mask_fractions: list[float] = []
    batches = max(1, config.eval.eval_batches)
    with torch.no_grad():
        for _ in range(batches):
            batch, loss_mask = dataset.next_batch_with_loss_mask(config.train.batch_size, config.train.seq_len, device)
            with maybe_autocast(device, dtype):
                out = model(batch, labels=batch, global_step=global_step, loss_mask=loss_mask)
            losses.append(_float(out["next_token_loss"]))
            if loss_mask is not None:
                mask_fractions.append(float(loss_mask.mean().detach().cpu()))
    model.train()
    metrics = {"val_next_token_loss": sum(losses) / len(losses)}
    if mask_fractions:
        metrics["val_loss_mask_fraction"] = sum(mask_fractions) / len(mask_fractions)
    return metrics


def inferred_loss_mask_path(token_path: str) -> Path | None:
    path = Path(token_path)
    candidate = path.with_name(f"{path.stem}_loss_mask{path.suffix}")
    return candidate if candidate.exists() else None


def best_validation_from_log(log_path: Path) -> tuple[float | None, int | None]:
    if not log_path.exists():
        return None, None
    best_loss: float | None = None
    best_step: int | None = None
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "val_next_token_loss" not in row:
            continue
        loss = float(row["val_next_token_loss"])
        if best_loss is None or loss < best_loss:
            best_loss = loss
            best_step = int(row.get("global_step", row.get("step", -1)))
    return best_loss, best_step


def last_validation_lr_decay_scale_from_log(log_path: Path) -> float:
    if not log_path.exists():
        return 1.0
    scale = 1.0
    for line in log_path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "validation_lr_decay_scale" in row:
            scale = float(row["validation_lr_decay_scale"])
        if "validation_lr_decay_new_scale" in row:
            scale = float(row["validation_lr_decay_new_scale"])
    return scale


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
    parser.add_argument(
        "--reset-resume-step",
        action="store_true",
        help="Load --resume weights but start this run at step 0. Use for model-only SFT continuation checkpoints.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-every", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument(
        "--save-best",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save checkpoints/best.pt whenever validation loss improves.",
    )
    parser.add_argument(
        "--early-stop-patience-evals",
        type=int,
        default=None,
        help="Stop after this many validation checks without enough improvement; 0 disables.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=None,
        help="Minimum validation-loss decrease required to reset early-stop patience.",
    )
    parser.add_argument(
        "--early-stop-min-step",
        type=int,
        default=None,
        help="Do not early-stop before this global step.",
    )
    parser.add_argument(
        "--restore-best-on-finish",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Before writing last.pt, restore the best validation checkpoint state.",
    )
    parser.add_argument(
        "--validation-lr-decay-patience-evals",
        type=int,
        default=None,
        help="Reduce LR after this many validation checks without enough improvement; 0 disables.",
    )
    parser.add_argument(
        "--validation-lr-decay-factor",
        type=float,
        default=None,
        help="Multiplicative LR scale applied when validation LR backoff triggers.",
    )
    parser.add_argument(
        "--validation-lr-decay-min-scale",
        type=float,
        default=None,
        help="Smallest multiplicative LR scale allowed by validation LR backoff.",
    )
    parser.add_argument(
        "--validation-lr-decay-min-step",
        type=int,
        default=None,
        help="Do not apply validation LR backoff before this global step.",
    )
    parser.add_argument(
        "--sync-dir",
        default=None,
        help="Optional mounted or external-sync directory that receives a copy of the run after saves.",
    )
    parser.add_argument("--sync-every", type=int, default=0, help="Also sync every N steps; 0 disables step sync.")
    parser.add_argument(
        "--mlops-project-path",
        default=os.environ.get("RAAM_MLOPS_PROJECT_PATH"),
        help="Optional project path whose .mlops/experiments tracker receives live run metrics.",
    )
    parser.add_argument("--mlops-run-id", default=None, help="Optional stable .mlops run id for resume/continuation.")
    args = parser.parse_args()

    config = load_config(args.config)
    tokenizer = AgentCoderTokenizer.load(args.tokenizer)
    config.vocab_size = tokenizer.vocab_size
    resolve_copy_head_token_ids(config, tokenizer)
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
    if args.save_best is not None:
        config.train.save_best = args.save_best
    if args.early_stop_patience_evals is not None:
        config.train.early_stop_patience_evals = args.early_stop_patience_evals
    if args.early_stop_min_delta is not None:
        config.train.early_stop_min_delta = args.early_stop_min_delta
    if args.early_stop_min_step is not None:
        config.train.early_stop_min_step = args.early_stop_min_step
    if args.restore_best_on_finish is not None:
        config.train.restore_best_on_finish = args.restore_best_on_finish
    if args.validation_lr_decay_patience_evals is not None:
        config.train.validation_lr_decay_patience_evals = args.validation_lr_decay_patience_evals
    if args.validation_lr_decay_factor is not None:
        config.train.validation_lr_decay_factor = args.validation_lr_decay_factor
    if args.validation_lr_decay_min_scale is not None:
        config.train.validation_lr_decay_min_scale = args.validation_lr_decay_min_scale
    if args.validation_lr_decay_min_step is not None:
        config.train.validation_lr_decay_min_step = args.validation_lr_decay_min_step

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
        if args.reset_resume_step:
            start_step = 0

    train_loss_mask_path = inferred_loss_mask_path(args.train_bin)
    val_loss_mask_path = inferred_loss_mask_path(args.val_bin)
    train_data = PackedTokenDataset(args.train_bin, seed=config.train.seed, loss_mask_path=train_loss_mask_path)
    val_data = PackedTokenDataset(args.val_bin, seed=config.train.seed + 1, loss_mask_path=val_loss_mask_path)
    est_flops = estimate_flops_per_token(config)
    manifest = {
        "model_name": config.model_name,
        "config_path": args.config,
        "config_hash": config_hash(config),
        "tokenizer_path": args.tokenizer,
        "train_bin": args.train_bin,
        "val_bin": args.val_bin,
        "train_loss_mask_bin": str(train_loss_mask_path) if train_loss_mask_path else None,
        "val_loss_mask_bin": str(val_loss_mask_path) if val_loss_mask_path else None,
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "param_count_total": count_parameters(model),
        "param_count_non_embedding": count_non_embedding_parameters(model),
        "estimated_flops_per_token": est_flops,
        "resume_from": args.resume,
        "resume_start_step": start_step,
        "resume_optimizer_loaded": resume_optimizer_loaded,
        "reset_resume_step": args.reset_resume_step,
        "resume_mode": "optimizer" if resume_optimizer_loaded else ("model_only" if args.resume else "none"),
        "save_best": config.train.save_best,
        "early_stop_patience_evals": config.train.early_stop_patience_evals,
        "early_stop_min_delta": config.train.early_stop_min_delta,
        "early_stop_min_step": config.train.early_stop_min_step,
        "restore_best_on_finish": config.train.restore_best_on_finish,
        "validation_lr_decay_patience_evals": config.train.validation_lr_decay_patience_evals,
        "validation_lr_decay_factor": config.train.validation_lr_decay_factor,
        "validation_lr_decay_min_scale": config.train.validation_lr_decay_min_scale,
        "validation_lr_decay_min_step": config.train.validation_lr_decay_min_step,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sync_run_dir(output_dir, sync_root)
    mlops_project_path = Path(args.mlops_project_path).expanduser().resolve() if args.mlops_project_path else None
    mlops_run_id = args.mlops_run_id or stable_run_id(output_dir, prefix="live")
    if mlops_project_path is not None:
        start_mlops_run(
            mlops_project_path,
            mlops_run_id,
            params={
                **manifest,
                "output_dir": str(output_dir),
                "run_kind": "live_training",
                "run_source": "scripts/train.py",
            },
        )
        for artifact_path in (Path(args.config), Path(args.tokenizer), output_dir / "manifest.json"):
            add_artifact_reference(mlops_project_path, mlops_run_id, artifact_path, kind="training_setup", copy=False)
    log_path = output_dir / "train_log.jsonl"
    mode = "a" if args.resume else "w"
    best_checkpoint = ckpt_dir / "best.pt"
    best_val_loss, best_val_step = (
        best_validation_from_log(log_path) if args.resume and config.train.save_best else (None, None)
    )
    no_improve_eval_count = 0
    validation_lr_no_improve_count = 0
    validation_lr_decay_count = 0
    validation_lr_decay_scale = (
        last_validation_lr_decay_scale_from_log(log_path)
        if args.resume and config.train.validation_lr_decay_patience_evals > 0
        else 1.0
    )
    stopped_early = False
    early_stop_step: int | None = None
    grad_accum = max(1, int(getattr(config.train, "gradient_accumulation_steps", 1)))
    tokens_per_step = config.train.batch_size * config.train.seq_len * grad_accum
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    interrupted = False
    failed = False
    try:
        with log_path.open(mode) as log_fh:
            for step in range(start_step, config.train.steps):
                scheduled_lr = lr_for_step(
                    config.train.lr,
                    step,
                    config.train.warmup_steps,
                    total_steps=config.train.steps,
                    cosine_decay=config.train.cosine_decay,
                    min_lr=config.train.min_lr,
                    decay_start_step=config.train.lr_decay_start_step,
                    decay_end_step=config.train.lr_decay_end_step,
                )
                lr = scheduled_lr
                if config.train.validation_lr_decay_patience_evals > 0:
                    lr = max(float(config.train.min_lr), scheduled_lr * validation_lr_decay_scale)
                for group in optimizer.param_groups:
                    group["lr"] = lr
                optimizer.zero_grad(set_to_none=True)
                start = time.perf_counter()
                last_out = None
                last_loss_mask = None
                for _ in range(grad_accum):
                    batch, loss_mask = train_data.next_batch_with_loss_mask(
                        config.train.batch_size,
                        config.train.seq_len,
                        device,
                    )
                    with maybe_autocast(device, dtype):
                        out = model(batch, labels=batch, global_step=step, loss_mask=loss_mask)
                        loss = out["loss"] / grad_accum
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at step {step}: {loss.detach().cpu().item()}")
                    loss.backward()
                    last_out = out
                    last_loss_mask = loss_mask
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
                    "scheduled_learning_rate": scheduled_lr,
                    "validation_lr_decay_scale": validation_lr_decay_scale,
                    "validation_lr_decay_count": validation_lr_decay_count,
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
                    "copy_head_enabled": bool(aux.get("copy_head_enabled", False)),
                }
                if last_loss_mask is not None:
                    metrics["loss_mask_fraction"] = float(last_loss_mask.mean().detach().cpu())
                did_eval = step % max(config.train.eval_every, 1) == 0 or step == config.train.steps - 1
                if did_eval:
                    metrics.update(evaluate(model, val_data, config, device, dtype, step))
                    current_val = float(metrics["val_next_token_loss"])
                    min_delta = max(0.0, float(config.train.early_stop_min_delta))
                    previous_best = best_val_loss
                    meaningful_improvement = previous_best is None or current_val < (previous_best - min_delta)
                    new_best = previous_best is None or current_val <= previous_best
                    if config.train.save_best and new_best:
                        best_val_loss = current_val
                        best_val_step = step
                        metrics["best_val_next_token_loss"] = best_val_loss
                        metrics["best_val_step"] = best_val_step
                        save_checkpoint(
                            best_checkpoint,
                            model,
                            optimizer,
                            step,
                            config,
                            args.tokenizer,
                            metadata={
                                "checkpoint_kind": "best",
                                "best_metric": "val_next_token_loss",
                                "best_metric_value": best_val_loss,
                                "best_metric_step": best_val_step,
                            },
                        )
                        if mlops_project_path is not None:
                            add_artifact_reference(
                                mlops_project_path,
                                mlops_run_id,
                                best_checkpoint,
                                kind="best_checkpoint_reference",
                                copy=False,
                            )
                        sync_run_dir(output_dir, sync_root)
                    if meaningful_improvement:
                        no_improve_eval_count = 0
                        validation_lr_no_improve_count = 0
                    elif previous_best is not None:
                        no_improve_eval_count += 1
                        validation_lr_no_improve_count += 1
                    if best_val_loss is not None:
                        metrics["best_val_next_token_loss"] = best_val_loss
                        metrics["best_val_step"] = best_val_step
                        metrics["no_improve_eval_count"] = no_improve_eval_count
                        metrics["validation_lr_no_improve_count"] = validation_lr_no_improve_count
                    lr_decay_patience = int(config.train.validation_lr_decay_patience_evals)
                    lr_decay_can_apply = step >= int(config.train.validation_lr_decay_min_step)
                    lr_decay_factor = float(config.train.validation_lr_decay_factor)
                    lr_decay_min_scale = float(config.train.validation_lr_decay_min_scale)
                    if (
                        lr_decay_patience > 0
                        and lr_decay_can_apply
                        and validation_lr_no_improve_count >= lr_decay_patience
                        and 0.0 < lr_decay_factor < 1.0
                        and validation_lr_decay_scale > lr_decay_min_scale
                    ):
                        old_scale = validation_lr_decay_scale
                        validation_lr_decay_scale = max(lr_decay_min_scale, validation_lr_decay_scale * lr_decay_factor)
                        validation_lr_no_improve_count = 0
                        validation_lr_decay_count += 1
                        metrics["validation_lr_decay_applied"] = True
                        metrics["validation_lr_decay_old_scale"] = old_scale
                        metrics["validation_lr_decay_new_scale"] = validation_lr_decay_scale
                        metrics["validation_lr_decay_count"] = validation_lr_decay_count
                        metrics["validation_lr_no_improve_count"] = validation_lr_no_improve_count
                    patience = int(config.train.early_stop_patience_evals)
                    can_stop = step >= int(config.train.early_stop_min_step)
                    if patience > 0 and can_stop and no_improve_eval_count >= patience:
                        stopped_early = True
                        early_stop_step = step
                        metrics["stopped_early"] = True
                        metrics["early_stop_reason"] = "validation_loss_no_improvement"
                log_fh.write(json.dumps(metrics, sort_keys=True) + "\n")
                log_fh.flush()
                if mlops_project_path is not None:
                    append_metrics(mlops_project_path, mlops_run_id, metrics, step=step)
                print(json.dumps(metrics, sort_keys=True))
                if stopped_early:
                    break
                if config.train.save_every and (
                    step % config.train.save_every == 0 or step == config.train.steps - 1
                ):
                    step_checkpoint = ckpt_dir / f"step_{step:06d}.pt"
                    save_checkpoint(step_checkpoint, model, optimizer, step, config, args.tokenizer)
                    if mlops_project_path is not None:
                        add_artifact_reference(
                            mlops_project_path,
                            mlops_run_id,
                            step_checkpoint,
                            kind="checkpoint_reference",
                            copy=False,
                        )
                    sync_run_dir(output_dir, sync_root)
                elif args.sync_every and step % args.sync_every == 0:
                    sync_run_dir(output_dir, sync_root)
    except KeyboardInterrupt:
        interrupted = True
    except Exception:
        failed = True
        raise
    finally:
        final_step = max(start_step - 1, min(config.train.steps - 1, step if "step" in locals() else start_step - 1))
        if best_val_loss is not None:
            manifest.update(
                {
                    "best_checkpoint": str(best_checkpoint) if best_checkpoint.exists() else None,
                    "best_val_loss": best_val_loss,
                    "best_val_step": best_val_step,
                }
            )
        restored_best_on_finish = False
        if config.train.restore_best_on_finish and best_checkpoint.exists():
            checkpoint = torch.load(best_checkpoint, map_location=device)
            model.load_state_dict(checkpoint["model_state"])
            if "optimizer_state" in checkpoint and checkpoint["optimizer_state"] is not None:
                optimizer.load_state_dict(checkpoint["optimizer_state"])
            final_step = int(checkpoint.get("step", best_val_step if best_val_step is not None else final_step))
            restored_best_on_finish = True
        manifest.update(
            {
                "final_checkpoint_step": final_step,
                "stopped_early": stopped_early,
                "early_stop_step": early_stop_step,
                "restored_best_on_finish": restored_best_on_finish,
                "validation_lr_decay_scale": validation_lr_decay_scale,
                "validation_lr_decay_count": validation_lr_decay_count,
            }
        )
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        last_checkpoint = ckpt_dir / "last.pt"
        save_checkpoint(
            last_checkpoint,
            model,
            optimizer,
            final_step,
            config,
            args.tokenizer,
            metadata={
                "checkpoint_kind": "last",
                "restored_from_best": restored_best_on_finish,
                "stopped_early": stopped_early,
                "early_stop_step": early_stop_step,
            },
        )
        if mlops_project_path is not None:
            add_artifact_reference(mlops_project_path, mlops_run_id, log_path, kind="train_log", copy=False)
            add_artifact_reference(mlops_project_path, mlops_run_id, last_checkpoint, kind="checkpoint_reference", copy=False)
            finish_mlops_run(
                mlops_project_path,
                mlops_run_id,
                status="failed" if failed or interrupted else "success",
            )
        sync_run_dir(output_dir, sync_root)
        print(f"saved_checkpoint path={ckpt_dir / 'last.pt'} step={final_step} interrupted={interrupted}")


if __name__ == "__main__":
    main()
