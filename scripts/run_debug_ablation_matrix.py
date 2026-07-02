#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raam_lm.config import load_config
from raam_lm.train_utils import run_training, resolve_device, seed_all


CONFIGS = [
    "configs/debug/transformer_tiny.yaml",
    "configs/debug/pure_mamba_like_tiny.yaml",
    "configs/debug/raam_tiny.yaml",
    "configs/debug/raam_no_compression.yaml",
    "configs/debug/raam_no_anchors.yaml",
    "configs/debug/raam_no_attention_islands.yaml",
    "configs/debug/raam_no_mtp.yaml",
]


def causal_smoke(model, config, device: torch.device) -> bool:
    model.eval()
    seed_all(config.train.seed + 123)
    with torch.no_grad():
        x = torch.randint(0, config.vocab_size, (2, config.train.seq_len), device=device)
        for cutoff in [1, config.compression.block_size - 1, config.compression.block_size, config.compression.block_size + 1, config.train.seq_len // 2]:
            cutoff = max(1, min(cutoff, config.train.seq_len - 1))
            y = x.clone()
            y[:, cutoff:] = (y[:, cutoff:] + 37) % config.vocab_size
            a = model(x)["logits"][:, :cutoff]
            b = model(y)["logits"][:, :cutoff]
            if (a - b).abs().max().item() > 1e-4:
                return False
    return True


def parse_seeds(raw: str) -> list[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def mean_std(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0}
    return {
        "mean": float(statistics.mean(values)),
        "std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
    }


def aggregate_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_config: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_config.setdefault(str(row["config"]), []).append(row)
    aggregates = []
    for cfg_path, cfg_rows in by_config.items():
        first = cfg_rows[0]
        final_next = [float(row["final_next_token_loss"]) for row in cfg_rows]
        final_train = [float(row["final_train_loss"]) for row in cfg_rows]
        tps = [float(row["tokens_per_sec"]) for row in cfg_rows]
        aggregates.append(
            {
                "config": cfg_path,
                "model": first["model"],
                "seeds": [int(row["seed"]) for row in cfg_rows],
                "runs": len(cfg_rows),
                "params": first["params"],
                "non_embedding_params": first["non_embedding_params"],
                "estimated_flops_per_token": first["estimated_flops_per_token"],
                "final_next_token_loss_mean": mean_std(final_next)["mean"],
                "final_next_token_loss_std": mean_std(final_next)["std"],
                "final_train_loss_mean": mean_std(final_train)["mean"],
                "final_train_loss_std": mean_std(final_train)["std"],
                "tokens_per_sec_mean": mean_std(tps)["mean"],
                "tokens_per_sec_std": mean_std(tps)["std"],
                "all_causal_tests_passed": all(bool(row["causal_test_status"]) for row in cfg_rows),
                "any_nan": any(bool(row["nan_status"]) for row in cfg_rows),
            }
        )
    aggregates.sort(key=lambda row: float(row["final_next_token_loss_mean"]))
    return aggregates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seeds", default="17")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", default="runs/debug_ablation_matrix/summary.json")
    args = parser.parse_args()
    rows = []
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    for seed in seeds:
        for cfg_path in CONFIGS:
            config = load_config(cfg_path)
            config.train.seed = seed
            config.train.output_dir = str(output_path.parent / f"{Path(cfg_path).stem}_seed{seed}")
            print(f"running config={cfg_path} seed={seed} steps={args.steps}", flush=True)
            result = run_training(
                config,
                steps=args.steps,
                device_override=args.device,
                log_path=output_path.parent / f"{Path(cfg_path).stem}_seed{seed}.jsonl",
                print_logs=not args.quiet,
            )
            device = resolve_device(args.device)
            causal_ok = causal_smoke(result["model"], config, device)
            last = result["last_metrics"]
            rows.append(
                {
                    "config": cfg_path,
                    "model": config.model_name,
                    "seed": seed,
                    "steps": args.steps,
                    "tokens_seen": last["tokens_seen"],
                    "params": result["param_count_total"],
                    "non_embedding_params": result["param_count_non_embedding"],
                    "estimated_flops_per_token": result["estimated_flops_per_token"],
                    "estimated_total_flops_seen": last["estimated_total_flops_seen"],
                    "final_smoke_loss": last["train_loss"],
                    "final_train_loss": last["train_loss"],
                    "final_next_token_loss": last["next_token_loss"],
                    "final_recon_loss": last["recon_loss"],
                    "final_mtp_loss_h2": last["mtp_loss_h2"],
                    "final_mtp_loss_h3": last["mtp_loss_h3"],
                    "final_mtp_loss_h4": last["mtp_loss_h4"],
                    "enabled_mtp_horizons": last["enabled_mtp_horizons"],
                    "tokens_per_sec": last["tokens_per_sec"],
                    "peak_memory": last["peak_memory_allocated_mb"],
                    "nan_status": result["nan_status"],
                    "causal_test_status": causal_ok,
                }
            )
    payload = {"rows": rows, "aggregates": aggregate_rows(rows)}
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
