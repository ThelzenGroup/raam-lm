#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_agentcoder_atomic_cardinality_sweep import parse_positive_int_list, write_aggregate


ROOT = Path(__file__).resolve().parents[1]
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class ConfigSpec:
    label: str
    path: str


def parse_config_specs(raw_specs: list[str]) -> list[ConfigSpec]:
    specs: list[ConfigSpec] = []
    seen: set[str] = set()
    for raw in raw_specs:
        if "=" not in raw:
            raise argparse.ArgumentTypeError(f"config spec must be label=path, got {raw!r}")
        label, path = raw.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise argparse.ArgumentTypeError(f"config spec must be label=path, got {raw!r}")
        if not LABEL_RE.fullmatch(label):
            raise argparse.ArgumentTypeError(
                f"config label {label!r} must start with an alphanumeric character and contain only letters, digits, '_' or '-'"
            )
        if label in seen:
            raise argparse.ArgumentTypeError(f"duplicate config label: {label}")
        specs.append(ConfigSpec(label=label, path=path))
        seen.add(label)
    if not specs:
        raise argparse.ArgumentTypeError("at least one config spec is required")
    return specs


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_repeatability_row(
    *,
    config: ConfigSpec,
    seed: int,
    train_records: int,
    eval_cases: int,
    run_dir: Path,
    child_summary: dict[str, Any],
) -> dict[str, Any]:
    rows = child_summary.get("rows") or []
    child_row = rows[0] if rows else {}
    return {
        "config_label": config.label,
        "config": config.path,
        "seed": seed,
        "requested_train_records": train_records,
        "requested_eval_cases": eval_cases,
        "train_records": child_row.get("train_records"),
        "eval_cases": child_row.get("eval_cases"),
        "pass_count": child_row.get("pass_count"),
        "case_count": child_row.get("case_count"),
        "pass_rate": child_row.get("pass_rate"),
        "behavior_accuracy": child_row.get("behavior_accuracy"),
        "train_loss": child_row.get("train_loss"),
        "val_next_token_loss": child_row.get("val_next_token_loss"),
        "tokens_seen": child_row.get("tokens_seen"),
        "tokens_per_sec": child_row.get("tokens_per_sec"),
        "step_time_ms": child_row.get("step_time_ms"),
        "train_tokens": child_row.get("train_tokens"),
        "val_tokens": child_row.get("val_tokens"),
        "mirror_val": child_row.get("mirror_val"),
        "eval_mode": child_row.get("eval_mode"),
        "param_count_non_embedding": child_row.get("param_count_non_embedding"),
        "estimated_flops_per_token": child_row.get("estimated_flops_per_token"),
        "run_dir": str(run_dir),
        "summary_json": str(run_dir / "summary.json"),
        "atomic_eval": child_row.get("atomic_eval"),
    }


def _float_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    return float(value) if value is not None else 0.0


def _int_value(row: dict[str, Any], key: str) -> int:
    value = row.get(key)
    return int(value) if value is not None else 0


def summarize_by_config(rows: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    labels = sorted({str(row["config_label"]) for row in rows})
    for label in labels:
        label_rows = [row for row in rows if row["config_label"] == label]
        pass_rates = [_float_value(row, "pass_rate") for row in label_rows]
        pass_counts = [_int_value(row, "pass_count") for row in label_rows]
        case_counts = [_int_value(row, "case_count") for row in label_rows]
        val_losses = [_float_value(row, "val_next_token_loss") for row in label_rows if row.get("val_next_token_loss") is not None]
        token_rates = [_float_value(row, "tokens_per_sec") for row in label_rows if row.get("tokens_per_sec") is not None]
        summary[label] = {
            "runs": len(label_rows),
            "seeds": [int(row["seed"]) for row in sorted(label_rows, key=lambda row: int(row["seed"]))],
            "mean_pass_rate": sum(pass_rates) / len(pass_rates),
            "min_pass_rate": min(pass_rates),
            "max_pass_rate": max(pass_rates),
            "mean_pass_count": sum(pass_counts) / len(pass_counts),
            "min_pass_count": min(pass_counts),
            "max_pass_count": max(pass_counts),
            "total_pass_count": sum(pass_counts),
            "total_case_count": sum(case_counts),
            "all_passed": all(rate >= threshold for rate in pass_rates),
            "mean_val_next_token_loss": (sum(val_losses) / len(val_losses)) if val_losses else None,
            "mean_tokens_per_sec": (sum(token_rates) / len(token_rates)) if token_rates else None,
        }
    return summary


def first_failure_by_config(rows: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, Any] | None]:
    failures: dict[str, dict[str, Any] | None] = {}
    labels = sorted({str(row["config_label"]) for row in rows})
    for label in labels:
        label_rows = sorted(
            [row for row in rows if row["config_label"] == label],
            key=lambda row: int(row["seed"]),
        )
        failures[label] = next((row for row in label_rows if _float_value(row, "pass_rate") < threshold), None)
    return failures


def run_seed_gate(
    *,
    args: argparse.Namespace,
    config: ConfigSpec,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/run_agentcoder_atomic_cardinality_sweep.py",
        "--models",
        "raam",
        "--raam-config",
        config.path,
        "--train-records",
        str(args.train_records),
        "--eval-cases",
        str(args.eval_cases),
        "--output-dir",
        str(run_dir),
        "--device",
        args.device,
        "--vocab-size",
        str(args.vocab_size),
        "--seq-len",
        str(args.seq_len),
        "--val-fraction",
        str(args.val_fraction),
        "--eval-mode",
        args.eval_mode,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-pass-rate",
        str(args.min_pass_rate),
        "--seed",
        str(seed),
        "--failure-threshold",
        str(args.failure_threshold),
        "--clean",
    ]
    if args.steps is not None:
        cmd.extend(["--steps", str(args.steps)])
    if args.eval_batches is not None:
        cmd.extend(["--eval-batches", str(args.eval_batches)])
    cmd.append("--mirror-val" if args.mirror_val else "--no-mirror-val")
    if args.fail_on_gate:
        cmd.append("--fail-on-gate")

    run(cmd)
    child_summary = json.loads((run_dir / "summary.json").read_text())
    return build_repeatability_row(
        config=config,
        seed=seed,
        train_records=args.train_records,
        eval_cases=args.eval_cases,
        run_dir=run_dir,
        child_summary=child_summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run atomic RAAM anchor configs across seeds for repeatability."
    )
    parser.add_argument("--output-dir", default="runs/agentcoder_atomic_anchor_seed_sweep")
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="One or more label=path pairs, for example learned=configs/...yaml hybrid1=configs/...yaml.",
    )
    parser.add_argument("--seeds", type=parse_positive_int_list, default=parse_positive_int_list("17,29,41"))
    parser.add_argument("--train-records", type=int, default=64)
    parser.add_argument("--eval-cases", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--eval-mode", choices=["mirror", "heldout", "ladder"], default="mirror")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--mirror-val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--failure-threshold", type=float, default=1.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    try:
        configs = parse_config_specs(args.configs)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc)) from exc

    if args.train_records < 1:
        raise SystemExit("--train-records must be positive")
    if args.eval_cases < 1:
        raise SystemExit("--eval-cases must be positive")

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.clean:
        raise SystemExit(f"{output_dir} already exists and is not empty; use --clean to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "format": "agentcoder-atomic-anchor-seed-sweep-v1",
        "note": "Mirrored atomic copy repeatability sweep across RAAM anchor configs; this is a diagnostic control, not a benchmark.",
        "configs": [spec.__dict__ for spec in configs],
        "seeds": args.seeds,
        "train_records": args.train_records,
        "eval_cases": args.eval_cases,
        "steps": args.steps,
        "eval_mode": args.eval_mode,
        "mirror_val": args.mirror_val,
        "failure_threshold": args.failure_threshold,
        "rows": rows,
        "by_config": {},
        "first_failure_below_threshold": {},
    }
    write_aggregate(output_dir, payload)

    for config in configs:
        for seed in args.seeds:
            run_dir = output_dir / f"{config.label}_seed{seed:03d}"
            row = run_seed_gate(args=args, config=config, seed=seed, run_dir=run_dir)
            rows.append(row)
            payload["by_config"] = summarize_by_config(rows, args.failure_threshold)
            payload["first_failure_below_threshold"] = first_failure_by_config(rows, args.failure_threshold)
            write_aggregate(output_dir, payload)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
