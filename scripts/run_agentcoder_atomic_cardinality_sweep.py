#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CONFIG_BY_MODEL = {
    "raam": "configs/scratch/raam_agentcoder_atomic_copy_gate.yaml",
    "transformer": "configs/scratch/transformer_agentcoder_atomic_copy_gate.yaml",
}


def parse_positive_int_list(raw: str) -> list[int]:
    values: list[int] = []
    seen: set[int] = set()
    for part in raw.replace(",", " ").split():
        try:
            value = int(part)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid positive integer: {part!r}") from exc
        if value < 1:
            raise argparse.ArgumentTypeError("all cardinalities must be positive")
        if value not in seen:
            values.append(value)
            seen.add(value)
    if not values:
        raise argparse.ArgumentTypeError("at least one cardinality is required")
    return values


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def build_row(model: str, requested_train_records: int, requested_eval_cases: int, summary: dict[str, Any]) -> dict[str, Any]:
    last_train_row = summary.get("last_train_row") or {}
    return {
        "model": model,
        "requested_train_records": requested_train_records,
        "requested_eval_cases": requested_eval_cases,
        "train_records": summary.get("train_records"),
        "eval_cases": summary.get("eval_cases"),
        "pass_count": summary.get("pass_count"),
        "case_count": summary.get("case_count"),
        "pass_rate": summary.get("pass_rate"),
        "behavior_accuracy": summary.get("behavior_accuracy"),
        "train_loss": last_train_row.get("train_loss"),
        "val_next_token_loss": last_train_row.get("val_next_token_loss"),
        "tokens_seen": last_train_row.get("tokens_seen"),
        "tokens_per_sec": last_train_row.get("tokens_per_sec"),
        "step_time_ms": last_train_row.get("step_time_ms"),
        "train_tokens": summary.get("train_tokens"),
        "val_tokens": summary.get("val_tokens"),
        "mirror_val": summary.get("mirror_val"),
        "assistant_loss_only": summary.get("assistant_loss_only"),
        "train_loss_tokens": summary.get("train_loss_tokens"),
        "val_loss_tokens": summary.get("val_loss_tokens"),
        "eval_mode": summary.get("eval_mode"),
        "param_count_non_embedding": summary.get("param_count_non_embedding"),
        "estimated_flops_per_token": summary.get("estimated_flops_per_token"),
        "config": summary.get("config"),
        "summary_json": str(Path(str(summary.get("output_dir", "."))) / "summary.json"),
        "atomic_eval": summary.get("atomic_eval"),
    }


def first_failure_by_model(rows: list[dict[str, Any]], threshold: float) -> dict[str, dict[str, Any] | None]:
    failures: dict[str, dict[str, Any] | None] = {}
    models = sorted({str(row["model"]) for row in rows})
    for model in models:
        model_rows = [row for row in rows if row["model"] == model]
        model_rows.sort(key=lambda row: int(row["requested_train_records"]))
        failures[model] = next(
            (row for row in model_rows if float(row.get("pass_rate") or 0.0) < threshold),
            None,
        )
    return failures


def write_aggregate(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def run_gate(
    *,
    args: argparse.Namespace,
    model: str,
    train_records: int,
    eval_cases: int,
    run_dir: Path,
) -> dict[str, Any]:
    config = args.raam_config if model == "raam" else args.transformer_config
    cmd = [
        sys.executable,
        "scripts/run_agentcoder_atomic_copy_gate.py",
        "--config",
        config,
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
        "--train-records",
        str(train_records),
        "--eval-cases",
        str(eval_cases),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-pass-rate",
        str(args.min_pass_rate),
        "--seed",
        str(args.seed),
        "--clean",
    ]
    if args.steps is not None:
        cmd.extend(["--steps", str(args.steps)])
    if args.eval_batches is not None:
        cmd.extend(["--eval-batches", str(args.eval_batches)])
    cmd.append("--mirror-val" if args.mirror_val else "--no-mirror-val")
    if not args.fail_on_gate:
        cmd.append("--no-fail")

    run(cmd)
    summary = json.loads((run_dir / "summary.json").read_text())
    return build_row(model, train_records, eval_cases, summary)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the atomic copy gate across binding-cardinality values."
    )
    parser.add_argument("--output-dir", default="runs/agentcoder_atomic_cardinality_sweep")
    parser.add_argument("--models", nargs="+", choices=sorted(CONFIG_BY_MODEL), default=["raam", "transformer"])
    parser.add_argument("--train-records", type=parse_positive_int_list, default=parse_positive_int_list("1,2,4,8,16,32,64"))
    parser.add_argument("--eval-cases", type=int, default=None, help="Defaults to the current train-record count.")
    parser.add_argument("--raam-config", default=CONFIG_BY_MODEL["raam"])
    parser.add_argument("--transformer-config", default=CONFIG_BY_MODEL["transformer"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--eval-mode", choices=["mirror", "heldout", "ladder"], default="mirror")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mirror-val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--failure-threshold", type=float, default=1.0)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    if args.eval_cases is not None and args.eval_cases < 1:
        raise SystemExit("--eval-cases must be positive")

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.clean:
        raise SystemExit(f"{output_dir} already exists and is not empty; use --clean to replace it")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    payload: dict[str, Any] = {
        "format": "agentcoder-atomic-cardinality-sweep-v1",
        "note": "Mirrored atomic copy binding sweep; this is a diagnostic control, not a benchmark.",
        "models": args.models,
        "train_record_values": args.train_records,
        "eval_cases": args.eval_cases,
        "eval_cases_policy": "fixed" if args.eval_cases is not None else "match_train_records",
        "eval_mode": args.eval_mode,
        "mirror_val": args.mirror_val,
        "failure_threshold": args.failure_threshold,
        "rows": rows,
        "first_failure_below_threshold": {},
    }
    write_aggregate(output_dir, payload)

    for train_records in args.train_records:
        eval_cases = args.eval_cases if args.eval_cases is not None else train_records
        for model in args.models:
            run_dir = output_dir / f"{model}_n{train_records:03d}"
            row = run_gate(args=args, model=model, train_records=train_records, eval_cases=eval_cases, run_dir=run_dir)
            rows.append(row)
            payload["first_failure_below_threshold"] = first_failure_by_model(rows, args.failure_threshold)
            write_aggregate(output_dir, payload)

    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
