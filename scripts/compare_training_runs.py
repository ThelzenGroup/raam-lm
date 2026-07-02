#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def read_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def summarize_run(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    agentic_eval = read_json(run_dir / "agentic_eval.json")
    logs = read_log(run_dir / "train_log.jsonl")
    first = logs[0] if logs else {}
    last = logs[-1] if logs else {}
    val_rows = [row for row in logs if "val_next_token_loss" in row]
    first_val = val_rows[0].get("val_next_token_loss") if val_rows else None
    last_val = val_rows[-1].get("val_next_token_loss") if val_rows else None
    return {
        "run_dir": str(run_dir),
        "model_name": manifest.get("model_name") or run_dir.name,
        "config_path": manifest.get("config_path"),
        "config_hash": manifest.get("config_hash"),
        "param_count_total": manifest.get("param_count_total"),
        "param_count_non_embedding": manifest.get("param_count_non_embedding"),
        "estimated_flops_per_token": manifest.get("estimated_flops_per_token"),
        "steps_logged": len(logs),
        "first_step": first.get("global_step"),
        "last_step": last.get("global_step"),
        "tokens_seen": last.get("tokens_seen"),
        "first_train_loss": first.get("train_loss"),
        "last_train_loss": last.get("train_loss"),
        "first_val_next_token_loss": first_val,
        "last_val_next_token_loss": last_val,
        "val_loss_delta": (last_val - first_val) if first_val is not None and last_val is not None else None,
        "last_tokens_per_sec": last.get("tokens_per_sec"),
        "peak_memory_allocated_mb": last.get("peak_memory_allocated_mb"),
        "compression_ratio": last.get("compression_ratio"),
        "mixer_backend": last.get("mixer_backend"),
        "agentic_eval_next_token_validation_loss": agentic_eval.get("next_token_validation_loss"),
        "json_tool_call_validity": agentic_eval.get("json_tool_call_validity"),
        "mean_patch_apply_rate": agentic_eval.get("mean_patch_apply_rate"),
    }


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    headers = [
        "model_name",
        "last_step",
        "tokens_seen",
        "last_val_next_token_loss",
        "val_loss_delta",
        "last_tokens_per_sec",
        "peak_memory_allocated_mb",
        "param_count_non_embedding",
        "estimated_flops_per_token",
    ]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        values = []
        for key in headers:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched RAAM-AgentCoder training runs.")
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--output-json", default="runs/stage3_baselines/summary.json")
    parser.add_argument("--output-md", default="runs/stage3_baselines/summary.md")
    args = parser.parse_args()

    rows = [summarize_run(Path(path)) for path in args.run_dirs]
    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({"runs": rows}, indent=2, sort_keys=True) + "\n")
    write_markdown(rows, md_path)
    print(json.dumps({"runs": rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
