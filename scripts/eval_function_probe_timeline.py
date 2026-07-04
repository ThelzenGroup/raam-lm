#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def checkpoint_sort_key(path: Path) -> tuple[int, int, str]:
    name = path.name
    match = re.fullmatch(r"step_(\d+)\.pt", name)
    if match:
        return (0, int(match.group(1)), name)
    if name == "best.pt":
        return (1, 0, name)
    if name == "last.pt":
        return (2, 0, name)
    return (3, 0, name)


def list_checkpoints(checkpoint_dir: str | Path, pattern: str = "*.pt") -> list[Path]:
    root = Path(checkpoint_dir)
    checkpoints = [path for path in root.glob(pattern) if path.is_file()]
    return sorted(checkpoints, key=checkpoint_sort_key)


def run_probe_eval(args: argparse.Namespace, checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    stem = checkpoint.stem
    output = output_dir / f"{stem}_probe_eval.json"
    summary_output = output_dir / f"{stem}_probe_summary.json"
    command = [
        sys.executable,
        str(ROOT / "scripts/eval_function_memorization_probe.py"),
        "--config",
        str(args.config),
        "--tokenizer",
        str(args.tokenizer),
        "--checkpoint",
        str(checkpoint),
        "--cases-json",
        str(args.cases_json),
        "--output",
        str(output),
        "--summary-output",
        str(summary_output),
        "--device",
        args.device,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--no-fail",
    ]
    if args.target_functions:
        command.extend(["--target-functions", args.target_functions])
    subprocess.run(command, cwd=ROOT, check=True)
    summary = json.loads(summary_output.read_text())
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_label": checkpoint.name,
        "eval_output": str(output),
        "summary_output": str(summary_output),
        **summary,
    }


def summarize_timeline(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first_target_pass = None
    first_full_pass = None
    for row in rows:
        target_count = int(row.get("target_probe_pass_count", 0))
        target_cases = int(row.get("target_probe_case_count", 0))
        anchor_count = int(row.get("anchor_probe_pass_count", 0))
        anchor_cases = int(row.get("anchor_probe_case_count", 0))
        if first_target_pass is None and target_cases and target_count == target_cases:
            first_target_pass = row["checkpoint"]
        if (
            first_full_pass is None
            and target_cases
            and anchor_cases
            and target_count == target_cases
            and anchor_count == anchor_cases
        ):
            first_full_pass = row["checkpoint"]
    return {
        "checkpoint_count": len(rows),
        "first_all_targets_pass_checkpoint": first_target_pass,
        "first_all_targets_and_anchors_pass_checkpoint": first_full_pass,
        "best_target_probe_pass_count": max((int(row.get("target_probe_pass_count", 0)) for row in rows), default=0),
        "best_anchor_probe_pass_count": max((int(row.get("anchor_probe_pass_count", 0)) for row in rows), default=0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact function probes across saved checkpoints.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--checkpoint-pattern", default="*.pt")
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--output", default="runs/function_probe_timeline.json")
    parser.add_argument("--eval-output-dir", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--target-functions", default=None)
    args = parser.parse_args()

    checkpoints = list_checkpoints(args.checkpoint_dir, args.checkpoint_pattern)
    if not checkpoints:
        raise SystemExit(f"no checkpoints found in {args.checkpoint_dir} matching {args.checkpoint_pattern}")
    output = Path(args.output)
    eval_output_dir = Path(args.eval_output_dir) if args.eval_output_dir else output.with_suffix("")
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    rows = [run_probe_eval(args, checkpoint, eval_output_dir) for checkpoint in checkpoints]
    payload = {
        "checkpoint_dir": str(args.checkpoint_dir),
        "checkpoint_pattern": args.checkpoint_pattern,
        "cases_json": str(args.cases_json),
        "summary": summarize_timeline(rows),
        "checkpoints": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
