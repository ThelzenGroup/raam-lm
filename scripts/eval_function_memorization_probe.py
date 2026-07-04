#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_agentcoder_function_repair_sft import ANCHOR_FLOOR_FUNCTIONS, TARGET_FUNCTIONS


def run_eval(args: argparse.Namespace, output: Path) -> dict:
    command = [
        sys.executable,
        str(ROOT / "scripts/eval_coding_ladder.py"),
        "--config",
        str(args.config),
        "--tokenizer",
        str(args.tokenizer),
        "--checkpoint",
        str(args.checkpoint),
        "--cases-json",
        str(args.cases_json),
        "--output",
        str(output),
        "--device",
        args.device,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--temperature",
        "0.0",
        "--top-k",
        "0",
        "--no-fail",
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    return json.loads(output.read_text())


def parse_function_names(raw: str | None, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return names or list(default)


def probe_case_names(payload: dict, *, target_functions: list[str] | None = None) -> tuple[list[str], list[str]]:
    rows = {str(row["name"]): row for row in payload.get("results", [])}
    probe_rows = [row for row in payload.get("results", []) if str(row.get("name", "")).startswith("probe_exact_")]
    if probe_rows and any("topic" in row for row in probe_rows):
        anchor_case_names = sorted(
            str(row["name"]) for row in probe_rows if str(row.get("topic")) == "anchor_tiny_function"
        )
        target_case_names = sorted(
            str(row["name"]) for row in probe_rows if str(row.get("topic")) != "anchor_tiny_function"
        )
    else:
        target_case_names = [f"probe_exact_{name}" for name in (target_functions or TARGET_FUNCTIONS)]
        anchor_case_names = [f"probe_exact_{name}" for name in ANCHOR_FLOOR_FUNCTIONS]
    return target_case_names, anchor_case_names


def summarize_probe(payload: dict, *, target_functions: list[str] | None = None) -> dict:
    rows = {str(row["name"]): row for row in payload.get("results", [])}
    target_case_names, anchor_case_names = probe_case_names(payload, target_functions=target_functions)
    target_passed = [name for name in target_case_names if rows.get(name, {}).get("passed")]
    anchor_passed = [name for name in anchor_case_names if rows.get(name, {}).get("passed")]
    return {
        "probe_pass_count": int(payload.get("pass_count", 0)),
        "probe_case_count": int(payload.get("case_count", 0)),
        "target_probe_pass_count": len(target_passed),
        "target_probe_case_count": len(target_case_names),
        "target_probe_passed_cases": target_passed,
        "anchor_probe_pass_count": len(anchor_passed),
        "anchor_probe_case_count": len(anchor_case_names),
        "anchor_probe_passed_cases": anchor_passed,
        "failed_cases": payload.get("failed_cases", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate exact train-like function memorization probes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--output", default="runs/function_memorization_probe_eval.json")
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument(
        "--target-functions",
        default=None,
        help="Optional comma-separated target function names, used only if eval rows lack topic metadata.",
    )
    parser.add_argument("--require-target-pass-count", type=int, default=0)
    parser.add_argument("--require-anchor-pass-count", type=int, default=0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    payload = run_eval(args, output)
    target_functions = parse_function_names(args.target_functions, TARGET_FUNCTIONS) if args.target_functions else None
    summary = {
        "metadata": payload.get("metadata", {}),
        "eval_output": str(output),
        "cases_json": str(args.cases_json),
        "checkpoint": str(args.checkpoint),
        **summarize_probe(payload, target_functions=target_functions),
    }
    summary_output = Path(args.summary_output) if args.summary_output else output.with_name("function_probe_summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))

    target_ok = summary["target_probe_pass_count"] >= args.require_target_pass_count
    anchor_ok = summary["anchor_probe_pass_count"] >= args.require_anchor_pass_count
    if not args.no_fail and (not target_ok or not anchor_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
