#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_agentcoder_function_repair_sft import (
    ANCHOR_FLOOR_FUNCTIONS,
    FORMAT,
    REQUIRED_TOPICS,
    STAGE,
    TARGET_FUNCTIONS,
    assert_train_eval_disjoint,
    build_selected_base_records,
    expanded_eval_cases,
    expand_records,
    memorization_probe_cases,
    parse_target_functions,
    required_topics_for_targets,
    rendered_prompt_for_row,
    validate_records,
    write_jsonl,
)


REQUIRED_TINY_FLOOR_CASES = ["ladder_is_even", "ladder_is_odd", "ladder_filter_even"]
TARGET_EVAL_CASES = [
    "ladder_count_even",
    "ladder_safe_int",
    "ladder_parse_port",
    "medium_count_even_negatives",
    "medium_safe_int_defaults",
    "medium_parse_port_range",
]


def write_cases(path: Path, cases: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}, indent=2, sort_keys=True) + "\n")


def run_eval(args: argparse.Namespace, cases_path: Path, eval_output: Path) -> dict:
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
        str(cases_path),
        "--output",
        str(eval_output),
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
    return json.loads(eval_output.read_text())


def assert_baseline_gate(payload: dict, min_pass_count: int) -> dict[str, int]:
    pass_count = int(payload.get("pass_count", 0))
    if pass_count < min_pass_count:
        raise SystemExit(f"baseline pass_count {pass_count} is below required {min_pass_count}")
    results = {str(row.get("name")): row for row in payload.get("results", [])}
    missing_tiny = [name for name in REQUIRED_TINY_FLOOR_CASES if name not in results]
    if missing_tiny:
        raise SystemExit(f"baseline eval missing tiny-floor cases: {missing_tiny}")
    failed_tiny = [name for name in REQUIRED_TINY_FLOOR_CASES if not results[name].get("passed")]
    if failed_tiny:
        raise SystemExit(f"baseline tiny-floor cases failed: {failed_tiny}")
    target_pass_count = sum(1 for name in TARGET_EVAL_CASES if results.get(name, {}).get("passed"))
    tiny_pass_count = sum(1 for name in REQUIRED_TINY_FLOOR_CASES if results.get(name, {}).get("passed"))
    return {
        "baseline_pass_count": pass_count,
        "baseline_tiny_pass_count": tiny_pass_count,
        "baseline_target_function_pass_count": target_pass_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight RAAM function-only medium repair data and baseline eval.")
    parser.add_argument("--output-dir", default="runs/function_repair_preflight")
    parser.add_argument("--config", default="configs/scratch/raam_agentcoder_100m_function_repair_sft.yaml")
    parser.add_argument(
        "--tokenizer",
        default="runs/vast_backups/coding_ladder_repair_stop_control_20260704T_remote/current/train/tokenizer.json",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "runs/vast_backups/coding_ladder_repair_stop_control_20260704T_remote/current/train/checkpoints/"
            "model_only_coding_ladder_stop_control_fp16.pt"
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--target-repeats", type=int, default=30)
    parser.add_argument("--anchor-repeats", type=int, default=18)
    parser.add_argument("--target-prompt-limit", type=int, default=12)
    parser.add_argument("--anchor-prompt-limit", type=int, default=8)
    parser.add_argument(
        "--target-functions",
        default=",".join(TARGET_FUNCTIONS),
        help="Comma-separated target function names to include before anchors.",
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--min-baseline-pass-count", type=int, default=3)
    parser.add_argument("--skip-checkpoint-eval", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generated_dir = output_dir / "generated"
    train_path = generated_dir / "function_train.jsonl"
    cases_path = generated_dir / "medium_eval_cases.json"
    probe_cases_path = generated_dir / "function_probe_cases.json"
    manifest_path = generated_dir / "function_manifest.json"
    eval_output = output_dir / "baseline_medium_eval.json"

    for path in [Path(args.config), Path(args.tokenizer), Path(args.checkpoint)]:
        if not path.exists():
            raise SystemExit(f"required path does not exist: {path}")

    selected_targets = parse_target_functions(args.target_functions)
    required_topics = required_topics_for_targets(selected_targets)
    base_records = build_selected_base_records(
        target_prompt_limit=args.target_prompt_limit,
        anchor_prompt_limit=args.anchor_prompt_limit,
        target_functions=selected_targets,
    )
    rows = expand_records(
        base_records,
        target_repeats=args.target_repeats,
        anchor_repeats=args.anchor_repeats,
        seed=args.seed,
    )
    validation = validate_records(rows, target_functions=selected_targets)
    cases = expanded_eval_cases()
    probe_cases = memorization_probe_cases(selected_targets)
    overlaps = assert_train_eval_disjoint(rows, cases)
    write_jsonl(train_path, rows, strip_validation=True)
    write_cases(cases_path, cases)
    write_cases(probe_cases_path, probe_cases)

    topic_counts = Counter(str(row.get("topic", "unknown")) for row in rows)
    behavior_counts = Counter(str(row.get("behavior", "unknown")) for row in rows)
    train_prompts = {rendered_prompt_for_row(row) for row in rows}
    eval_prompts = {str(case["prompt"]) for case in cases}
    manifest = {
        "format": FORMAT,
        "stage": STAGE,
        "seed": args.seed,
        "target_repeats": args.target_repeats,
        "anchor_repeats": args.anchor_repeats,
        "target_prompt_limit": args.target_prompt_limit,
        "anchor_prompt_limit": args.anchor_prompt_limit,
        "train_records": len(rows),
        "base_records": len(base_records),
        "eval_cases": len(cases),
        "probe_cases": len(probe_cases),
        "topic_counts": dict(sorted(topic_counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "target_functions": selected_targets,
        "target_eval_cases": TARGET_EVAL_CASES,
        "tiny_anchor_floor_functions": ANCHOR_FLOOR_FUNCTIONS,
        "required_topics": required_topics,
        "final_nonempty_count": sum(1 for row in rows if row.get("final")),
        "validation": validation,
        "exact_train_eval_prompt_overlaps": overlaps,
        "train_prompt_count": len(train_prompts),
        "eval_prompt_count": len(eval_prompts),
        "train_output": str(train_path),
        "cases_output": str(cases_path),
        "probe_cases_output": str(probe_cases_path),
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "config": str(args.config),
    }

    payload = None
    baseline_summary = None
    if not args.skip_checkpoint_eval:
        payload = run_eval(args, cases_path, eval_output)
        baseline_summary = assert_baseline_gate(payload, args.min_baseline_pass_count)
        manifest.update(baseline_summary)
        manifest["baseline_eval"] = str(eval_output)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(
        json.dumps(
            {
                "preflight": "passed",
                "format": FORMAT,
                "stage": STAGE,
                "train_records": len(rows),
                "eval_cases": len(cases),
                "tiny_anchor_floor_counts": validation["tiny_anchor_floor_counts"],
                "target_function_counts": {name: validation["function_counts"][name] for name in selected_targets},
                "baseline_pass_count": payload.get("pass_count") if payload else None,
                "baseline_summary": baseline_summary,
                "baseline_eval": str(eval_output) if payload else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
