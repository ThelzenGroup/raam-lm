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

from scripts.make_agentcoder_medium_repair_sft import (
    ANCHOR_FLOOR_FUNCTIONS,
    FORMAT,
    REQUIRED_TOPICS,
    build_base_records,
    expanded_eval_cases,
    expand_records,
    rendered_prompt_for_row,
    validate_records,
    write_jsonl,
)


REQUIRED_TINY_FLOOR = ["ladder_is_even", "ladder_is_odd", "ladder_filter_even"]


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


def assert_baseline_gate(payload: dict, min_pass_count: int) -> None:
    pass_count = int(payload.get("pass_count", 0))
    if pass_count < min_pass_count:
        raise SystemExit(f"baseline pass_count {pass_count} is below required {min_pass_count}")
    results = {str(row.get("name")): row for row in payload.get("results", [])}
    missing = [name for name in REQUIRED_TINY_FLOOR if name not in results]
    if missing:
        raise SystemExit(f"baseline eval missing tiny-floor cases: {missing}")
    failed = [name for name in REQUIRED_TINY_FLOOR if not results[name].get("passed")]
    if failed:
        raise SystemExit(f"baseline tiny-floor cases failed: {failed}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight RAAM medium-repair data and starting-checkpoint eval.")
    parser.add_argument("--output-dir", default="runs/medium_repair_preflight")
    parser.add_argument("--config", default="configs/scratch/raam_agentcoder_100m_medium_repair_v2_sft.yaml")
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
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--anchor-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--min-baseline-pass-count", type=int, default=3)
    parser.add_argument("--skip-checkpoint-eval", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    generated_dir = output_dir / "generated"
    train_path = generated_dir / "medium_train.jsonl"
    cases_path = generated_dir / "medium_eval_cases.json"
    manifest_path = generated_dir / "medium_manifest.json"
    eval_output = output_dir / "baseline_medium_eval.json"

    for path in [Path(args.config), Path(args.tokenizer), Path(args.checkpoint)]:
        if not path.exists():
            raise SystemExit(f"required path does not exist: {path}")

    base_records = build_base_records()
    rows = expand_records(base_records, repeats=args.repeats, anchor_repeats=args.anchor_repeats, seed=args.seed)
    validation = validate_records(rows)
    cases = expanded_eval_cases()
    train_prompts = {rendered_prompt_for_row(row) for row in rows}
    eval_prompts = {str(case["prompt"]) for case in cases}
    overlaps = sorted(train_prompts & eval_prompts)
    if overlaps:
        raise SystemExit(f"train/eval exact prompt overlap: {overlaps[:3]}")

    write_jsonl(train_path, rows, strip_validation=True)
    write_cases(cases_path, cases)
    topic_counts = Counter(str(row.get("topic", "unknown")) for row in rows)
    behavior_counts = Counter(str(row.get("behavior", "unknown")) for row in rows)
    train_prompts_hashable = sorted(train_prompts)
    eval_prompts_hashable = sorted(eval_prompts)
    manifest = {
        "format": FORMAT,
        "seed": args.seed,
        "repeats": args.repeats,
        "anchor_repeats": args.anchor_repeats,
        "train_records": len(rows),
        "base_records": len(base_records),
        "eval_cases": len(cases),
        "topic_counts": dict(sorted(topic_counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "tiny_anchor_records": topic_counts.get("anchor_tiny_function", 0),
        "tiny_anchor_ratio": topic_counts.get("anchor_tiny_function", 0) / len(rows) if rows else 0.0,
        "tiny_anchor_floor_functions": ANCHOR_FLOOR_FUNCTIONS,
        "required_topics": REQUIRED_TOPICS,
        "final_nonempty_count": sum(1 for row in rows if row.get("final")),
        "validation": validation,
        "exact_train_eval_prompt_overlaps": overlaps,
        "train_prompt_count": len(train_prompts_hashable),
        "eval_prompt_count": len(eval_prompts_hashable),
        "train_output": str(train_path),
        "cases_output": str(cases_path),
        "checkpoint": str(args.checkpoint),
        "tokenizer": str(args.tokenizer),
        "config": str(args.config),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    payload = None
    if not args.skip_checkpoint_eval:
        payload = run_eval(args, cases_path, eval_output)
        assert_baseline_gate(payload, args.min_baseline_pass_count)

    print(
        json.dumps(
            {
                "preflight": "passed",
                "train_records": len(rows),
                "eval_cases": len(cases),
                "tiny_anchor_floor_counts": validation["tiny_anchor_floor_counts"],
                "baseline_pass_count": payload.get("pass_count") if payload else None,
                "baseline_eval": str(eval_output) if payload else None,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
