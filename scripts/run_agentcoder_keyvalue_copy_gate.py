#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raam_lm.config import load_config
from raam_lm.tokenization import AgentCoderTokenizer
from scripts.make_agentcoder_keyvalue_copy_sft import (
    COMPLETION_MODES,
    VALUE_BOUNDARY_CLOSE,
    VALUE_BOUNDARY_OPEN,
)
from scripts.run_agentcoder_slotcopy_gate import read_last_train_row, summarize_ladder, summarize_slot_families


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def read_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload.get("cases", payload) if isinstance(payload, dict) else payload


def render_training_record(record: dict[str, Any]) -> str:
    parts: list[str] = []
    if record.get("system"):
        parts.append(f"<|system|>\n{record['system']}\n")
    if record.get("repo_context"):
        parts.append(f"<|repo_context|>\n{record['repo_context']}\n")
    for message in record.get("messages", []):
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"<|{role}|>\n{content}\n")
    for step in record.get("trace", []):
        kind = step.get("type", "assistant")
        content = step.get("content", "")
        if kind == "tool_call":
            parts.append(f"<|tool|>\n{content}\n")
        elif kind == "tool_result":
            parts.append(f"<|tool_result|>\n{content}\n")
        elif kind == "patch":
            parts.append(f"<|patch|>\n{content}\n")
        elif kind == "test_output":
            parts.append(f"<|test_output|>\n{content}\n")
        else:
            parts.append(f"<|assistant|>\n{content}\n")
    if record.get("final"):
        parts.append(f"<|final|>\n{record['final']}\n")
    return "\n".join(parts).strip() + "\n"


def expected_completion(case: dict[str, Any], completion_mode: str, value_boundaries: bool) -> str:
    keys = [str(key) for key in case.get("expected_key_sequence", case.get("target_keys", []))]
    values = [str(value) for value in case.get("expected_value_sequence", [])]
    if completion_mode == "key_only":
        return "\n".join(keys)
    if completion_mode == "value_only":
        if value_boundaries:
            values = [f"{VALUE_BOUNDARY_OPEN}{value}{VALUE_BOUNDARY_CLOSE}" for value in values]
        return "\n".join(values)
    slots = case.get("expected_slots", {})
    return "\n".join(f"{key}={slots[key]}" for key in keys if key in slots)


def build_length_report(
    *,
    train_records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
    tokenizer: AgentCoderTokenizer,
    completion_mode: str,
    value_boundaries: bool,
) -> dict[str, Any]:
    train_lengths = []
    for index, record in enumerate(train_records):
        ids = tokenizer.encode(render_training_record(record), add_bos=True, add_eos=True)
        train_lengths.append(
            {
                "index": index,
                "source_row_index": record.get("source_row_index"),
                "target_keys": record.get("target_keys"),
                "token_length": len(ids),
            }
        )

    eval_lengths = []
    for case in cases:
        prompt_len = len(tokenizer.encode(case["prompt"], add_bos=True, add_eos=False))
        completion = expected_completion(case, completion_mode, value_boundaries)
        completion_len = len(tokenizer.encode(completion, add_bos=False, add_eos=True))
        eval_lengths.append(
            {
                "name": case["name"],
                "eval_tier": case.get("eval_tier"),
                "target_keys": case.get("target_keys"),
                "prompt_tokens": prompt_len,
                "expected_completion_tokens": completion_len,
                "full_tokens": prompt_len + completion_len,
            }
        )

    max_train = max(train_lengths, key=lambda row: row["token_length"])
    max_eval = max(eval_lengths, key=lambda row: row["full_tokens"])
    return {
        "max_train_record_tokens": max_train["token_length"],
        "max_train_record": max_train,
        "max_eval_prompt_tokens": max(row["prompt_tokens"] for row in eval_lengths),
        "max_eval_full_tokens": max_eval["full_tokens"],
        "max_eval_case": max_eval,
    }


def validate_sequence_windows(args: argparse.Namespace, train_jsonl: Path, cases_json: Path, tokenizer_path: Path) -> dict[str, Any]:
    tokenizer = AgentCoderTokenizer.load(tokenizer_path)
    config = load_config(args.config)
    train_records = read_jsonl(train_jsonl)
    cases = read_cases(cases_json)
    report = build_length_report(
        train_records=train_records,
        cases=cases,
        tokenizer=tokenizer,
        completion_mode=args.completion_mode,
        value_boundaries=args.value_boundaries,
    )
    report.update(
        {
            "train_seq_len": args.seq_len,
            "eval_max_seq_len": config.max_seq_len,
        }
    )
    problems = []
    if report["max_train_record_tokens"] > args.seq_len:
        problems.append(
            "training seq_len "
            f"{args.seq_len} is shorter than the longest generated training record "
            f"({report['max_train_record_tokens']} tokens)"
        )
    if report["max_eval_full_tokens"] > config.max_seq_len:
        problems.append(
            "config max_seq_len "
            f"{config.max_seq_len} is shorter than the longest prompt plus expected completion "
            f"({report['max_eval_full_tokens']} tokens)"
        )
    if problems:
        formatted = json.dumps(report, indent=2, sort_keys=True)
        raise ValueError("key-value copy gate sequence window is too short: " + "; ".join(problems) + "\n" + formatted)
    print(json.dumps({"length_preflight": report}, indent=2, sort_keys=True), flush=True)
    return report


def build_pack_command(args: argparse.Namespace, train_jsonl: Path, tokenizer: Path, packed: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/pack_dataset.py",
        str(train_jsonl),
        "--tokenizer",
        str(tokenizer),
        "--output-dir",
        str(packed),
        "--seq-len",
        str(args.seq_len),
        "--val-fraction",
        str(args.val_fraction),
        "--seed",
        str(args.seed),
    ]
    if args.mirror_val:
        cmd.append("--mirror-val")
    if args.assistant_loss_only:
        cmd.append("--assistant-loss-only")
    return cmd


def build_train_command(args: argparse.Namespace, packed: Path, tokenizer: Path, train_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        args.config,
        "--train-bin",
        str(packed / "train.bin"),
        "--val-bin",
        str(packed / "val.bin"),
        "--tokenizer",
        str(tokenizer),
        "--output-dir",
        str(train_dir),
        "--device",
        args.device,
        "--seq-len",
        str(args.seq_len),
        "--seed",
        str(args.seed),
    ]
    if args.steps is not None:
        cmd.extend(["--steps", str(args.steps)])
    if args.eval_batches is not None:
        cmd.extend(["--eval-batches", str(args.eval_batches)])
    if args.mlops_project_path:
        cmd.extend(["--mlops-project-path", args.mlops_project_path])
        if args.mlops_run_id:
            cmd.extend(["--mlops-run-id", args.mlops_run_id])
    return cmd


def summarize_train_log(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    val_rows = [row for row in rows if "val_next_token_loss" in row]
    summary: dict[str, Any] = {"train_log_rows": len(rows)}
    if rows:
        last = rows[-1]
        summary.update(
            {
                "final_step": last.get("global_step"),
                "final_tokens_per_sec": last.get("tokens_per_sec"),
                "final_train_loss": last.get("train_loss"),
            }
        )
    if val_rows:
        best = min(val_rows, key=lambda row: row["val_next_token_loss"])
        final = val_rows[-1]
        summary.update(
            {
                "best_val_loss": best["val_next_token_loss"],
                "best_val_step": best.get("global_step"),
                "final_val_loss": final["val_next_token_loss"],
                "final_val_step": final.get("global_step"),
            }
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the structured key-value AgentCoder copy gate.")
    parser.add_argument("--config", default="configs/scratch/raam_agentcoder_keyvalue_copy_gate.yaml")
    parser.add_argument("--output-dir", default="runs/agentcoder_keyvalue_copy_gate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument(
        "--eval-mode",
        choices=["mirror", "covered", "heldout", "ladder", "coverage_ladder"],
        default="ladder",
    )
    parser.add_argument("--completion-mode", choices=COMPLETION_MODES, default="keyvalue")
    parser.add_argument("--train-records", type=int, default=96)
    parser.add_argument("--train-variants-per-row", type=int, default=1)
    parser.add_argument("--eval-cases", type=int, default=64)
    parser.add_argument("--target-fields", type=int, default=3)
    parser.add_argument("--distractor-fields", type=int, default=4)
    parser.add_argument("--value-boundaries", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mirror-val", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--assistant-loss-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlops-project-path", default="")
    parser.add_argument("--mlops-run-id", default="")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_dir = output_dir / "generated"
    train_jsonl = generated_dir / "keyvalue_train.jsonl"
    cases_json = generated_dir / "keyvalue_eval_cases.json"
    data_manifest = generated_dir / "keyvalue_manifest.json"
    tokenizer = output_dir / "tokenizer.json"
    packed = output_dir / "packed"
    train_dir = output_dir / "train"
    eval_json = output_dir / "keyvalue_eval.json"
    summary_json = output_dir / "summary.json"

    run(
        [
            sys.executable,
            "scripts/make_agentcoder_keyvalue_copy_sft.py",
            "--train-output",
            str(train_jsonl),
            "--cases-output",
            str(cases_json),
            "--manifest-output",
            str(data_manifest),
            "--seed",
            str(args.seed),
            "--eval-mode",
            args.eval_mode,
            "--completion-mode",
            args.completion_mode,
            "--train-records",
            str(args.train_records),
            "--train-variants-per-row",
            str(args.train_variants_per_row),
            "--eval-cases",
            str(args.eval_cases),
            "--target-fields",
            str(args.target_fields),
            "--distractor-fields",
            str(args.distractor_fields),
        ]
        + (["--value-boundaries"] if args.value_boundaries else [])
    )
    run(
        [
            sys.executable,
            "scripts/train_tokenizer.py",
            str(train_jsonl),
            "--output",
            str(tokenizer),
            "--vocab-size",
            str(args.vocab_size),
        ]
    )
    length_report = validate_sequence_windows(args, train_jsonl, cases_json, tokenizer)
    run(build_pack_command(args, train_jsonl, tokenizer, packed))
    run(build_train_command(args, packed, tokenizer, train_dir))

    checkpoint = train_dir / "checkpoints" / "last.pt"
    eval_cmd = [
        sys.executable,
        "scripts/eval_overfit_sanity.py",
        "--config",
        args.config,
        "--tokenizer",
        str(tokenizer),
        "--checkpoint",
        str(checkpoint),
        "--device",
        args.device,
        "--cases-json",
        str(cases_json),
        "--output",
        str(eval_json),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--min-pass-rate",
        str(args.min_pass_rate),
    ]
    if args.no_fail:
        eval_cmd.append("--no-fail")
    run(eval_cmd)

    eval_payload = json.loads(eval_json.read_text())
    data_payload = json.loads(data_manifest.read_text())
    packed_manifest = json.loads((packed / "manifest.json").read_text())
    train_manifest = json.loads((train_dir / "manifest.json").read_text())
    summary: dict[str, Any] = {
        "config": args.config,
        "output_dir": str(output_dir),
        "generated_train": str(train_jsonl),
        "generated_cases": str(cases_json),
        "generated_manifest": str(data_manifest),
        "behavior_counts": data_payload["behavior_counts"],
        "train_slot_family_counts": data_payload["train_slot_family_counts"],
        "eval_slot_family_counts": data_payload["eval_slot_family_counts"],
        "eval_tier_counts": data_payload.get("eval_tier_counts"),
        "eval_target_key_counts": data_payload.get("eval_target_key_counts"),
        "eval_value_format_counts": data_payload.get("eval_value_format_counts"),
        "eval_mode": data_payload.get("eval_mode", args.eval_mode),
        "completion_mode": data_payload.get("completion_mode", args.completion_mode),
        "slot_family_summary": summarize_slot_families(eval_payload),
        "slot_ladder_summary": summarize_ladder(eval_payload),
        "tokenizer": str(tokenizer),
        "packed_manifest": str(packed / "manifest.json"),
        "train_manifest": str(train_dir / "manifest.json"),
        "checkpoint": str(checkpoint),
        "keyvalue_eval": str(eval_json),
        "pass_count": eval_payload["pass_count"],
        "case_count": eval_payload["case_count"],
        "pass_rate": eval_payload["pass_rate"],
        "behavior_confusion": eval_payload.get("behavior_confusion"),
        "behavior_accuracy": eval_payload.get("behavior_accuracy"),
        "behavior_correct_count": eval_payload.get("behavior_correct_count"),
        "behavior_labeled_cases": eval_payload.get("behavior_labeled_cases"),
        "key_sequence_accuracy": eval_payload.get("key_sequence_accuracy"),
        "key_sequence_correct_count": eval_payload.get("key_sequence_correct_count"),
        "key_sequence_labeled_cases": eval_payload.get("key_sequence_labeled_cases"),
        "value_sequence_accuracy": eval_payload.get("value_sequence_accuracy"),
        "value_sequence_correct_count": eval_payload.get("value_sequence_correct_count"),
        "value_sequence_labeled_cases": eval_payload.get("value_sequence_labeled_cases"),
        "last_train_row": read_last_train_row(train_dir / "train_log.jsonl"),
        "train_records": data_payload["train_records"],
        "base_train_rows": data_payload.get("base_train_rows", args.train_records),
        "train_variants_per_row": data_payload.get("train_variants_per_row", args.train_variants_per_row),
        "target_fields": data_payload.get("target_fields", args.target_fields),
        "distractor_fields": data_payload.get("distractor_fields", args.distractor_fields),
        "value_boundaries": data_payload.get("value_boundaries", args.value_boundaries),
        "eval_cases": data_payload["eval_cases"],
        "train_tokens": packed_manifest["train_tokens"],
        "val_tokens": packed_manifest["val_tokens"],
        "train_docs": packed_manifest.get("train_docs"),
        "val_docs": packed_manifest.get("val_docs"),
        "mirror_val": packed_manifest.get("mirror_val", False),
        "assistant_loss_only": packed_manifest.get("assistant_loss_only", False),
        "train_loss_tokens": packed_manifest.get("train_loss_tokens"),
        "val_loss_tokens": packed_manifest.get("val_loss_tokens"),
        "param_count_non_embedding": train_manifest["param_count_non_embedding"],
        "estimated_flops_per_token": train_manifest["estimated_flops_per_token"],
        "length_preflight": length_report,
    }
    summary.update(summarize_train_log(train_dir / "train_log.jsonl"))
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
