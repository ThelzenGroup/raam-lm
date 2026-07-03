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

from scripts.make_agentcoder_keyvalue_copy_sft import COMPLETION_MODES
from scripts.run_agentcoder_slotcopy_gate import read_last_train_row, summarize_ladder, summarize_slot_families


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


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
    return cmd


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
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
