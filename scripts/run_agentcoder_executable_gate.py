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


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_last_train_row(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[-1] if rows else {}


def build_data_command(args: argparse.Namespace, generated_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/make_agentcoder_executable_sft.py",
        "--output-dir",
        str(generated_dir),
        "--ladder-repeats",
        str(args.ladder_repeats),
        "--curated-anchor-repeats",
        str(args.curated_anchor_repeats),
        "--eval-source-fraction",
        str(args.eval_source_fraction),
        "--min-average-score",
        str(args.min_average_score),
        "--max-answer-chars",
        str(args.max_answer_chars),
        "--max-tests-chars",
        str(args.max_tests_chars),
        "--max-function-lines",
        str(args.max_function_lines),
        "--max-diff-lines",
        str(args.max_diff_lines),
        "--max-file-chars",
        str(args.max_file_chars),
        "--seed",
        str(args.seed),
    ]
    if args.use_hf:
        cmd.append("--use-hf")
        cmd.extend(["--opencode-limit", str(args.opencode_limit)])
        cmd.extend(["--scotch-limit", str(args.scotch_limit)])
        cmd.extend(["--coderm-unittest-limit", str(args.coderm_unittest_limit)])
        cmd.extend(["--commitpackft-limit", str(args.commitpackft_limit)])
        cmd.extend(["--opencode-config", args.opencode_config])
        cmd.extend(["--scotch-config", args.scotch_config])
        cmd.extend(["--coderm-unittest-config", args.coderm_unittest_config])
        cmd.extend(["--commitpackft-config", args.commitpackft_config])
    if not args.include_ladder_eval:
        cmd.append("--no-include-ladder-eval")
    for behavior in args.train_behavior:
        cmd.extend(["--train-behavior", behavior])
    for topic in args.train_topic_contains:
        cmd.extend(["--train-topic-contains", topic])
    for behavior in args.eval_expected_behavior:
        cmd.extend(["--eval-expected-behavior", behavior])
    for topic in args.eval_topic_contains:
        cmd.extend(["--eval-topic-contains", topic])
    if not args.require_opencode_test_signal:
        cmd.append("--no-require-opencode-test-signal")
    return cmd


def resolve_data_paths(args: argparse.Namespace, output_dir: Path) -> tuple[Path, Path, Path]:
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = output_dir / "generated"
        run(build_data_command(args, data_dir))
    return (
        data_dir / "agentcoder_executable_train.jsonl",
        data_dir / "agentcoder_executable_eval_cases.json",
        data_dir / "agentcoder_executable_manifest.json",
    )


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
    if args.eval_every is not None:
        cmd.extend(["--eval-every", str(args.eval_every)])
    if args.save_best:
        cmd.append("--save-best")
    if args.restore_best_on_finish:
        cmd.append("--restore-best-on-finish")
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
    parser = argparse.ArgumentParser(description="Run a bounded executable-code RAAM-AgentCoder gate.")
    parser.add_argument("--config", default="configs/scratch/raam_agentcoder_executable_tiny_gate.yaml")
    parser.add_argument("--output-dir", default="runs/agentcoder_executable_tiny_gate/raam")
    parser.add_argument("--data-dir", default="", help="Existing executable SFT data dir; skips data generation when set.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=2048)
    parser.add_argument("--seq-len", type=int, default=384)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--mirror-val", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--assistant-loss-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--restore-best-on-finish", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mlops-project-path", default=str(ROOT))
    parser.add_argument("--mlops-run-id", default="")
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-fail", action="store_true")

    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--opencode-limit", type=int, default=20)
    parser.add_argument("--scotch-limit", type=int, default=30)
    parser.add_argument("--coderm-unittest-limit", type=int, default=5)
    parser.add_argument("--commitpackft-limit", type=int, default=20)
    parser.add_argument("--opencode-config", default="train")
    parser.add_argument("--scotch-config", default="python")
    parser.add_argument("--coderm-unittest-config", default="default")
    parser.add_argument("--commitpackft-config", default="python")
    parser.add_argument("--ladder-repeats", type=int, default=2)
    parser.add_argument("--curated-anchor-repeats", type=int, default=0)
    parser.add_argument("--include-ladder-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-behavior",
        action="append",
        default=[],
        help="Keep only generated training records with this behavior. Repeat for multiple behaviors.",
    )
    parser.add_argument(
        "--train-topic-contains",
        action="append",
        default=[],
        help="Keep only generated training records whose topic contains this substring. Repeat for multiple substrings.",
    )
    parser.add_argument(
        "--eval-expected-behavior",
        action="append",
        default=[],
        help="Keep/evaluate only eval cases with this expected_behavior. Repeat for multiple behaviors.",
    )
    parser.add_argument(
        "--eval-topic-contains",
        action="append",
        default=[],
        help="Keep/evaluate only eval cases whose topic contains this substring. Repeat for multiple substrings.",
    )
    parser.add_argument("--eval-source-fraction", type=float, default=0.05)
    parser.add_argument("--min-average-score", type=float, default=0.8)
    parser.add_argument("--require-opencode-test-signal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-answer-chars", type=int, default=2400)
    parser.add_argument("--max-tests-chars", type=int, default=5000)
    parser.add_argument("--max-function-lines", type=int, default=80)
    parser.add_argument("--max-diff-lines", type=int, default=80)
    parser.add_argument("--max-file-chars", type=int, default=4000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl, cases_json, data_manifest = resolve_data_paths(args, output_dir)
    tokenizer = output_dir / "tokenizer.json"
    packed = output_dir / "packed"
    train_dir = output_dir / "train"
    eval_json = output_dir / "executable_eval.json"
    summary_json = output_dir / "summary.json"

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
        "scripts/eval_coding_ladder.py",
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
    for behavior in args.eval_expected_behavior:
        eval_cmd.extend(["--expected-behavior", behavior])
    for topic in args.eval_topic_contains:
        eval_cmd.extend(["--topic-contains", topic])
    if args.no_fail:
        eval_cmd.append("--no-fail")
    run(eval_cmd)

    eval_payload = json.loads(eval_json.read_text())
    data_payload = json.loads(data_manifest.read_text())
    packed_manifest = json.loads((packed / "manifest.json").read_text())
    train_manifest = json.loads((train_dir / "manifest.json").read_text())
    summary = {
        "config": args.config,
        "output_dir": str(output_dir),
        "generated_train": str(train_jsonl),
        "generated_cases": str(cases_json),
        "generated_manifest": str(data_manifest),
        "behavior_counts": data_payload["behavior_counts"],
        "source_dataset_counts": data_payload.get("source_dataset_counts", {}),
        "tokenizer": str(tokenizer),
        "packed_manifest": str(packed / "manifest.json"),
        "train_manifest": str(train_dir / "manifest.json"),
        "checkpoint": str(checkpoint),
        "executable_eval": str(eval_json),
        "pass_count": eval_payload["pass_count"],
        "case_count": eval_payload["case_count"],
        "pass_rate": eval_payload["pass_rate"],
        "function_pass_count": eval_payload.get("function_pass_count"),
        "json_pass_count": eval_payload.get("json_pass_count"),
        "patch_pass_count": eval_payload.get("patch_pass_count"),
        "nonsense_fail_count": eval_payload.get("nonsense_fail_count"),
        "failed_cases": eval_payload.get("failed_cases", []),
        "last_train_row": read_last_train_row(train_dir / "train_log.jsonl"),
        "train_records": data_payload["train_records"],
        "eval_cases": data_payload["eval_cases"],
        "train_behavior": args.train_behavior,
        "train_topic_contains": args.train_topic_contains,
        "eval_expected_behavior": args.eval_expected_behavior,
        "eval_topic_contains": args.eval_topic_contains,
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
        "mlops_project_path": args.mlops_project_path,
        "mlops_run_id": args.mlops_run_id,
    }
    summary.update(summarize_train_log(train_dir / "train_log.jsonl"))
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
