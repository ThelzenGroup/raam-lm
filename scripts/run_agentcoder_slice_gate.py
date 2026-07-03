#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def read_last_train_row(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows[-1] if rows else {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small non-mirrored AgentCoder slice gate.")
    parser.add_argument("--data", default="examples/agentcoder_slice_train.jsonl")
    parser.add_argument("--cases-json", default="examples/agentcoder_slice_eval_cases.json")
    parser.add_argument("--config", default="configs/scratch/raam_agentcoder_slice_gate.yaml")
    parser.add_argument("--output-dir", default="runs/agentcoder_slice_gate")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--seq-len", type=int, default=192)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--eval-batches", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--min-pass-rate", type=float, default=0.34)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = output_dir / "tokenizer.json"
    packed = output_dir / "packed"
    train_dir = output_dir / "train"
    eval_json = output_dir / "slice_eval.json"
    qualitative_json = output_dir / "qualitative_samples.json"
    qualitative_md = output_dir / "qualitative_samples.md"
    summary_json = output_dir / "summary.json"

    run(
        [
            sys.executable,
            "scripts/train_tokenizer.py",
            args.data,
            "--output",
            str(tokenizer),
            "--vocab-size",
            str(args.vocab_size),
        ]
    )
    run(
        [
            sys.executable,
            "scripts/pack_dataset.py",
            args.data,
            "--tokenizer",
            str(tokenizer),
            "--output-dir",
            str(packed),
            "--seq-len",
            str(args.seq_len),
            "--val-fraction",
            str(args.val_fraction),
        ]
    )
    train_cmd = [
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
    ]
    if args.steps is not None:
        train_cmd.extend(["--steps", str(args.steps)])
    if args.eval_batches is not None:
        train_cmd.extend(["--eval-batches", str(args.eval_batches)])
    run(train_cmd)

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
        args.cases_json,
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
    run(
        [
            sys.executable,
            "scripts/qualitative_checkpoint_inspect.py",
            "--config",
            args.config,
            "--tokenizer",
            str(tokenizer),
            "--checkpoint",
            str(checkpoint),
            "--device",
            args.device,
            "--output-json",
            str(qualitative_json),
            "--output-md",
            str(qualitative_md),
            "--max-new-tokens",
            "96",
            "--temperature",
            "0.0",
            "--top-k",
            "0",
        ]
    )

    eval_payload = json.loads(eval_json.read_text())
    packed_manifest = json.loads((packed / "manifest.json").read_text())
    train_manifest = json.loads((train_dir / "manifest.json").read_text())
    summary = {
        "data": args.data,
        "cases_json": args.cases_json,
        "config": args.config,
        "output_dir": str(output_dir),
        "tokenizer": str(tokenizer),
        "packed_manifest": str(packed / "manifest.json"),
        "train_manifest": str(train_dir / "manifest.json"),
        "checkpoint": str(checkpoint),
        "slice_eval": str(eval_json),
        "qualitative_json": str(qualitative_json),
        "qualitative_md": str(qualitative_md),
        "pass_count": eval_payload["pass_count"],
        "case_count": eval_payload["case_count"],
        "pass_rate": eval_payload["pass_rate"],
        "last_train_row": read_last_train_row(train_dir / "train_log.jsonl"),
        "train_tokens": packed_manifest["train_tokens"],
        "val_tokens": packed_manifest["val_tokens"],
        "train_docs": packed_manifest.get("train_docs"),
        "val_docs": packed_manifest.get("val_docs"),
        "mirror_val": packed_manifest.get("mirror_val", False),
        "param_count_non_embedding": train_manifest["param_count_non_embedding"],
        "estimated_flops_per_token": train_manifest["estimated_flops_per_token"],
    }
    summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
