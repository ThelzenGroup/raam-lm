#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any


SYSTEM = "You are RAAM-AgentCoder, a precise copying assistant."
DEFAULT_SEED = 17
TRAIN_RECORDS = 64
EVAL_CASES = 32


def spec(index: int) -> dict[str, str]:
    return {
        "symbol": f"copy_symbol_{index:03d}",
        "file": f"copy_file_{index:03d}.py",
    }


def repo_context(row: dict[str, str]) -> str:
    return f"symbol: {row['symbol']}\nfile: {row['file']}"


def user_prompt() -> str:
    return "Copy the symbol and file exactly. Return only symbol=<value> and file=<value>."


def assistant_text(row: dict[str, str]) -> str:
    return f"symbol={row['symbol']}\nfile={row['file']}"


def chat_prompt(row: dict[str, str]) -> str:
    return (
        "<|system|>\n"
        f"{SYSTEM} Return only exact key=value lines.\n\n"
        "<|repo_context|>\n"
        f"{repo_context(row)}\n\n"
        "<|user|>\n"
        f"{user_prompt()}\n\n"
        "<|assistant|>\n"
    )


def record(row: dict[str, str]) -> dict[str, Any]:
    return {
        "behavior": "copy_slot_values",
        "slot_family": "atomic_repo_pair_copy",
        "expected_slots": {"symbol": row["symbol"], "file": row["file"]},
        "system": f"{SYSTEM} Return only exact key=value lines.",
        "repo_context": repo_context(row),
        "messages": [{"role": "user", "content": user_prompt()}],
        "trace": [{"type": "assistant", "content": assistant_text(row)}],
    }


def case(name: str, row: dict[str, str], eval_tier: str) -> dict[str, Any]:
    return {
        "name": name,
        "prompt": chat_prompt(row),
        "required_substrings": [f"symbol={row['symbol']}", f"file={row['file']}"],
        "forbidden_substrings": [],
        "expected_behavior": "copy_slot_values",
        "slot_family": "atomic_repo_pair_copy",
        "expected_slots": {"symbol": row["symbol"], "file": row["file"]},
        "eval_tier": eval_tier,
    }


def build_train_records(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    del seed
    return [record(spec(index)) for index in range(TRAIN_RECORDS)]


def build_eval_cases(seed: int = DEFAULT_SEED, eval_mode: str = "mirror") -> list[dict[str, Any]]:
    del seed
    if eval_mode == "mirror":
        return [case(f"atomic_mirror_{index:03d}", spec(index), "mirror_slot") for index in range(EVAL_CASES)]
    if eval_mode == "heldout":
        start = TRAIN_RECORDS
        return [case(f"atomic_heldout_{index:03d}", spec(start + index), "heldout_slot") for index in range(EVAL_CASES)]
    if eval_mode == "ladder":
        mirror = [case(f"atomic_mirror_{index:03d}", spec(index), "mirror_slot") for index in range(EVAL_CASES)]
        heldout = [
            case(f"atomic_heldout_{index:03d}", spec(TRAIN_RECORDS + index), "heldout_slot")
            for index in range(EVAL_CASES)
        ]
        return mirror + heldout
    raise ValueError("eval_mode must be 'mirror', 'heldout', or 'ladder'")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an atomic AgentCoder copy task.")
    parser.add_argument("--train-output", default="examples/agentcoder_atomic_copy_train.jsonl")
    parser.add_argument("--cases-output", default="examples/agentcoder_atomic_copy_cases.json")
    parser.add_argument("--manifest-output", default="examples/agentcoder_atomic_copy_manifest.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-mode", choices=["mirror", "heldout", "ladder"], default="mirror")
    args = parser.parse_args()

    train_records = build_train_records(seed=args.seed)
    eval_cases = build_eval_cases(seed=args.seed, eval_mode=args.eval_mode)
    train_path = Path(args.train_output)
    cases_path = Path(args.cases_output)
    manifest_path = Path(args.manifest_output)
    write_jsonl(train_path, train_records)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(json.dumps({"cases": eval_cases}, indent=2, sort_keys=True) + "\n")

    behavior_counts = Counter(str(row["behavior"]) for row in train_records)
    train_family_counts = Counter(str(row["slot_family"]) for row in train_records)
    eval_family_counts = Counter(str(row["slot_family"]) for row in eval_cases)
    eval_tier_counts = Counter(str(row["eval_tier"]) for row in eval_cases)
    manifest = {
        "format": "agentcoder-atomic-copy-sft-v1",
        "note": "Atomic no-decoy symbol/file copy supervision; not a benchmark dataset.",
        "seed": args.seed,
        "eval_mode": args.eval_mode,
        "train_output": str(train_path),
        "cases_output": str(cases_path),
        "train_records": len(train_records),
        "eval_cases": len(eval_cases),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "train_slot_family_counts": dict(sorted(train_family_counts.items())),
        "eval_slot_family_counts": dict(sorted(eval_family_counts.items())),
        "eval_tier_counts": dict(sorted(eval_tier_counts.items())),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
