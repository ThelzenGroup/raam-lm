#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any


SYSTEM = "You are RAAM-AgentCoder, a precise repo-context key-value copying assistant."
DEFAULT_SEED = 17
TRAIN_RECORDS = 96
EVAL_CASES = 64
TARGET_FIELDS = 3
DISTRACTOR_FIELDS = 4

KEYS = [
    "symbol",
    "file",
    "helper",
    "module",
    "route",
    "fixture",
    "setting",
    "adapter",
    "service",
    "endpoint",
    "task",
    "parser",
]
VALUE_FORMATS = ["identifier", "filename", "flag", "caseid"]
FLAGS = ["enabled", "disabled", "visible", "hidden", "allowed", "blocked"]


def value_for(index: int, key: str, value_format: str) -> str:
    stem = f"copy_{key}_{index:03d}"
    if value_format == "identifier":
        return stem
    if value_format == "filename":
        return f"{stem}.py"
    if value_format == "flag":
        return f"{FLAGS[index % len(FLAGS)]}_{key}_{index:03d}"
    if value_format == "caseid":
        return f"case_{key}_{index:03d}"
    raise ValueError(f"unknown value format: {value_format}")


def spec(index: int) -> dict[str, Any]:
    fields: list[dict[str, str]] = []
    for key_offset, key in enumerate(KEYS):
        value_format = VALUE_FORMATS[(index + key_offset) % len(VALUE_FORMATS)]
        fields.append(
            {
                "key": key,
                "value": value_for(index, key, value_format),
                "value_format": value_format,
            }
        )
    return {"index": index, "fields": fields}


def choose_fields(row: dict[str, Any], rng: random.Random) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    fields = list(row["fields"])
    targets = rng.sample(fields, TARGET_FIELDS)
    remaining = [field for field in fields if field not in targets]
    distractors = rng.sample(remaining, DISTRACTOR_FIELDS)
    return targets, distractors


def repo_context(targets: list[dict[str, str]], distractors: list[dict[str, str]], rng: random.Random) -> str:
    rows = [*targets, *distractors]
    rng.shuffle(rows)
    return "\n".join(f"{field['key']}: {field['value']}" for field in rows)


def user_prompt(targets: list[dict[str, str]]) -> str:
    keys = ", ".join(field["key"] for field in targets)
    return f"Copy these repo_context fields exactly and in order: {keys}. Return only key=value lines."


def assistant_text(targets: list[dict[str, str]]) -> str:
    return "\n".join(f"{field['key']}={field['value']}" for field in targets)


def chat_prompt(targets: list[dict[str, str]], distractors: list[dict[str, str]], rng: random.Random) -> str:
    return (
        "<|system|>\n"
        f"{SYSTEM} Return only exact key=value lines.\n\n"
        "<|repo_context|>\n"
        f"{repo_context(targets, distractors, rng)}\n\n"
        "<|user|>\n"
        f"{user_prompt(targets)}\n\n"
        "<|assistant|>\n"
    )


def expected_slots(targets: list[dict[str, str]]) -> dict[str, str]:
    return {field["key"]: field["value"] for field in targets}


def forbidden_substrings(distractors: list[dict[str, str]]) -> list[str]:
    return [f"{field['key']}={field['value']}" for field in distractors]


def record(row: dict[str, Any], seed: int) -> dict[str, Any]:
    rng = random.Random(seed + int(row["index"]) * 17)
    targets, distractors = choose_fields(row, rng)
    return {
        "behavior": "copy_slot_values",
        "slot_family": "keyvalue_repo_copy",
        "expected_slots": expected_slots(targets),
        "system": f"{SYSTEM} Return only exact key=value lines.",
        "repo_context": repo_context(targets, distractors, rng),
        "messages": [{"role": "user", "content": user_prompt(targets)}],
        "trace": [{"type": "assistant", "content": assistant_text(targets)}],
    }


def case(name: str, row: dict[str, Any], eval_tier: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed + int(row["index"]) * 17)
    targets, distractors = choose_fields(row, rng)
    return {
        "name": name,
        "prompt": chat_prompt(targets, distractors, rng),
        "required_substrings": [f"{field['key']}={field['value']}" for field in targets],
        "forbidden_substrings": forbidden_substrings(distractors),
        "expected_behavior": "copy_slot_values",
        "slot_family": "keyvalue_repo_copy",
        "expected_slots": expected_slots(targets),
        "eval_tier": eval_tier,
        "value_formats": sorted({field["value_format"] for field in targets}),
        "target_keys": [field["key"] for field in targets],
    }


def build_train_records(seed: int = DEFAULT_SEED, train_records: int = TRAIN_RECORDS) -> list[dict[str, Any]]:
    if train_records < 1:
        raise ValueError("train_records must be positive")
    return [record(spec(index), seed=seed) for index in range(train_records)]


def build_eval_cases(
    seed: int = DEFAULT_SEED,
    eval_mode: str = "ladder",
    train_records: int = TRAIN_RECORDS,
    eval_cases: int = EVAL_CASES,
) -> list[dict[str, Any]]:
    if eval_mode not in {"mirror", "heldout", "ladder"}:
        raise ValueError("eval_mode must be 'mirror', 'heldout', or 'ladder'")
    if train_records < 1:
        raise ValueError("train_records must be positive")
    if eval_cases < 1:
        raise ValueError("eval_cases must be positive")
    cases: list[dict[str, Any]] = []
    if eval_mode in {"mirror", "ladder"}:
        cases.extend(
            case(
                f"keyvalue_seen_{index:03d}",
                spec(index % train_records),
                "seen_slot",
                seed=seed,
            )
            for index in range(eval_cases)
        )
    if eval_mode in {"heldout", "ladder"}:
        cases.extend(
            case(
                f"keyvalue_heldout_{index:03d}",
                spec(train_records + index),
                "heldout_slot",
                seed=seed,
            )
            for index in range(eval_cases)
        )
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a structured key-value AgentCoder copy task.")
    parser.add_argument("--train-output", default="examples/agentcoder_keyvalue_copy_train.jsonl")
    parser.add_argument("--cases-output", default="examples/agentcoder_keyvalue_copy_cases.json")
    parser.add_argument("--manifest-output", default="examples/agentcoder_keyvalue_copy_manifest.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-mode", choices=["mirror", "heldout", "ladder"], default="ladder")
    parser.add_argument("--train-records", type=int, default=TRAIN_RECORDS)
    parser.add_argument("--eval-cases", type=int, default=EVAL_CASES)
    args = parser.parse_args()

    train_records = build_train_records(seed=args.seed, train_records=args.train_records)
    eval_cases = build_eval_cases(
        seed=args.seed,
        eval_mode=args.eval_mode,
        train_records=args.train_records,
        eval_cases=args.eval_cases,
    )
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
    key_counts = Counter(key for row in eval_cases for key in row["target_keys"])
    format_counts = Counter(fmt for row in eval_cases for fmt in row["value_formats"])
    manifest = {
        "format": "agentcoder-keyvalue-copy-sft-v1",
        "note": "Structured synthetic repo_context key-value copy task; not a benchmark dataset.",
        "seed": args.seed,
        "eval_mode": args.eval_mode,
        "train_output": str(train_path),
        "cases_output": str(cases_path),
        "train_records": len(train_records),
        "eval_cases": len(eval_cases),
        "requested_train_records": args.train_records,
        "requested_eval_cases": args.eval_cases,
        "target_fields": TARGET_FIELDS,
        "distractor_fields": DISTRACTOR_FIELDS,
        "available_keys": KEYS,
        "value_formats": VALUE_FORMATS,
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "train_slot_family_counts": dict(sorted(train_family_counts.items())),
        "eval_slot_family_counts": dict(sorted(eval_family_counts.items())),
        "eval_tier_counts": dict(sorted(eval_tier_counts.items())),
        "eval_target_key_counts": dict(sorted(key_counts.items())),
        "eval_value_format_counts": dict(sorted(format_counts.items())),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
