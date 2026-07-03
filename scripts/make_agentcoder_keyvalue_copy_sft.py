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
TRAIN_VARIANTS_PER_ROW = 1
EVAL_CASES = 64
TARGET_FIELDS = 3
DISTRACTOR_FIELDS = 4
EVAL_MODES = ["mirror", "covered", "heldout", "ladder", "coverage_ladder"]
COMPLETION_MODES = ["keyvalue", "key_only"]

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


def spec_with_seen_values(index: int, value_index: int, seed: int) -> dict[str, Any]:
    """Use held-out prompt selection with values that appear in train records."""
    train_rng = random.Random(seed + value_index * 17)
    targets, distractors = choose_fields(spec(value_index), train_rng)
    return {"index": index, "value_index": value_index, "fields": [*targets, *distractors]}


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


def user_prompt(targets: list[dict[str, str]], completion_mode: str = "keyvalue") -> str:
    keys = ", ".join(field["key"] for field in targets)
    if completion_mode == "key_only":
        return f"List these repo_context keys exactly and in order: {keys}. Return only one key per line."
    return f"Copy these repo_context fields exactly and in order: {keys}. Return only key=value lines."


def assistant_text(targets: list[dict[str, str]], completion_mode: str = "keyvalue") -> str:
    if completion_mode == "key_only":
        return "\n".join(field["key"] for field in targets)
    return "\n".join(f"{field['key']}={field['value']}" for field in targets)


def chat_prompt(
    targets: list[dict[str, str]],
    distractors: list[dict[str, str]],
    rng: random.Random,
    completion_mode: str = "keyvalue",
) -> str:
    response_instruction = (
        "Return only exact key names, one per line."
        if completion_mode == "key_only"
        else "Return only exact key=value lines."
    )
    return (
        "<|system|>\n"
        f"{SYSTEM} {response_instruction}\n\n"
        "<|repo_context|>\n"
        f"{repo_context(targets, distractors, rng)}\n\n"
        "<|user|>\n"
        f"{user_prompt(targets, completion_mode=completion_mode)}\n\n"
        "<|assistant|>\n"
    )


def expected_slots(targets: list[dict[str, str]]) -> dict[str, str]:
    return {field["key"]: field["value"] for field in targets}


def forbidden_substrings(distractors: list[dict[str, str]]) -> list[str]:
    return [f"{field['key']}={field['value']}" for field in distractors]


def variant_seed(seed: int, row_index: int, variant_index: int = 0) -> int:
    return seed + row_index * 17 + variant_index * 1009


def record(
    row: dict[str, Any],
    seed: int,
    variant_index: int = 0,
    completion_mode: str = "keyvalue",
) -> dict[str, Any]:
    if completion_mode not in COMPLETION_MODES:
        raise ValueError(f"completion_mode must be one of {', '.join(COMPLETION_MODES)}")
    row_index = int(row["index"])
    rng = random.Random(variant_seed(seed, row_index, variant_index))
    targets, distractors = choose_fields(row, rng)
    response_instruction = (
        "Return only exact key names, one per line."
        if completion_mode == "key_only"
        else "Return only exact key=value lines."
    )
    expected_behavior = "copy_key_sequence" if completion_mode == "key_only" else "copy_slot_values"
    slot_family = "keyvalue_repo_key_only" if completion_mode == "key_only" else "keyvalue_repo_copy"
    return {
        "behavior": expected_behavior,
        "slot_family": slot_family,
        "source_row_index": row_index,
        "train_variant_index": variant_index,
        "target_keys": [field["key"] for field in targets],
        "expected_slots": expected_slots(targets),
        "system": f"{SYSTEM} {response_instruction}",
        "repo_context": repo_context(targets, distractors, rng),
        "messages": [{"role": "user", "content": user_prompt(targets, completion_mode=completion_mode)}],
        "trace": [{"type": "assistant", "content": assistant_text(targets, completion_mode=completion_mode)}],
    }


def case(
    name: str,
    row: dict[str, Any],
    eval_tier: str,
    seed: int,
    completion_mode: str = "keyvalue",
) -> dict[str, Any]:
    if completion_mode not in COMPLETION_MODES:
        raise ValueError(f"completion_mode must be one of {', '.join(COMPLETION_MODES)}")
    rng = random.Random(seed + int(row["index"]) * 17)
    targets, distractors = choose_fields(row, rng)
    expected_behavior = "copy_key_sequence" if completion_mode == "key_only" else "copy_slot_values"
    slot_family = "keyvalue_repo_key_only" if completion_mode == "key_only" else "keyvalue_repo_copy"
    required_substrings = (
        [field["key"] for field in targets]
        if completion_mode == "key_only"
        else [f"{field['key']}={field['value']}" for field in targets]
    )
    forbidden = (
        [field["key"] for field in distractors]
        if completion_mode == "key_only"
        else forbidden_substrings(distractors)
    )
    return {
        "name": name,
        "prompt": chat_prompt(targets, distractors, rng, completion_mode=completion_mode),
        "required_substrings": required_substrings,
        "forbidden_substrings": forbidden,
        "expected_behavior": expected_behavior,
        "slot_family": slot_family,
        "expected_slots": expected_slots(targets),
        "expected_key_sequence": [field["key"] for field in targets],
        "enforce_key_sequence": True,
        "eval_tier": eval_tier,
        "value_formats": sorted({field["value_format"] for field in targets}),
        "target_keys": [field["key"] for field in targets],
    }


def build_train_records(
    seed: int = DEFAULT_SEED,
    train_records: int = TRAIN_RECORDS,
    train_variants_per_row: int = TRAIN_VARIANTS_PER_ROW,
    completion_mode: str = "keyvalue",
) -> list[dict[str, Any]]:
    if train_records < 1:
        raise ValueError("train_records must be positive")
    if train_variants_per_row < 1:
        raise ValueError("train_variants_per_row must be positive")
    if completion_mode not in COMPLETION_MODES:
        raise ValueError(f"completion_mode must be one of {', '.join(COMPLETION_MODES)}")
    records: list[dict[str, Any]] = []
    for index in range(train_records):
        row = spec(index)
        for variant_index in range(train_variants_per_row):
            records.append(record(row, seed=seed, variant_index=variant_index, completion_mode=completion_mode))
    return records


def build_eval_cases(
    seed: int = DEFAULT_SEED,
    eval_mode: str = "ladder",
    train_records: int = TRAIN_RECORDS,
    eval_cases: int = EVAL_CASES,
    completion_mode: str = "keyvalue",
) -> list[dict[str, Any]]:
    if eval_mode not in EVAL_MODES:
        raise ValueError(f"eval_mode must be one of {', '.join(EVAL_MODES)}")
    if train_records < 1:
        raise ValueError("train_records must be positive")
    if eval_cases < 1:
        raise ValueError("eval_cases must be positive")
    if completion_mode not in COMPLETION_MODES:
        raise ValueError(f"completion_mode must be one of {', '.join(COMPLETION_MODES)}")
    cases: list[dict[str, Any]] = []
    if eval_mode in {"mirror", "ladder", "coverage_ladder"}:
        cases.extend(
            case(
                f"keyvalue_seen_{index:03d}",
                spec(index % train_records),
                "seen_slot",
                seed=seed,
                completion_mode=completion_mode,
            )
            for index in range(eval_cases)
        )
    if eval_mode in {"covered", "coverage_ladder"}:
        cases.extend(
            case(
                f"keyvalue_covered_{index:03d}",
                spec_with_seen_values(train_records + index, index % train_records, seed=seed),
                "covered_value_slot",
                seed=seed,
                completion_mode=completion_mode,
            )
            for index in range(eval_cases)
        )
    if eval_mode in {"heldout", "ladder", "coverage_ladder"}:
        cases.extend(
            case(
                f"keyvalue_heldout_{index:03d}",
                spec(train_records + index),
                "heldout_slot",
                seed=seed,
                completion_mode=completion_mode,
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
    parser.add_argument("--eval-mode", choices=EVAL_MODES, default="ladder")
    parser.add_argument("--completion-mode", choices=COMPLETION_MODES, default="keyvalue")
    parser.add_argument("--train-records", type=int, default=TRAIN_RECORDS)
    parser.add_argument("--train-variants-per-row", type=int, default=TRAIN_VARIANTS_PER_ROW)
    parser.add_argument("--eval-cases", type=int, default=EVAL_CASES)
    args = parser.parse_args()

    train_records = build_train_records(
        seed=args.seed,
        train_records=args.train_records,
        train_variants_per_row=args.train_variants_per_row,
        completion_mode=args.completion_mode,
    )
    eval_cases = build_eval_cases(
        seed=args.seed,
        eval_mode=args.eval_mode,
        train_records=args.train_records,
        eval_cases=args.eval_cases,
        completion_mode=args.completion_mode,
    )
    train_path = Path(args.train_output)
    cases_path = Path(args.cases_output)
    manifest_path = Path(args.manifest_output)
    write_jsonl(train_path, train_records)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(json.dumps({"cases": eval_cases}, indent=2, sort_keys=True) + "\n")

    behavior_counts = Counter(str(row["behavior"]) for row in train_records)
    train_family_counts = Counter(str(row["slot_family"]) for row in train_records)
    train_variant_counts = Counter(str(row.get("train_variant_index", 0)) for row in train_records)
    train_source_row_counts = Counter(str(row.get("source_row_index", "")) for row in train_records)
    eval_family_counts = Counter(str(row["slot_family"]) for row in eval_cases)
    eval_tier_counts = Counter(str(row["eval_tier"]) for row in eval_cases)
    key_counts = Counter(key for row in eval_cases for key in row["target_keys"])
    format_counts = Counter(fmt for row in eval_cases for fmt in row["value_formats"])
    manifest = {
        "format": "agentcoder-keyvalue-copy-sft-v1",
        "note": "Structured synthetic repo_context key-value copy task; not a benchmark dataset.",
        "seed": args.seed,
        "eval_mode": args.eval_mode,
        "completion_mode": args.completion_mode,
        "train_output": str(train_path),
        "cases_output": str(cases_path),
        "train_records": len(train_records),
        "eval_cases": len(eval_cases),
        "requested_train_records": args.train_records,
        "base_train_rows": args.train_records,
        "train_variants_per_row": args.train_variants_per_row,
        "requested_eval_cases": args.eval_cases,
        "target_fields": TARGET_FIELDS,
        "distractor_fields": DISTRACTOR_FIELDS,
        "available_keys": KEYS,
        "value_formats": VALUE_FORMATS,
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "train_slot_family_counts": dict(sorted(train_family_counts.items())),
        "train_variant_counts": dict(sorted(train_variant_counts.items())),
        "train_source_row_count": len(train_source_row_counts),
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
