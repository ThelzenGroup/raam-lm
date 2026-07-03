#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, TypeVar

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.make_agentcoder_slotcopy_sft import (
    DEFAULT_SEED,
    EVAL_PER_FAMILY,
    SEEN_EVAL_PER_FAMILY,
    TRAIN_PER_FAMILY,
    arithmetic_specs,
    literal_specs,
    repo_specs,
    split_specs,
)


SYSTEM = "You are RAAM-AgentCoder, a concise software-engineering assistant."
T = TypeVar("T")


def record(
    slot_family: str,
    expected_slots: dict[str, str],
    user: str,
    assistant: str,
    *,
    repo_context: str,
) -> dict[str, Any]:
    return {
        "behavior": "copy_slot_values",
        "slot_family": slot_family,
        "expected_slots": expected_slots,
        "system": (
            f"{SYSTEM} Copy exact slot values from the current context. "
            "Return only the requested key=value lines."
        ),
        "repo_context": repo_context,
        "messages": [{"role": "user", "content": user}],
        "trace": [{"type": "assistant", "content": assistant}],
    }


def case(
    name: str,
    prompt: str,
    required: list[str],
    *,
    slot_family: str,
    expected_slots: dict[str, str],
    forbidden_substrings: list[str],
    eval_tier: str,
) -> dict[str, Any]:
    return {
        "name": name,
        "prompt": prompt,
        "required_substrings": required,
        "forbidden_substrings": forbidden_substrings,
        "expected_behavior": "copy_slot_values",
        "slot_family": slot_family,
        "expected_slots": expected_slots,
        "eval_tier": eval_tier,
    }


def chat_prompt(user: str, repo_context: str) -> str:
    return (
        "<|system|>\n"
        f"{SYSTEM} Copy exact slot values from the current context. Return only the requested key=value lines.\n\n"
        "<|repo_context|>\n"
        f"{repo_context}\n\n"
        "<|user|>\n"
        f"{user}\n\n"
        "<|assistant|>\n"
    )


def unique(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def sample_decoys(rng: random.Random, specs: list[T], target: T, count: int) -> list[T]:
    pool = [row for row in specs if row is not target]
    return rng.sample(pool, min(count, len(pool)))


def block(lines: list[str]) -> str:
    return "\n".join(lines)


def repo_lookup_context(
    rng: random.Random,
    target: dict[str, str],
    specs: list[dict[str, str]],
) -> tuple[str, list[str]]:
    decoys = sample_decoys(rng, specs, target, 3)
    entries = [
        block(
            [
                "entry: target",
                f"symbol: {target['symbol']}",
                f"file: {target['impl_file']}",
                f"definition: def {target['symbol']}(value):",
            ]
        )
    ]
    forbidden: list[str] = []
    for idx, decoy in enumerate(decoys, 1):
        entries.append(
            block(
                [
                    f"entry: decoy_{idx}",
                    f"symbol: {decoy['symbol']}",
                    f"file: {decoy['impl_file']}",
                    f"definition: def {decoy['symbol']}(value):",
                ]
            )
        )
        forbidden.extend([f"symbol={decoy['symbol']}", f"file={decoy['impl_file']}"])
    rng.shuffle(entries)
    return "\n---\n".join(entries), unique(forbidden)


def repo_lookup_user(symbol: str) -> str:
    return (
        f"Copy-only repo lookup for requested symbol `{symbol}`. "
        "Return exactly two lines: symbol=<requested symbol> and file=<defining file>."
    )


def repo_lookup_record(
    rng: random.Random,
    target: dict[str, str],
    specs: list[dict[str, str]],
) -> dict[str, Any]:
    repo_context, _ = repo_lookup_context(rng, target, specs)
    expected = {"symbol": target["symbol"], "file": target["impl_file"]}
    assistant = f"symbol={target['symbol']}\nfile={target['impl_file']}"
    return record("repo_lookup_copy", expected, repo_lookup_user(target["symbol"]), assistant, repo_context=repo_context)


def repo_lookup_case(
    rng: random.Random,
    index: int,
    target: dict[str, str],
    specs: list[dict[str, str]],
    *,
    eval_tier: str,
) -> dict[str, Any]:
    repo_context, forbidden = repo_lookup_context(rng, target, specs)
    expected = {"symbol": target["symbol"], "file": target["impl_file"]}
    return case(
        f"copy_{eval_tier}_repo_lookup_{index:02d}_{target['symbol']}",
        chat_prompt(repo_lookup_user(target["symbol"]), repo_context),
        [f"symbol={target['symbol']}", f"file={target['impl_file']}"],
        slot_family="repo_lookup_copy",
        expected_slots=expected,
        forbidden_substrings=forbidden,
        eval_tier=eval_tier,
    )


def arithmetic_context(
    rng: random.Random,
    target: dict[str, str],
    specs: list[dict[str, str]],
) -> tuple[str, list[str]]:
    decoys = sample_decoys(rng, specs, target, 3)
    entries = [
        block(
            [
                "entry: target",
                f"file: {target['path']}",
                f"helper: {target['helper']}",
                f"return: {target['good_expr']}",
                f"test: {target['test_path']}",
            ]
        )
    ]
    forbidden: list[str] = []
    for idx, decoy in enumerate(decoys, 1):
        entries.append(
            block(
                [
                    f"entry: decoy_{idx}",
                    f"file: {decoy['path']}",
                    f"helper: {decoy['helper']}",
                    f"return: {decoy['good_expr']}",
                    f"test: {decoy['test_path']}",
                ]
            )
        )
        forbidden.extend(
            [
                f"file={decoy['path']}",
                f"helper={decoy['helper']}",
                f"return={decoy['good_expr']}",
                f"test={decoy['test_path']}",
            ]
        )
    rng.shuffle(entries)
    return "\n---\n".join(entries), unique(forbidden)


def arithmetic_user(path: str, helper: str) -> str:
    return (
        f"Copy-only arithmetic slot task for `{path}` and `{helper}`. "
        "Return exactly four lines: file=<file>, helper=<helper>, return=<return expression>, test=<test path>."
    )


def arithmetic_record(
    rng: random.Random,
    target: dict[str, str],
    specs: list[dict[str, str]],
) -> dict[str, Any]:
    repo_context, _ = arithmetic_context(rng, target, specs)
    expected = {
        "file": target["path"],
        "helper": target["helper"],
        "return": target["good_expr"],
        "test": target["test_path"],
    }
    assistant = (
        f"file={target['path']}\n"
        f"helper={target['helper']}\n"
        f"return={target['good_expr']}\n"
        f"test={target['test_path']}"
    )
    return record(
        "patch_return_copy",
        expected,
        arithmetic_user(target["path"], target["helper"]),
        assistant,
        repo_context=repo_context,
    )


def arithmetic_case(
    rng: random.Random,
    index: int,
    target: dict[str, str],
    specs: list[dict[str, str]],
    *,
    eval_tier: str,
) -> dict[str, Any]:
    repo_context, forbidden = arithmetic_context(rng, target, specs)
    expected = {
        "file": target["path"],
        "helper": target["helper"],
        "return": target["good_expr"],
        "test": target["test_path"],
    }
    return case(
        f"copy_{eval_tier}_patch_return_{index:02d}_{target['helper']}",
        chat_prompt(arithmetic_user(target["path"], target["helper"]), repo_context),
        [
            f"file={target['path']}",
            f"helper={target['helper']}",
            f"return={target['good_expr']}",
            f"test={target['test_path']}",
        ],
        slot_family="patch_return_copy",
        expected_slots=expected,
        forbidden_substrings=forbidden,
        eval_tier=eval_tier,
    )


def literal_context(
    rng: random.Random,
    target: dict[str, str],
    specs: list[dict[str, str]],
) -> tuple[str, list[str]]:
    decoys = sample_decoys(rng, specs, target, 3)
    entries = [
        block(
            [
                "entry: target",
                f"file: {target['path']}",
                f"helper: {target['helper']}",
                f"literal: {target['new_literal']}",
                f"test: {target['test_path']}",
            ]
        )
    ]
    forbidden: list[str] = []
    for idx, decoy in enumerate(decoys, 1):
        entries.append(
            block(
                [
                    f"entry: decoy_{idx}",
                    f"file: {decoy['path']}",
                    f"helper: {decoy['helper']}",
                    f"literal: {decoy['new_literal']}",
                    f"test: {decoy['test_path']}",
                ]
            )
        )
        forbidden.extend(
            [
                f"file={decoy['path']}",
                f"helper={decoy['helper']}",
                f"literal={decoy['new_literal']}",
                f"test={decoy['test_path']}",
            ]
        )
    rng.shuffle(entries)
    return "\n---\n".join(entries), unique(forbidden)


def literal_user(path: str, helper: str) -> str:
    return (
        f"Copy-only literal slot task for `{path}` and `{helper}`. "
        "Return exactly four lines: file=<file>, helper=<helper>, literal=<literal>, test=<test path>."
    )


def literal_record(
    rng: random.Random,
    target: dict[str, str],
    specs: list[dict[str, str]],
) -> dict[str, Any]:
    repo_context, _ = literal_context(rng, target, specs)
    expected = {
        "file": target["path"],
        "helper": target["helper"],
        "literal": target["new_literal"],
        "test": target["test_path"],
    }
    assistant = (
        f"file={target['path']}\n"
        f"helper={target['helper']}\n"
        f"literal={target['new_literal']}\n"
        f"test={target['test_path']}"
    )
    return record(
        "patch_literal_copy",
        expected,
        literal_user(target["path"], target["helper"]),
        assistant,
        repo_context=repo_context,
    )


def literal_case(
    rng: random.Random,
    index: int,
    target: dict[str, str],
    specs: list[dict[str, str]],
    *,
    eval_tier: str,
) -> dict[str, Any]:
    repo_context, forbidden = literal_context(rng, target, specs)
    expected = {
        "file": target["path"],
        "helper": target["helper"],
        "literal": target["new_literal"],
        "test": target["test_path"],
    }
    return case(
        f"copy_{eval_tier}_patch_literal_{index:02d}_{target['helper']}",
        chat_prompt(literal_user(target["path"], target["helper"]), repo_context),
        [
            f"file={target['path']}",
            f"helper={target['helper']}",
            f"literal={target['new_literal']}",
            f"test={target['test_path']}",
        ],
        slot_family="patch_literal_copy",
        expected_slots=expected,
        forbidden_substrings=forbidden,
        eval_tier=eval_tier,
    )


def build_train_records(seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    repo_train, _ = split_specs(repo_specs(), seed + 1)
    arithmetic_train, _ = split_specs(arithmetic_specs(), seed + 2)
    literal_train, _ = split_specs(literal_specs(), seed + 3)
    rows: list[dict[str, Any]] = []
    rows.extend(repo_lookup_record(rng, target, repo_train) for target in repo_train)
    rows.extend(arithmetic_record(rng, target, arithmetic_train) for target in arithmetic_train)
    rows.extend(literal_record(rng, target, literal_train) for target in literal_train)
    return rows


def build_eval_cases(seed: int = DEFAULT_SEED, eval_mode: str = "ladder") -> list[dict[str, Any]]:
    if eval_mode not in {"heldout", "ladder"}:
        raise ValueError("eval_mode must be 'heldout' or 'ladder'")
    rng = random.Random(seed + 2000)
    repo_train, repo_eval = split_specs(repo_specs(), seed + 1)
    arithmetic_train, arithmetic_eval = split_specs(arithmetic_specs(), seed + 2)
    literal_train, literal_eval = split_specs(literal_specs(), seed + 3)
    all_repo = repo_train + repo_eval
    all_arithmetic = arithmetic_train + arithmetic_eval
    all_literal = literal_train + literal_eval
    cases: list[dict[str, Any]] = []
    if eval_mode == "ladder":
        cases.extend(
            repo_lookup_case(rng, i, target, all_repo, eval_tier="seen_slot")
            for i, target in enumerate(repo_train[:SEEN_EVAL_PER_FAMILY])
        )
        cases.extend(
            arithmetic_case(rng, i, target, all_arithmetic, eval_tier="seen_slot")
            for i, target in enumerate(arithmetic_train[:SEEN_EVAL_PER_FAMILY])
        )
        cases.extend(
            literal_case(rng, i, target, all_literal, eval_tier="seen_slot")
            for i, target in enumerate(literal_train[:SEEN_EVAL_PER_FAMILY])
        )
    cases.extend(
        repo_lookup_case(rng, i, target, all_repo, eval_tier="heldout_slot")
        for i, target in enumerate(repo_eval)
    )
    cases.extend(
        arithmetic_case(rng, i, target, all_arithmetic, eval_tier="heldout_slot")
        for i, target in enumerate(arithmetic_eval)
    )
    cases.extend(
        literal_case(rng, i, target, all_literal, eval_tier="heldout_slot")
        for i, target in enumerate(literal_eval)
    )
    return cases


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate copy-only AgentCoder slot binding data.")
    parser.add_argument("--train-output", default="examples/agentcoder_copyonly_sft_train.jsonl")
    parser.add_argument("--cases-output", default="examples/agentcoder_copyonly_eval_cases.json")
    parser.add_argument("--manifest-output", default="examples/agentcoder_copyonly_manifest.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-mode", choices=["heldout", "ladder"], default="ladder")
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
    eval_tier_counts = Counter(str(row.get("eval_tier", "heldout_slot")) for row in eval_cases)
    manifest = {
        "format": "agentcoder-copyonly-sft-v1",
        "note": "Deterministic synthetic copy-only slot binding supervision; not a benchmark dataset.",
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
        "train_per_family": TRAIN_PER_FAMILY,
        "eval_per_family": EVAL_PER_FAMILY,
        "seen_eval_per_family": SEEN_EVAL_PER_FAMILY if args.eval_mode == "ladder" else 0,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
