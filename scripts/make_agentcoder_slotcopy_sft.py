#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
from typing import Any, TypeVar


SYSTEM = "You are RAAM-AgentCoder, a concise software-engineering assistant."
DEFAULT_SEED = 17
TRAIN_PER_FAMILY = 48
EVAL_PER_FAMILY = 16
SEEN_EVAL_PER_FAMILY = 16
T = TypeVar("T")


def record(
    behavior: str,
    slot_family: str,
    expected_slots: dict[str, str],
    user: str,
    assistant: str,
    final: str,
    *,
    system_suffix: str,
    repo_context: str,
) -> dict[str, Any]:
    return {
        "behavior": behavior,
        "slot_family": slot_family,
        "expected_slots": expected_slots,
        "system": f"{SYSTEM} {system_suffix}".strip(),
        "repo_context": repo_context,
        "messages": [{"role": "user", "content": user}],
        "trace": [{"type": "assistant", "content": assistant}],
        "final": final,
    }


def case(
    name: str,
    prompt: str,
    required: list[str],
    *,
    expected_behavior: str,
    slot_family: str,
    expected_slots: dict[str, str],
    forbidden_substrings: list[str],
    eval_tier: str = "heldout_slot",
) -> dict[str, Any]:
    return {
        "name": name,
        "prompt": prompt,
        "required_substrings": required,
        "forbidden_substrings": forbidden_substrings,
        "expected_behavior": expected_behavior,
        "slot_family": slot_family,
        "expected_slots": expected_slots,
        "eval_tier": eval_tier,
    }


def chat_prompt(system_suffix: str, user: str, repo_context: str) -> str:
    return (
        f"<|system|>\n{SYSTEM} {system_suffix}".strip()
        + "\n\n<|repo_context|>\n"
        + repo_context
        + "\n\n<|user|>\n"
        + user
        + "\n\n<|assistant|>\n"
    )


def code_file(path: str, code: str) -> tuple[str, str]:
    return path, code


def format_repo(files: list[tuple[str, str]]) -> str:
    return "\n".join(f"file: {path}\n```python\n{code}\n```" for path, code in files)


def patch_text(path: str, before: str, after: str, header: str) -> str:
    return (
        f"{header}\n"
        "```diff\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@\n"
        f"-{before}\n"
        f"+{after}\n"
        "```"
    )


def unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


COMMON_REPO_DECOYS = [
    ("calc.py", "def add(a, b):\n    return a + b", "add"),
    ("names.py", "def slugify(text):\n    return text.lower().replace(' ', '-')", "slugify"),
    ("config.py", "def parse_port(value):\n    return int(value)", "parse_port"),
    ("cli_parser.py", "def build_parser():\n    return object()", "build_parser"),
]


def repo_specs() -> list[dict[str, str]]:
    verbs = [
        "normalize",
        "format",
        "clean",
        "parse",
        "load",
        "render",
        "validate",
        "resolve",
        "encode",
        "decode",
        "extract",
        "merge",
    ]
    nouns = [
        "title",
        "path",
        "email",
        "user",
        "invoice",
        "header",
        "slug",
        "token",
        "config",
        "payload",
        "summary",
        "route",
    ]
    specs: list[dict[str, str]] = []
    for index, noun in enumerate(nouns):
        for verb in verbs:
            symbol = f"{verb}_{noun}"
            module = f"{noun}_{verb}_{index:02d}"
            specs.append(
                {
                    "symbol": symbol,
                    "source_file": f"use_{noun}_{verb}_{index:02d}.py",
                    "source_code": f"from {module} import {symbol}\nRESULT = {symbol}(' sample ')",
                    "impl_file": f"{module}.py",
                    "impl_code": f"def {symbol}(value):\n    return value.strip()",
                }
            )
    return specs


def arithmetic_specs() -> list[dict[str, str]]:
    helpers = [
        "add",
        "combine",
        "merge",
        "sum",
        "accumulate",
        "increase",
        "join",
        "append",
        "total",
        "rollup",
        "blend",
        "apply",
    ]
    nouns = [
        "count",
        "score",
        "balance",
        "quota",
        "points",
        "credits",
        "items",
        "total",
        "amount",
        "stock",
        "usage",
        "capacity",
    ]
    arg_patterns = [
        ("left, right", "left - right", "left + right"),
        ("total, delta", "total - delta", "total + delta"),
        ("current, increment", "current - increment", "current + increment"),
        ("base, extra", "base - extra", "base + extra"),
    ]
    specs: list[dict[str, str]] = []
    for index, noun in enumerate(nouns):
        for helper_prefix in helpers:
            args, bad_expr, good_expr = arg_patterns[(index + len(helper_prefix)) % len(arg_patterns)]
            helper = f"{helper_prefix}_{noun}"
            path = f"{noun}_{helper_prefix}_{index:02d}.py"
            before = f"def {helper}({args}):\n    return {bad_expr}"
            after = f"def {helper}({args}):\n    return {good_expr}"
            specs.append(
                {
                    "path": path,
                    "helper": helper,
                    "args": args,
                    "bad_expr": bad_expr,
                    "good_expr": good_expr,
                    "before": before,
                    "after": after,
                    "test_path": f"tests/test_{noun}_{helper_prefix}_{index:02d}.py",
                }
            )
    return specs


def literal_specs() -> list[dict[str, str]]:
    nouns = [
        "cache",
        "signup",
        "billing",
        "delete",
        "debug",
        "export",
        "import",
        "search",
        "notify",
        "archive",
        "sync",
        "deploy",
    ]
    helpers = [
        "enabled",
        "allowed",
        "active",
        "visible",
        "ready",
        "permitted",
        "open",
        "public",
    ]
    literals = [
        ("off", "on"),
        ("false", "true"),
        ("disabled", "enabled"),
        ("blocked", "allowed"),
        ("closed", "open"),
        ("hidden", "visible"),
    ]
    specs: list[dict[str, str]] = []
    for index, noun in enumerate(nouns):
        for helper_suffix in helpers:
            old_literal, new_literal = literals[(index + len(helper_suffix)) % len(literals)]
            helper = f"{noun}_{helper_suffix}"
            path = f"{noun}_{helper_suffix}_{index:02d}.py"
            before = f"def {helper}(value):\n    return value == '{old_literal}'"
            after = f"def {helper}(value):\n    return value == '{new_literal}'"
            specs.append(
                {
                    "path": path,
                    "helper": helper,
                    "old_literal": old_literal,
                    "new_literal": new_literal,
                    "before": before,
                    "after": after,
                    "test_path": f"tests/test_{noun}_{helper_suffix}_{index:02d}.py",
                }
            )
    return specs


def sample_decoys(rng: random.Random, specs: list[T], target: T, count: int) -> list[T]:
    pool = [row for row in specs if row is not target]
    return rng.sample(pool, count)


def repo_context_for_repo_lookup(rng: random.Random, target: dict[str, str], specs: list[dict[str, str]]) -> tuple[str, list[str]]:
    spec_decoys = sample_decoys(rng, specs, target, 3)
    files = [
        code_file(target["source_file"], target["source_code"]),
        code_file(target["impl_file"], target["impl_code"]),
    ]
    forbidden: list[str] = []
    for path, code, symbol in COMMON_REPO_DECOYS:
        files.append(code_file(path, code))
        forbidden.extend([symbol, path])
    for decoy in spec_decoys:
        files.append(code_file(decoy["impl_file"], decoy["impl_code"]))
        forbidden.extend([decoy["symbol"], decoy["impl_file"]])
    rng.shuffle(files)
    return format_repo(files), unique(forbidden)


def repo_lookup_prompt(symbol: str) -> str:
    return (
        f"Repo lookup task. Requested symbol: {symbol}. Read repo_context. "
        f"Find `def {symbol}`. Start the answer with the exact requested symbol `{symbol}`. "
        "Ignore unrelated definitions even if they look familiar."
    )


def repo_lookup_record(rng: random.Random, target: dict[str, str], specs: list[dict[str, str]]) -> dict[str, Any]:
    repo_context, _ = repo_context_for_repo_lookup(rng, target, specs)
    symbol = target["symbol"]
    impl_file = target["impl_file"]
    expected_slots = {"symbol": symbol, "file": impl_file}
    return record(
        "repo_context_lookup",
        "repo_lookup",
        expected_slots,
        repo_lookup_prompt(symbol),
        f"{symbol} is implemented in {impl_file}. That file contains def {symbol}.",
        f"The implementation is in {impl_file}.",
        system_suffix=(
            "Use repo context when it is provided. Copy the exact requested symbol and defining file. "
            "Do not reuse symbols or files from earlier examples."
        ),
        repo_context=repo_context,
    )


def repo_lookup_case(
    rng: random.Random,
    index: int,
    target: dict[str, str],
    specs: list[dict[str, str]],
    *,
    eval_tier: str = "heldout_slot",
) -> dict[str, Any]:
    repo_context, forbidden = repo_context_for_repo_lookup(rng, target, specs)
    symbol = target["symbol"]
    impl_file = target["impl_file"]
    prompt = chat_prompt(
        (
            "Use repo context when it is provided. Copy the exact requested symbol and defining file. "
            "Do not reuse symbols or files from earlier examples."
        ),
        repo_lookup_prompt(symbol),
        repo_context,
    )
    return case(
        f"slot_{eval_tier}_repo_lookup_{index:02d}_{symbol}",
        prompt,
        [f"{symbol} is implemented in {impl_file}", f"def {symbol}"],
        expected_behavior="repo_context_lookup",
        slot_family="repo_lookup",
        expected_slots={"symbol": symbol, "file": impl_file},
        forbidden_substrings=forbidden,
        eval_tier=eval_tier,
    )


def patch_context_for_arithmetic(
    rng: random.Random,
    target: dict[str, str],
    specs: list[dict[str, str]],
) -> tuple[str, list[str]]:
    decoys = sample_decoys(rng, specs, target, 3)
    files = [code_file(target["path"], target["before"])]
    forbidden: list[str] = []
    for decoy in decoys:
        files.append(code_file(decoy["path"], decoy["before"]))
        forbidden.extend([decoy["path"], decoy["helper"]])
    rng.shuffle(files)
    return format_repo(files), unique(forbidden)


def arithmetic_prompt(target: dict[str, str]) -> str:
    return (
        f"Patch task for `{target['path']}`. Copy the shown `{target['helper']}` function from repo_context. "
        f"Change `return {target['bad_expr']}` to `return {target['good_expr']}`. "
        "Emit the diff first, then the focused pytest command."
    )


def arithmetic_record(rng: random.Random, target: dict[str, str], specs: list[dict[str, str]]) -> dict[str, Any]:
    repo_context, _ = patch_context_for_arithmetic(rng, target, specs)
    expected_slots = {
        "file": target["path"],
        "helper": target["helper"],
        "return_expr": target["good_expr"],
        "test": target["test_path"],
    }
    assistant = (
        patch_text(target["path"], target["before"], target["after"], "Arithmetic slot-copy patch.")
        + f"\nTest command: `pytest {target['test_path']} -q`"
    )
    return record(
        "patch_addition",
        "patch_return",
        expected_slots,
        arithmetic_prompt(target),
        assistant,
        f"Patched {target['path']} and verified with pytest {target['test_path']} -q.",
        system_suffix=(
            "Patch the exact file and helper requested by the user. Copy the return expression from repo_context. "
            "Do not reuse another arithmetic patch example."
        ),
        repo_context=repo_context,
    )


def arithmetic_case(
    rng: random.Random,
    index: int,
    target: dict[str, str],
    specs: list[dict[str, str]],
    *,
    eval_tier: str = "heldout_slot",
) -> dict[str, Any]:
    repo_context, forbidden = patch_context_for_arithmetic(rng, target, specs)
    prompt = chat_prompt(
        (
            "Patch the exact file and helper requested by the user. Copy the return expression from repo_context. "
            "Do not reuse another arithmetic patch example."
        ),
        arithmetic_prompt(target),
        repo_context,
    )
    return case(
        f"slot_{eval_tier}_patch_return_{index:02d}_{target['helper']}",
        prompt,
        [
            f"--- a/{target['path']}",
            f"def {target['helper']}({target['args']}):",
            f"return {target['good_expr']}",
            f"pytest {target['test_path']} -q",
        ],
        expected_behavior="patch_addition",
        slot_family="patch_return",
        expected_slots={
            "file": target["path"],
            "helper": target["helper"],
            "return_expr": target["good_expr"],
            "test": target["test_path"],
        },
        forbidden_substrings=forbidden,
        eval_tier=eval_tier,
    )


def patch_context_for_literal(
    rng: random.Random,
    target: dict[str, str],
    specs: list[dict[str, str]],
) -> tuple[str, list[str]]:
    decoys = sample_decoys(rng, specs, target, 3)
    files = [code_file(target["path"], target["before"])]
    forbidden: list[str] = []
    for decoy in decoys:
        files.append(code_file(decoy["path"], decoy["before"]))
        forbidden.extend([decoy["path"], decoy["helper"], f"== '{decoy['new_literal']}'"])
    rng.shuffle(files)
    return format_repo(files), unique(forbidden)


def literal_prompt(target: dict[str, str]) -> str:
    return (
        f"Boolean flag slot-copy task for `{target['path']}`. Copy the shown `{target['helper']}` helper. "
        f"Change only the comparison literal from `{target['old_literal']}` to `{target['new_literal']}`. "
        "Emit the diff first, then the focused pytest command."
    )


def literal_record(rng: random.Random, target: dict[str, str], specs: list[dict[str, str]]) -> dict[str, Any]:
    repo_context, _ = patch_context_for_literal(rng, target, specs)
    expected_slots = {
        "file": target["path"],
        "helper": target["helper"],
        "literal": target["new_literal"],
        "test": target["test_path"],
    }
    assistant = (
        patch_text(target["path"], target["before"], target["after"], "Boolean flag slot-copy patch.")
        + f"\nTest command: `pytest {target['test_path']} -q`"
    )
    return record(
        "patch_boolean_flag",
        "patch_literal",
        expected_slots,
        literal_prompt(target),
        assistant,
        f"Patched {target['path']} and verified with pytest {target['test_path']} -q.",
        system_suffix=(
            "Boolean flag patch only. Copy the exact file, helper, and enabled literal from repo_context. "
            "Do not reuse another flag patch example."
        ),
        repo_context=repo_context,
    )


def literal_case(
    rng: random.Random,
    index: int,
    target: dict[str, str],
    specs: list[dict[str, str]],
    *,
    eval_tier: str = "heldout_slot",
) -> dict[str, Any]:
    repo_context, forbidden = patch_context_for_literal(rng, target, specs)
    prompt = chat_prompt(
        (
            "Boolean flag patch only. Copy the exact file, helper, and enabled literal from repo_context. "
            "Do not reuse another flag patch example."
        ),
        literal_prompt(target),
        repo_context,
    )
    return case(
        f"slot_{eval_tier}_patch_literal_{index:02d}_{target['helper']}",
        prompt,
        [
            f"--- a/{target['path']}",
            f"def {target['helper']}(value):",
            f"return value == '{target['new_literal']}'",
            f"pytest {target['test_path']} -q",
        ],
        expected_behavior="patch_boolean_flag",
        slot_family="patch_literal",
        expected_slots={
            "file": target["path"],
            "helper": target["helper"],
            "literal": target["new_literal"],
            "test": target["test_path"],
        },
        forbidden_substrings=forbidden,
        eval_tier=eval_tier,
    )


def split_specs(specs: list[T], seed: int) -> tuple[list[T], list[T]]:
    rng = random.Random(seed)
    shuffled = list(specs)
    rng.shuffle(shuffled)
    train = shuffled[:TRAIN_PER_FAMILY]
    eval_rows = shuffled[TRAIN_PER_FAMILY : TRAIN_PER_FAMILY + EVAL_PER_FAMILY]
    return train, eval_rows


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


def build_eval_cases(seed: int = DEFAULT_SEED, eval_mode: str = "heldout") -> list[dict[str, Any]]:
    if eval_mode not in {"heldout", "ladder"}:
        raise ValueError("eval_mode must be 'heldout' or 'ladder'")
    rng = random.Random(seed + 1000)
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
    parser = argparse.ArgumentParser(description="Generate a larger AgentCoder slot-copy SFT curriculum and held-out eval.")
    parser.add_argument("--train-output", default="examples/agentcoder_slotcopy_sft_train.jsonl")
    parser.add_argument("--cases-output", default="examples/agentcoder_slotcopy_eval_cases.json")
    parser.add_argument("--manifest-output", default="examples/agentcoder_slotcopy_manifest.json")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--eval-mode", choices=["heldout", "ladder"], default="heldout")
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
        "format": "agentcoder-slotcopy-sft-v1",
        "note": "Deterministic synthetic slot-copy supervision for repo-context diagnostics; not a benchmark dataset.",
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
