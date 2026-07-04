#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.eval_coding_ladder import score_case
from scripts.make_agentcoder_coding_ladder_sft import SYSTEM, chat_prompt, fenced_python, record
from scripts.make_agentcoder_medium_repair_sft import (
    ANCHOR_FLOOR_FUNCTIONS,
    ANCHOR_FUNCTIONS,
    FUNCTION_SPECS,
    expanded_eval_cases,
    forbid_family_contamination,
    function_case,
    rendered_prompt_for_row,
)


FORMAT = "agentcoder-medium-function-repair-v1"
STAGE = "medium_function_only"
DEFAULT_TARGET_FUNCTIONS = ["count_even", "safe_int", "parse_port"]
TARGET_FUNCTIONS = list(DEFAULT_TARGET_FUNCTIONS)
REQUIRED_TOPICS = [*DEFAULT_TARGET_FUNCTIONS, "anchor_tiny_function"]
FORBIDDEN_ANSWER_FRAGMENTS = [
    "```diff",
    "Test command:",
    '"cmd"',
    "pytest",
    "--- a/",
    "+++ b/",
    "<|assistant|>",
    "<|user|>",
    "<|system|>",
]
FUNCTION_SYSTEM_SUFFIX = (
    "Function-completion task only. Return one complete, valid Python function in a Python code block. "
    "Do not output patches, pytest tests, shell commands, JSON, explanations, or trailing text."
)


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def assistant_text(row: dict[str, Any]) -> str:
    trace = row.get("trace", [])
    if not trace:
        return ""
    return str(trace[0].get("content", ""))


def parse_target_functions(raw: str | None) -> list[str]:
    if raw is None:
        return list(DEFAULT_TARGET_FUNCTIONS)
    names = [item.strip() for item in raw.split(",") if item.strip()]
    return names or list(DEFAULT_TARGET_FUNCTIONS)


def required_topics_for_targets(target_functions: list[str]) -> list[str]:
    return [*target_functions, "anchor_tiny_function"]


def target_function_specs(target_functions: list[str] | None = None) -> list[dict[str, Any]]:
    target_functions = list(target_functions or DEFAULT_TARGET_FUNCTIONS)
    specs = [spec for spec in FUNCTION_SPECS if str(spec["name"]) in target_functions]
    found = {str(spec["name"]) for spec in specs}
    missing = sorted(set(target_functions) - found)
    if missing:
        raise ValueError(f"missing target function specs: {missing}")
    return specs


def function_prompts(spec: dict[str, Any]) -> list[str]:
    name = str(spec["name"])
    arglist = str(spec["arglist"])
    description = str(spec["description"])
    signature = f"def {name}({arglist}):"
    return [
        f"Implement `{name}`. {description} Return exactly one Python code block.",
        f"Complete this Python helper and stop after the code:\n```python\n{signature}\n```",
        f"Write valid Python for `{signature}`. No explanation or test code.",
        f"Fill in `{name}` so that it satisfies this behavior: {description}",
        f"Provide only the complete Python function `{signature}`.",
        f"Define `{name}` with the requested signature. {description}",
        f"Replace the missing body with a correct implementation:\n```python\n{signature}\n    pass\n```",
        f"Return a compact Python implementation of `{name}`; do not include prose.",
        f"Create the full function for `{name}` using signature `{signature}`.",
        f"Implement a robust `{name}` function. Behavior: {description}",
        f"Write one fenced Python block containing `{signature}` and its body.",
        f"Finish `{name}`. The answer should contain code only.",
    ]


def make_validation_case(spec: dict[str, Any]) -> dict[str, Any]:
    case = function_case(spec)
    forbid_family_contamination(case, "function_completion")
    forbidden = set(str(value) for value in case.get("forbidden_substrings", []))
    forbidden.update(FORBIDDEN_ANSWER_FRAGMENTS)
    case["forbidden_substrings"] = sorted(forbidden)
    case["max_completion_chars"] = 1100
    return case


def build_function_records(specs: list[dict[str, Any]], *, prompt_limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        prompts = function_prompts(spec)
        if prompt_limit is not None:
            prompts = prompts[:prompt_limit]
        case = make_validation_case(spec)
        for index, prompt in enumerate(prompts):
            row = record(
                str(spec["topic"]),
                "function_completion",
                prompt,
                fenced_python(str(spec["code"])),
                "",
                system_suffix=FUNCTION_SYSTEM_SUFFIX,
                source=f"function-repair:{spec['name']}:{index}",
            )
            row["_validation_case"] = case
            rows.append(row)
    return rows


def memorization_probe_cases(target_functions: list[str] | None = None) -> list[dict[str, Any]]:
    """Exact train-like probes used to diagnose generation after Stage A SFT.

    These prompts intentionally overlap the Stage A train prompt family. They are
    not promotion gates and must not be mixed into held-out medium eval scoring.
    """

    cases: list[dict[str, Any]] = []
    for spec in [*target_function_specs(target_functions), *ANCHOR_FUNCTIONS]:
        case = make_validation_case(spec)
        case.update(
            {
                "name": f"probe_exact_{spec['name']}",
                "topic": spec["topic"],
                "prompt": chat_prompt(FUNCTION_SYSTEM_SUFFIX, function_prompts(spec)[0]),
                "probe_kind": "exact_train_like_function_generation",
            }
        )
        cases.append(case)
    return cases


def build_base_records(*, target_prompt_limit: int = 12, anchor_prompt_limit: int = 8) -> list[dict[str, Any]]:
    return build_selected_base_records(
        target_prompt_limit=target_prompt_limit,
        anchor_prompt_limit=anchor_prompt_limit,
        target_functions=DEFAULT_TARGET_FUNCTIONS,
    )


def build_selected_base_records(
    *,
    target_prompt_limit: int = 12,
    anchor_prompt_limit: int = 8,
    target_functions: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(build_function_records(target_function_specs(target_functions), prompt_limit=target_prompt_limit))
    rows.extend(build_function_records(ANCHOR_FUNCTIONS, prompt_limit=anchor_prompt_limit))
    return rows


def validate_records(rows: list[dict[str, Any]], *, target_functions: list[str] | None = None) -> dict[str, Any]:
    target_functions = list(target_functions or DEFAULT_TARGET_FUNCTIONS)
    required_topics = set(required_topics_for_targets(target_functions))
    invalid: list[dict[str, Any]] = []
    topics = Counter()
    functions = Counter()
    anchors = Counter()
    behaviors = Counter()
    for index, row in enumerate(rows):
        source = str(row.get("_curriculum_source", ""))
        behavior = str(row.get("behavior", "unknown"))
        topic = str(row.get("topic", "unknown"))
        text = assistant_text(row)
        topics[topic] += 1
        behaviors[behavior] += 1

        if row.get("final") not in ("", None):
            invalid.append({"index": index, "source": source, "reason": "non_empty_final"})
            continue
        if behavior != "function_completion":
            invalid.append({"index": index, "source": source, "reason": f"bad_behavior:{behavior}"})
            continue
        if topic not in required_topics:
            invalid.append({"index": index, "source": source, "reason": f"bad_topic:{topic}"})
            continue
        present_forbidden = [fragment for fragment in FORBIDDEN_ANSWER_FRAGMENTS if fragment in text]
        if present_forbidden:
            invalid.append(
                {
                    "index": index,
                    "source": source,
                    "reason": "answer_family_contamination",
                    "present_forbidden": present_forbidden,
                }
            )
            continue

        case = row.get("_validation_case")
        if not isinstance(case, dict):
            invalid.append({"index": index, "source": source, "reason": "missing_validation_case"})
            continue
        if "expected_patch" in case or "expected_json" in case or case.get("python_syntax"):
            invalid.append({"index": index, "source": source, "reason": "non_function_validation_case"})
            continue
        scored = score_case(case, text)
        if not scored["passed"]:
            invalid.append(
                {
                    "index": index,
                    "source": source,
                    "reason": "score_case_failed",
                    "missing": scored.get("missing_required_substrings"),
                    "present_forbidden": scored.get("present_forbidden_substrings"),
                    "function_failures": scored.get("function_failures"),
                    "trailing_text": scored.get("trailing_text_after_code_block"),
                }
            )
            continue

        function_name = str(case.get("python_function", {}).get("name", "unknown"))
        functions[function_name] += 1
        if topic == "anchor_tiny_function":
            anchors[function_name] += 1

    missing_targets = [name for name in target_functions if functions[name] == 0]
    missing_anchors = [name for name in ANCHOR_FLOOR_FUNCTIONS if anchors[name] == 0]
    if missing_targets:
        invalid.append({"reason": "missing_target_functions", "missing": missing_targets})
    if missing_anchors:
        invalid.append({"reason": "missing_tiny_anchor_floor", "missing": missing_anchors})
    if invalid:
        raise ValueError(f"{len(invalid)} function-repair records failed validation: {invalid[:5]}")
    return {
        "validated_records": len(rows),
        "validated_topic_counts": dict(sorted(topics.items())),
        "validated_behavior_counts": dict(sorted(behaviors.items())),
        "function_counts": dict(sorted(functions.items())),
        "tiny_anchor_floor_counts": dict(sorted(anchors.items())),
    }


def expand_records(
    base_records: list[dict[str, Any]],
    *,
    target_repeats: int,
    anchor_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, item in enumerate(base_records):
        repeat_count = anchor_repeats if item.get("topic") == "anchor_tiny_function" else target_repeats
        for repeat in range(max(1, repeat_count)):
            row = dict(item)
            row["_function_repeat"] = repeat
            row["_function_base_index"] = index
            rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def assert_train_eval_disjoint(rows: list[dict[str, Any]], cases: list[dict[str, Any]]) -> list[str]:
    train_prompts = {rendered_prompt_for_row(row) for row in rows}
    eval_prompts = {str(case["prompt"]) for case in cases}
    overlaps = sorted(train_prompts & eval_prompts)
    if overlaps:
        raise ValueError(f"train/eval exact prompt overlap: {overlaps[:3]}")
    return overlaps


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, strip_validation: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_rows = []
    for row in rows:
        item = dict(row)
        if strip_validation:
            item.pop("_validation_case", None)
        clean_rows.append(item)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in clean_rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Stage A RAAM medium-function-only repair SFT data.")
    parser.add_argument("--train-output", default="examples/agentcoder_function_repair_train.jsonl")
    parser.add_argument("--cases-output", default="examples/agentcoder_function_repair_eval_cases.json")
    parser.add_argument("--probe-cases-output", default="examples/agentcoder_function_repair_probe_cases.json")
    parser.add_argument("--manifest-output", default="examples/agentcoder_function_repair_manifest.json")
    parser.add_argument("--target-repeats", type=int, default=30)
    parser.add_argument("--anchor-repeats", type=int, default=18)
    parser.add_argument("--target-prompt-limit", type=int, default=12)
    parser.add_argument("--anchor-prompt-limit", type=int, default=8)
    parser.add_argument(
        "--target-functions",
        default=",".join(DEFAULT_TARGET_FUNCTIONS),
        help="Comma-separated target function names to include before anchors.",
    )
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--keep-validation-fields", action="store_true")
    args = parser.parse_args()

    selected_targets = parse_target_functions(args.target_functions)
    required_topics = required_topics_for_targets(selected_targets)
    base_records = build_selected_base_records(
        target_prompt_limit=args.target_prompt_limit,
        anchor_prompt_limit=args.anchor_prompt_limit,
        target_functions=selected_targets,
    )
    rows = expand_records(
        base_records,
        target_repeats=args.target_repeats,
        anchor_repeats=args.anchor_repeats,
        seed=args.seed,
    )
    validation = validate_records(rows, target_functions=selected_targets)
    cases = expanded_eval_cases()
    probe_cases = memorization_probe_cases(selected_targets)
    exact_prompt_overlaps = assert_train_eval_disjoint(rows, cases)

    topics = {str(row.get("topic")) for row in rows}
    missing_topics = sorted(set(required_topics) - topics)
    if missing_topics:
        raise ValueError(f"missing required topics: {missing_topics}")

    train_path = Path(args.train_output)
    cases_path = Path(args.cases_output)
    probe_cases_path = Path(args.probe_cases_output)
    manifest_path = Path(args.manifest_output)
    write_jsonl(train_path, rows, strip_validation=not args.keep_validation_fields)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(json.dumps({"cases": cases}, indent=2, sort_keys=True) + "\n")
    probe_cases_path.parent.mkdir(parents=True, exist_ok=True)
    probe_cases_path.write_text(json.dumps({"cases": probe_cases}, indent=2, sort_keys=True) + "\n")

    train_prompts = {rendered_prompt_for_row(row) for row in rows}
    eval_prompts = {str(case["prompt"]) for case in cases}
    topic_counts = Counter(str(row.get("topic", "unknown")) for row in rows)
    behavior_counts = Counter(str(row.get("behavior", "unknown")) for row in rows)
    final_nonempty_count = sum(1 for row in rows if row.get("final"))
    manifest = {
        "format": FORMAT,
        "stage": STAGE,
        "seed": args.seed,
        "train_output": str(train_path),
        "cases_output": str(cases_path),
        "probe_cases_output": str(probe_cases_path),
        "train_records": len(rows),
        "base_records": len(base_records),
        "target_repeats": args.target_repeats,
        "anchor_repeats": args.anchor_repeats,
        "target_prompt_limit": args.target_prompt_limit,
        "anchor_prompt_limit": args.anchor_prompt_limit,
        "eval_cases": len(cases),
        "probe_cases": len(probe_cases),
        "topic_counts": dict(sorted(topic_counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "target_functions": selected_targets,
        "tiny_anchor_floor_functions": ANCHOR_FLOOR_FUNCTIONS,
        "required_topics": required_topics,
        "final_nonempty_count": final_nonempty_count,
        "validation": validation,
        "exact_train_eval_prompt_overlaps": exact_prompt_overlaps,
        "train_prompt_hashes": sorted(stable_hash(prompt) for prompt in train_prompts),
        "eval_prompt_hashes": sorted(stable_hash(prompt) for prompt in eval_prompts),
        "note": (
            "Stage A function-only repair data. This intentionally excludes patch, pytest, JSON, and shell-command "
            "examples until count_even, safe_int, parse_port, is_even, is_odd, and filter_even are stable."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
