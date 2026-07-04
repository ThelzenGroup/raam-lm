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

from scripts.make_agentcoder_curated_sft import SYSTEM, build_train_records as build_curated_train_records


FORMAT = "agentcoder-coding-ladder-repair-v1"
REQUIRED_LADDER_TOPICS = [
    "is_even",
    "is_odd",
    "count_even",
    "filter_even",
    "safe_int",
    "parse_port",
    "one_file_bug_fix",
    "pytest_generation",
    "json_command",
]


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chat_prompt(system_suffix: str, user: str, repo_context: str | None = None) -> str:
    parts = [f"<|system|>\n{SYSTEM} {system_suffix}".strip() + "\n"]
    if repo_context:
        parts.append(f"\n<|repo_context|>\n{repo_context}\n")
    parts.append(f"\n<|user|>\n{user}\n\n<|assistant|>\n")
    return "".join(parts)


def record(
    topic: str,
    behavior: str,
    user: str,
    assistant: str,
    final: str,
    *,
    system_suffix: str,
    repo_context: str | None = None,
    source: str,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "topic": topic,
        "behavior": behavior,
        "system": f"{SYSTEM} {system_suffix}".strip(),
        "messages": [{"role": "user", "content": user}],
        "trace": [{"type": "assistant", "content": assistant}],
        "final": final,
        "_curriculum_source": source,
    }
    if repo_context:
        item["repo_context"] = repo_context
    return item


def fenced_python(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def fenced_diff(diff: str) -> str:
    return f"```diff\n{diff.rstrip()}\n```"


FUNCTION_SPECS = [
    {
        "topic": "is_even",
        "name": "is_even",
        "arglist": "n",
        "description": "Return True when n is even and False otherwise.",
        "code": "def is_even(n):\n    return n % 2 == 0",
        "required": ["def is_even(n):", "return n % 2 == 0"],
        "tests": [
            {"args": [2], "expected": True},
            {"args": [3], "expected": False},
            {"args": [0], "expected": True},
        ],
    },
    {
        "topic": "is_odd",
        "name": "is_odd",
        "arglist": "n",
        "description": "Return True when n is odd and False otherwise.",
        "code": "def is_odd(n):\n    return n % 2 == 1",
        "required": ["def is_odd(n):", "return n % 2 == 1"],
        "tests": [
            {"args": [3], "expected": True},
            {"args": [4], "expected": False},
            {"args": [0], "expected": False},
        ],
    },
    {
        "topic": "count_even",
        "name": "count_even",
        "arglist": "numbers",
        "description": "Count how many integers in numbers are even.",
        "code": "def count_even(numbers):\n    return sum(1 for n in numbers if n % 2 == 0)",
        "required": ["def count_even(numbers):", "sum(1 for n in numbers", "n % 2 == 0"],
        "tests": [
            {"args": [[1, 2, 4, 5]], "expected": 2},
            {"args": [[1, 3, 5]], "expected": 0},
            {"args": [[0, -2, 7, 8]], "expected": 3},
        ],
    },
    {
        "topic": "filter_even",
        "name": "filter_even",
        "arglist": "numbers",
        "description": "Return a list containing only the even integers from numbers.",
        "code": "def filter_even(numbers):\n    return [n for n in numbers if n % 2 == 0]",
        "required": ["def filter_even(numbers):", "[n for n in numbers", "n % 2 == 0"],
        "tests": [
            {"args": [[1, 2, 4, 5]], "expected": [2, 4]},
            {"args": [[1, 3, 5]], "expected": []},
            {"args": [[0, -2, 7, 8]], "expected": [0, -2, 8]},
        ],
    },
    {
        "topic": "safe_int",
        "name": "safe_int",
        "arglist": "value, default=None",
        "description": "Convert value to int; return default when conversion fails.",
        "code": (
            "def safe_int(value, default=None):\n"
            "    try:\n"
            "        return int(value)\n"
            "    except (TypeError, ValueError):\n"
            "        return default"
        ),
        "required": ["def safe_int(value, default=None):", "try:", "except (TypeError, ValueError):", "return default"],
        "tests": [
            {"args": ["42"], "expected": 42},
            {"args": ["bad", 7], "expected": 7},
            {"args": [None, -1], "expected": -1},
        ],
    },
    {
        "topic": "parse_port",
        "name": "parse_port",
        "arglist": "value",
        "description": "Parse a TCP port and reject non-numeric or out-of-range values.",
        "code": (
            "def parse_port(value):\n"
            "    try:\n"
            "        port = int(str(value).strip())\n"
            "    except (TypeError, ValueError):\n"
            "        raise ValueError(\"port must be an integer\")\n"
            "    if port < 1 or port > 65535:\n"
            "        raise ValueError(\"port must be between 1 and 65535\")\n"
            "    return port"
        ),
        "required": [
            "def parse_port(value):",
            "int(str(value).strip())",
            "65535",
            "raise ValueError",
            "return port",
        ],
        "tests": [
            {"args": ["8080"], "expected": 8080},
            {"args": [" 443 "], "expected": 443},
            {"args": ["0"], "raises": "ValueError"},
            {"args": ["65536"], "raises": "ValueError"},
            {"args": ["abc"], "raises": "ValueError"},
        ],
    },
]


PATCH_SPECS = [
    {
        "topic": "one_file_bug_fix",
        "behavior": "patch_addition",
        "path": "calc.py",
        "before": "def add(a, b):\n    return a - b",
        "after": "def add(a, b):\n    return a + b",
        "removed": "return a - b",
        "added": "return a + b",
        "test": "pytest tests/test_calc.py -q",
        "bug": "subtraction where addition is required",
    },
    {
        "topic": "one_file_bug_fix",
        "behavior": "patch_off_by_one",
        "path": "window.py",
        "before": "def first_items(items, limit):\n    return items[:limit + 1]",
        "after": "def first_items(items, limit):\n    return items[:limit]",
        "removed": "return items[:limit + 1]",
        "added": "return items[:limit]",
        "test": "pytest tests/test_window.py -q",
        "bug": "an off-by-one slice includes too many items",
    },
    {
        "topic": "one_file_bug_fix",
        "behavior": "patch_boolean_flag",
        "path": "flags.py",
        "before": "def is_enabled(value):\n    return value == 'false'",
        "after": "def is_enabled(value):\n    return value == 'true'",
        "removed": "return value == 'false'",
        "added": "return value == 'true'",
        "test": "pytest tests/test_flags.py -q",
        "bug": "the enabled flag is inverted",
    },
]


def make_patch_answer(spec: dict[str, str]) -> str:
    diff = (
        f"--- a/{spec['path']}\n"
        f"+++ b/{spec['path']}\n"
        "@@\n"
        f"-{spec['removed']}\n"
        f"+{spec['added']}"
    )
    return f"The bug is {spec['bug']}.\n{fenced_diff(diff)}\nTest command: `{spec['test']}`"


def build_ladder_base_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in FUNCTION_SPECS:
        prompts = [
            (
                f"Complete this Python function. Return only the code block.\n"
                f"```python\ndef {spec['name']}({spec['arglist']}):\n```"
            ),
            f"Write a concise Python implementation of `{spec['name']}`. {spec['description']} Return only code.",
            (
                f"Implement `{spec['name']}` in Python with no explanation. "
                f"Signature: `def {spec['name']}({spec['arglist']}):`"
            ),
            f"Fill in the function body for `{spec['name']}`: {spec['description']}",
        ]
        for prompt_index, prompt in enumerate(prompts):
            records.append(
                record(
                    str(spec["topic"]),
                    "function_completion",
                    prompt,
                    fenced_python(str(spec["code"])),
                    "",
                    system_suffix=(
                        "Write complete, valid Python functions. Stop after the requested code; "
                        "do not add unrelated prose or trailing text."
                    ),
                    source=f"ladder:function:{spec['name']}:{prompt_index}",
                )
            )

    for spec in PATCH_SPECS:
        repo = f"file: {spec['path']}\n```python\n{spec['before']}\n```"
        prompts = [
            (
                f"Fix the one-file bug in `{spec['path']}`. Emit a unified diff first, "
                f"then give the focused pytest command."
            ),
            (
                f"Patch `{spec['path']}` only. Replace `{spec['removed']}` with `{spec['added']}` "
                "and include the test command."
            ),
            f"Given repo_context, produce the minimal diff for `{spec['path']}` and name the pytest command.",
        ]
        for prompt_index, prompt in enumerate(prompts):
            records.append(
                record(
                    str(spec["topic"]),
                    str(spec["behavior"]),
                    prompt,
                    make_patch_answer(spec),
                    "",
                    system_suffix="For patch tasks, output a unified diff plus one focused pytest command.",
                    repo_context=repo,
                    source=f"ladder:patch:{spec['path']}:{prompt_index}",
                )
            )

    pytest_records = [
        (
            "count_even",
            "def count_even(numbers):\n    return sum(1 for n in numbers if n % 2 == 0)",
            (
                "```python\n"
                "from counters import count_even\n\n"
                "def test_count_even_counts_matching_values():\n"
                "    assert count_even([1, 2, 4, 5]) == 2\n\n"
                "def test_count_even_handles_no_even_values():\n"
                "    assert count_even([1, 3, 5]) == 0\n"
                "```"
            ),
        ),
        (
            "safe_int",
            "def safe_int(value, default=None):\n    try:\n        return int(value)\n    except (TypeError, ValueError):\n        return default",
            (
                "```python\n"
                "from parsers import safe_int\n\n"
                "def test_safe_int_converts_numeric_text():\n"
                "    assert safe_int('12') == 12\n\n"
                "def test_safe_int_returns_default_on_bad_input():\n"
                "    assert safe_int('bad', default=7) == 7\n"
                "```"
            ),
        ),
        (
            "parse_port",
            "def parse_port(value):\n    port = int(str(value).strip())\n    if port < 1 or port > 65535:\n        raise ValueError('bad port')\n    return port",
            (
                "```python\n"
                "import pytest\n"
                "from config import parse_port\n\n"
                "def test_parse_port_accepts_valid_port():\n"
                "    assert parse_port('8080') == 8080\n\n"
                "def test_parse_port_rejects_out_of_range_port():\n"
                "    with pytest.raises(ValueError):\n"
                "        parse_port('65536')\n"
                "```"
            ),
        ),
    ]
    for index, (name, implementation, tests) in enumerate(pytest_records):
        records.append(
            record(
                "pytest_generation",
                "pytest_generation",
                f"Write focused pytest tests for this function `{name}`.\n```python\n{implementation}\n```",
                tests,
                "",
                system_suffix="Write small pytest tests with concrete assertions. Return only test code.",
                source=f"ladder:pytest:{name}:{index}",
            )
        )

    json_commands = [
        ("Return strict JSON for running the quiet Python test suite.", {"cmd": "python -m pytest -q"}),
        ("Return one JSON object for finding Python source files.", {"cmd": "find . -type f -name '*.py'"}),
        ("Give strict JSON for checking syntax of the ladder eval script.", {"cmd": "python -m py_compile scripts/eval_coding_ladder.py"}),
        ("Return JSON only for searching parse_port in scripts and tests.", {"cmd": "grep -R \"parse_port\" -n scripts tests"}),
        ("Return strict JSON for listing generated JSONL files under runs.", {"cmd": "find runs -type f -name '*.jsonl'"}),
    ]
    for index, (prompt, payload) in enumerate(json_commands):
        records.append(
            record(
                "json_command",
                "json_tool_command",
                prompt,
                json.dumps(payload, sort_keys=True),
                "",
                system_suffix="When the user asks for JSON, return one valid JSON object and no prose.",
                source=f"ladder:json:{index}",
            )
        )

    return records


def build_eval_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for spec in FUNCTION_SPECS:
        cases.append(
            {
                "name": f"ladder_{spec['topic']}",
                "topic": spec["topic"],
                "prompt": chat_prompt(
                    (
                        "Write complete, valid Python functions. Stop after the requested code; "
                        "do not add unrelated prose or trailing text."
                    ),
                    f"Implement this function as valid Python and stop after the code:\n```python\ndef {spec['name']}({spec['arglist']}):\n```",
                ),
                "required_substrings": spec["required"],
                "forbidden_substrings": ["Now, we can", "communicationService", "compromising", "Cahoon"],
                "expected_behavior": "function_completion",
                "python_function": {"name": spec["name"], "tests": spec["tests"]},
                "no_trailing_text": True,
                "max_completion_chars": 900,
            }
        )

    for index, spec in enumerate(PATCH_SPECS[:2]):
        cases.append(
            {
                "name": f"ladder_patch_{index}_{Path(spec['path']).stem}",
                "topic": "one_file_bug_fix",
                "prompt": chat_prompt(
                    "For patch tasks, output a unified diff plus one focused pytest command.",
                    (
                        f"Fix `{spec['path']}` using a minimal unified diff. "
                        f"The correct line is `{spec['added']}`. Include the focused pytest command."
                    ),
                    f"file: {spec['path']}\n```python\n{spec['before']}\n```",
                ),
                "required_substrings": [f"--- a/{spec['path']}", f"+++ b/{spec['path']}", spec["added"], spec["test"]],
                "expected_patch": {
                    "path": spec["path"],
                    "removed": spec["removed"],
                    "added": spec["added"],
                },
                "expected_behavior": spec["behavior"],
                "no_nonsense": True,
                "max_completion_chars": 1200,
            }
        )

    cases.append(
        {
            "name": "ladder_pytest_count_even",
            "topic": "pytest_generation",
            "prompt": chat_prompt(
                "Write small pytest tests with concrete assertions. Return only test code.",
                (
                    "Write pytest tests for count_even. Cover a list with two even values "
                    "and a list with no even values."
                ),
                "file: counters.py\n```python\ndef count_even(numbers):\n    return sum(1 for n in numbers if n % 2 == 0)\n```",
            ),
            "required_substrings": [
                "def test_count_even",
                "assert count_even([1, 2, 4, 5]) == 2",
                "assert count_even([1, 3, 5]) == 0",
            ],
            "python_syntax": True,
            "expected_behavior": "pytest_generation",
            "no_trailing_text": True,
            "max_completion_chars": 1000,
        }
    )
    cases.append(
        {
            "name": "ladder_json_pytest",
            "topic": "json_command",
            "prompt": chat_prompt(
                "When the user asks for JSON, return one valid JSON object and no prose.",
                "Return strict JSON with the quiet command for running Python tests.",
            ),
            "required_substrings": ["python -m pytest -q"],
            "expected_json": {"cmd": "python -m pytest -q"},
            "expected_behavior": "json_tool_command",
            "strict_json": True,
            "no_nonsense": True,
            "max_completion_chars": 200,
        }
    )
    return cases


def expand_records(
    base_records: list[dict[str, Any]],
    *,
    ladder_repeats: int,
    curated_anchor_repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for repeat in range(max(1, ladder_repeats)):
        for index, item in enumerate(base_records):
            row = dict(item)
            row["_ladder_repeat"] = repeat
            row["_ladder_base_index"] = index
            records.append(row)
    curated = build_curated_train_records()
    for repeat in range(max(0, curated_anchor_repeats)):
        for index, item in enumerate(curated):
            row = dict(item)
            row["_curriculum_source"] = f"curated_anchor:{repeat}:{index}"
            row["_ladder_repeat"] = repeat
            records.append(row)
    rng = random.Random(seed)
    rng.shuffle(records)
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAAM-AgentCoder coding-ladder repair SFT data.")
    parser.add_argument("--train-output", default="examples/agentcoder_coding_ladder_train.jsonl")
    parser.add_argument("--cases-output", default="examples/agentcoder_coding_ladder_eval_cases.json")
    parser.add_argument("--manifest-output", default="examples/agentcoder_coding_ladder_manifest.json")
    parser.add_argument("--ladder-repeats", type=int, default=10)
    parser.add_argument("--curated-anchor-repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    base_records = build_ladder_base_records()
    train_records = expand_records(
        base_records,
        ladder_repeats=args.ladder_repeats,
        curated_anchor_repeats=args.curated_anchor_repeats,
        seed=args.seed,
    )
    eval_cases = build_eval_cases()

    train_prompts = {
        str(message.get("content", ""))
        for row in train_records
        for message in row.get("messages", [])
        if message.get("role") == "user"
    }
    eval_prompts = {str(case["prompt"]) for case in eval_cases}
    exact_prompt_overlaps = sorted(train_prompts & eval_prompts)

    missing_topics = sorted(set(REQUIRED_LADDER_TOPICS) - {str(row.get("topic")) for row in train_records})
    if missing_topics:
        raise ValueError(f"missing required ladder topics in train records: {missing_topics}")
    eval_missing_topics = sorted(set(REQUIRED_LADDER_TOPICS) - {str(case.get("topic")) for case in eval_cases})
    if eval_missing_topics:
        raise ValueError(f"missing required ladder topics in eval cases: {eval_missing_topics}")
    if exact_prompt_overlaps:
        raise ValueError(f"train/eval exact prompt overlap: {exact_prompt_overlaps[:3]}")

    train_path = Path(args.train_output)
    cases_path = Path(args.cases_output)
    manifest_path = Path(args.manifest_output)
    write_jsonl(train_path, train_records)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(json.dumps({"cases": eval_cases}, indent=2, sort_keys=True) + "\n")

    behavior_counts = Counter(str(row.get("behavior", "unknown")) for row in train_records)
    topic_counts = Counter(str(row.get("topic", "unknown")) for row in train_records)
    manifest = {
        "format": FORMAT,
        "seed": args.seed,
        "train_output": str(train_path),
        "cases_output": str(cases_path),
        "train_records": len(train_records),
        "ladder_base_records": len(base_records),
        "ladder_repeats": args.ladder_repeats,
        "curated_anchor_repeats": args.curated_anchor_repeats,
        "curated_anchor_records": 96 * max(0, args.curated_anchor_repeats),
        "eval_cases": len(eval_cases),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "required_ladder_topics": REQUIRED_LADDER_TOPICS,
        "train_prompt_hashes": sorted(stable_hash(prompt) for prompt in train_prompts),
        "eval_prompt_hashes": sorted(stable_hash(prompt) for prompt in eval_prompts),
        "exact_train_eval_prompt_overlaps": exact_prompt_overlaps,
        "note": (
            "Deterministic repair SFT data for small Python function, patch, pytest, and JSON-command behavior. "
            "Eval prompts are held out by exact prompt text and scored by scripts/eval_coding_ladder.py."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
