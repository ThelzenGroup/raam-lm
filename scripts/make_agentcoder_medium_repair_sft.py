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
from scripts.make_agentcoder_coding_ladder_sft import (
    SYSTEM,
    build_eval_cases as build_base_ladder_eval_cases,
    chat_prompt,
    fenced_diff,
    fenced_python,
    record,
)


FORMAT = "agentcoder-medium-frontier-repair-v2"
REQUIRED_TOPICS = [
    "count_even",
    "safe_int",
    "parse_port",
    "one_file_bug_fix",
    "pytest_generation",
    "json_command",
    "anchor_tiny_function",
]
ANCHOR_FLOOR_FUNCTIONS = ["is_even", "is_odd", "filter_even"]
ANSWER_FAMILY_FORBIDDEN = {
    "function_completion": ["```diff", "Test command:", '"cmd"', "pytest ", "from counters import", "from parsers import"],
    "patch_addition": ["```python", '"cmd":'],
    "patch_off_by_one": ["```python", '"cmd":'],
    "patch_boolean_flag": ["```python", '"cmd":'],
    "patch_safe_int_default": ["```python", '"cmd":'],
    "patch_parse_port_range": ["```python", '"cmd":'],
    "pytest_generation": ["```diff", '"cmd":', "Test command:"],
    "json_tool_command": ["```", "Test command:", "def ", "pytest tests/"],
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def assistant_text(row: dict[str, Any]) -> str:
    trace = row.get("trace", [])
    if not trace:
        return ""
    return str(trace[0].get("content", ""))


def rendered_prompt_for_row(row: dict[str, Any]) -> str:
    parts = [f"<|system|>\n{row.get('system', SYSTEM)}\n"]
    if row.get("repo_context"):
        parts.append(f"\n<|repo_context|>\n{row['repo_context']}\n")
    for message in row.get("messages", []):
        parts.append(f"\n<|{message.get('role', 'user')}|>\n{message.get('content', '')}\n")
    parts.append("\n<|assistant|>\n")
    return "".join(parts)


def function_case(spec: dict[str, Any], *, name: str | None = None) -> dict[str, Any]:
    return {
        "name": name or f"train_{spec['name']}",
        "required_substrings": spec["required"],
        "python_function": {"name": spec["name"], "tests": spec["tests"]},
        "expected_behavior": "function_completion",
        "no_trailing_text": True,
    }


def python_syntax_case(required: list[str], behavior: str, *, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "required_substrings": required,
        "python_syntax": True,
        "expected_behavior": behavior,
        "no_trailing_text": True,
    }


def json_case(expected: dict[str, Any], *, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "required_substrings": [],
        "expected_json": expected,
        "expected_behavior": "json_tool_command",
        "strict_json": True,
        "no_nonsense": True,
    }


def patch_case(spec: dict[str, str], *, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "required_substrings": [f"--- a/{spec['path']}", f"+++ b/{spec['path']}", spec["test"]],
        "expected_patch": {
            "path": spec["path"],
            "removed": spec["removed"],
            "added": spec["added"],
        },
        "expected_behavior": spec["behavior"],
        "no_nonsense": True,
    }


def forbid_family_contamination(case: dict[str, Any], behavior: str) -> None:
    forbidden = list(case.get("forbidden_substrings", []))
    forbidden.extend(ANSWER_FAMILY_FORBIDDEN.get(behavior, []))
    case["forbidden_substrings"] = sorted(set(str(item) for item in forbidden))


FUNCTION_SPECS = [
    {
        "topic": "count_even",
        "name": "count_even",
        "arglist": "numbers",
        "description": "Return the number of even integers in numbers.",
        "code": "def count_even(numbers):\n    return sum(1 for n in numbers if n % 2 == 0)",
        "required": ["def count_even(numbers):", "sum(1 for n in numbers", "n % 2 == 0"],
        "tests": [
            {"args": [[1, 2, 4, 5]], "expected": 2},
            {"args": [[1, 3, 5]], "expected": 0},
            {"args": [[0, -2, 7, 8]], "expected": 3},
            {"args": [[]], "expected": 0},
        ],
    },
    {
        "topic": "count_even",
        "name": "count_even_loop",
        "arglist": "numbers",
        "description": "Count even integers using an explicit loop.",
        "code": (
            "def count_even_loop(numbers):\n"
            "    total = 0\n"
            "    for n in numbers:\n"
            "        if n % 2 == 0:\n"
            "            total += 1\n"
            "    return total"
        ),
        "required": ["def count_even_loop(numbers):", "for n in numbers:", "total += 1", "return total"],
        "tests": [
            {"args": [[2, 3, 4]], "expected": 2},
            {"args": [[1, 3, 5]], "expected": 0},
            {"args": [[-4, -3, 0]], "expected": 2},
        ],
    },
    {
        "topic": "count_even",
        "name": "count_positive",
        "arglist": "numbers",
        "description": "Return the number of positive integers in numbers.",
        "code": "def count_positive(numbers):\n    return sum(1 for n in numbers if n > 0)",
        "required": ["def count_positive(numbers):", "sum(1 for n in numbers", "n > 0"],
        "tests": [
            {"args": [[-1, 0, 2, 3]], "expected": 2},
            {"args": [[-3, -2]], "expected": 0},
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
            {"args": [" 7 "], "expected": 7},
            {"args": ["bad", 9], "expected": 9},
            {"args": [None, -1], "expected": -1},
        ],
    },
    {
        "topic": "safe_int",
        "name": "safe_positive_int",
        "arglist": "value, default=0",
        "description": "Convert value to a positive int, otherwise return default.",
        "code": (
            "def safe_positive_int(value, default=0):\n"
            "    try:\n"
            "        number = int(value)\n"
            "    except (TypeError, ValueError):\n"
            "        return default\n"
            "    if number <= 0:\n"
            "        return default\n"
            "    return number"
        ),
        "required": ["def safe_positive_int(value, default=0):", "try:", "number <= 0", "return default"],
        "tests": [
            {"args": ["5"], "expected": 5},
            {"args": ["0", 10], "expected": 10},
            {"args": ["bad", 3], "expected": 3},
        ],
    },
    {
        "topic": "safe_int",
        "name": "safe_float",
        "arglist": "value, default=0.0",
        "description": "Convert value to float; return default when conversion fails.",
        "code": (
            "def safe_float(value, default=0.0):\n"
            "    try:\n"
            "        return float(value)\n"
            "    except (TypeError, ValueError):\n"
            "        return default"
        ),
        "required": ["def safe_float(value, default=0.0):", "try:", "float(value)", "except (TypeError, ValueError):"],
        "tests": [
            {"args": ["1.5"], "expected": 1.5},
            {"args": ["bad", 2.25], "expected": 2.25},
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
        "required": ["def parse_port(value):", "int(str(value).strip())", "65535", "raise ValueError", "return port"],
        "tests": [
            {"args": ["8080"], "expected": 8080},
            {"args": [" 443 "], "expected": 443},
            {"args": [22], "expected": 22},
            {"args": ["0"], "raises": "ValueError"},
            {"args": ["65536"], "raises": "ValueError"},
            {"args": ["abc"], "raises": "ValueError"},
        ],
    },
    {
        "topic": "parse_port",
        "name": "parse_percent",
        "arglist": "value",
        "description": "Parse an integer percent between 0 and 100 inclusive.",
        "code": (
            "def parse_percent(value):\n"
            "    text = str(value).strip().removesuffix('%')\n"
            "    try:\n"
            "        percent = int(text)\n"
            "    except (TypeError, ValueError):\n"
            "        raise ValueError(\"percent must be an integer\")\n"
            "    if percent < 0 or percent > 100:\n"
            "        raise ValueError(\"percent must be between 0 and 100\")\n"
            "    return percent"
        ),
        "required": ["def parse_percent(value):", "removesuffix('%')", "raise ValueError", "return percent"],
        "tests": [
            {"args": ["85%"], "expected": 85},
            {"args": ["0"], "expected": 0},
            {"args": ["101"], "raises": "ValueError"},
        ],
    },
    {
        "topic": "parse_port",
        "name": "parse_bool",
        "arglist": "value",
        "description": "Parse common true/false text into a boolean.",
        "code": (
            "def parse_bool(value):\n"
            "    text = str(value).strip().lower()\n"
            "    if text in {\"1\", \"true\", \"yes\", \"on\"}:\n"
            "        return True\n"
            "    if text in {\"0\", \"false\", \"no\", \"off\"}:\n"
            "        return False\n"
            "    raise ValueError(\"expected boolean text\")"
        ),
        "required": ["def parse_bool(value):", "strip().lower()", "return True", "raise ValueError"],
        "tests": [
            {"args": ["yes"], "expected": True},
            {"args": ["OFF"], "expected": False},
            {"args": ["maybe"], "raises": "ValueError"},
        ],
    },
]


ANCHOR_FUNCTIONS = [
    {
        "topic": "anchor_tiny_function",
        "name": "is_even",
        "arglist": "n",
        "description": "Return True when n is even.",
        "code": "def is_even(n):\n    return n % 2 == 0",
        "required": ["def is_even(n):", "return n % 2 == 0"],
        "tests": [{"args": [2], "expected": True}, {"args": [3], "expected": False}],
    },
    {
        "topic": "anchor_tiny_function",
        "name": "is_odd",
        "arglist": "n",
        "description": "Return True when n is odd.",
        "code": "def is_odd(n):\n    return n % 2 == 1",
        "required": ["def is_odd(n):", "return n % 2 == 1"],
        "tests": [{"args": [3], "expected": True}, {"args": [4], "expected": False}, {"args": [0], "expected": False}],
    },
    {
        "topic": "anchor_tiny_function",
        "name": "filter_even",
        "arglist": "numbers",
        "description": "Return only the even integers.",
        "code": "def filter_even(numbers):\n    return [n for n in numbers if n % 2 == 0]",
        "required": ["def filter_even(numbers):", "[n for n in numbers", "n % 2 == 0"],
        "tests": [{"args": [[1, 2, 4]], "expected": [2, 4]}],
    },
]


PATCH_SPECS = [
    {
        "topic": "one_file_bug_fix",
        "behavior": "patch_addition",
        "path": "calc.py",
        "before": "def add(a, b):\n    return a - b",
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
        "removed": "return value == 'false'",
        "added": "return value == 'true'",
        "test": "pytest tests/test_flags.py -q",
        "bug": "the enabled flag is inverted",
    },
    {
        "topic": "one_file_bug_fix",
        "behavior": "patch_safe_int_default",
        "path": "parsers.py",
        "before": "def safe_int(value, default=None):\n    return int(value)",
        "removed": "return int(value)",
        "added": (
            "try:\n"
            "        return int(value)\n"
            "    except (TypeError, ValueError):\n"
            "        return default"
        ),
        "test": "pytest tests/test_parsers.py -q",
        "bug": "bad integer input raises instead of returning the default",
    },
    {
        "topic": "one_file_bug_fix",
        "behavior": "patch_parse_port_range",
        "path": "config.py",
        "before": "def parse_port(value):\n    port = int(value)\n    return port",
        "removed": "return port",
        "added": (
            "if port < 1 or port > 65535:\n"
            "        raise ValueError(\"port must be between 1 and 65535\")\n"
            "    return port"
        ),
        "test": "pytest tests/test_config.py -q",
        "bug": "the parser accepts out-of-range port numbers",
    },
]


def make_patch_answer(spec: dict[str, str]) -> str:
    removed = "\n".join(f"-{line}" for line in spec["removed"].splitlines())
    added = "\n".join(f"+{line}" for line in spec["added"].splitlines())
    diff = (
        f"--- a/{spec['path']}\n"
        f"+++ b/{spec['path']}\n"
        "@@\n"
        f"{removed}\n"
        f"{added}"
    )
    return f"{fenced_diff(diff)}\nTest command: `{spec['test']}`"


def function_prompts(spec: dict[str, Any]) -> list[str]:
    return [
        (
            f"Implement `{spec['name']}` as valid Python. {spec['description']} "
            "Return only one Python code block."
        ),
        (
            f"Complete this function and stop after the code:\n"
            f"```python\ndef {spec['name']}({spec['arglist']}):\n```"
        ),
        f"Write `def {spec['name']}({spec['arglist']}):` with correct error handling where needed. No explanation.",
        f"Fill in `{spec['name']}`. The function should: {spec['description']}",
    ]


def build_function_records(specs: list[dict[str, Any]], *, prompt_limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        prompts = function_prompts(spec)
        if prompt_limit is not None:
            prompts = prompts[:prompt_limit]
        case = function_case(spec)
        forbid_family_contamination(case, "function_completion")
        for index, prompt in enumerate(prompts):
            row = record(
                str(spec["topic"]),
                "function_completion",
                prompt,
                fenced_python(str(spec["code"])),
                "",
                system_suffix=(
                    "Write complete, valid Python. Stop immediately after the requested code block. "
                    "Do not add prose, test commands, patches, or trailing text."
                ),
                source=f"medium:function:{spec['name']}:{index}",
            )
            row["_validation_case"] = case
            rows.append(row)
    return rows


def build_patch_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PATCH_SPECS:
        repo = f"file: {spec['path']}\n```python\n{spec['before']}\n```"
        prompts = [
            f"PATCH TASK. Fix `{spec['path']}` with the minimal unified diff and include the focused pytest command.",
            f"PATCH TASK. Patch only `{spec['path']}`. Use exact unified diff headers and include `{spec['test']}`.",
            f"PATCH TASK. Given repo_context, repair the bug in `{spec['path']}`. Emit diff then test command.",
            f"PATCH TASK. Produce a one-file patch for `{spec['path']}`; do not emit Python function code outside the diff.",
        ]
        case = patch_case(spec, name=f"train_patch_{Path(spec['path']).stem}")
        forbid_family_contamination(case, str(spec["behavior"]))
        for index, prompt in enumerate(prompts):
            row = record(
                str(spec["topic"]),
                str(spec["behavior"]),
                prompt,
                make_patch_answer(spec),
                "",
                system_suffix=(
                    "Patch task only. Output a short bug sentence, one diff fenced as diff, and one focused pytest command. "
                    "Do not output standalone Python code, JSON, or unrelated functions."
                ),
                repo_context=repo,
                source=f"medium:patch:{spec['path']}:{index}",
            )
            row["_validation_case"] = case
            rows.append(row)
    return rows


PYTEST_SPECS = [
    {
        "name": "count_even",
        "module": "counters",
        "implementation": "def count_even(numbers):\n    return sum(1 for n in numbers if n % 2 == 0)",
        "tests": (
            "from counters import count_even\n\n"
            "def test_count_even_counts_matching_values():\n"
            "    assert count_even([1, 2, 4, 5]) == 2\n\n"
            "def test_count_even_handles_no_even_values():\n"
            "    assert count_even([1, 3, 5]) == 0\n\n"
            "def test_count_even_handles_empty_list():\n"
            "    assert count_even([]) == 0"
        ),
        "required": [
            "def test_count_even",
            "assert count_even([1, 2, 4, 5]) == 2",
            "assert count_even([1, 3, 5]) == 0",
        ],
    },
    {
        "name": "safe_int",
        "module": "parsers",
        "implementation": (
            "def safe_int(value, default=None):\n"
            "    try:\n"
            "        return int(value)\n"
            "    except (TypeError, ValueError):\n"
            "        return default"
        ),
        "tests": (
            "from parsers import safe_int\n\n"
            "def test_safe_int_converts_numeric_text():\n"
            "    assert safe_int('12') == 12\n\n"
            "def test_safe_int_returns_default_on_bad_input():\n"
            "    assert safe_int('bad', default=7) == 7\n\n"
            "def test_safe_int_returns_default_for_none():\n"
            "    assert safe_int(None, default=-1) == -1"
        ),
        "required": [
            "def test_safe_int",
            "assert safe_int('12') == 12",
            "assert safe_int('bad', default=7) == 7",
        ],
    },
    {
        "name": "parse_port",
        "module": "config",
        "implementation": (
            "def parse_port(value):\n"
            "    port = int(str(value).strip())\n"
            "    if port < 1 or port > 65535:\n"
            "        raise ValueError('bad port')\n"
            "    return port"
        ),
        "tests": (
            "import pytest\n"
            "from config import parse_port\n\n"
            "def test_parse_port_accepts_valid_port():\n"
            "    assert parse_port('8080') == 8080\n\n"
            "def test_parse_port_strips_whitespace():\n"
            "    assert parse_port(' 443 ') == 443\n\n"
            "def test_parse_port_rejects_out_of_range_port():\n"
            "    with pytest.raises(ValueError):\n"
            "        parse_port('65536')"
        ),
        "required": [
            "def test_parse_port",
            "assert parse_port('8080') == 8080",
            "with pytest.raises(ValueError):",
        ],
    },
]


def build_pytest_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in PYTEST_SPECS:
        prompts = [
            f"Write focused pytest tests for `{spec['name']}`. Cover success and failure cases.",
            f"Create a pytest file for `{spec['name']}` using concrete assertions only.",
            f"Given this implementation, write pytest tests and return only test code.\n```python\n{spec['implementation']}\n```",
        ]
        case = python_syntax_case(spec["required"], "pytest_generation", name=f"train_pytest_{spec['name']}")
        forbid_family_contamination(case, "pytest_generation")
        for index, prompt in enumerate(prompts):
            row = record(
                "pytest_generation",
                "pytest_generation",
                prompt,
                fenced_python(spec["tests"]),
                "",
                system_suffix="Pytest task only. Write small pytest tests with concrete assertions. Return only test code.",
                source=f"medium:pytest:{spec['name']}:{index}",
            )
            row["_validation_case"] = case
            rows.append(row)
    return rows


JSON_COMMANDS = [
    ("Return strict JSON for running all Python tests quietly.", {"cmd": "python -m pytest -q"}),
    ("Return JSON only for checking syntax of the medium repair generator.", {"cmd": "python -m py_compile scripts/make_agentcoder_medium_repair_sft.py"}),
    ("Return one JSON object for running the coding ladder tests.", {"cmd": "python -m pytest -q tests/test_coding_ladder.py"}),
    ("Return strict JSON for searching parse_port usage.", {"cmd": "grep -R \"parse_port\" -n scripts tests src"}),
    ("Return JSON only for listing Python files under scripts.", {"cmd": "find scripts -type f -name '*.py'"}),
    ("Return strict JSON for compiling the eval script.", {"cmd": "python -m py_compile scripts/eval_coding_ladder.py"}),
]


def build_json_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (prompt, payload) in enumerate(JSON_COMMANDS):
        case = json_case(payload, name=f"train_json_{index}")
        forbid_family_contamination(case, "json_tool_command")
        row = record(
            "json_command",
            "json_tool_command",
            prompt,
            json.dumps(payload, sort_keys=True),
            "",
            system_suffix="When the user asks for JSON, return exactly one valid JSON object and no prose.",
            source=f"medium:json:{index}",
        )
        row["_validation_case"] = case
        rows.append(row)
    return rows


def build_base_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rows.extend(build_function_records(FUNCTION_SPECS))
    rows.extend(build_patch_records())
    rows.extend(build_pytest_records())
    rows.extend(build_json_records())
    rows.extend(build_function_records(ANCHOR_FUNCTIONS, prompt_limit=2))
    return rows


def expanded_eval_cases() -> list[dict[str, Any]]:
    cases = build_base_ladder_eval_cases()
    for case in cases:
        forbid_family_contamination(case, str(case.get("expected_behavior", "")))
    extra_function_specs = [
        {
            **FUNCTION_SPECS[0],
            "eval_name": "medium_count_even_negatives",
            "eval_prompt": "Write `count_even(numbers)` as valid Python. It must handle negatives, zero, and empty lists. Return only code.",
        },
        {
            **FUNCTION_SPECS[3],
            "eval_name": "medium_safe_int_defaults",
            "eval_prompt": "Implement `safe_int(value, default=None)`. Bad input must return the default. Return only code.",
        },
        {
            **FUNCTION_SPECS[6],
            "eval_name": "medium_parse_port_range",
            "eval_prompt": "Implement `parse_port(value)`. Strip text, convert to int, and reject values outside 1..65535. Return only code.",
        },
    ]
    for spec in extra_function_specs:
        case = function_case(spec, name=spec["eval_name"])
        forbid_family_contamination(case, "function_completion")
        case.update(
            {
                "topic": spec["topic"],
                "prompt": chat_prompt(
                    (
                        "Write complete, valid Python functions. Stop after the requested code; "
                        "do not add unrelated prose or trailing text."
                    ),
                    spec["eval_prompt"],
                ),
                "forbidden_substrings": ["communicationService", "compromising", "Cahoon", "Now, we can"],
                "max_completion_chars": 1100,
            }
        )
        cases.append(case)

    for index, spec in enumerate(PATCH_SPECS[2:5], start=2):
        case = patch_case(spec, name=f"medium_patch_{index}_{Path(spec['path']).stem}")
        forbid_family_contamination(case, str(spec["behavior"]))
        case.update(
            {
                "topic": "one_file_bug_fix",
                "prompt": chat_prompt(
                    "For patch tasks, output exact unified diff headers, a hunk marker, and one focused pytest command.",
                    f"Fix `{spec['path']}` with a minimal unified diff. Include `{spec['test']}`.",
                    f"file: {spec['path']}\n```python\n{spec['before']}\n```",
                ),
                "max_completion_chars": 1400,
            }
        )
        cases.append(case)

    for spec in PYTEST_SPECS[1:]:
        case = python_syntax_case(spec["required"], "pytest_generation", name=f"medium_pytest_{spec['name']}")
        forbid_family_contamination(case, "pytest_generation")
        case.update(
            {
                "topic": "pytest_generation",
                "prompt": chat_prompt(
                    "Write small pytest tests with concrete assertions. Return only test code.",
                    f"Write pytest tests for `{spec['name']}`. Cover success and invalid input behavior.",
                    f"file: {spec['module']}.py\n```python\n{spec['implementation']}\n```",
                ),
                "max_completion_chars": 1200,
            }
        )
        cases.append(case)

    extra_json = [
        ("medium_json_pycompile", {"cmd": "python -m py_compile scripts/make_agentcoder_medium_repair_sft.py"}),
        ("medium_json_pytest_ladder", {"cmd": "python -m pytest -q tests/test_coding_ladder.py"}),
    ]
    for name, payload in extra_json:
        case = json_case(payload, name=name)
        forbid_family_contamination(case, "json_tool_command")
        case.update(
            {
                "topic": "json_command",
                "prompt": chat_prompt(
                    "When the user asks for JSON, return exactly one valid JSON object and no prose.",
                    f"Return strict JSON for this command: {payload['cmd']}",
                ),
                "max_completion_chars": 240,
            }
        )
        cases.append(case)
    return cases


def validate_records(rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid: list[dict[str, Any]] = []
    counts = Counter()
    topics = Counter()
    anchor_names = Counter()
    for index, row in enumerate(rows):
        if row.get("final") not in ("", None):
            invalid.append({"index": index, "source": row.get("_curriculum_source"), "reason": "non_empty_final"})
            continue
        case = row.get("_validation_case")
        if not isinstance(case, dict):
            invalid.append({"index": index, "source": row.get("_curriculum_source"), "reason": "missing_validation_case"})
            continue
        scored = score_case(case, assistant_text(row))
        counts[str(row.get("behavior", "unknown"))] += 1
        topics[str(row.get("topic", "unknown"))] += 1
        if row.get("topic") == "anchor_tiny_function":
            source = str(row.get("_curriculum_source", ""))
            for name in ANCHOR_FLOOR_FUNCTIONS:
                if f":{name}:" in source:
                    anchor_names[name] += 1
        if not scored["passed"]:
            invalid.append(
                {
                    "index": index,
                    "source": row.get("_curriculum_source"),
                    "reason": "score_case_failed",
                    "missing": scored.get("missing_required_substrings"),
                    "function_failures": scored.get("function_failures"),
                    "patch_failures": scored.get("patch_failures"),
                    "json_ok": scored.get("json_ok"),
                }
            )
    if invalid:
        raise ValueError(f"{len(invalid)} generated records failed validation: {invalid[:5]}")
    missing_anchors = [name for name in ANCHOR_FLOOR_FUNCTIONS if anchor_names[name] == 0]
    if missing_anchors:
        raise ValueError(f"missing tiny anchor floor records: {missing_anchors}")
    return {
        "validated_records": len(rows),
        "validated_behavior_counts": dict(sorted(counts.items())),
        "validated_topic_counts": dict(sorted(topics.items())),
        "tiny_anchor_floor_counts": dict(sorted(anchor_names.items())),
    }


def expand_records(base_records: list[dict[str, Any]], *, repeats: int, anchor_repeats: int, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for repeat in range(max(1, repeats)):
        for index, item in enumerate(base_records):
            if item.get("topic") == "anchor_tiny_function" and repeat >= max(1, anchor_repeats):
                continue
            row = dict(item)
            row["_medium_repeat"] = repeat
            row["_medium_base_index"] = index
            rows.append(row)
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


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
    parser = argparse.ArgumentParser(description="Generate focused RAAM medium-function repair SFT data.")
    parser.add_argument("--train-output", default="examples/agentcoder_medium_repair_train.jsonl")
    parser.add_argument("--cases-output", default="examples/agentcoder_medium_repair_eval_cases.json")
    parser.add_argument("--manifest-output", default="examples/agentcoder_medium_repair_manifest.json")
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--anchor-repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--keep-validation-fields", action="store_true")
    args = parser.parse_args()

    base_records = build_base_records()
    rows = expand_records(base_records, repeats=args.repeats, anchor_repeats=args.anchor_repeats, seed=args.seed)
    validation = validate_records(rows)
    cases = expanded_eval_cases()

    train_prompts = {rendered_prompt_for_row(row) for row in rows}
    eval_prompts = {str(case["prompt"]) for case in cases}
    exact_prompt_overlaps = sorted(train_prompts & eval_prompts)
    if exact_prompt_overlaps:
        raise ValueError(f"train/eval exact prompt overlap: {exact_prompt_overlaps[:3]}")

    topics = {str(row.get("topic")) for row in rows}
    missing = sorted(set(REQUIRED_TOPICS) - topics)
    if missing:
        raise ValueError(f"missing required topics: {missing}")

    train_path = Path(args.train_output)
    cases_path = Path(args.cases_output)
    manifest_path = Path(args.manifest_output)
    write_jsonl(train_path, rows, strip_validation=not args.keep_validation_fields)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(json.dumps({"cases": cases}, indent=2, sort_keys=True) + "\n")

    topic_counts = Counter(str(row.get("topic", "unknown")) for row in rows)
    behavior_counts = Counter(str(row.get("behavior", "unknown")) for row in rows)
    final_nonempty_count = sum(1 for row in rows if row.get("final"))
    tiny_anchor_records = topic_counts.get("anchor_tiny_function", 0)
    tiny_anchor_ratio = tiny_anchor_records / len(rows) if rows else 0.0
    manifest = {
        "format": FORMAT,
        "seed": args.seed,
        "train_output": str(train_path),
        "cases_output": str(cases_path),
        "train_records": len(rows),
        "base_records": len(base_records),
        "repeats": args.repeats,
        "anchor_repeats": args.anchor_repeats,
        "eval_cases": len(cases),
        "topic_counts": dict(sorted(topic_counts.items())),
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "tiny_anchor_records": tiny_anchor_records,
        "tiny_anchor_ratio": tiny_anchor_ratio,
        "tiny_anchor_floor_functions": ANCHOR_FLOOR_FUNCTIONS,
        "required_topics": REQUIRED_TOPICS,
        "final_nonempty_count": final_nonempty_count,
        "validation": validation,
        "train_prompt_hashes": sorted(stable_hash(prompt) for prompt in train_prompts),
        "eval_prompt_hashes": sorted(stable_hash(prompt) for prompt in eval_prompts),
        "exact_train_eval_prompt_overlaps": exact_prompt_overlaps,
        "note": (
            "Focused no-final medium-function repair data for count_even, safe_int, parse_port, "
            "strict unified diffs, pytest generation, JSON command output, and a stronger tiny-function floor."
        ),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
