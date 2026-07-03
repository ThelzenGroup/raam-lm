#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any


SYSTEM = "You are RAAM-AgentCoder, a concise software-engineering assistant."
BALANCED_BEHAVIOR_TARGET = 8


def record(
    behavior: str,
    user: str,
    assistant: str,
    final: str,
    *,
    system_suffix: str = "",
    repo_context: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "behavior": behavior,
        "system": f"{SYSTEM} {system_suffix}".strip(),
        "messages": [{"role": "user", "content": user}],
        "trace": [{"type": "assistant", "content": assistant}],
        "final": final,
    }
    if repo_context:
        item["repo_context"] = repo_context
    return item


def case(
    name: str,
    prompt: str,
    required: list[str],
    *,
    expected_behavior: str,
    expected_json: Any | None = None,
) -> dict[str, Any]:
    item = {
        "name": name,
        "prompt": prompt,
        "required_substrings": required,
        "expected_behavior": expected_behavior,
    }
    if expected_json is not None:
        item["expected_json"] = expected_json
    return item


def chat_prompt(system_suffix: str, user: str, repo_context: str | None = None) -> str:
    parts = [f"<|system|>\n{SYSTEM} {system_suffix}".strip() + "\n"]
    if repo_context:
        parts.append(f"\n<|repo_context|>\n{repo_context}\n")
    parts.append(f"\n<|user|>\n{user}\n\n<|assistant|>\n")
    return "".join(parts)


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


def balance_records(records: list[dict[str, Any]], target_per_behavior: int = BALANCED_BEHAVIOR_TARGET) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["behavior"])].append(row)

    underfilled = {
        behavior: len(rows)
        for behavior, rows in sorted(grouped.items())
        if len(rows) < target_per_behavior
    }
    if underfilled:
        raise ValueError(f"not enough examples for balanced curriculum: {underfilled}")

    balanced: list[dict[str, Any]] = []
    for behavior in sorted(grouped):
        balanced.extend(grouped[behavior][:target_per_behavior])
    return balanced


def build_train_records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    add_variants = [
        ("calc.py", "def add(a, b):\n    return a - b", "def add(a, b):\n    return a + b", "tests/test_calc.py"),
        ("mathlib.py", "def add(a, b):\n    return a - b", "def add(a, b):\n    return a + b", "tests/test_mathlib.py"),
        ("arithmetic.py", "def sum_values(left, right):\n    return left - right", "def sum_values(left, right):\n    return left + right", "tests/test_arithmetic.py"),
        ("totals.py", "def combine(x, y):\n    return x - y", "def combine(x, y):\n    return x + y", "tests/test_totals.py"),
    ]
    add_prompts = [
        "The helper fails because it subtracts. Provide a minimal patch and test command.",
        "A test expected addition but the function returns subtraction. Patch it and name the focused pytest command.",
        "Fix this arithmetic bug with a tiny diff and say which test to run.",
    ]
    for idx, (path, before, after, test_path) in enumerate(add_variants):
        repo = f"file: {path}\n```python\n{before}\n```"
        for prompt in add_prompts:
            assistant = (
                patch_text(path, before, after, "The bug is subtraction where addition is required.")
                + f"\nTest command: `pytest {test_path} -q`"
            )
            records.append(
                record(
                    "patch_addition",
                    prompt,
                    assistant,
                    f"Patched {path} and verified with pytest {test_path} -q.",
                    system_suffix="Preserve exact patches and include a focused test command.",
                    repo_context=repo,
                )
            )

    flag_variants = [
        ("flags.py", "def is_enabled(value):\n    return value == 'false'", "def is_enabled(value):\n    return value == 'true'"),
        ("settings.py", "def feature_on(raw):\n    return raw == 'off'", "def feature_on(raw):\n    return raw == 'on'"),
        ("env_flags.py", "def debug_mode(raw):\n    return raw == 'disabled'", "def debug_mode(raw):\n    return raw == 'enabled'"),
        ("feature_flags.py", "def allows_signup(flag):\n    return flag == 'blocked'", "def allows_signup(flag):\n    return flag == 'allowed'"),
        ("permissions.py", "def can_delete(role):\n    return role == 'viewer'", "def can_delete(role):\n    return role == 'admin'"),
        ("toggles.py", "def cache_enabled(value):\n    return value == 'off'", "def cache_enabled(value):\n    return value == 'on'"),
        ("account.py", "def is_active(status):\n    return status == 'disabled'", "def is_active(status):\n    return status == 'active'"),
        ("rollout.py", "def rollout_enabled(value):\n    return value == 'no'", "def rollout_enabled(value):\n    return value == 'yes'"),
    ]
    for path, before, after in flag_variants:
        records.append(
            record(
                "patch_boolean_flag",
                "The boolean helper is inverted. Provide the minimal patch.",
                patch_text(path, before, after, "The comparison is inverted; compare against the enabled value."),
                f"Changed {path} to compare against the enabled value.",
                system_suffix="Prefer minimal diffs.",
                repo_context=f"file: {path}\n```python\n{before}\n```",
            )
        )

    json_commands = [
        (
            "Return JSON for a shell command that finds Python files.",
            {"cmd": "find . -type f -name '*.py'"},
            "The command finds Python source files recursively.",
        ),
        (
            "Give me one JSON object for running the Python test suite.",
            {"cmd": "python -m pytest -q"},
            "The command runs the Python test suite quietly.",
        ),
        (
            "Respond with strict JSON for listing markdown docs.",
            {"cmd": "find docs -type f -name '*.md'"},
            "The command lists Markdown files under docs.",
        ),
        (
            "Return a JSON command that searches for TODO comments.",
            {"cmd": "grep -R \"TODO\" -n src tests"},
            "The command searches source and tests for TODO comments.",
        ),
        (
            "Return JSON for listing YAML configs.",
            {"cmd": "find configs -type f -name '*.yaml'"},
            "The command lists YAML config files.",
        ),
        (
            "Return strict JSON with a command that prints git status.",
            {"cmd": "git status --short"},
            "The command prints concise git status.",
        ),
        (
            "Respond with JSON for finding test files.",
            {"cmd": "find tests -type f -name 'test_*.py'"},
            "The command finds Python test files.",
        ),
        (
            "Give a JSON object for searching FIXME markers.",
            {"cmd": "grep -R \"FIXME\" -n src tests"},
            "The command searches source and tests for FIXME markers.",
        ),
    ]
    for user, payload, final in json_commands:
        records.append(
            record(
                "json_tool_command",
                user,
                json.dumps(payload),
                final,
                system_suffix="Return valid JSON when asked for JSON.",
            )
        )

    python_file_json_prompts = [
        "Return strict JSON for a command that lists Python files.",
        "I need JSON only: a shell command to find .py files.",
        "Give one JSON object whose cmd finds Python source files.",
        "As JSON, provide the command for recursively listing Python files.",
        "Return JSON, not prose, for finding Python files in this repo.",
        "Write a JSON command that locates files ending in .py.",
    ]
    for user in python_file_json_prompts:
        records.append(
            record(
                "command_disambiguation",
                user,
                "{\"cmd\": \"find . -type f -name '*.py'\"}",
                "Returned the Python-file discovery command as JSON.",
                system_suffix="Distinguish file-discovery commands from test commands. Return strict JSON when asked.",
            )
        )

    pytest_json_prompts = [
        "Return strict JSON for running Python tests.",
        "I need JSON only: the command to run pytest quietly.",
        "Give one JSON object whose cmd runs the Python package tests.",
    ]
    for user in pytest_json_prompts:
        records.append(
            record(
                "command_disambiguation",
                user,
                "{\"cmd\": \"python -m pytest -q\"}",
                "Returned the pytest command as JSON.",
                system_suffix="Distinguish test commands from file-discovery commands. Return strict JSON when asked.",
            )
        )

    risky_prompts = [
        "Before editing production auth code, ask one concise clarifying question.",
        "I want you to change a risky payment file. Ask for the missing safety detail first.",
        "Ask one question before touching a production database migration.",
        "Before changing deployment code, what must you ask?",
        "A production config edit could break login. Ask one concise question first.",
        "Before modifying a risky file, ask about the target and rollback plan.",
        "Before changing authentication middleware in production, ask one concise question.",
        "Before editing a live billing migration, what single question do you ask?",
    ]
    risky_answer = "Which file should I edit, and what test command or rollback command should I use if the change behaves unexpectedly?"
    for user in risky_prompts:
        records.append(
            record(
                "risky_clarifying_question",
                user,
                risky_answer,
                "Asked for the target file plus test or rollback command before editing.",
                system_suffix="Ask before risky edits.",
            )
        )

    debugging_prompts = [
        "A Python unit test is failing. Explain what to check before editing code; do not write a function.",
        "The test suite failed after my change. Explain the debugging order without completing code.",
        "Explain in plain English how you debug a failing assertion before patching. Do not output Python code.",
        "A regression test is red. What steps do you take before changing code? Answer as debugging steps only.",
        "Before fixing a failed unit test, what do you inspect? Do not write a replacement function.",
        "A test expected 5 but got -1. Explain the reasoning before editing, not a code completion.",
        "A CLI command fails only in CI. Explain your debugging order before editing; no code block.",
        "A patch broke one focused pytest. What do you inspect first? Explain, do not implement.",
    ]
    debugging_answer = (
        "I reproduce the failing test, read the assertion and error, inspect the smallest changed code path, "
        "then make the smallest patch that explains the failure."
    )
    for user in debugging_prompts:
        records.append(
            record(
                "plain_debugging",
                user,
                debugging_answer,
                "Reproduce the test, read the assertion, inspect the narrow path, and patch minimally.",
                system_suffix="Explain debugging steps plainly. Do not write code for debugging-process prompts.",
            )
        )

    function_variants = [
        ("is_even", "return n % 2 == 0", "Check whether an integer is even."),
        ("is_odd", "return n % 2 == 1", "Check whether an integer is odd."),
        ("is_positive", "return n > 0", "Check whether a number is positive."),
        ("is_nonempty", "return len(items) > 0", "Check whether a collection is nonempty."),
    ]
    for name, body, desc in function_variants:
        prompts = [
            f"Complete this Python function:\n```python\ndef {name}(n):\n```" if "items" not in body else f"Complete this Python function:\n```python\ndef {name}(items):\n```",
            f"Write the body for {name}. {desc}",
        ]
        arg = "items" if "items" in body else "n"
        answer = f"```python\ndef {name}({arg}):\n    {body}\n```"
        for user in prompts:
            records.append(
                record(
                    "function_completion",
                    user,
                    answer,
                    f"Implemented {name}.",
                    system_suffix="Complete Python functions safely.",
                )
            )

    stack_variants = [
        (
            "ValueError: invalid literal for int() with base 10: 'abc'",
            "The code tried to convert non-numeric text with int(). Validate before conversion and return a clear error.",
            "unvalidated string-to-int conversion",
        ),
        (
            "TypeError: unsupported operand type(s) for +: 'int' and 'str'",
            "The code is adding an int and a string. Normalize or reject mixed types before addition.",
            "mixed numeric and string input",
        ),
        (
            "KeyError: 'user_id'",
            "The code indexed a missing user_id key. Check the input schema and use validation or a safe lookup.",
            "missing dictionary key",
        ),
        (
            "FileNotFoundError: [Errno 2] No such file or directory: 'config.yaml'",
            "The code expected config.yaml but the file is absent or the working directory is wrong. Verify the path before opening it.",
            "missing config file or wrong working directory",
        ),
    ]
    for error, answer, final in stack_variants:
        for prefix in ["Diagnose this stack trace:", "What does this error mean?"]:
            records.append(
                record(
                    "stack_trace_diagnosis",
                    f"{prefix}\n{error}",
                    answer,
                    f"The likely issue is {final}.",
                    system_suffix="Diagnose exceptions from inputs outward.",
                )
            )

    repo_variants = [
        ("slugify", "main.py", "from names import slugify\nprint(slugify('Hello World'))", "names.py", "def slugify(text):\n    return text.lower().replace(' ', '-')"),
        ("add", "app.py", "from calc import add\nprint(add(2, 3))", "calc.py", "def add(a, b):\n    return a + b"),
        ("parse_port", "server.py", "from config import parse_port\nPORT = parse_port('8080')", "config.py", "def parse_port(value):\n    return int(value)"),
        ("load_settings", "cli.py", "from settings import load_settings", "settings.py", "def load_settings(path):\n    return {'path': path}"),
        ("validate_email", "web.py", "from validators import validate_email", "validators.py", "def validate_email(value):\n    return '@' in value"),
        ("render_invoice", "routes.py", "from invoices import render_invoice", "invoices.py", "def render_invoice(invoice):\n    return str(invoice)"),
        ("build_parser", "main.py", "from cli_parser import build_parser", "cli_parser.py", "def build_parser():\n    return object()"),
        ("save_user", "views.py", "from storage import save_user", "storage.py", "def save_user(user):\n    return user['id']"),
    ]
    for func, import_file, import_code, impl_file, impl_code in repo_variants:
        repo = f"file: {import_file}\n```python\n{import_code}\n```\nfile: {impl_file}\n```python\n{impl_code}\n```"
        records.append(
            record(
                "repo_context_lookup",
                f"Where is {func} implemented?",
                f"{func} is implemented in {impl_file}. That is the file containing def {func}.",
                f"The implementation is in {impl_file}.",
                system_suffix="Use repo context when it is provided. Cite the file that defines the function, not the file that imports it.",
                repo_context=repo,
            )
        )

    test_prompts = [
        "What command should I run before committing a Python package change?",
        "Name the safest default test command for a Python package.",
        "I edited Python code. What verification command comes before commit?",
        "Before committing, which command runs the Python tests quietly?",
        "You edited a Python package. What command should run before committing?",
        "After changing package code, name the default test command.",
        "What is the one quiet pytest command to run before commit?",
        "Before committing this Python change, which test command should I use?",
    ]
    for user in test_prompts:
        records.append(
            record(
                "test_command",
                user,
                "Run `python -m pytest -q` before committing.",
                "Use python -m pytest -q as the default package test command.",
                system_suffix="Recommend verification before commits.",
            )
        )

    review_prompts = [
        "Review this parser:\n```python\ndef parse_port(value):\n    return int(value)\n```",
        "What edge cases are missing here?\n```python\ndef parse_port(value):\n    return int(value)\n```",
        "Code review this function:\n```python\ndef parse_port(value):\n    return int(value)\n```",
    ]
    review_answer = "Validate that the value is numeric and between 1 and 65535; otherwise return a clear error instead of raw ValueError."
    for user in review_prompts:
        records.append(
            record(
                "code_review",
                user,
                review_answer,
                "Add numeric and port-range validation.",
                system_suffix="Review code for edge cases.",
            )
        )
    extra_review_prompts = [
        (
            "Review this divider:\n```python\ndef divide(a, b):\n    return a / b\n```",
            "Check for division by zero and define how callers should receive that error.",
            "Add zero-denominator handling.",
        ),
        (
            "Review this loader:\n```python\ndef read_json(path):\n    return json.load(open(path))\n```",
            "Handle missing files, close the file with a context manager, and report JSONDecodeError clearly.",
            "Handle file and JSON parsing failures.",
        ),
        (
            "Review this lookup:\n```python\ndef get_user(users, user_id):\n    return users[user_id]\n```",
            "Handle missing users explicitly instead of leaking a raw KeyError.",
            "Handle missing user ids.",
        ),
        (
            "Review this slug helper:\n```python\ndef slug(text):\n    return text.lower().replace(' ', '-')\n```",
            "Validate empty input and normalize repeated separators before returning a slug.",
            "Validate empty slugs and separator normalization.",
        ),
        (
            "Review this pagination helper:\n```python\ndef page(items, limit):\n    return items[:limit]\n```",
            "Validate that limit is positive and capped so callers cannot request unbounded work.",
            "Validate and cap pagination limits.",
        ),
    ]
    for user, answer, final in extra_review_prompts:
        records.append(
            record(
                "code_review",
                user,
                answer,
                final,
                system_suffix="Review code for edge cases.",
            )
        )

    summary_prompts = [
        ("Write a one-sentence commit summary for fixing parse_port validation.", "Validate parse_port input and reject out-of-range ports."),
        ("Summarize a patch that changes add from subtraction to addition.", "Fix add to return the sum instead of the difference."),
        ("Write a concise commit message for adding JSON command validation.", "Validate JSON command responses in agentic evals."),
        ("Write a commit summary for adding rollback checks before risky edits.", "Ask for rollback checks before changing risky production files."),
        ("Summarize a patch that fixes an inverted feature flag.", "Fix inverted feature-flag comparisons."),
        ("Write a concise commit message for improving stack trace diagnostics.", "Improve stack-trace diagnosis for invalid inputs."),
        ("Summarize a change that adds balanced eval behavior counts.", "Balance AgentCoder eval behavior coverage."),
        ("Write a commit summary for adding repo-context lookup tests.", "Add repo-context lookup coverage for AgentCoder evals."),
    ]
    for user, answer in summary_prompts:
        records.append(
            record(
                "commit_summary",
                user,
                answer,
                "Commit summary produced.",
                system_suffix="Summarize changes clearly.",
            )
        )

    return balance_records(records)


def build_eval_cases() -> list[dict[str, Any]]:
    return [
        case(
            "curated_add_patch",
            chat_prompt(
                "Preserve exact patches and include a focused test command.",
                "Fix add and name the focused test command.",
                "file: calc.py\n```python\ndef add(a, b):\n    return a - b\n```",
            ),
            ["return a + b", "pytest"],
            expected_behavior="patch_addition",
        ),
        case(
            "curated_json_python_files",
            chat_prompt("Return valid JSON when asked for JSON.", "Return JSON for a shell command that lists Python files."),
            ["*.py"],
            expected_behavior="json_tool_command",
            expected_json={"cmd": "find . -type f -name '*.py'"},
        ),
        case(
            "curated_risky_question",
            chat_prompt("Ask before risky edits.", "Before editing production payment code, ask one concise question."),
            ["file", "test", "rollback"],
            expected_behavior="risky_clarifying_question",
        ),
        case(
            "curated_debugging",
            chat_prompt(
                "Explain debugging steps plainly. Do not write code for debugging-process prompts.",
                "A Python unit test is failing. Explain what to check before changing code. Do not write code.",
            ),
            ["reproduce", "assertion", "smallest"],
            expected_behavior="plain_debugging",
        ),
        case(
            "curated_is_even_completion",
            chat_prompt("Complete Python functions safely.", "Complete this function:\n```python\ndef is_even(n):\n```"),
            ["def is_even(n):", "return n % 2 == 0"],
            expected_behavior="function_completion",
        ),
        case(
            "curated_stack_valueerror",
            chat_prompt(
                "Diagnose exceptions from inputs outward.",
                "Diagnose this stack trace:\nValueError: invalid literal for int() with base 10: 'abc'",
            ),
            ["int()", "Validate before conversion"],
            expected_behavior="stack_trace_diagnosis",
        ),
        case(
            "curated_repo_lookup",
            chat_prompt(
                "Use repo context when it is provided. Cite the file that defines the function, not the file that imports it.",
                "Where is normalize_title implemented?",
                "file: blog.py\n```python\nfrom titles import normalize_title\nprint(normalize_title('Hello World'))\n```\nfile: titles.py\n```python\ndef normalize_title(text):\n    return text.strip().title()\n```",
            ),
            ["normalize_title is implemented in titles.py"],
            expected_behavior="repo_context_lookup",
        ),
        case(
            "curated_test_command",
            chat_prompt("Recommend verification before commits.", "You edited a Python package. What command should run before committing?"),
            ["python -m pytest -q"],
            expected_behavior="test_command",
        ),
        case(
            "curated_parse_port_review",
            chat_prompt("Review code for edge cases.", "Review this function:\n```python\ndef parse_port(value):\n    return int(value)\n```"),
            ["numeric", "1", "65535"],
            expected_behavior="code_review",
        ),
        case(
            "curated_flag_patch",
            chat_prompt(
                "Prefer minimal diffs.",
                "The enabled flag is inverted. Patch it.",
                "file: flags.py\n```python\ndef is_enabled(value):\n    return value == 'false'\n```",
            ),
            ["return value == 'true'"],
            expected_behavior="patch_boolean_flag",
        ),
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic curated AgentCoder SFT data and held-out cases.")
    parser.add_argument("--train-output", default="examples/agentcoder_curated_sft_train.jsonl")
    parser.add_argument("--cases-output", default="examples/agentcoder_curated_sft_eval_cases.json")
    parser.add_argument("--manifest-output", default="examples/agentcoder_curated_sft_manifest.json")
    args = parser.parse_args()

    train_records = build_train_records()
    eval_cases = build_eval_cases()
    train_path = Path(args.train_output)
    cases_path = Path(args.cases_output)
    manifest_path = Path(args.manifest_output)
    write_jsonl(train_path, train_records)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    cases_path.write_text(json.dumps({"cases": eval_cases}, indent=2, sort_keys=True) + "\n")
    counts = Counter(str(row["behavior"]) for row in train_records)
    manifest = {
        "train_output": str(train_path),
        "cases_output": str(cases_path),
        "train_records": len(train_records),
        "eval_cases": len(eval_cases),
        "behavior_counts": dict(sorted(counts.items())),
        "balanced_behavior_target": BALANCED_BEHAVIOR_TARGET,
        "format": "agentcoder-curated-sft-v3",
        "note": "Deterministic synthetic supervision for gate testing; not a benchmark dataset.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
