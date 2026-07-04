#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
import ast
import difflib
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Iterable, Iterator
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.make_agentcoder_coding_ladder_sft import (
    build_eval_cases as build_ladder_eval_cases,
    build_ladder_base_records,
    expand_records as expand_ladder_records,
    write_jsonl,
)


FORMAT = "agentcoder-executable-coding-sft-v1"
SYSTEM = (
    "You are RAAM-AgentCoder, a precise coding assistant. Produce concise, valid code, "
    "tests, diffs, or strict tool JSON as requested."
)

SOURCE_ORDER = ["ladder", "opencode", "scotch", "coderm_unittest", "commitpackft"]
DATASET_VIEWER_BASE = "https://datasets-server.huggingface.co"
SOURCE_DATASET_NAMES = {
    "opencode": "nvidia/OpenCodeInstruct",
    "scotch": "Samip/Scotch",
    "coderm_unittest": "KAKA22/CodeRM-UnitTest",
    "commitpackft": "bigcode/commitpackft",
}


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        value = coerce_text(row.get(key)).strip()
        if value:
            return value
    return ""


def iter_jsonl(path: str | Path | None) -> Iterator[dict[str, Any]]:
    if not path:
        return
    with Path(path).open("r", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no} must contain a JSON object")
            yield row


def extract_fenced_code(text: str) -> str:
    if "```python" in text:
        chunk = text.split("```python", 1)[1]
        return chunk.split("```", 1)[0].strip()
    if "```" in text:
        chunk = text.split("```", 1)[1]
        return chunk.split("```", 1)[0].strip()
    return text.strip()


def fenced_python(code: str) -> str:
    return f"```python\n{code.rstrip()}\n```"


def fenced_diff(diff: str) -> str:
    return f"```diff\n{diff.rstrip()}\n```"


def python_language(row: dict[str, Any]) -> bool:
    fields = [
        row.get("language"),
        row.get("lang"),
        row.get("programming_language"),
        row.get("repo_language"),
        row.get("domain"),
        row.get("tags"),
    ]
    haystack = " ".join(coerce_text(field).lower() for field in fields)
    if not haystack.strip() or "python" in haystack or haystack.strip() == "py":
        return True
    codeish = first_text(
        row,
        [
            "output",
            "answer",
            "solution",
            "response",
            "completion",
            "code",
            "code_ground_truth",
            "function",
            "func_code",
            "old_contents",
            "new_contents",
        ],
    ).lower()
    return "```python" in codeish or "\ndef " in f"\n{codeish}"


def parsed_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


def syntax_ok(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def line_count(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def parse_score(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def row_passes_execution_filter(row: dict[str, Any], *, min_average_score: float) -> bool:
    score = None
    for key in ["average_test_score", "avg_test_score", "test_score", "score"]:
        score = parse_score(row.get(key))
        if score is not None:
            break
    if score is not None and score < min_average_score:
        return False

    status = coerce_text(
        row.get("execution_status")
        or row.get("status")
        or row.get("passed")
        or row.get("is_correct")
    ).strip().lower()
    if status in {"false", "failed", "fail", "0", "incorrect", "error"}:
        return False
    return True


def normalize_structured_tests(raw: Any) -> list[dict[str, Any]]:
    raw = maybe_json(raw)
    if isinstance(raw, dict):
        raw = raw.get("tests") or raw.get("unit_tests") or raw.get("cases") or []
    if not isinstance(raw, list):
        return []

    tests: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        if "args" in item or "expected" in item or "raises" in item:
            test = {
                "args": item.get("args", []),
                "kwargs": item.get("kwargs", {}),
            }
            if "raises" in item:
                test["raises"] = item["raises"]
            else:
                test["expected"] = item.get("expected")
            tests.append(test)
            continue
        if "input" in item and "output" in item:
            value = item["input"]
            args = value if isinstance(value, list) else [value]
            tests.append({"args": args, "kwargs": {}, "expected": item["output"]})
    return tests


def normalize_assertion_tests(raw: Any, *, max_tests: int = 8, max_chars: int = 4000) -> list[str]:
    raw = maybe_json(raw)
    candidates: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, str):
                candidates.append(item)
            elif isinstance(item, dict):
                code = first_text(item, ["code", "test_code", "unit_test", "tests"])
                if code:
                    candidates.append(code)
    elif isinstance(raw, str):
        candidates.append(raw)

    tests: list[str] = []
    total_chars = 0
    for candidate in candidates:
        for line in str(candidate).splitlines():
            stripped = line.strip()
            if not stripped.startswith("assert "):
                continue
            if total_chars + len(stripped) > max_chars:
                return tests
            tests.append(stripped)
            total_chars += len(stripped)
            if len(tests) >= max_tests:
                return tests
    return tests


def test_payload(row: dict[str, Any]) -> Any:
    for key in ["unit_tests", "tests", "test_cases", "public_tests", "visible_tests"]:
        if row.get(key) not in (None, ""):
            return row[key]
    return None


def test_text(row: dict[str, Any]) -> str:
    raw = test_payload(row)
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return json.dumps(raw, ensure_ascii=True, sort_keys=True)


def compact_preview(value: Any, *, max_chars: int = 180) -> str:
    text = coerce_text(value).replace("\n", "\\n")
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}..."


def row_example(row: dict[str, Any], *, source_id: str) -> dict[str, Any]:
    return {
        "source_id": source_id,
        "keys": sorted(str(key) for key in row.keys()),
        "question": compact_preview(row.get("question") or row.get("instruction") or row.get("prompt") or row.get("input")),
        "code": compact_preview(
            row.get("code_ground_truth")
            or row.get("func_code")
            or row.get("function")
            or row.get("solution")
            or row.get("output")
            or row.get("new_contents")
        ),
        "tests": compact_preview(test_payload(row)),
    }


def source_dataset_name(source_key: str) -> str:
    return SOURCE_DATASET_NAMES.get(source_key, source_key)


def make_record(
    *,
    topic: str,
    behavior: str,
    user: str,
    assistant: str,
    source_dataset: str,
    source_id: str,
    repo_context: str = "",
    license_value: str = "",
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "topic": topic,
        "behavior": behavior,
        "source_dataset": source_dataset,
        "source_id": source_id,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": user.strip()}],
        "trace": [{"type": "assistant", "content": assistant.strip()}],
        "final": "",
        "_curriculum_source": f"{source_dataset}:{source_id}",
    }
    if repo_context.strip():
        record["repo_context"] = repo_context.strip()
    if license_value.strip():
        record["license"] = license_value.strip()
    return record


def row_source_id(row: dict[str, Any], fallback_index: int) -> str:
    for key in ["id", "task_id", "problem_id", "instance_id", "repo", "path", "file_path"]:
        value = coerce_text(row.get(key)).strip()
        if value:
            return value[:120]
    return f"row_{fallback_index:06d}"


def opencode_row_to_record(
    row: dict[str, Any],
    *,
    source_id: str,
    min_average_score: float,
    require_test_signal: bool,
    max_answer_chars: int,
) -> dict[str, Any] | None:
    if not python_language(row):
        return None
    if not row_passes_execution_filter(row, min_average_score=min_average_score):
        return None

    prompt = first_text(row, ["question", "instruction", "problem", "prompt", "input"])
    answer = first_text(row, ["answer", "solution", "response", "output", "completion", "code"])
    if not prompt or not answer:
        return None
    tests = test_text(row)
    if require_test_signal and not tests and parse_score(row.get("average_test_score")) is None:
        return None
    if len(answer) > max_answer_chars:
        return None

    user = prompt
    if tests and len(tests) < 1600:
        user = f"{prompt}\n\nUse these tests as the behavioral contract:\n{tests}"
    code = extract_fenced_code(answer)
    behavior = "function_completion" if parsed_function_name(code) else "code_generation"
    assistant = fenced_python(code) if behavior == "function_completion" and "```" not in answer else answer
    return make_record(
        topic="opencode_python",
        behavior=behavior,
        user=user,
        assistant=assistant,
        source_dataset="nvidia/OpenCodeInstruct",
        source_id=source_id,
        license_value=coerce_text(row.get("license")),
    )


def scotch_row_to_record(
    row: dict[str, Any],
    *,
    source_id: str,
    max_function_lines: int,
) -> dict[str, Any] | None:
    if not python_language(row):
        return None
    code = first_text(row, ["func_code", "function", "function_body", "code", "content"])
    code = extract_fenced_code(code)
    if not code or line_count(code) > max_function_lines or not syntax_ok(code):
        return None
    function_name = parsed_function_name(code)
    if not function_name:
        return None
    docstring = first_text(row, ["docstring", "documentation", "description", "summary"])
    signature = first_text(row, ["signature", "function_signature"]) or f"{function_name}(...)"
    path = first_text(row, ["path", "file_path", "repo_path"])
    prompt = (
        f"Implement the Python function `{signature}`."
        + (f"\nDocstring/behavior:\n{docstring}" if docstring else "")
        + "\nReturn only a Python code block."
    )
    repo_context = "\n".join(bit for bit in [f"path: {path}" if path else "", f"repo: {row.get('repo', '')}"] if bit.strip())
    return make_record(
        topic="scotch_function",
        behavior="function_completion",
        user=prompt,
        assistant=fenced_python(code),
        source_dataset="Samip/Scotch",
        source_id=source_id,
        repo_context=repo_context,
        license_value=coerce_text(row.get("license")),
    )


def coderm_row_to_records(
    row: dict[str, Any],
    *,
    source_id: str,
    max_code_chars: int,
    max_tests_chars: int,
) -> list[dict[str, Any]]:
    if not python_language(row):
        return []
    code = first_text(row, ["code_ground_truth", "ground_truth", "canonical_solution", "solution", "code", "answer"])
    tests = coderm_test_code(row, max_tests_chars=max_tests_chars)
    code = extract_fenced_code(code)
    if not code or not tests or len(code) > max_code_chars or len(tests) > max_tests_chars:
        return []
    if not syntax_ok(code):
        return []
    records = [
        make_record(
            topic="coderm_pytest",
            behavior="pytest_generation",
            user=f"Write focused pytest tests for this Python code:\n{fenced_python(code)}",
            assistant=fenced_python(tests) if "```" not in tests else tests,
            source_dataset="KAKA22/CodeRM-UnitTest",
            source_id=source_id,
            license_value=coerce_text(row.get("license")),
        )
    ]
    prompt = first_text(row, ["question", "prompt", "instruction", "problem"])
    if prompt:
        records.append(
            make_record(
                topic="coderm_implementation",
                behavior="function_completion",
                user=f"{prompt}\n\nThe implementation should satisfy these tests:\n{tests}",
                assistant=fenced_python(code),
                source_dataset="KAKA22/CodeRM-UnitTest",
                source_id=f"{source_id}:implementation",
                license_value=coerce_text(row.get("license")),
            )
        )
    return records


def coderm_test_code(row: dict[str, Any], *, max_tests_chars: int) -> str:
    raw = maybe_json(test_payload(row) or row.get("test") or row.get("test_code"))
    snippets: list[tuple[float, int, str]] = []
    if isinstance(raw, list):
        for item in raw:
            if not isinstance(item, dict):
                continue
            code = first_text(item, ["code", "test_code", "unit_test", "tests"])
            if not code:
                continue
            far = parse_score(item.get("FAR"))
            frr = parse_score(item.get("FRR"))
            quality_penalty = (far or 0.0) + (frr or 0.0)
            snippets.append((quality_penalty, len(code), code.strip()))
    elif isinstance(raw, dict):
        code = first_text(raw, ["code", "test_code", "unit_test", "tests"])
        if code:
            snippets.append((0.0, len(code), code.strip()))
    elif isinstance(raw, str):
        code = raw.strip()
        if code:
            snippets.append((0.0, len(code), code))

    for _, _, code in sorted(snippets):
        if len(code) <= max_tests_chars and syntax_ok(code):
            return code
    for _, _, code in sorted(snippets):
        if len(code) <= max_tests_chars:
            return code
    return ""


def coderm_skip_reason(row: dict[str, Any], *, max_code_chars: int, max_tests_chars: int) -> str:
    if not python_language(row):
        return "non_python"
    code = extract_fenced_code(
        first_text(row, ["code_ground_truth", "ground_truth", "canonical_solution", "solution", "code", "answer"])
    )
    if not code:
        return "missing_code"
    if len(code) > max_code_chars:
        return "code_too_long"
    if not syntax_ok(code):
        return "code_syntax_error"
    tests = coderm_test_code(row, max_tests_chars=max_tests_chars)
    if not tests:
        raw_tests = first_text(row, ["unit_tests", "tests", "test", "test_code"])
        if raw_tests and len(raw_tests) > max_tests_chars:
            return "tests_too_long_or_no_short_snippet"
        return "missing_tests"
    return "unknown_converter_rejection"


def converter_skip_reason(source_key: str, row: dict[str, Any], args: argparse.Namespace) -> str:
    if source_key == "coderm_unittest":
        return coderm_skip_reason(row, max_code_chars=args.max_answer_chars, max_tests_chars=args.max_tests_chars)
    if not python_language(row):
        return "non_python"
    if source_key == "opencode":
        if not row_passes_execution_filter(row, min_average_score=args.min_average_score):
            return "execution_filter_failed"
        prompt = first_text(row, ["question", "instruction", "problem", "prompt", "input"])
        answer = first_text(row, ["answer", "solution", "response", "output", "completion", "code"])
        if not prompt:
            return "missing_prompt"
        if not answer:
            return "missing_answer"
        if args.require_opencode_test_signal and not test_text(row) and parse_score(row.get("average_test_score")) is None:
            return "missing_test_signal"
        if len(answer) > args.max_answer_chars:
            return "answer_too_long"
    if source_key == "scotch":
        code = extract_fenced_code(first_text(row, ["func_code", "function", "function_body", "code", "content"]))
        if not code:
            return "missing_function_code"
        if line_count(code) > args.max_function_lines:
            return "function_too_long"
        if not syntax_ok(code):
            return "function_syntax_error"
        if not parsed_function_name(code):
            return "missing_function_def"
    if source_key == "commitpackft":
        old = first_text(row, ["old_contents", "old_content", "before", "old_file"])
        new = first_text(row, ["new_contents", "new_content", "after", "new_file"])
        if not old or not new:
            return "missing_old_or_new_contents"
        if old == new:
            return "unchanged_contents"
        if len(old) > args.max_file_chars or len(new) > args.max_file_chars:
            return "file_too_long"
    return "unknown_converter_rejection"


def commitpackft_row_to_record(
    row: dict[str, Any],
    *,
    source_id: str,
    max_diff_lines: int,
    max_file_chars: int,
) -> dict[str, Any] | None:
    if not python_language(row):
        return None
    old = first_text(row, ["old_contents", "old_content", "before", "old_file"])
    new = first_text(row, ["new_contents", "new_content", "after", "new_file"])
    if not old or not new or old == new:
        return None
    if len(old) > max_file_chars or len(new) > max_file_chars:
        return None
    path = first_text(row, ["path", "file_path", "filename"]) or "file.py"
    if not path.endswith(".py"):
        path = f"{Path(path).stem or 'file'}.py"
    diff_lines = list(
        difflib.unified_diff(
            old.splitlines(),
            new.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
    )
    if not diff_lines or len(diff_lines) > max_diff_lines:
        return None
    message = first_text(row, ["commit_message", "message", "subject", "instruction"]) or "Apply the requested Python edit."
    return make_record(
        topic="commitpackft_patch",
        behavior="patch_generation",
        user=f"Apply this commit intent to `{path}` as a minimal unified diff:\n{message}",
        assistant=fenced_diff("\n".join(diff_lines)),
        source_dataset="bigcode/commitpackft",
        source_id=source_id,
        repo_context=f"file: {path}\n```python\n{old}\n```",
        license_value=coerce_text(row.get("license")),
    )


def eval_case_from_structured_row(
    row: dict[str, Any],
    *,
    source_dataset: str,
    source_id: str,
    max_answer_chars: int,
) -> dict[str, Any] | None:
    prompt = first_text(row, ["question", "instruction", "problem", "prompt", "input"])
    answer = first_text(row, ["answer", "solution", "response", "output", "completion", "code", "canonical_solution"])
    tests = normalize_structured_tests(test_payload(row))
    assert_tests = normalize_assertion_tests(test_payload(row), max_chars=max_answer_chars)
    if not prompt or not answer or not tests:
        if not prompt or not answer or not assert_tests:
            return None
    code = extract_fenced_code(answer)
    if len(code) > max_answer_chars or not syntax_ok(code):
        return None
    function_name = coerce_text(row.get("function_name") or row.get("entry_point")).strip() or parsed_function_name(code)
    if not function_name:
        return None
    case: dict[str, Any] = {
        "name": f"{source_dataset.replace('/', '_').replace('-', '_')}_{stable_hash(source_id)}",
        "topic": f"{source_dataset}:heldout_function",
        "prompt": (
            "<|system|>\n"
            "Write complete, valid Python functions. Stop after the requested code; do not add explanations.\n\n"
            "<|user|>\n"
            f"{prompt}\n\n<|assistant|>\n"
        ),
        "required_substrings": [f"def {function_name}"],
        "forbidden_substrings": ["```diff", "pytest", "<|user|>", "<|assistant|>"],
        "expected_behavior": "function_completion",
        "no_trailing_text": True,
        "max_completion_chars": max_answer_chars,
        "source_dataset": source_dataset,
        "source_id": source_id,
    }
    if tests:
        case["python_function"] = {"name": function_name, "tests": tests}
    else:
        case["python_assert_tests"] = assert_tests
        case["python_syntax"] = True
    return case


def eval_case_from_coderm_row(
    row: dict[str, Any],
    *,
    source_id: str,
    max_answer_chars: int,
    max_tests_chars: int,
) -> dict[str, Any] | None:
    code = extract_fenced_code(first_text(row, ["code_ground_truth", "ground_truth", "canonical_solution", "solution", "code"]))
    tests = coderm_test_code(row, max_tests_chars=max_tests_chars)
    if not code or not tests or len(code) > max_answer_chars:
        return None
    if not syntax_ok(code):
        return None
    function_name = parsed_function_name(code)
    required = ["def test_", "assert"]
    if function_name:
        required.append(function_name)
    return {
        "name": f"KAKA22_CodeRM_UnitTest_{stable_hash(source_id)}",
        "topic": "KAKA22/CodeRM-UnitTest:heldout_pytest",
        "prompt": (
            "<|system|>\n"
            "Write small Python unit tests with concrete assertions. Return only valid test code.\n\n"
            "<|user|>\n"
            f"Write focused pytest-style tests for this Python code:\n{fenced_python(code)}\n\n<|assistant|>\n"
        ),
        "required_substrings": required,
        "forbidden_substrings": ["```diff", "<|user|>", "<|assistant|>"],
        "python_syntax": True,
        "expected_behavior": "pytest_generation",
        "no_trailing_text": True,
        "max_completion_chars": max_tests_chars,
        "source_dataset": "KAKA22/CodeRM-UnitTest",
        "source_id": source_id,
    }


def is_eval_only(row: dict[str, Any]) -> bool:
    split = coerce_text(row.get("split") or row.get("partition") or row.get("eval_tier")).strip().lower()
    return bool(row.get("eval_only")) or split in {"eval", "validation", "test", "heldout"}


def maybe_take_eval(
    row: dict[str, Any],
    *,
    rng: random.Random,
    eval_source_fraction: float,
) -> bool:
    if is_eval_only(row):
        return True
    return eval_source_fraction > 0 and rng.random() < eval_source_fraction


def dataset_viewer_json(endpoint: str, params: dict[str, Any], *, page_delay_sec: float = 0.0) -> dict[str, Any]:
    url = f"{DATASET_VIEWER_BASE}/{endpoint}?{urlencode(params)}"
    request = Request(url, headers={"User-Agent": "raam-lm-executable-sft-builder/1.0"})
    payload: dict[str, Any] | None = None
    last_error: Exception | None = None
    for attempt in range(4):
        if page_delay_sec > 0:
            time.sleep(page_delay_sec)
        try:
            with urlopen(request, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            break
        except HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            last_error = RuntimeError(f"Dataset Viewer request failed HTTP {exc.code}: {url}\n{body[:1000]}")
            if exc.code not in {429, 500, 502, 503, 504} or attempt == 3:
                raise last_error from exc
            if exc.code == 429:
                time.sleep(15 * (attempt + 1))
        except (URLError, TimeoutError) as exc:
            last_error = RuntimeError(f"Dataset Viewer request failed: {url}: {exc}")
            if attempt == 3:
                raise last_error from exc
        time.sleep(2**attempt)
    if payload is None:
        raise RuntimeError(f"Dataset Viewer request failed: {url}: {last_error}")
    if isinstance(payload, dict) and payload.get("error"):
        raise RuntimeError(f"Dataset Viewer returned error for {url}: {payload['error']}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"Dataset Viewer returned non-object payload for {url}")
    return payload


def load_hf_viewer_rows(
    dataset: str,
    config: str,
    split: str,
    *,
    page_size: int = 100,
    page_delay_sec: float = 0.0,
) -> Iterator[dict[str, Any]]:
    offset = 0
    while True:
        payload = dataset_viewer_json(
            "rows",
            {
                "dataset": dataset,
                "config": config or "default",
                "split": split,
                "offset": offset,
                "length": min(max(1, page_size), 100),
            },
            page_delay_sec=page_delay_sec,
        )
        rows = payload.get("rows", [])
        if not rows:
            break
        for item in rows:
            row = item.get("row") if isinstance(item, dict) else None
            if isinstance(row, dict):
                yield row
        offset += len(rows)
        total = payload.get("num_rows_total")
        if isinstance(total, int) and offset >= total:
            break


def load_hf_rows(dataset: str, config: str, split: str, *, page_delay_sec: float = 0.0) -> Iterable[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return load_hf_viewer_rows(dataset, config, split, page_delay_sec=page_delay_sec)
    args = () if not config or config == "default" else (config,)
    return load_dataset(dataset, *args, split=split, streaming=True)


def limit_rows(rows: Iterable[dict[str, Any]], limit: int) -> Iterator[dict[str, Any]]:
    if limit <= 0:
        return
    emitted = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        yield row
        emitted += 1
        if emitted >= limit:
            break


def train_user_prompts(records: Iterable[dict[str, Any]]) -> set[str]:
    prompts: set[str] = set()
    for row in records:
        for message in row.get("messages", []):
            if message.get("role") == "user":
                prompts.add(str(message.get("content", "")))
    return prompts


def eval_user_prompt(case: dict[str, Any]) -> str:
    prompt = str(case.get("prompt", ""))
    marker = "<|user|>\n"
    if marker not in prompt:
        return prompt
    after = prompt.split(marker, 1)[1]
    return after.split("\n\n<|assistant|>", 1)[0].strip()


def write_cases(path: Path, cases: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cases": cases}, indent=2, sort_keys=True) + "\n")


def filter_eval_cases(
    cases: list[dict[str, Any]],
    *,
    expected_behaviors: list[str] | None = None,
    topic_contains: list[str] | None = None,
) -> list[dict[str, Any]]:
    behaviors = {value for value in (expected_behaviors or []) if value}
    topic_needles = [value for value in (topic_contains or []) if value]
    filtered = cases
    if behaviors:
        filtered = [case for case in filtered if str(case.get("expected_behavior", "")) in behaviors]
    if topic_needles:
        filtered = [
            case
            for case in filtered
            if any(needle in str(case.get("topic", "")) for needle in topic_needles)
        ]
    if (behaviors or topic_needles) and not filtered:
        raise ValueError("eval filters removed all cases")
    return filtered


def filter_train_records(
    records: list[dict[str, Any]],
    *,
    behaviors: list[str] | None = None,
    topic_contains: list[str] | None = None,
) -> list[dict[str, Any]]:
    behavior_set = {value for value in (behaviors or []) if value}
    topic_needles = [value for value in (topic_contains or []) if value]
    filtered = records
    if behavior_set:
        filtered = [record for record in filtered if str(record.get("behavior", "")) in behavior_set]
    if topic_needles:
        filtered = [
            record
            for record in filtered
            if any(needle in str(record.get("topic", "")) for needle in topic_needles)
        ]
    if (behavior_set or topic_needles) and not filtered:
        raise ValueError("train filters removed all records")
    return filtered


def add_rows_from_source(
    *,
    source_key: str,
    rows: Iterable[dict[str, Any]],
    records: list[dict[str, Any]],
    eval_cases: list[dict[str, Any]],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict[str, Any]:
    input_rows = 0
    train_added = 0
    eval_added = 0
    skipped = 0
    skip_reasons: Counter[str] = Counter()
    eval_skip_reasons: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        input_rows += 1
        source_id = row_source_id(row, index)
        if len(examples) < 3:
            examples.append(row_example(row, source_id=source_id))
        if maybe_take_eval(row, rng=rng, eval_source_fraction=args.eval_source_fraction):
            if source_key == "coderm_unittest":
                case = eval_case_from_coderm_row(
                    row,
                    source_id=source_id,
                    max_answer_chars=args.max_answer_chars,
                    max_tests_chars=args.max_tests_chars,
                )
            else:
                case = eval_case_from_structured_row(
                    row,
                    source_dataset=source_dataset_name(source_key),
                    source_id=source_id,
                    max_answer_chars=args.max_answer_chars,
                )
            if case is not None:
                eval_cases.append(case)
                eval_added += 1
                continue
            eval_skip_reasons["unsupported_eval_case"] += 1
            if is_eval_only(row):
                skipped += 1
                skip_reasons["eval_only_without_supported_eval_case"] += 1
                continue

        before = len(records)
        if source_key == "opencode":
            record = opencode_row_to_record(
                row,
                source_id=source_id,
                min_average_score=args.min_average_score,
                require_test_signal=args.require_opencode_test_signal,
                max_answer_chars=args.max_answer_chars,
            )
            if record:
                records.append(record)
        elif source_key == "scotch":
            record = scotch_row_to_record(row, source_id=source_id, max_function_lines=args.max_function_lines)
            if record:
                records.append(record)
        elif source_key == "coderm_unittest":
            records.extend(
                coderm_row_to_records(
                    row,
                    source_id=source_id,
                    max_code_chars=args.max_answer_chars,
                    max_tests_chars=args.max_tests_chars,
                )
            )
        elif source_key == "commitpackft":
            record = commitpackft_row_to_record(
                row,
                source_id=source_id,
                max_diff_lines=args.max_diff_lines,
                max_file_chars=args.max_file_chars,
            )
            if record:
                records.append(record)
        else:
            raise ValueError(f"unknown source key: {source_key}")
        added = len(records) - before
        if added:
            train_added += added
        else:
            skipped += 1
            skip_reasons[converter_skip_reason(source_key, row, args)] += 1
    return {
        "input_rows": input_rows,
        "train_records_added": train_added,
        "eval_cases_added": eval_added,
        "skipped_rows": skipped,
        "skip_reasons": dict(sorted(skip_reasons.items())),
        "eval_skip_reasons": dict(sorted(eval_skip_reasons.items())),
        "sample_examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a small executable-code RAAM-AgentCoder SFT mix from local fixtures or optional HF streams."
    )
    parser.add_argument("--output-dir", default="runs/agentcoder_executable_sft_data")
    parser.add_argument("--train-output", default="")
    parser.add_argument("--cases-output", default="")
    parser.add_argument("--manifest-output", default="")
    parser.add_argument("--opencode-jsonl", default="")
    parser.add_argument("--scotch-jsonl", default="")
    parser.add_argument("--coderm-unittest-jsonl", default="")
    parser.add_argument("--commitpackft-jsonl", default="")
    parser.add_argument("--use-hf", action="store_true")
    parser.add_argument("--opencode-limit", type=int, default=0)
    parser.add_argument("--scotch-limit", type=int, default=0)
    parser.add_argument("--coderm-unittest-limit", type=int, default=0)
    parser.add_argument("--commitpackft-limit", type=int, default=0)
    parser.add_argument("--opencode-config", default="train")
    parser.add_argument("--opencode-split", default="train")
    parser.add_argument("--scotch-config", default="python")
    parser.add_argument("--scotch-split", default="train")
    parser.add_argument("--coderm-unittest-config", default="default")
    parser.add_argument("--coderm-unittest-split", default="train")
    parser.add_argument("--commitpackft-config", default="python")
    parser.add_argument("--commitpackft-split", default="train")
    parser.add_argument("--hf-page-delay-sec", type=float, default=0.0)
    parser.add_argument("--ladder-repeats", type=int, default=4)
    parser.add_argument("--curated-anchor-repeats", type=int, default=0)
    parser.add_argument("--include-ladder-eval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--train-behavior",
        action="append",
        default=[],
        help="Keep only training records with this behavior. Repeat for multiple behaviors.",
    )
    parser.add_argument(
        "--train-topic-contains",
        action="append",
        default=[],
        help="Keep only training records whose topic contains this substring. Repeat for multiple substrings.",
    )
    parser.add_argument(
        "--eval-expected-behavior",
        action="append",
        default=[],
        help="Keep only eval cases with this expected_behavior. Repeat for multiple behaviors.",
    )
    parser.add_argument(
        "--eval-topic-contains",
        action="append",
        default=[],
        help="Keep only eval cases whose topic contains this substring. Repeat for multiple substrings.",
    )
    parser.add_argument("--eval-source-fraction", type=float, default=0.0)
    parser.add_argument("--min-average-score", type=float, default=0.8)
    parser.add_argument("--require-opencode-test-signal", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-answer-chars", type=int, default=3000)
    parser.add_argument("--max-tests-chars", type=int, default=3000)
    parser.add_argument("--max-function-lines", type=int, default=80)
    parser.add_argument("--max-diff-lines", type=int, default=80)
    parser.add_argument("--max-file-chars", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_output = Path(args.train_output) if args.train_output else output_dir / "agentcoder_executable_train.jsonl"
    cases_output = Path(args.cases_output) if args.cases_output else output_dir / "agentcoder_executable_eval_cases.json"
    manifest_output = Path(args.manifest_output) if args.manifest_output else output_dir / "agentcoder_executable_manifest.json"

    rng = random.Random(args.seed)
    records = expand_ladder_records(
        build_ladder_base_records(),
        ladder_repeats=args.ladder_repeats,
        curated_anchor_repeats=args.curated_anchor_repeats,
        seed=args.seed,
    )
    eval_cases = build_ladder_eval_cases() if args.include_ladder_eval else []
    sources: dict[str, Any] = {
        "ladder": {
            "train_records_added": len(records),
            "eval_cases_added": len(eval_cases),
            "ladder_repeats": args.ladder_repeats,
            "curated_anchor_repeats": args.curated_anchor_repeats,
        }
    }

    local_sources = {
        "opencode": args.opencode_jsonl,
        "scotch": args.scotch_jsonl,
        "coderm_unittest": args.coderm_unittest_jsonl,
        "commitpackft": args.commitpackft_jsonl,
    }
    for key, path in local_sources.items():
        if path:
            sources[key] = add_rows_from_source(
                source_key=key,
                rows=iter_jsonl(path),
                records=records,
                eval_cases=eval_cases,
                args=args,
                rng=rng,
            )

    if args.use_hf:
        hf_specs = {
            "opencode": ("nvidia/OpenCodeInstruct", args.opencode_config, args.opencode_split, args.opencode_limit),
            "scotch": ("Samip/Scotch", args.scotch_config, args.scotch_split, args.scotch_limit),
            "coderm_unittest": (
                "KAKA22/CodeRM-UnitTest",
                args.coderm_unittest_config,
                args.coderm_unittest_split,
                args.coderm_unittest_limit,
            ),
            "commitpackft": ("bigcode/commitpackft", args.commitpackft_config, args.commitpackft_split, args.commitpackft_limit),
        }
        for key, (dataset, config, split, limit) in hf_specs.items():
            if limit <= 0:
                continue
            rows = limit_rows(load_hf_rows(dataset, config, split, page_delay_sec=args.hf_page_delay_sec), limit)
            sources[f"{key}:hf"] = add_rows_from_source(
                source_key=key,
                rows=rows,
                records=records,
                eval_cases=eval_cases,
                args=args,
                rng=rng,
            )

    train_records_before_filter = len(records)
    records = filter_train_records(
        records,
        behaviors=args.train_behavior,
        topic_contains=args.train_topic_contains,
    )

    eval_cases_before_filter = len(eval_cases)
    eval_cases = filter_eval_cases(
        eval_cases,
        expected_behaviors=args.eval_expected_behavior,
        topic_contains=args.eval_topic_contains,
    )

    train_prompts = train_user_prompts(records)
    eval_prompts = {eval_user_prompt(case) for case in eval_cases}
    prompt_overlaps = sorted(train_prompts & eval_prompts)
    if prompt_overlaps:
        raise ValueError(f"train/eval exact user-prompt overlap: {prompt_overlaps[:5]}")

    behavior_counts = Counter(str(row.get("behavior", "unknown")) for row in records)
    topic_counts = Counter(str(row.get("topic", "unknown")) for row in records)
    dataset_counts = Counter(str(row.get("source_dataset", "local_ladder")) for row in records)
    eval_dataset_counts = Counter(str(case.get("source_dataset", "local_ladder")) for case in eval_cases)
    eval_behavior_counts = Counter(str(case.get("expected_behavior", "unknown")) for case in eval_cases)
    write_jsonl(train_output, records)
    write_cases(cases_output, eval_cases)

    manifest = {
        "format": FORMAT,
        "seed": args.seed,
        "train_output": str(train_output),
        "cases_output": str(cases_output),
        "train_records": len(records),
        "eval_cases": len(eval_cases),
        "sources": sources,
        "source_order": SOURCE_ORDER,
        "behavior_counts": dict(sorted(behavior_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "source_dataset_counts": dict(sorted(dataset_counts.items())),
        "eval_source_dataset_counts": dict(sorted(eval_dataset_counts.items())),
        "eval_behavior_counts": dict(sorted(eval_behavior_counts.items())),
        "train_user_prompt_count": len(train_prompts),
        "eval_user_prompt_count": len(eval_prompts),
        "train_user_prompt_hashes": sorted(stable_hash(prompt) for prompt in train_prompts),
        "eval_user_prompt_hashes": sorted(stable_hash(prompt) for prompt in eval_prompts),
        "exact_train_eval_user_prompt_overlaps": prompt_overlaps,
        "filters": {
            "min_average_score": args.min_average_score,
            "require_opencode_test_signal": args.require_opencode_test_signal,
            "max_answer_chars": args.max_answer_chars,
            "max_tests_chars": args.max_tests_chars,
            "max_function_lines": args.max_function_lines,
            "max_diff_lines": args.max_diff_lines,
            "max_file_chars": args.max_file_chars,
            "eval_source_fraction": args.eval_source_fraction,
            "train_behavior": args.train_behavior,
            "train_topic_contains": args.train_topic_contains,
            "train_records_before_filter": train_records_before_filter,
            "eval_expected_behavior": args.eval_expected_behavior,
            "eval_topic_contains": args.eval_topic_contains,
            "eval_cases_before_filter": eval_cases_before_filter,
        },
        "notes": [
            "SWE-bench gold patches are not used by this builder.",
            "Public-source code is filtered and formatted for SFT; arbitrary public code is not executed during data preparation.",
            "Held-out eval cases use structured JSON args/expected tests when a source row provides them.",
            "CodeRM held-out eval cases use syntax-checked unit-test generation prompts from extracted test snippets.",
            "Each source entry includes skip_reasons, eval_skip_reasons, and sample_examples for data-pipeline audits.",
        ],
    }
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
