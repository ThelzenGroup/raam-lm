from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.eval_coding_ladder import score_case
from scripts.make_agentcoder_executable_sft import (
    coderm_test_code,
    coderm_row_to_records,
    commitpackft_row_to_record,
    eval_case_from_coderm_row,
    eval_case_from_structured_row,
    filter_eval_cases,
    filter_train_records,
    load_hf_viewer_rows,
    opencode_row_to_record,
    scotch_row_to_record,
)


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n")


def test_source_converters_filter_and_format_executable_records():
    opencode = {
        "language": "python",
        "question": "Implement double(n).",
        "solution": "def double(n):\n    return n * 2",
        "average_test_score": 1.0,
        "unit_tests": [{"args": [3], "expected": 6}],
    }
    record = opencode_row_to_record(
        opencode,
        source_id="double",
        min_average_score=0.8,
        require_test_signal=True,
        max_answer_chars=1000,
    )
    assert record is not None
    assert record["source_dataset"] == "nvidia/OpenCodeInstruct"
    assert record["behavior"] == "function_completion"
    assert "Use these tests" in record["messages"][0]["content"]

    low_score = dict(opencode, average_test_score=0.2)
    assert (
        opencode_row_to_record(
            low_score,
            source_id="bad",
            min_average_score=0.8,
            require_test_signal=True,
            max_answer_chars=1000,
        )
        is None
    )
    generic_domain_opencode = dict(opencode)
    generic_domain_opencode.pop("language")
    generic_domain_opencode["domain"] = "generic"
    assert (
        opencode_row_to_record(
            generic_domain_opencode,
            source_id="generic-python",
            min_average_score=0.8,
            require_test_signal=True,
            max_answer_chars=1000,
        )
        is not None
    )

    scotch = {
        "language": "python",
        "signature": "triple(n)",
        "docstring": "Return n times three.",
        "func_code": "def triple(n):\n    return n * 3",
        "path": "maths.py",
        "license": "MIT",
    }
    scotch_record = scotch_row_to_record(scotch, source_id="triple", max_function_lines=20)
    assert scotch_record is not None
    assert scotch_record["topic"] == "scotch_function"
    assert "path: maths.py" in scotch_record["repo_context"]

    coderm = {
        "language": "python",
        "question": "Implement clamp.",
        "code_ground_truth": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))",
        "unit_tests": "from solution import clamp\n\ndef test_clamp_bounds():\n    assert clamp(5, 1, 3) == 3\n",
    }
    coderm_records = coderm_row_to_records(coderm, source_id="clamp", max_code_chars=1000, max_tests_chars=1000)
    assert {row["behavior"] for row in coderm_records} == {"pytest_generation", "function_completion"}

    coderm_list_tests = {
        "question": "Implement add_one.",
        "code_ground_truth": "def add_one(x):\n    return x + 1",
        "unit_tests": json.dumps(
            [
                {
                    "ut_id": 0,
                    "code": "import unittest\n\nclass TestAddOne(unittest.TestCase):\n    def test_positive(self):\n        self.assertEqual(add_one(2), 3)\n",
                    "FAR": 0.0,
                    "FRR": 0.0,
                }
            ]
        ),
    }
    assert "TestAddOne" in coderm_test_code(coderm_list_tests, max_tests_chars=1000)
    coderm_list_records = coderm_row_to_records(
        coderm_list_tests,
        source_id="add_one",
        max_code_chars=1000,
        max_tests_chars=1000,
    )
    assert {row["behavior"] for row in coderm_list_records} == {"pytest_generation", "function_completion"}

    coderm_case = eval_case_from_coderm_row(
        coderm_list_tests,
        source_id="add_one",
        max_answer_chars=1000,
        max_tests_chars=1000,
    )
    assert coderm_case is not None
    assert coderm_case["source_dataset"] == "KAKA22/CodeRM-UnitTest"
    assert coderm_case["expected_behavior"] == "pytest_generation"
    assert "add_one" in coderm_case["required_substrings"]

    commit = {
        "language": "python",
        "path": "calc.py",
        "commit_message": "Fix add to use addition.",
        "old_contents": "def add(a, b):\n    return a - b\n",
        "new_contents": "def add(a, b):\n    return a + b\n",
    }
    patch_record = commitpackft_row_to_record(commit, source_id="calc", max_diff_lines=20, max_file_chars=1000)
    assert patch_record is not None
    assert patch_record["behavior"] == "patch_generation"
    assert "--- a/calc.py" in patch_record["trace"][0]["content"]
    assert "+    return a + b" in patch_record["trace"][0]["content"]


def test_hf_viewer_rows_yield_dataset_server_row_payloads(monkeypatch):
    calls = []

    def fake_dataset_viewer_json(endpoint: str, params: dict, **kwargs):
        calls.append((endpoint, dict(params)))
        if params["offset"] == 0:
            return {
                "rows": [
                    {"row_idx": 0, "row": {"id": "a", "language": "python"}},
                    {"row_idx": 1, "row": {"id": "b", "language": "python"}},
                ],
                "num_rows_total": 3,
            }
        return {
            "rows": [{"row_idx": 2, "row": {"id": "c", "language": "python"}}],
            "num_rows_total": 3,
        }

    monkeypatch.setattr("scripts.make_agentcoder_executable_sft.dataset_viewer_json", fake_dataset_viewer_json)

    rows = list(load_hf_viewer_rows("nvidia/OpenCodeInstruct", "train", "train", page_size=2))

    assert [row["id"] for row in rows] == ["a", "b", "c"]
    assert calls[0][1]["config"] == "train"
    assert calls[1][1]["offset"] == 2


def test_structured_eval_case_scores_known_answer():
    row = {
        "question": "Write the function square(n).",
        "solution": "def square(n):\n    return n * n",
        "function_name": "square",
        "unit_tests": [
            {"args": [4], "expected": 16},
            {"args": [-3], "expected": 9},
        ],
    }
    case = eval_case_from_structured_row(
        row,
        source_dataset="nvidia/OpenCodeInstruct",
        source_id="square",
        max_answer_chars=1000,
    )
    assert case is not None
    completion = "```python\ndef square(n):\n    return n * n\n```"
    scored = score_case(case, completion)
    assert scored["passed"] is True
    assert scored["function_tests_ok"] is True


def test_assertion_eval_case_scores_known_answer():
    row = {
        "input": "Write a function inc(n).",
        "output": "```python\ndef inc(n):\n    return n + 1\n```",
        "unit_tests": [
            "\nassert inc(1) == 2\n",
            "\nassert inc(-1) == 0\n",
        ],
    }
    case = eval_case_from_structured_row(
        row,
        source_dataset="nvidia/OpenCodeInstruct",
        source_id="inc",
        max_answer_chars=1000,
    )
    assert case is not None
    assert case["python_assert_tests"] == ["assert inc(1) == 2", "assert inc(-1) == 0"]
    completion = "```python\ndef inc(n):\n    return n + 1\n```"
    scored = score_case(case, completion)
    assert scored["passed"] is True
    assert scored["assert_tests_ok"] is True


def test_executable_sft_cli_writes_manifest_and_heldout_cases(tmp_path: Path):
    opencode_path = tmp_path / "opencode.jsonl"
    scotch_path = tmp_path / "scotch.jsonl"
    coderm_path = tmp_path / "coderm.jsonl"
    commit_path = tmp_path / "commitpackft.jsonl"

    write_jsonl(
        opencode_path,
        [
            {
                "language": "python",
                "question": "Implement double(n).",
                "solution": "def double(n):\n    return n * 2",
                "average_test_score": 1.0,
                "unit_tests": [{"args": [3], "expected": 6}],
            },
            {
                "language": "python",
                "split": "eval",
                "question": "Implement square(n).",
                "solution": "def square(n):\n    return n * n",
                "function_name": "square",
                "unit_tests": [{"args": [5], "expected": 25}],
            },
        ],
    )
    write_jsonl(
        scotch_path,
        [
            {
                "language": "python",
                "signature": "triple(n)",
                "docstring": "Return n times three.",
                "func_code": "def triple(n):\n    return n * 3",
            }
        ],
    )
    write_jsonl(
        commit_path,
        [
            {
                "language": "python",
                "path": "calc.py",
                "commit_message": "Fix add to use addition.",
                "old_contents": "def add(a, b):\n    return a - b\n",
                "new_contents": "def add(a, b):\n    return a + b\n",
            }
        ],
    )
    write_jsonl(
        coderm_path,
        [
            {
                "task_id": 7,
                "split": "eval",
                "question": "Implement add_one.",
                "code_ground_truth": "def add_one(x):\n    return x + 1",
                "unit_tests": json.dumps(
                    [
                        {
                            "ut_id": 0,
                            "code": "def test_add_one_positive():\n    assert add_one(2) == 3\n",
                            "FAR": 0.0,
                            "FRR": 0.0,
                        }
                    ]
                ),
            },
            {
                "task_id": 8,
                "question": "Implement negate.",
                "code_ground_truth": "def negate(flag):\n    return not flag",
                "unit_tests": json.dumps(
                    [
                        {
                            "ut_id": 0,
                            "code": "def test_negate_true():\n    assert negate(True) is False\n",
                            "FAR": 0.0,
                            "FRR": 0.0,
                        }
                    ]
                ),
            },
        ],
    )

    out = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/make_agentcoder_executable_sft.py"),
            "--output-dir",
            str(out),
            "--opencode-jsonl",
            str(opencode_path),
            "--scotch-jsonl",
            str(scotch_path),
            "--coderm-unittest-jsonl",
            str(coderm_path),
            "--commitpackft-jsonl",
            str(commit_path),
            "--ladder-repeats",
            "1",
            "--curated-anchor-repeats",
            "0",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((out / "agentcoder_executable_manifest.json").read_text())
    cases = json.loads((out / "agentcoder_executable_eval_cases.json").read_text())["cases"]
    train_rows = [json.loads(line) for line in (out / "agentcoder_executable_train.jsonl").read_text().splitlines()]

    assert manifest["format"] == "agentcoder-executable-coding-sft-v1"
    assert manifest["exact_train_eval_user_prompt_overlaps"] == []
    assert manifest["sources"]["opencode"]["train_records_added"] == 1
    assert manifest["sources"]["opencode"]["eval_cases_added"] == 1
    assert manifest["sources"]["scotch"]["train_records_added"] == 1
    assert manifest["sources"]["coderm_unittest"]["train_records_added"] == 2
    assert manifest["sources"]["coderm_unittest"]["eval_cases_added"] == 1
    assert manifest["sources"]["commitpackft"]["train_records_added"] == 1
    assert manifest["sources"]["coderm_unittest"]["sample_examples"]
    assert manifest["eval_source_dataset_counts"]["KAKA22/CodeRM-UnitTest"] == 1
    assert any(case.get("python_function", {}).get("name") == "square" for case in cases)
    assert any(case.get("source_dataset") == "KAKA22/CodeRM-UnitTest" for case in cases)
    assert any(row.get("source_dataset") == "bigcode/commitpackft" for row in train_rows)


def test_eval_case_filter_keeps_function_only_cases():
    cases = [
        {"name": "fn", "topic": "function", "expected_behavior": "function_completion"},
        {"name": "patch", "topic": "patch", "expected_behavior": "patch_addition"},
        {"name": "json", "topic": "json_command", "expected_behavior": "json_tool_command"},
    ]

    filtered = filter_eval_cases(cases, expected_behaviors=["function_completion"])

    assert [case["name"] for case in filtered] == ["fn"]


def test_eval_case_filter_fails_when_all_cases_removed():
    cases = [{"name": "patch", "topic": "patch", "expected_behavior": "patch_addition"}]

    try:
        filter_eval_cases(cases, expected_behaviors=["function_completion"])
    except ValueError as exc:
        assert "removed all cases" in str(exc)
    else:
        raise AssertionError("expected empty eval filter to fail")


def test_train_record_filter_keeps_function_only_records():
    records = [
        {"topic": "function", "behavior": "function_completion"},
        {"topic": "patch", "behavior": "patch_generation"},
        {"topic": "json", "behavior": "json_tool_command"},
    ]

    filtered = filter_train_records(records, behaviors=["function_completion"])

    assert filtered == [{"topic": "function", "behavior": "function_completion"}]


def test_executable_sft_cli_can_write_function_only_eval_cases(tmp_path: Path):
    out = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/make_agentcoder_executable_sft.py"),
            "--output-dir",
            str(out),
            "--ladder-repeats",
            "1",
            "--curated-anchor-repeats",
            "0",
            "--eval-expected-behavior",
            "function_completion",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((out / "agentcoder_executable_manifest.json").read_text())
    cases = json.loads((out / "agentcoder_executable_eval_cases.json").read_text())["cases"]

    assert cases
    assert {case["expected_behavior"] for case in cases} == {"function_completion"}
    assert manifest["filters"]["eval_expected_behavior"] == ["function_completion"]
    assert manifest["filters"]["eval_cases_before_filter"] > len(cases)


def test_executable_sft_cli_can_write_function_only_train_and_eval(tmp_path: Path):
    out = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/make_agentcoder_executable_sft.py"),
            "--output-dir",
            str(out),
            "--ladder-repeats",
            "1",
            "--curated-anchor-repeats",
            "0",
            "--train-behavior",
            "function_completion",
            "--eval-expected-behavior",
            "function_completion",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((out / "agentcoder_executable_manifest.json").read_text())
    cases = json.loads((out / "agentcoder_executable_eval_cases.json").read_text())["cases"]
    train_rows = [json.loads(line) for line in (out / "agentcoder_executable_train.jsonl").read_text().splitlines()]

    assert train_rows
    assert {row["behavior"] for row in train_rows} == {"function_completion"}
    assert {case["expected_behavior"] for case in cases} == {"function_completion"}
    assert manifest["filters"]["train_behavior"] == ["function_completion"]
    assert manifest["filters"]["train_records_before_filter"] > len(train_rows)
