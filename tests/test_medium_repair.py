from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_coding_ladder import score_case
from scripts.make_agentcoder_medium_repair_sft import (
    ANCHOR_FLOOR_FUNCTIONS,
    FORMAT,
    REQUIRED_TOPICS,
    build_base_records,
    expanded_eval_cases,
    expand_records,
    rendered_prompt_for_row,
    validate_records,
    write_jsonl,
)


def test_medium_repair_generator_uses_no_final_and_covers_failed_frontier(tmp_path: Path):
    base = build_base_records()
    rows = expand_records(base, repeats=2, anchor_repeats=1, seed=31)
    cases = expanded_eval_cases()

    assert {str(row["topic"]) for row in rows} >= set(REQUIRED_TOPICS)
    assert all(not row.get("final") for row in rows)
    validation = validate_records(rows)
    assert validation["validated_records"] == len(rows)
    assert set(validation["tiny_anchor_floor_counts"]) == set(ANCHOR_FLOOR_FUNCTIONS)
    assert all(count > 0 for count in validation["tiny_anchor_floor_counts"].values())

    train_prompts = {rendered_prompt_for_row(row) for row in rows}
    eval_prompts = {str(case["prompt"]) for case in cases}
    assert train_prompts.isdisjoint(eval_prompts)

    out = tmp_path / "medium.jsonl"
    write_jsonl(out, rows, strip_validation=True)
    first = json.loads(out.read_text().splitlines()[0])
    assert "_validation_case" not in first
    assert FORMAT == "agentcoder-medium-frontier-repair-v2"


def test_medium_repair_family_guards_and_tiny_floor_eval_cases():
    rows = expand_records(build_base_records(), repeats=4, anchor_repeats=4, seed=31)
    anchor_rows = [row for row in rows if row["topic"] == "anchor_tiny_function"]
    assert len(anchor_rows) >= 20
    assert len(anchor_rows) / len(rows) >= 0.05

    for row in rows:
        case = row["_validation_case"]
        forbidden = set(case.get("forbidden_substrings", []))
        behavior = row["behavior"]
        if behavior == "function_completion":
            assert "```diff" in forbidden
            assert "Test command:" in forbidden
        elif behavior == "json_tool_command":
            assert "```" in forbidden
            assert "def " in forbidden
        elif behavior == "pytest_generation":
            assert "```diff" in forbidden
            assert '"cmd":' in forbidden

    cases = {case["name"]: case for case in expanded_eval_cases()}
    for name, answer in {
        "ladder_is_even": "```python\ndef is_even(n):\n    return n % 2 == 0\n```",
        "ladder_is_odd": "```python\ndef is_odd(n):\n    return n % 2 == 1\n```",
        "ladder_filter_even": "```python\ndef filter_even(numbers):\n    return [n for n in numbers if n % 2 == 0]\n```",
    }.items():
        assert name in cases
        assert score_case(cases[name], answer)["passed"] is True


def test_medium_eval_scores_frontier_function_json_and_patch():
    count_case = next(case for case in expanded_eval_cases() if case["name"] == "medium_count_even_negatives")
    count_answer = "```python\ndef count_even(numbers):\n    return sum(1 for n in numbers if n % 2 == 0)\n```"
    assert score_case(count_case, count_answer)["passed"] is True

    json_case = next(case for case in expanded_eval_cases() if case["name"] == "medium_json_pytest_ladder")
    assert score_case(json_case, '{"cmd": "python -m pytest -q tests/test_coding_ladder.py"}')["passed"] is True
    assert score_case(json_case, '{"cmd": "python -m pytest -q tests/test_coding_ladder.py"}\nextra')["passed"] is False

    patch_case = next(case for case in expanded_eval_cases() if case["name"] == "medium_patch_2_flags")
    good_patch = (
        "The bug is inverted.\n"
        "```diff\n"
        "--- a/flags.py\n"
        "+++ b/flags.py\n"
        "@@\n"
        "-return value == 'false'\n"
        "+return value == 'true'\n"
        "```\n"
        "Test command: `pytest tests/test_flags.py -q`"
    )
    assert score_case(patch_case, good_patch)["passed"] is True
    assert score_case(patch_case, good_patch.replace("@@\n", ""))["passed"] is False
