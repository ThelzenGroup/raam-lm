from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_coding_ladder import score_case
from scripts.make_agentcoder_coding_ladder_sft import (
    REQUIRED_LADDER_TOPICS,
    build_eval_cases,
    build_ladder_base_records,
    expand_records,
    write_jsonl,
)


def test_coding_ladder_generator_covers_required_topics_and_keeps_eval_held_out(tmp_path: Path):
    base = build_ladder_base_records()
    records = expand_records(base, ladder_repeats=2, curated_anchor_repeats=1, seed=17)
    cases = build_eval_cases()

    train_topics = {str(row.get("topic")) for row in records}
    eval_topics = {str(case.get("topic")) for case in cases}
    assert set(REQUIRED_LADDER_TOPICS) <= train_topics
    assert set(REQUIRED_LADDER_TOPICS) <= eval_topics

    train_prompts = {
        str(message.get("content", ""))
        for row in records
        for message in row.get("messages", [])
        if message.get("role") == "user"
    }
    eval_prompts = {str(case["prompt"]) for case in cases}
    assert train_prompts.isdisjoint(eval_prompts)

    out = tmp_path / "ladder.jsonl"
    write_jsonl(out, records)
    lines = out.read_text().splitlines()
    assert len(lines) == len(records)
    assert all(json.loads(line) for line in lines)


def test_eval_scores_working_function_and_rejects_trailing_nonsense():
    case = {
        "name": "count_even",
        "required_substrings": ["def count_even(numbers):", "n % 2 == 0"],
        "python_function": {
            "name": "count_even",
            "tests": [
                {"args": [[1, 2, 4, 5]], "expected": 2},
                {"args": [[1, 3, 5]], "expected": 0},
            ],
        },
        "expected_behavior": "function_completion",
        "no_trailing_text": True,
    }
    good = "```python\ndef count_even(numbers):\n    return sum(1 for n in numbers if n % 2 == 0)\n```"
    assert score_case(case, good)["passed"] is True

    bad = good + "\n\nNow, we can be a communicationService to the colonial parser."
    scored = score_case(case, bad)
    assert scored["passed"] is False
    assert scored["nonsense_present"] is True
    assert scored["no_trailing_text_ok"] is False


def test_eval_scores_strict_json_and_patch_format():
    json_case = {
        "name": "json_pytest",
        "required_substrings": ["python -m pytest -q"],
        "expected_json": {"cmd": "python -m pytest -q"},
        "expected_behavior": "json_tool_command",
        "strict_json": True,
    }
    assert score_case(json_case, '{"cmd": "python -m pytest -q"}')["passed"] is True
    assert score_case(json_case, '{"cmd": "python -m pytest -q"}\nextra')["passed"] is False

    patch_case = {
        "name": "patch_add",
        "required_substrings": ["--- a/calc.py", "+++ b/calc.py", "return a + b"],
        "expected_patch": {"path": "calc.py", "removed": "return a - b", "added": "return a + b"},
        "expected_behavior": "patch_addition",
    }
    patch = "```diff\n--- a/calc.py\n+++ b/calc.py\n@@\n-return a - b\n+return a + b\n```"
    assert score_case(patch_case, patch)["passed"] is True
    assert score_case(patch_case, "return a + b")["passed"] is False
