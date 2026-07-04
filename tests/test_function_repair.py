from __future__ import annotations

import json
from pathlib import Path

from scripts.eval_coding_ladder import score_case
from scripts.make_agentcoder_function_repair_sft import (
    ANCHOR_FLOOR_FUNCTIONS,
    FORMAT,
    REQUIRED_TOPICS,
    STAGE,
    TARGET_FUNCTIONS,
    assert_train_eval_disjoint,
    build_base_records,
    build_selected_base_records,
    expanded_eval_cases,
    expand_records,
    memorization_probe_cases,
    required_topics_for_targets,
    rendered_prompt_for_row,
    validate_records,
    write_jsonl,
)
from scripts.eval_function_memorization_probe import summarize_probe
from scripts.eval_function_probe_timeline import list_checkpoints, summarize_timeline


def test_function_repair_generator_is_function_only_and_no_final(tmp_path: Path):
    base = build_base_records()
    rows = expand_records(base, target_repeats=2, anchor_repeats=1, seed=31)
    validation = validate_records(rows)
    cases = expanded_eval_cases()

    assert FORMAT == "agentcoder-medium-function-repair-v1"
    assert STAGE == "medium_function_only"
    assert {str(row["topic"]) for row in rows} == set(REQUIRED_TOPICS)
    assert {str(row["behavior"]) for row in rows} == {"function_completion"}
    assert all(not row.get("final") for row in rows)
    assert validation["validated_records"] == len(rows)
    assert set(validation["tiny_anchor_floor_counts"]) == set(ANCHOR_FLOOR_FUNCTIONS)
    assert all(validation["tiny_anchor_floor_counts"][name] > 0 for name in ANCHOR_FLOOR_FUNCTIONS)
    assert all(validation["function_counts"][name] > 0 for name in TARGET_FUNCTIONS)

    assert assert_train_eval_disjoint(rows, cases) == []
    train_prompts = {rendered_prompt_for_row(row) for row in rows}
    eval_prompts = {str(case["prompt"]) for case in cases}
    assert train_prompts.isdisjoint(eval_prompts)

    out = tmp_path / "function.jsonl"
    write_jsonl(out, rows, strip_validation=True)
    first = json.loads(out.read_text().splitlines()[0])
    assert "_validation_case" not in first


def test_function_repair_rejects_patch_pytest_json_contamination():
    rows = expand_records(build_base_records(), target_repeats=1, anchor_repeats=1, seed=31)
    forbidden = ["```diff", "Test command:", '"cmd"', "pytest", "--- a/", "+++ b/"]
    for row in rows:
        answer = row["trace"][0]["content"]
        assert not any(fragment in answer for fragment in forbidden)
        case = row["_validation_case"]
        assert "expected_patch" not in case
        assert "expected_json" not in case
        assert not case.get("python_syntax")
        assert "python_function" in case

    contaminated = dict(rows[0])
    contaminated["trace"] = [{"type": "assistant", "content": rows[0]["trace"][0]["content"] + "\nTest command: `pytest -q`"}]
    try:
        validate_records([contaminated])
    except ValueError as exc:
        assert "answer_family_contamination" in str(exc)
    else:
        raise AssertionError("contaminated function answer passed validation")


def test_function_repair_expanded_eval_keeps_tiny_and_target_function_cases():
    cases = {case["name"]: case for case in expanded_eval_cases()}
    expected_names = {
        "ladder_is_even",
        "ladder_is_odd",
        "ladder_filter_even",
        "ladder_count_even",
        "ladder_safe_int",
        "ladder_parse_port",
        "medium_count_even_negatives",
        "medium_safe_int_defaults",
        "medium_parse_port_range",
    }
    assert expected_names <= set(cases)

    good_answers = {
        "medium_count_even_negatives": "```python\ndef count_even(numbers):\n    return sum(1 for n in numbers if n % 2 == 0)\n```",
        "medium_safe_int_defaults": (
            "```python\n"
            "def safe_int(value, default=None):\n"
            "    try:\n"
            "        return int(value)\n"
            "    except (TypeError, ValueError):\n"
            "        return default\n"
            "```"
        ),
        "medium_parse_port_range": (
            "```python\n"
            "def parse_port(value):\n"
            "    try:\n"
            "        port = int(str(value).strip())\n"
            "    except (TypeError, ValueError):\n"
            "        raise ValueError(\"port must be an integer\")\n"
            "    if port < 1 or port > 65535:\n"
            "        raise ValueError(\"port must be between 1 and 65535\")\n"
            "    return port\n"
            "```"
        ),
    }
    for name, answer in good_answers.items():
        assert score_case(cases[name], answer)["passed"] is True


def test_function_repair_memorization_probe_is_train_like_but_separate_from_gate():
    rows = expand_records(build_base_records(), target_repeats=1, anchor_repeats=1, seed=31)
    train_prompts = {rendered_prompt_for_row(row) for row in rows}
    probes = memorization_probe_cases()

    assert {case["name"] for case in probes} == {
        "probe_exact_count_even",
        "probe_exact_safe_int",
        "probe_exact_parse_port",
        "probe_exact_is_even",
        "probe_exact_is_odd",
        "probe_exact_filter_even",
    }
    assert {str(case["prompt"]) for case in probes} <= train_prompts
    assert all(case["probe_kind"] == "exact_train_like_function_generation" for case in probes)
    assert all(case["expected_behavior"] == "function_completion" for case in probes)


def test_function_repair_can_select_single_target_for_diagnostic_probe():
    selected_targets = ["count_even"]
    rows = expand_records(
        build_selected_base_records(target_functions=selected_targets, target_prompt_limit=2, anchor_prompt_limit=1),
        target_repeats=2,
        anchor_repeats=1,
        seed=37,
    )
    validation = validate_records(rows, target_functions=selected_targets)
    probes = memorization_probe_cases(selected_targets)

    assert required_topics_for_targets(selected_targets) == ["count_even", "anchor_tiny_function"]
    assert {str(row["topic"]) for row in rows} == {"count_even", "anchor_tiny_function"}
    assert validation["function_counts"]["count_even"] == 4
    assert "safe_int" not in validation["function_counts"]
    assert "parse_port" not in validation["function_counts"]
    assert {case["name"] for case in probes} == {
        "probe_exact_count_even",
        "probe_exact_is_even",
        "probe_exact_is_odd",
        "probe_exact_filter_even",
    }


def test_function_repair_can_select_safe_int_for_diagnostic_probe():
    selected_targets = ["safe_int"]
    rows = expand_records(
        build_selected_base_records(target_functions=selected_targets, target_prompt_limit=2, anchor_prompt_limit=1),
        target_repeats=2,
        anchor_repeats=1,
        seed=37,
    )
    validation = validate_records(rows, target_functions=selected_targets)
    probes = memorization_probe_cases(selected_targets)

    assert required_topics_for_targets(selected_targets) == ["safe_int", "anchor_tiny_function"]
    assert {str(row["topic"]) for row in rows} == {"safe_int", "anchor_tiny_function"}
    assert validation["function_counts"]["safe_int"] == 4
    assert "count_even" not in validation["function_counts"]
    assert "parse_port" not in validation["function_counts"]
    assert {case["name"] for case in probes} == {
        "probe_exact_safe_int",
        "probe_exact_is_even",
        "probe_exact_is_odd",
        "probe_exact_filter_even",
    }


def test_function_memorization_probe_summary_counts_targets_and_anchors():
    payload = {
        "pass_count": 4,
        "case_count": 6,
        "failed_cases": ["probe_exact_safe_int", "probe_exact_parse_port"],
        "results": [
            {"name": "probe_exact_count_even", "passed": True},
            {"name": "probe_exact_safe_int", "passed": False},
            {"name": "probe_exact_parse_port", "passed": False},
            {"name": "probe_exact_is_even", "passed": True},
            {"name": "probe_exact_is_odd", "passed": True},
            {"name": "probe_exact_filter_even", "passed": True},
        ],
    }
    summary = summarize_probe(payload)
    assert summary["target_probe_pass_count"] == 1
    assert summary["target_probe_case_count"] == 3
    assert summary["anchor_probe_pass_count"] == 3
    assert summary["anchor_probe_case_count"] == 3
    assert summary["failed_cases"] == ["probe_exact_safe_int", "probe_exact_parse_port"]


def test_function_memorization_probe_summary_uses_probe_topics_when_available():
    payload = {
        "pass_count": 4,
        "case_count": 4,
        "failed_cases": [],
        "results": [
            {"name": "probe_exact_count_even", "topic": "count_even", "passed": True},
            {"name": "probe_exact_is_even", "topic": "anchor_tiny_function", "passed": True},
            {"name": "probe_exact_is_odd", "topic": "anchor_tiny_function", "passed": True},
            {"name": "probe_exact_filter_even", "topic": "anchor_tiny_function", "passed": True},
        ],
    }
    summary = summarize_probe(payload)
    assert summary["target_probe_pass_count"] == 1
    assert summary["target_probe_case_count"] == 1
    assert summary["anchor_probe_pass_count"] == 3
    assert summary["anchor_probe_case_count"] == 3


def test_function_probe_timeline_orders_checkpoints_and_summarizes_first_pass(tmp_path: Path):
    for name in ["last.pt", "step_000100.pt", "best.pt", "step_000050.pt"]:
        (tmp_path / name).write_text("")

    assert [path.name for path in list_checkpoints(tmp_path)] == [
        "step_000050.pt",
        "step_000100.pt",
        "best.pt",
        "last.pt",
    ]

    summary = summarize_timeline(
        [
            {
                "checkpoint": "step_000050.pt",
                "target_probe_pass_count": 0,
                "target_probe_case_count": 1,
                "anchor_probe_pass_count": 3,
                "anchor_probe_case_count": 3,
            },
            {
                "checkpoint": "step_000100.pt",
                "target_probe_pass_count": 1,
                "target_probe_case_count": 1,
                "anchor_probe_pass_count": 3,
                "anchor_probe_case_count": 3,
            },
        ]
    )
    assert summary["checkpoint_count"] == 2
    assert summary["first_all_targets_pass_checkpoint"] == "step_000100.pt"
    assert summary["first_all_targets_and_anchors_pass_checkpoint"] == "step_000100.pt"
