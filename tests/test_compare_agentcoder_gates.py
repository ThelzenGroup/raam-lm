from __future__ import annotations

import json
from pathlib import Path

from scripts.compare_agentcoder_gates import infer_behavior, summarize_run, write_markdown


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def test_summarize_curated_gate_with_inferred_confusion(tmp_path):
    run_dir = tmp_path / "run"
    write_json(
        run_dir / "summary.json",
        {
            "pass_count": 1,
            "case_count": 2,
            "pass_rate": 0.5,
            "train_records": 12,
            "train_tokens": 100,
            "val_tokens": 20,
            "last_train_row": {
                "train_loss": 0.1,
                "val_next_token_loss": 0.2,
                "tokens_per_sec": 123.0,
            },
        },
    )
    write_json(
        run_dir / "curated_eval.json",
        {
            "results": [
                {
                    "name": "curated_json_python_files",
                    "passed": True,
                    "completion": "{\"cmd\": \"find . -type f -name '*.py'\"}",
                    "missing_required_substrings": [],
                    "json_ok": True,
                },
                {
                    "name": "curated_risky_question",
                    "passed": False,
                    "completion": "{\"cmd\": \"find . -type f -name '*.py'\"}",
                    "missing_required_substrings": ["rollback"],
                    "json_ok": None,
                },
            ]
        },
    )

    row = summarize_run(run_dir)

    assert row["pass_rate"] == 0.5
    assert row["behavior_accuracy"] == 0.5
    assert row["behavior_confusion"]["json_tool_command"]["json_tool_command"] == 1
    assert row["behavior_confusion"]["risky_clarifying_question"]["json_tool_command"] == 1
    assert row["failed_cases"][0]["expected_behavior"] == "risky_clarifying_question"
    assert row["failed_cases"][0]["predicted_behavior"] == "json_tool_command"


def test_write_markdown_includes_failures(tmp_path):
    output = tmp_path / "report.md"
    write_markdown(
        [
            {
                "run_name": "run_a",
                "pass_count": 1,
                "case_count": 2,
                "pass_rate": 0.5,
                "behavior_accuracy": 0.5,
                "failed_cases": [
                    {
                        "name": "curated_risky_question",
                        "expected_behavior": "risky_clarifying_question",
                        "predicted_behavior": "json_tool_command",
                        "missing_required_substrings": ["rollback"],
                        "json_ok": None,
                        "completion_preview": "{\"cmd\": \"find .\"}",
                    }
                ],
                "behavior_confusion": {
                    "risky_clarifying_question": {"json_tool_command": 1},
                },
            }
        ],
        output,
    )

    text = output.read_text()
    assert "AgentCoder Gate Comparison" in text
    assert "curated_risky_question" in text
    assert "risky_clarifying_question -> json_tool_command: 1" in text


def test_port_review_inference_beats_generic_valueerror():
    completion = "Validate that the value is numeric and between 1 and 65535; otherwise return a clear error instead of raw ValueError."

    assert infer_behavior(completion) == "code_review"
