from __future__ import annotations

import importlib.util
from pathlib import Path


def load_prepare_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_agentcoder_research_data.py"
    spec = importlib.util.spec_from_file_location("prepare_agentcoder_research_data", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_openhands_row_to_record_preserves_agent_roles_and_patch():
    mod = load_prepare_module()
    row = {
        "instance_id": "owner__repo-1",
        "repo": "owner/repo",
        "license": "MIT",
        "language": "python",
        "resolved": 1,
        "trajectory": [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": "Fix the bug."},
            {"role": "assistant", "content": "I will inspect the tests."},
            {"role": "tool", "content": "pytest failed"},
        ],
        "model_patch": "diff --git a/app.py b/app.py\n",
    }

    record = mod.openhands_row_to_record(
        row,
        source="unit-test",
        resolved_only=True,
        languages={"python"},
    )

    assert record is not None
    assert record["system"] == "You are an agent."
    assert [msg["role"] for msg in record["messages"]] == ["user", "assistant", "tool_result"]
    assert "repo: owner/repo" in record["repo_context"]
    assert record["trace"][0]["type"] == "patch"


def test_wildchat_row_filters_toxic_and_normalizes_roles():
    mod = load_prepare_module()
    toxic = {"toxic": True, "language": "English", "conversation": [{"role": "user", "content": "hi"}]}
    assert mod.wildchat_row_to_record(toxic, english_only=True) is None

    row = {
        "toxic": False,
        "language": "English",
        "conversation": [
            {"role": "user", "content": "Write a Python function."},
            {"role": "assistant", "content": "def add(a, b):\n    return a + b"},
        ],
    }
    record = mod.wildchat_row_to_record(row, english_only=True)

    assert record is not None
    assert [msg["role"] for msg in record["messages"]] == ["user", "assistant"]


def test_oasst_rows_to_records_builds_parent_child_pairs():
    mod = load_prepare_module()
    rows = [
        {
            "message_id": "m1",
            "parent_id": None,
            "role": "prompter",
            "lang": "en",
            "review_result": True,
            "deleted": False,
            "text": "Explain unit tests.",
        },
        {
            "message_id": "m2",
            "parent_id": "m1",
            "role": "assistant",
            "lang": "en",
            "review_result": True,
            "deleted": False,
            "text": "Unit tests verify small behaviors.",
        },
    ]

    records = list(mod.oasst_rows_to_records(rows, english_only=True))

    assert len(records) == 1
    assert records[0]["messages"][0]["role"] == "user"
    assert records[0]["messages"][1]["role"] == "assistant"
