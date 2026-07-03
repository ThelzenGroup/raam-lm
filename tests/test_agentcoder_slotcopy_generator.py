from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.make_agentcoder_slotcopy_sft import (
    EVAL_PER_FAMILY,
    SEEN_EVAL_PER_FAMILY,
    TRAIN_PER_FAMILY,
    build_eval_cases,
    build_train_records,
)
from scripts.run_agentcoder_slotcopy_gate import summarize_ladder, summarize_slot_families


ROOT = Path(__file__).resolve().parents[1]


def test_slotcopy_generator_has_disjoint_balanced_families():
    records = build_train_records(seed=17)
    cases = build_eval_cases(seed=17)

    assert len(records) == TRAIN_PER_FAMILY * 3
    assert len(cases) == EVAL_PER_FAMILY * 3
    assert {row["slot_family"] for row in records} == {"repo_lookup", "patch_return", "patch_literal"}
    assert {row["slot_family"] for row in cases} == {"repo_lookup", "patch_return", "patch_literal"}

    for family in {"repo_lookup", "patch_return", "patch_literal"}:
        assert sum(1 for row in records if row["slot_family"] == family) == TRAIN_PER_FAMILY
        assert sum(1 for row in cases if row["slot_family"] == family) == EVAL_PER_FAMILY

    train_slot_tuples = {
        (row["slot_family"], tuple(sorted(row["expected_slots"].items())))
        for row in records
    }
    eval_slot_tuples = {
        (row["slot_family"], tuple(sorted(row["expected_slots"].items())))
        for row in cases
    }
    assert train_slot_tuples.isdisjoint(eval_slot_tuples)
    assert all(row["required_substrings"] for row in cases)
    assert all(row["forbidden_substrings"] for row in cases)
    assert all("expected_slots" in row for row in cases)
    assert {row["eval_tier"] for row in cases} == {"heldout_slot"}


def test_slotcopy_ladder_has_seen_and_heldout_tiers():
    records = build_train_records(seed=17)
    cases = build_eval_cases(seed=17, eval_mode="ladder")

    assert len(cases) == (SEEN_EVAL_PER_FAMILY + EVAL_PER_FAMILY) * 3
    assert {row["eval_tier"] for row in cases} == {"seen_slot", "heldout_slot"}

    train_slot_tuples = {
        (row["slot_family"], tuple(sorted(row["expected_slots"].items())))
        for row in records
    }
    seen_slot_tuples = {
        (row["slot_family"], tuple(sorted(row["expected_slots"].items())))
        for row in cases
        if row["eval_tier"] == "seen_slot"
    }
    heldout_slot_tuples = {
        (row["slot_family"], tuple(sorted(row["expected_slots"].items())))
        for row in cases
        if row["eval_tier"] == "heldout_slot"
    }

    assert seen_slot_tuples
    assert seen_slot_tuples.issubset(train_slot_tuples)
    assert heldout_slot_tuples.isdisjoint(train_slot_tuples)
    for family in {"repo_lookup", "patch_return", "patch_literal"}:
        assert sum(1 for row in cases if row["slot_family"] == family and row["eval_tier"] == "seen_slot") == SEEN_EVAL_PER_FAMILY
        assert sum(1 for row in cases if row["slot_family"] == family and row["eval_tier"] == "heldout_slot") == EVAL_PER_FAMILY


def test_slotcopy_cases_check_exact_slot_copying():
    cases = build_eval_cases(seed=17)
    repo_case = next(row for row in cases if row["slot_family"] == "repo_lookup")
    patch_case = next(row for row in cases if row["slot_family"] == "patch_return")
    literal_case = next(row for row in cases if row["slot_family"] == "patch_literal")

    assert "Start the answer with the exact requested symbol" in repo_case["prompt"]
    assert repo_case["expected_slots"]["symbol"] in repo_case["required_substrings"][0]
    assert repo_case["expected_slots"]["file"] in repo_case["required_substrings"][0]

    assert "Emit the diff first" in patch_case["prompt"]
    assert f"--- a/{patch_case['expected_slots']['file']}" in patch_case["required_substrings"]
    assert patch_case["expected_slots"]["helper"] in "\n".join(patch_case["required_substrings"])

    assert "Boolean flag slot-copy task" in literal_case["prompt"]
    assert literal_case["expected_slots"]["literal"] in "\n".join(literal_case["required_substrings"])


def test_slotcopy_cli_writes_manifest_and_cases(tmp_path):
    train = tmp_path / "train.jsonl"
    cases = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_agentcoder_slotcopy_sft.py",
            "--train-output",
            str(train),
            "--cases-output",
            str(cases),
            "--manifest-output",
            str(manifest),
        ],
        cwd=ROOT,
        check=True,
    )

    payload = json.loads(manifest.read_text())
    assert payload["format"] == "agentcoder-slotcopy-sft-v1"
    assert payload["eval_mode"] == "heldout"
    assert payload["train_records"] == TRAIN_PER_FAMILY * 3
    assert payload["eval_cases"] == EVAL_PER_FAMILY * 3
    assert payload["eval_tier_counts"] == {"heldout_slot": EVAL_PER_FAMILY * 3}
    assert payload["train_slot_family_counts"] == {
        "patch_literal": TRAIN_PER_FAMILY,
        "patch_return": TRAIN_PER_FAMILY,
        "repo_lookup": TRAIN_PER_FAMILY,
    }
    assert train.exists()
    assert cases.exists()


def test_slotcopy_family_summary_counts_passes_and_slot_errors():
    payload = {
        "results": [
            {"slot_family": "repo_lookup", "passed": True, "slot_error": False, "behavior_correct": True, "name": "r1"},
            {"slot_family": "repo_lookup", "passed": False, "slot_error": True, "behavior_correct": True, "name": "r2"},
            {"slot_family": "patch_return", "passed": False, "slot_error": False, "behavior_correct": False, "name": "p1"},
        ]
    }
    summary = summarize_slot_families(payload)

    assert summary["repo_lookup"]["case_count"] == 2
    assert summary["repo_lookup"]["pass_count"] == 1
    assert summary["repo_lookup"]["slot_error_count"] == 1
    assert summary["repo_lookup"]["behavior_accuracy"] == 1.0
    assert summary["patch_return"]["failed_cases"] == ["p1"]


def test_slotcopy_ladder_summary_splits_family_and_tier():
    payload = {
        "results": [
            {
                "slot_family": "repo_lookup",
                "eval_tier": "seen_slot",
                "passed": True,
                "slot_error": False,
                "behavior_correct": True,
                "name": "seen",
            },
            {
                "slot_family": "repo_lookup",
                "eval_tier": "heldout_slot",
                "passed": False,
                "slot_error": True,
                "behavior_correct": True,
                "name": "heldout",
            },
        ]
    }
    summary = summarize_ladder(payload)

    assert summary["repo_lookup"]["seen_slot"]["pass_rate"] == 1.0
    assert summary["repo_lookup"]["heldout_slot"]["pass_rate"] == 0.0
    assert summary["repo_lookup"]["heldout_slot"]["failed_cases"] == ["heldout"]
