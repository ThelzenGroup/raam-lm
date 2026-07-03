from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.make_agentcoder_keyvalue_copy_sft import (
    DISTRACTOR_FIELDS,
    EVAL_CASES,
    KEYS,
    TARGET_FIELDS,
    TRAIN_RECORDS,
    VALUE_FORMATS,
    build_eval_cases,
    build_train_records,
)
from scripts.run_agentcoder_slotcopy_gate import summarize_ladder, summarize_slot_families


ROOT = Path(__file__).resolve().parents[1]


def test_keyvalue_generator_builds_ladder_with_distractors_and_formats():
    records = build_train_records(seed=17, train_records=TRAIN_RECORDS)
    cases = build_eval_cases(seed=17, eval_mode="ladder", train_records=TRAIN_RECORDS, eval_cases=EVAL_CASES)

    assert len(records) == TRAIN_RECORDS
    assert len(cases) == EVAL_CASES * 2
    assert {row["slot_family"] for row in records} == {"keyvalue_repo_copy"}
    assert {row["slot_family"] for row in cases} == {"keyvalue_repo_copy"}
    assert {row["eval_tier"] for row in cases} == {"seen_slot", "heldout_slot"}
    assert all(len(row["expected_slots"]) == TARGET_FIELDS for row in records)
    assert all(len(row["expected_slots"]) == TARGET_FIELDS for row in cases)
    assert all(len(row["forbidden_substrings"]) == DISTRACTOR_FIELDS for row in cases)
    assert set(KEYS).issubset({key for row in cases for key in row["target_keys"]})
    assert set(VALUE_FORMATS).issubset({fmt for row in cases for fmt in row["value_formats"]})


def test_keyvalue_seen_and_heldout_slots_are_distinct():
    records = build_train_records(seed=17, train_records=TRAIN_RECORDS)
    cases = build_eval_cases(seed=17, eval_mode="ladder", train_records=TRAIN_RECORDS, eval_cases=EVAL_CASES)
    train_slots = {tuple(sorted(row["expected_slots"].items())) for row in records}
    seen_slots = {
        tuple(sorted(row["expected_slots"].items()))
        for row in cases
        if row["eval_tier"] == "seen_slot"
    }
    heldout_slots = {
        tuple(sorted(row["expected_slots"].items()))
        for row in cases
        if row["eval_tier"] == "heldout_slot"
    }

    assert seen_slots
    assert seen_slots.issubset(train_slots)
    assert heldout_slots.isdisjoint(train_slots)


def test_keyvalue_cases_require_exact_key_value_lines_and_forbid_distractors():
    cases = build_eval_cases(seed=17, eval_mode="heldout", train_records=16, eval_cases=8)
    row = cases[0]

    assert "Return only key=value lines" in row["prompt"]
    assert all("=" in needle for needle in row["required_substrings"])
    assert all("=" in needle for needle in row["forbidden_substrings"])
    for key, value in row["expected_slots"].items():
        assert f"{key}={value}" in row["required_substrings"]


def test_keyvalue_cli_writes_manifest_and_cases(tmp_path):
    train = tmp_path / "train.jsonl"
    cases = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_agentcoder_keyvalue_copy_sft.py",
            "--train-output",
            str(train),
            "--cases-output",
            str(cases),
            "--manifest-output",
            str(manifest),
            "--eval-mode",
            "ladder",
            "--train-records",
            "12",
            "--eval-cases",
            "8",
        ],
        cwd=ROOT,
        check=True,
    )

    payload = json.loads(manifest.read_text())
    assert payload["format"] == "agentcoder-keyvalue-copy-sft-v1"
    assert payload["eval_mode"] == "ladder"
    assert payload["train_records"] == 12
    assert payload["eval_cases"] == 16
    assert payload["target_fields"] == TARGET_FIELDS
    assert payload["distractor_fields"] == DISTRACTOR_FIELDS
    assert payload["eval_tier_counts"] == {"heldout_slot": 8, "seen_slot": 8}
    assert train.exists()
    assert cases.exists()


def test_keyvalue_summaries_count_tiers():
    payload = {
        "results": [
            {
                "slot_family": "keyvalue_repo_copy",
                "eval_tier": "seen_slot",
                "passed": True,
                "slot_error": False,
                "behavior_correct": True,
                "name": "seen",
            },
            {
                "slot_family": "keyvalue_repo_copy",
                "eval_tier": "heldout_slot",
                "passed": False,
                "slot_error": True,
                "behavior_correct": True,
                "name": "heldout",
            },
        ]
    }

    family = summarize_slot_families(payload)
    ladder = summarize_ladder(payload)

    assert family["keyvalue_repo_copy"]["pass_count"] == 1
    assert family["keyvalue_repo_copy"]["slot_error_count"] == 1
    assert ladder["keyvalue_repo_copy"]["seen_slot"]["pass_rate"] == 1.0
    assert ladder["keyvalue_repo_copy"]["heldout_slot"]["failed_cases"] == ["heldout"]


def test_keyvalue_key_follow_configs_enable_same_route():
    for path, model_name in [
        ("configs/scratch/raam_agentcoder_keyvalue_key_follow_gate.yaml", "raam"),
        ("configs/scratch/transformer_agentcoder_keyvalue_key_follow_gate.yaml", "transformer"),
    ]:
        payload = yaml.safe_load(Path(path).read_text())

        assert payload["model_name"] == model_name
        assert payload["copy_head"]["enabled"] is True
        assert payload["copy_head"]["key_follow_strength"] > 0
        assert payload["copy_head"]["key_follow_continuation_strength"] == 0.0
        assert payload["copy_head"]["key_follow_recent_tokens"] == 24
        assert payload["copy_head"]["key_follow_value_offset"] == 3
        assert payload["copy_head"]["key_follow_min_source_gap"] == 8
        assert payload["copy_head"]["key_follow_source_until_token_id"] == 5
        assert payload["copy_head"]["key_follow_align_value_offset"] is True
        assert payload["copy_head"]["key_follow_match_value_prefix"] is True
        assert payload["copy_head"]["key_follow_separator_token_id"] == 275
        assert payload["copy_head"]["key_follow_recent_after_token_id"] == 6
        assert payload["copy_head"]["key_follow_stop_token_ids"] == [23]
