from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

from scripts.make_agentcoder_keyvalue_copy_sft import (
    COMPLETION_MODES,
    DISTRACTOR_FIELDS,
    EVAL_CASES,
    KEYS,
    TARGET_FIELDS,
    TRAIN_RECORDS,
    TRAIN_VARIANTS_PER_ROW,
    VALUE_BOUNDARY_CLOSE,
    VALUE_BOUNDARY_OPEN,
    VALUE_FORMATS,
    build_eval_cases,
    build_train_records,
    strip_value_boundaries,
)
from scripts.run_agentcoder_slotcopy_gate import summarize_ladder, summarize_slot_families


ROOT = Path(__file__).resolve().parents[1]


def test_keyvalue_generator_builds_ladder_with_distractors_and_formats():
    records = build_train_records(seed=17, train_records=TRAIN_RECORDS)
    cases = build_eval_cases(seed=17, eval_mode="ladder", train_records=TRAIN_RECORDS, eval_cases=EVAL_CASES)

    assert len(records) == TRAIN_RECORDS
    assert TRAIN_VARIANTS_PER_ROW == 1
    assert COMPLETION_MODES == ["keyvalue", "key_only", "value_only"]
    assert len(cases) == EVAL_CASES * 2
    assert {row["slot_family"] for row in records} == {"keyvalue_repo_copy"}
    assert {row["slot_family"] for row in cases} == {"keyvalue_repo_copy"}
    assert {row["train_variant_index"] for row in records} == {0}
    assert {row["eval_tier"] for row in cases} == {"seen_slot", "heldout_slot"}
    assert all(len(row["expected_slots"]) == TARGET_FIELDS for row in records)
    assert all(len(row["expected_slots"]) == TARGET_FIELDS for row in cases)
    assert all(len(row["target_keys"]) == TARGET_FIELDS for row in records)
    assert all(row["enforce_key_sequence"] is True for row in cases)
    assert all(row["enforce_value_sequence"] is True for row in cases)
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


def test_keyvalue_train_variants_randomize_layouts_per_source_row():
    records = build_train_records(seed=17, train_records=4, train_variants_per_row=4)
    first_row_records = [row for row in records if row["source_row_index"] == 0]
    expected_slot_sets = {
        tuple(row["expected_slots"].items())
        for row in first_row_records
    }
    context_orders = {
        tuple(row["repo_context"].splitlines())
        for row in first_row_records
    }

    assert len(records) == 16
    assert len(first_row_records) == 4
    assert {row["train_variant_index"] for row in first_row_records} == {0, 1, 2, 3}
    assert len(expected_slot_sets) > 1
    assert len(context_orders) > 1


def test_keyvalue_key_only_mode_emits_requested_key_sequence_without_values():
    records = build_train_records(
        seed=17,
        train_records=4,
        train_variants_per_row=2,
        completion_mode="key_only",
    )
    cases = build_eval_cases(
        seed=17,
        eval_mode="coverage_ladder",
        train_records=4,
        eval_cases=2,
        completion_mode="key_only",
    )
    first_record = records[0]
    first_case = cases[0]
    completion = first_record["trace"][0]["content"]

    assert len(records) == 8
    assert {row["behavior"] for row in records} == {"copy_key_sequence"}
    assert {row["slot_family"] for row in records} == {"keyvalue_repo_key_only"}
    assert "=" not in completion
    assert completion.splitlines() == first_record["target_keys"]
    assert "one per line" in first_record["system"]
    assert "Return only one key per line" in first_record["messages"][0]["content"]
    assert first_case["expected_behavior"] == "copy_key_sequence"
    assert first_case["slot_family"] == "keyvalue_repo_key_only"
    assert first_case["required_substrings"] == first_case["target_keys"]
    assert first_case["expected_key_sequence"] == first_case["target_keys"]
    assert first_case["enforce_key_sequence"] is True
    assert first_case["enforce_value_sequence"] is False
    assert all("=" not in value for value in first_case["required_substrings"])
    assert "=" in first_case["forbidden_substrings"]
    assert {row["eval_tier"] for row in cases} == {
        "seen_slot",
        "covered_value_slot",
        "heldout_slot",
    }


def test_keyvalue_value_only_mode_emits_requested_values_without_keys():
    records = build_train_records(
        seed=17,
        train_records=4,
        train_variants_per_row=2,
        completion_mode="value_only",
    )
    cases = build_eval_cases(
        seed=17,
        eval_mode="coverage_ladder",
        train_records=4,
        eval_cases=2,
        completion_mode="value_only",
    )
    first_record = records[0]
    first_case = cases[0]
    completion = first_record["trace"][0]["content"]

    assert len(records) == 8
    assert {row["behavior"] for row in records} == {"copy_value_sequence"}
    assert {row["slot_family"] for row in records} == {"keyvalue_repo_value_only"}
    assert "=" not in completion
    assert completion.splitlines() == first_record["target_values"]
    assert first_record["target_values"] == [
        first_record["expected_slots"][key] for key in first_record["target_keys"]
    ]
    assert "one per line" in first_record["system"]
    assert "Return only one value per line" in first_record["messages"][0]["content"]
    assert first_case["expected_behavior"] == "copy_value_sequence"
    assert first_case["slot_family"] == "keyvalue_repo_value_only"
    assert first_case["required_substrings"] == first_case["expected_value_sequence"]
    assert first_case["expected_value_sequence"] == [
        first_case["expected_slots"][key] for key in first_case["target_keys"]
    ]
    assert first_case["enforce_key_sequence"] is False
    assert first_case["enforce_value_sequence"] is True
    assert all("=" not in value for value in first_case["required_substrings"])
    assert "=" in first_case["forbidden_substrings"]
    assert {row["eval_tier"] for row in cases} == {
        "seen_slot",
        "covered_value_slot",
        "heldout_slot",
    }


def test_keyvalue_value_only_can_target_one_value_for_curriculum():
    records = build_train_records(
        seed=17,
        train_records=4,
        train_variants_per_row=2,
        completion_mode="value_only",
        target_fields=1,
    )
    cases = build_eval_cases(
        seed=17,
        eval_mode="coverage_ladder",
        train_records=4,
        eval_cases=2,
        completion_mode="value_only",
        target_fields=1,
    )
    first_record = records[0]
    first_case = cases[0]

    assert len(records) == 8
    assert all(len(row["target_keys"]) == 1 for row in records)
    assert all(len(row["target_values"]) == 1 for row in records)
    assert all(len(row["expected_slots"]) == 1 for row in records)
    assert all(len(row["target_keys"]) == 1 for row in cases)
    assert all(len(row["expected_value_sequence"]) == 1 for row in cases)
    assert first_record["trace"][0]["content"] == first_record["target_values"][0]
    assert first_case["required_substrings"] == first_case["expected_value_sequence"]
    assert first_case["enforce_key_sequence"] is False
    assert first_case["enforce_value_sequence"] is True
    assert len(first_case["forbidden_substrings"]) == DISTRACTOR_FIELDS + 1
    assert "=" in first_case["forbidden_substrings"]
    assert {row["eval_tier"] for row in cases} == {
        "seen_slot",
        "covered_value_slot",
        "heldout_slot",
    }


def test_keyvalue_value_only_boundaries_wrap_and_keep_plain_expected_values():
    records = build_train_records(
        seed=17,
        train_records=4,
        train_variants_per_row=2,
        completion_mode="value_only",
        target_fields=1,
        value_boundaries=True,
    )
    cases = build_eval_cases(
        seed=17,
        eval_mode="coverage_ladder",
        train_records=4,
        eval_cases=2,
        completion_mode="value_only",
        target_fields=1,
        value_boundaries=True,
    )
    first_record = records[0]
    first_case = cases[0]
    completion = first_record["trace"][0]["content"]
    wrapped_value = f"{VALUE_BOUNDARY_OPEN}{first_record['target_values'][0]}{VALUE_BOUNDARY_CLOSE}"

    assert completion == wrapped_value
    assert wrapped_value in first_record["repo_context"]
    assert first_record["value_boundaries"] is True
    assert VALUE_BOUNDARY_OPEN in first_record["system"]
    assert VALUE_BOUNDARY_OPEN in first_record["messages"][0]["content"]
    assert first_case["value_boundaries"] is True
    assert first_case["required_substrings"] == [
        f"{VALUE_BOUNDARY_OPEN}{first_case['expected_value_sequence'][0]}{VALUE_BOUNDARY_CLOSE}"
    ]
    assert first_case["expected_value_sequence"] == [
        first_case["expected_slots"][key] for key in first_case["target_keys"]
    ]
    assert strip_value_boundaries(completion) == first_record["target_values"][0]


def test_keyvalue_coverage_ladder_adds_tokenizer_covered_tier():
    records = build_train_records(seed=17, train_records=TRAIN_RECORDS)
    cases = build_eval_cases(
        seed=17,
        eval_mode="coverage_ladder",
        train_records=TRAIN_RECORDS,
        eval_cases=EVAL_CASES,
    )
    train_values = set()
    for row in records:
        for line in row["repo_context"].splitlines():
            _, value = line.split(": ", 1)
            train_values.add(value)
        train_values.update(row["expected_slots"].values())
    covered_cases = [row for row in cases if row["eval_tier"] == "covered_value_slot"]
    heldout_cases = [row for row in cases if row["eval_tier"] == "heldout_slot"]

    assert len(cases) == EVAL_CASES * 3
    assert {row["eval_tier"] for row in cases} == {
        "seen_slot",
        "covered_value_slot",
        "heldout_slot",
    }
    assert covered_cases
    assert all(
        value in train_values
        for row in covered_cases
        for value in row["expected_slots"].values()
    )
    assert any(
        value not in train_values
        for row in heldout_cases
        for value in row["expected_slots"].values()
    )


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
            "coverage_ladder",
            "--train-records",
            "12",
            "--train-variants-per-row",
            "3",
            "--eval-cases",
            "8",
        ],
        cwd=ROOT,
        check=True,
    )

    payload = json.loads(manifest.read_text())
    assert payload["format"] == "agentcoder-keyvalue-copy-sft-v1"
    assert payload["eval_mode"] == "coverage_ladder"
    assert payload["base_train_rows"] == 12
    assert payload["train_variants_per_row"] == 3
    assert payload["train_records"] == 36
    assert payload["train_source_row_count"] == 12
    assert payload["train_variant_counts"] == {"0": 12, "1": 12, "2": 12}
    assert payload["eval_cases"] == 24
    assert payload["target_fields"] == TARGET_FIELDS
    assert payload["distractor_fields"] == DISTRACTOR_FIELDS
    assert payload["eval_tier_counts"] == {
        "covered_value_slot": 8,
        "heldout_slot": 8,
        "seen_slot": 8,
    }
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
                "key_sequence_correct": True,
                "value_sequence_correct": True,
                "name": "seen",
            },
            {
                "slot_family": "keyvalue_repo_copy",
                "eval_tier": "heldout_slot",
                "passed": False,
                "slot_error": True,
                "behavior_correct": True,
                "key_sequence_correct": False,
                "value_sequence_correct": False,
                "name": "heldout",
            },
        ]
    }

    family = summarize_slot_families(payload)
    ladder = summarize_ladder(payload)

    assert family["keyvalue_repo_copy"]["pass_count"] == 1
    assert family["keyvalue_repo_copy"]["slot_error_count"] == 1
    assert family["keyvalue_repo_copy"]["key_sequence_accuracy"] == 0.5
    assert family["keyvalue_repo_copy"]["value_sequence_accuracy"] == 0.5
    assert ladder["keyvalue_repo_copy"]["seen_slot"]["pass_rate"] == 1.0
    assert ladder["keyvalue_repo_copy"]["seen_slot"]["key_sequence_accuracy"] == 1.0
    assert ladder["keyvalue_repo_copy"]["seen_slot"]["value_sequence_accuracy"] == 1.0
    assert ladder["keyvalue_repo_copy"]["heldout_slot"]["failed_cases"] == ["heldout"]
    assert ladder["keyvalue_repo_copy"]["heldout_slot"]["key_sequence_accuracy"] == 0.0
    assert ladder["keyvalue_repo_copy"]["heldout_slot"]["value_sequence_accuracy"] == 0.0


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


def test_keyvalue_request_value_configs_enable_request_route():
    for path, model_name in [
        ("configs/scratch/raam_agentcoder_keyvalue_request_value_gate.yaml", "raam"),
        ("configs/scratch/transformer_agentcoder_keyvalue_request_value_gate.yaml", "transformer"),
    ]:
        payload = yaml.safe_load(Path(path).read_text())

        assert payload["model_name"] == model_name
        assert payload["copy_head"]["enabled"] is True
        assert payload["copy_head"]["key_follow_strength"] > 0
        assert payload["copy_head"]["key_follow_stop_token_ids"] == [23, 328]
        assert payload["copy_head"]["request_key_follow_strength"] == 12.0
        assert payload["copy_head"]["request_key_follow_continuation_strength"] == 24.0
        assert payload["copy_head"]["request_key_follow_recent_tokens"] == 64
        assert payload["copy_head"]["request_key_follow_after_token_id"] == 5
        assert payload["copy_head"]["request_key_follow_before_token_id"] == 6
        assert payload["copy_head"]["request_key_follow_value_span"] == 32
        assert payload["copy_head"]["request_key_follow_eval_only"] is True
        assert payload["copy_head"]["request_key_follow_source_after_token_id"] == 9
        assert payload["copy_head"]["request_key_follow_query_after_token_id"] == 271
        assert payload["copy_head"]["request_key_follow_query_before_token_ids"] == [273]
        assert payload["copy_head"]["request_key_follow_query_ignore_token_ids"] == [23, 269, 272, 328]
        assert payload["copy_head"]["request_key_follow_prompt_suffix_tokens"] == 1
