from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.make_agentcoder_atomic_copy_sft import (
    EVAL_CASES,
    TRAIN_RECORDS,
    build_eval_cases,
    build_train_records,
)


ROOT = Path(__file__).resolve().parents[1]


def slots(row: dict) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(row["expected_slots"].items()))


def test_atomic_copy_generator_is_single_family_and_no_decoys():
    records = build_train_records(seed=17)
    cases = build_eval_cases(seed=17, eval_mode="mirror")

    assert len(records) == TRAIN_RECORDS
    assert len(cases) == EVAL_CASES
    assert {row["slot_family"] for row in records} == {"atomic_repo_pair_copy"}
    assert {row["slot_family"] for row in cases} == {"atomic_repo_pair_copy"}
    assert {row["behavior"] for row in records} == {"copy_slot_values"}
    assert {row["expected_behavior"] for row in cases} == {"copy_slot_values"}
    assert {row["eval_tier"] for row in cases} == {"mirror_slot"}
    assert all(row["forbidden_substrings"] == [] for row in cases)
    assert all("---" not in row["repo_context"] for row in records)


def test_atomic_copy_mirror_and_heldout_splits_are_explicit():
    records = build_train_records(seed=17)
    mirror_cases = build_eval_cases(seed=17, eval_mode="mirror")
    heldout_cases = build_eval_cases(seed=17, eval_mode="heldout")
    ladder_cases = build_eval_cases(seed=17, eval_mode="ladder")

    train_slots = {slots(row) for row in records}
    mirror_slots = {slots(row) for row in mirror_cases}
    heldout_slots = {slots(row) for row in heldout_cases}

    assert mirror_slots.issubset(train_slots)
    assert heldout_slots.isdisjoint(train_slots)
    assert len(ladder_cases) == EVAL_CASES * 2
    assert {row["eval_tier"] for row in ladder_cases} == {"mirror_slot", "heldout_slot"}


def test_atomic_copy_single_record_mode_reuses_one_train_slot():
    records = build_train_records(seed=17, train_records=1)
    cases = build_eval_cases(seed=17, eval_mode="mirror", train_records=1, eval_cases=4)

    assert len(records) == 1
    assert len(cases) == 4
    assert {slots(row) for row in cases} == {slots(records[0])}
    assert [row["name"] for row in cases] == [
        "atomic_mirror_000",
        "atomic_mirror_001",
        "atomic_mirror_002",
        "atomic_mirror_003",
    ]


def test_atomic_copy_required_outputs_are_two_key_value_lines():
    case = build_eval_cases(seed=17, eval_mode="mirror")[0]

    assert case["required_substrings"] == [
        f"symbol={case['expected_slots']['symbol']}",
        f"file={case['expected_slots']['file']}",
    ]
    assert "Copy the symbol and file exactly" in case["prompt"]
    assert "```diff" not in case["prompt"]


def test_atomic_copy_cli_writes_manifest_and_cases(tmp_path):
    train = tmp_path / "train.jsonl"
    cases = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_agentcoder_atomic_copy_sft.py",
            "--train-output",
            str(train),
            "--cases-output",
            str(cases),
            "--manifest-output",
            str(manifest),
            "--eval-mode",
            "ladder",
            "--train-records",
            "1",
            "--eval-cases",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )

    payload = json.loads(manifest.read_text())
    assert payload["format"] == "agentcoder-atomic-copy-sft-v1"
    assert payload["eval_mode"] == "ladder"
    assert payload["train_records"] == 1
    assert payload["eval_cases"] == 4
    assert payload["requested_train_records"] == 1
    assert payload["requested_eval_cases"] == 2
    assert payload["eval_tier_counts"] == {
        "heldout_slot": 2,
        "mirror_slot": 2,
    }
    assert train.exists()
    assert cases.exists()
