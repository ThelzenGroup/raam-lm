from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from scripts.make_agentcoder_copyonly_sft import (
    EVAL_PER_FAMILY,
    SEEN_EVAL_PER_FAMILY,
    TRAIN_PER_FAMILY,
    build_eval_cases,
    build_train_records,
)


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {"repo_lookup_copy", "patch_return_copy", "patch_literal_copy"}


def slot_tuple(row: dict) -> tuple[str, tuple[tuple[str, str], ...]]:
    return row["slot_family"], tuple(sorted(row["expected_slots"].items()))


def test_copyonly_generator_has_balanced_ladder_tiers():
    records = build_train_records(seed=17)
    cases = build_eval_cases(seed=17, eval_mode="ladder")

    assert len(records) == TRAIN_PER_FAMILY * 3
    assert len(cases) == (SEEN_EVAL_PER_FAMILY + EVAL_PER_FAMILY) * 3
    assert {row["slot_family"] for row in records} == FAMILIES
    assert {row["slot_family"] for row in cases} == FAMILIES
    assert {row["eval_tier"] for row in cases} == {"seen_slot", "heldout_slot"}
    assert {row["behavior"] for row in records} == {"copy_slot_values"}
    assert {row["expected_behavior"] for row in cases} == {"copy_slot_values"}

    train_slots = {slot_tuple(row) for row in records}
    seen_slots = {slot_tuple(row) for row in cases if row["eval_tier"] == "seen_slot"}
    heldout_slots = {slot_tuple(row) for row in cases if row["eval_tier"] == "heldout_slot"}

    assert seen_slots.issubset(train_slots)
    assert heldout_slots.isdisjoint(train_slots)
    for family in FAMILIES:
        assert sum(1 for row in records if row["slot_family"] == family) == TRAIN_PER_FAMILY
        assert sum(1 for row in cases if row["slot_family"] == family and row["eval_tier"] == "seen_slot") == SEEN_EVAL_PER_FAMILY
        assert sum(1 for row in cases if row["slot_family"] == family and row["eval_tier"] == "heldout_slot") == EVAL_PER_FAMILY


def test_copyonly_cases_require_short_key_value_outputs():
    cases = build_eval_cases(seed=17, eval_mode="heldout")
    repo_case = next(row for row in cases if row["slot_family"] == "repo_lookup_copy")
    return_case = next(row for row in cases if row["slot_family"] == "patch_return_copy")
    literal_case = next(row for row in cases if row["slot_family"] == "patch_literal_copy")

    assert repo_case["required_substrings"] == [
        f"symbol={repo_case['expected_slots']['symbol']}",
        f"file={repo_case['expected_slots']['file']}",
    ]
    assert f"helper={return_case['expected_slots']['helper']}" in return_case["required_substrings"]
    assert f"return={return_case['expected_slots']['return']}" in return_case["required_substrings"]
    assert f"literal={literal_case['expected_slots']['literal']}" in literal_case["required_substrings"]
    assert all(row["forbidden_substrings"] for row in cases)
    assert all("```diff" not in "\n".join(row["required_substrings"]) for row in cases)


def test_copyonly_cli_writes_manifest_and_cases(tmp_path):
    train = tmp_path / "train.jsonl"
    cases = tmp_path / "cases.json"
    manifest = tmp_path / "manifest.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/make_agentcoder_copyonly_sft.py",
            "--train-output",
            str(train),
            "--cases-output",
            str(cases),
            "--manifest-output",
            str(manifest),
            "--eval-mode",
            "ladder",
        ],
        cwd=ROOT,
        check=True,
    )

    payload = json.loads(manifest.read_text())
    assert payload["format"] == "agentcoder-copyonly-sft-v1"
    assert payload["eval_mode"] == "ladder"
    assert payload["train_records"] == TRAIN_PER_FAMILY * 3
    assert payload["eval_cases"] == (SEEN_EVAL_PER_FAMILY + EVAL_PER_FAMILY) * 3
    assert payload["eval_tier_counts"] == {
        "heldout_slot": EVAL_PER_FAMILY * 3,
        "seen_slot": SEEN_EVAL_PER_FAMILY * 3,
    }
    assert train.exists()
    assert cases.exists()
