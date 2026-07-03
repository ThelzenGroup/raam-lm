from __future__ import annotations

import json

import pytest

from scripts.run_agentcoder_atomic_cardinality_sweep import (
    build_row,
    first_failure_by_model,
    parse_positive_int_list,
    write_aggregate,
)


def test_parse_positive_int_list_accepts_commas_spaces_and_dedupes():
    assert parse_positive_int_list("1, 2 4,4 8") == [1, 2, 4, 8]


@pytest.mark.parametrize("raw", ["", "0", "-1", "1,two"])
def test_parse_positive_int_list_rejects_invalid_values(raw):
    with pytest.raises(Exception):
        parse_positive_int_list(raw)


def test_build_row_keeps_cardinality_and_training_metrics():
    row = build_row(
        "raam",
        requested_train_records=8,
        requested_eval_cases=8,
        summary={
            "output_dir": "runs/sweep/raam_n008",
            "config": "configs/scratch/raam_agentcoder_atomic_copy_gate.yaml",
            "train_records": 8,
            "eval_cases": 8,
            "pass_count": 7,
            "case_count": 8,
            "pass_rate": 0.875,
            "behavior_accuracy": 1.0,
            "train_tokens": 784,
            "val_tokens": 784,
            "mirror_val": True,
            "eval_mode": "mirror",
            "param_count_non_embedding": 123,
            "estimated_flops_per_token": 456,
            "atomic_eval": "runs/sweep/raam_n008/atomic_eval.json",
            "last_train_row": {
                "train_loss": 0.12,
                "val_next_token_loss": 0.13,
                "tokens_seen": 999,
                "tokens_per_sec": 1234.5,
                "step_time_ms": 10.0,
            },
        },
    )

    assert row["model"] == "raam"
    assert row["requested_train_records"] == 8
    assert row["requested_eval_cases"] == 8
    assert row["pass_rate"] == 0.875
    assert row["train_loss"] == 0.12
    assert row["summary_json"] == "runs/sweep/raam_n008/summary.json"


def test_first_failure_by_model_uses_lowest_cardinality_below_threshold():
    rows = [
        {"model": "raam", "requested_train_records": 4, "pass_rate": 1.0},
        {"model": "raam", "requested_train_records": 8, "pass_rate": 0.875},
        {"model": "raam", "requested_train_records": 16, "pass_rate": 0.5},
        {"model": "transformer", "requested_train_records": 4, "pass_rate": 1.0},
        {"model": "transformer", "requested_train_records": 8, "pass_rate": 1.0},
    ]

    failures = first_failure_by_model(rows, threshold=1.0)

    assert failures["raam"]["requested_train_records"] == 8
    assert failures["transformer"] is None


def test_write_aggregate_writes_summary_json(tmp_path):
    write_aggregate(tmp_path, {"format": "agentcoder-atomic-cardinality-sweep-v1", "rows": []})

    assert json.loads((tmp_path / "summary.json").read_text()) == {
        "format": "agentcoder-atomic-cardinality-sweep-v1",
        "rows": [],
    }
