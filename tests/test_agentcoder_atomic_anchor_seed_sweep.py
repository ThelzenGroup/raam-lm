from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_agentcoder_atomic_copy_gate import build_pack_command, build_train_command
from scripts.run_agentcoder_atomic_anchor_seed_sweep import (
    ConfigSpec,
    build_repeatability_row,
    first_failure_by_config,
    parse_config_specs,
    summarize_by_config,
)


def test_parse_config_specs_accepts_label_path_pairs():
    specs = parse_config_specs(
        [
            "learned=configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml",
            "hybrid1=configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml",
        ]
    )

    assert specs == [
        ConfigSpec("learned", "configs/scratch/raam_agentcoder_atomic_anchor_attention_gate.yaml"),
        ConfigSpec("hybrid1", "configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml"),
    ]


@pytest.mark.parametrize(
    "raw",
    [
        ["missing_equals"],
        ["=configs/foo.yaml"],
        ["learned="],
        ["bad.label=configs/foo.yaml"],
        ["learned=configs/a.yaml", "learned=configs/b.yaml"],
        [],
    ],
)
def test_parse_config_specs_rejects_invalid_specs(raw):
    with pytest.raises(Exception):
        parse_config_specs(raw)


def test_build_repeatability_row_flattens_single_child_row(tmp_path):
    row = build_repeatability_row(
        config=ConfigSpec("hybrid1", "configs/hybrid.yaml"),
        seed=29,
        train_records=64,
        eval_cases=64,
        run_dir=tmp_path / "hybrid1_seed029",
        child_summary={
            "rows": [
                {
                    "train_records": 64,
                    "eval_cases": 64,
                    "pass_count": 61,
                    "case_count": 64,
                    "pass_rate": 0.953125,
                    "behavior_accuracy": 1.0,
                    "train_loss": 0.12,
                    "val_next_token_loss": 0.13,
                    "tokens_seen": 777,
                    "tokens_per_sec": 1234.5,
                    "step_time_ms": 10.0,
                    "train_tokens": 999,
                    "val_tokens": 999,
                    "mirror_val": True,
                    "eval_mode": "mirror",
                    "param_count_non_embedding": 123,
                    "estimated_flops_per_token": 456,
                    "atomic_eval": "runs/child/atomic_eval.json",
                }
            ]
        },
    )

    assert row["config_label"] == "hybrid1"
    assert row["seed"] == 29
    assert row["pass_count"] == 61
    assert row["pass_rate"] == 0.953125
    assert row["summary_json"].endswith("hybrid1_seed029/summary.json")


def test_summarize_by_config_computes_repeatability_stats():
    rows = [
        {"config_label": "learned", "seed": 17, "pass_rate": 1.0, "pass_count": 64, "case_count": 64, "val_next_token_loss": 0.4, "tokens_per_sec": 100.0},
        {"config_label": "learned", "seed": 29, "pass_rate": 0.875, "pass_count": 56, "case_count": 64, "val_next_token_loss": 0.5, "tokens_per_sec": 200.0},
        {"config_label": "hybrid1", "seed": 17, "pass_rate": 1.0, "pass_count": 64, "case_count": 64, "val_next_token_loss": 0.3, "tokens_per_sec": 300.0},
    ]

    summary = summarize_by_config(rows, threshold=1.0)

    assert summary["learned"]["runs"] == 2
    assert summary["learned"]["seeds"] == [17, 29]
    assert summary["learned"]["mean_pass_rate"] == pytest.approx(0.9375)
    assert summary["learned"]["min_pass_count"] == 56
    assert summary["learned"]["total_pass_count"] == 120
    assert summary["learned"]["total_case_count"] == 128
    assert summary["learned"]["all_passed"] is False
    assert summary["learned"]["mean_val_next_token_loss"] == pytest.approx(0.45)
    assert summary["learned"]["mean_tokens_per_sec"] == pytest.approx(150.0)
    assert summary["hybrid1"]["all_passed"] is True


def test_first_failure_by_config_uses_lowest_seed_below_threshold():
    rows = [
        {"config_label": "hybrid1", "seed": 29, "pass_rate": 0.875},
        {"config_label": "hybrid1", "seed": 17, "pass_rate": 1.0},
        {"config_label": "learned", "seed": 17, "pass_rate": 1.0},
    ]

    failures = first_failure_by_config(rows, threshold=1.0)

    assert failures["hybrid1"]["seed"] == 29
    assert failures["learned"] is None


def test_atomic_copy_gate_forwards_seed_to_packing_and_training():
    args = Namespace(
        config="configs/scratch/raam_agentcoder_atomic_hybrid1_anchor_attention_gate.yaml",
        device="cuda",
        seq_len=96,
        val_fraction=0.2,
        seed=41,
        mirror_val=True,
        assistant_loss_only=True,
        steps=2400,
        eval_batches=None,
    )
    pack_cmd = build_pack_command(
        args,
        Path("runs/gate/generated/atomic_train.jsonl"),
        Path("runs/gate/tokenizer.json"),
        Path("runs/gate/packed"),
    )
    train_cmd = build_train_command(args, Path("runs/gate/packed"), Path("runs/gate/tokenizer.json"), Path("runs/gate/train"))

    assert pack_cmd[pack_cmd.index("--seed") + 1] == "41"
    assert "--assistant-loss-only" in pack_cmd
    assert train_cmd[train_cmd.index("--seed") + 1] == "41"
    assert train_cmd[train_cmd.index("--steps") + 1] == "2400"
