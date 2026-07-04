from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

from scripts.make_agentcoder_curated_sft import build_eval_cases, build_train_records
from raam_lm.agent_data import (
    encode_agent_record_with_loss_mask,
    pack_binary_shards,
    pack_documents,
    read_int32_tokens,
    render_agent_record,
    write_int32_tokens,
)
from raam_lm.tokenization import AgentCoderTokenizer, train_agent_tokenizer


ROOT = Path(__file__).resolve().parents[1]


def write_tiny_agentic(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "system": "You are a coding agent.",
                        "repo_context": "file: calc.py\ndef add(a,b): return a-b",
                        "messages": [{"role": "user", "content": "Fix add."}],
                        "trace": [
                            {"type": "assistant", "content": "Plan then patch."},
                            {
                                "type": "patch",
                                "content": "--- a/calc.py\n+++ b/calc.py\n@@\n-return a-b\n+return a+b\n",
                            },
                        ],
                        "final": "Fixed add.",
                    }
                ),
                json.dumps(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "ValueError: invalid literal for int() with base 10: 'abc'",
                            }
                        ],
                        "final": "Validate before int conversion.",
                    }
                ),
            ]
        )
        + "\n"
    )


def test_agent_record_renderer():
    text = render_agent_record(
        {
            "repo_context": "file: app.py",
            "messages": [{"role": "user", "content": "Review this."}],
            "trace": [{"type": "tool_call", "content": "pytest -q"}],
            "final": "Looks good.",
        }
    )
    assert "<|repo_context|>" in text
    assert "<|user|>" in text
    assert "<|tool|>" in text
    assert "<|final|>" in text


def test_tokenizer_train_save_load(tmp_path):
    data = tmp_path / "tiny.jsonl"
    write_tiny_agentic(data)
    tokenizer = train_agent_tokenizer([data], vocab_size=384)
    out = tmp_path / "tokenizer.json"
    tokenizer.save(out)
    loaded = AgentCoderTokenizer.load(out)
    ids = loaded.encode("<|user|>\nFix calc.py\n", add_bos=True, add_eos=True)
    assert ids[0] == loaded.bos_token_id
    assert ids[-1] == loaded.eos_token_id
    assert loaded.vocab_size <= 384
    suppressed = loaded.generation_suppressed_token_ids()
    assert loaded.vocab["<|assistant|>"] in suppressed
    assert loaded.vocab["<|user|>"] in suppressed
    assert loaded.eos_token_id not in suppressed


def test_dataset_packing(tmp_path):
    data = tmp_path / "tiny.jsonl"
    write_tiny_agentic(data)
    tokenizer = train_agent_tokenizer([data], vocab_size=384)
    manifest = pack_documents([data], tokenizer, tmp_path / "packed", seq_len=32, val_fraction=0.5)
    assert manifest["train_tokens"] > 0
    assert manifest["val_tokens"] > 0
    assert (tmp_path / "packed" / "train.bin").exists()
    assert (tmp_path / "packed" / "val.bin").exists()


def test_dataset_packing_can_mirror_validation_for_overfit(tmp_path):
    data = tmp_path / "tiny.jsonl"
    write_tiny_agentic(data)
    tokenizer = train_agent_tokenizer([data], vocab_size=384)
    manifest = pack_documents(
        [data],
        tokenizer,
        tmp_path / "packed",
        seq_len=32,
        val_fraction=0.5,
        mirror_val=True,
    )

    assert manifest["mirror_val"] is True
    assert manifest["train_docs"] == manifest["val_docs"] == 2
    assert manifest["train_tokens"] == manifest["val_tokens"]


def test_assistant_loss_mask_scores_only_agent_outputs(tmp_path):
    data = tmp_path / "tiny.jsonl"
    write_tiny_agentic(data)
    tokenizer = train_agent_tokenizer([data], vocab_size=384)
    record = {
        "system": "Rules.",
        "repo_context": "file: calc.py",
        "messages": [{"role": "user", "content": "Fix add."}],
        "trace": [{"type": "assistant", "content": "symbol=add\nfile=calc.py"}],
    }

    ids, mask = encode_agent_record_with_loss_mask(record, tokenizer, add_bos=True, add_eos=True)
    decoded_scored = tokenizer.decode([token for token, keep in zip(ids, mask) if keep])

    assert len(ids) == len(mask)
    assert sum(mask) > 0
    assert "symbol" in decoded_scored
    assert "calc.py" in decoded_scored
    assert "Fix add" not in decoded_scored
    assert "file: calc.py" not in decoded_scored


def test_dataset_packing_writes_assistant_loss_masks(tmp_path):
    data = tmp_path / "tiny.jsonl"
    write_tiny_agentic(data)
    tokenizer = train_agent_tokenizer([data], vocab_size=384)
    manifest = pack_documents(
        [data],
        tokenizer,
        tmp_path / "packed_masked",
        seq_len=32,
        val_fraction=0.5,
        assistant_loss_only=True,
    )

    train_tokens = read_int32_tokens(tmp_path / "packed_masked" / "train.bin")
    train_mask = read_int32_tokens(tmp_path / "packed_masked" / "train_loss_mask.bin")

    assert manifest["assistant_loss_only"] is True
    assert manifest["train_loss_tokens"] == int(train_mask.sum().item())
    assert 0 < manifest["train_loss_tokens"] < manifest["train_tokens"]
    assert train_tokens.numel() == train_mask.numel()


def test_dataset_packing_can_focus_on_agent_records_only(tmp_path):
    data = tmp_path / "tiny.jsonl"
    text_jsonl = tmp_path / "plain_text.jsonl"
    notes = tmp_path / "notes.txt"
    write_tiny_agentic(data)
    text_jsonl.write_text(json.dumps({"text": "jsonl plain text should not count as an agent record"}) + "\n")
    notes.write_text("plain documentation that should not be packed in record-only mode\n")
    tokenizer = train_agent_tokenizer([data, text_jsonl, notes], vocab_size=384)

    manifest = pack_documents(
        [data, text_jsonl, notes],
        tokenizer,
        tmp_path / "packed_records_only",
        seq_len=32,
        val_fraction=0.5,
        assistant_loss_only=True,
        agent_records_only=True,
    )

    sources = [doc["source"] for doc in manifest["documents"]]
    assert manifest["agent_records_only"] is True
    assert manifest["source_type_counts"] == {"agent_records": 2, "plain_text": 2}
    assert all("notes.txt" not in source for source in sources)
    assert all("plain_text.jsonl" not in source for source in sources)
    assert manifest["train_loss_tokens"] and manifest["train_loss_tokens"] > 0


def test_dataset_packing_can_disable_plain_text_loss(tmp_path):
    data = tmp_path / "tiny.jsonl"
    notes = tmp_path / "notes.txt"
    write_tiny_agentic(data)
    notes.write_text("plain documentation tokens should be context only when unscored\n")
    tokenizer = train_agent_tokenizer([data, notes], vocab_size=384)

    scored = pack_documents(
        [data, notes],
        tokenizer,
        tmp_path / "packed_plain_scored",
        seq_len=32,
        val_fraction=0.5,
        mirror_val=True,
        assistant_loss_only=True,
        score_plain_text_loss=True,
    )
    unscored = pack_documents(
        [data, notes],
        tokenizer,
        tmp_path / "packed_plain_unscored",
        seq_len=32,
        val_fraction=0.5,
        mirror_val=True,
        assistant_loss_only=True,
        score_plain_text_loss=False,
    )

    assert scored["source_type_counts"] == {"agent_records": 2, "plain_text": 1}
    assert unscored["score_plain_text_loss"] is False
    assert unscored["train_tokens"] == scored["train_tokens"]
    assert unscored["train_loss_tokens"] < scored["train_loss_tokens"]


def test_dataset_packing_can_filter_long_documents(tmp_path):
    data = tmp_path / "tiny.jsonl"
    long_notes = tmp_path / "long_notes.txt"
    write_tiny_agentic(data)
    long_notes.write_text("oversized trajectory\n" + ("x" * 5000) + "\n")
    tokenizer = train_agent_tokenizer([data, long_notes], vocab_size=384)

    manifest = pack_documents(
        [data, long_notes],
        tokenizer,
        tmp_path / "packed_length_filtered",
        seq_len=32,
        val_fraction=0.5,
        max_document_chars=1000,
        max_documents=1,
    )

    assert manifest["max_document_chars"] == 1000
    assert manifest["max_documents"] == 1
    assert manifest["skipped_long_documents"] == 1
    assert manifest["capped_documents"] == 1
    assert manifest["train_docs"] == 1
    assert all("long_notes.txt" not in doc["source"] for doc in manifest["documents"])


def test_pack_dataset_cli_forwards_assistant_loss_only(tmp_path):
    data = tmp_path / "tiny.jsonl"
    tok = tmp_path / "tokenizer.json"
    packed = tmp_path / "packed_cli_masked"
    write_tiny_agentic(data)
    train_agent_tokenizer([data], vocab_size=384).save(tok)

    subprocess.run(
        [
            sys.executable,
            "scripts/pack_dataset.py",
            str(data),
            "--tokenizer",
            str(tok),
            "--output-dir",
            str(packed),
            "--seq-len",
            "32",
            "--val-fraction",
            "0.5",
            "--assistant-loss-only",
            "--agent-records-only",
            "--no-score-plain-text-loss",
        ],
        cwd=ROOT,
        check=True,
    )

    manifest = json.loads((packed / "manifest.json").read_text())
    assert manifest["assistant_loss_only"] is True
    assert manifest["agent_records_only"] is True
    assert manifest["score_plain_text_loss"] is False
    assert manifest["train_loss_tokens"] is not None
    assert (packed / "train_loss_mask.bin").exists()


def test_dataset_packing_skips_manifest_metadata(tmp_path):
    data_dir = tmp_path / "raw"
    data_dir.mkdir()
    data = data_dir / "tiny.jsonl"
    write_tiny_agentic(data)
    (data_dir / "manifest.json").write_text(json.dumps({"not": "training data"}) + "\n")

    tokenizer = train_agent_tokenizer([data_dir], vocab_size=384)
    manifest = pack_documents([data_dir], tokenizer, tmp_path / "packed", seq_len=32, val_fraction=0.5)

    sources = [doc["source"] for doc in manifest["documents"]]
    assert all("manifest.json" not in source for source in sources)


def test_binary_shard_packing(tmp_path):
    shard = tmp_path / "tokens.bin"
    write_int32_tokens(shard, list(range(20)))
    manifest = pack_binary_shards([shard], tmp_path / "packed_bin", seq_len=8, val_fraction=0.25)
    assert manifest["input_type"] == "binary_shards"
    assert manifest["train_tokens"] > 0
    assert manifest["val_tokens"] > 0
    assert (tmp_path / "packed_bin" / "train.bin").exists()


def test_curated_sft_generator_has_behavior_coverage():
    records = build_train_records()
    cases = build_eval_cases()
    behaviors = {row["behavior"] for row in records}
    case_names = {row["name"] for row in cases}
    behavior_counts = Counter(row["behavior"] for row in records)

    assert len(records) == 96
    assert len(cases) == 10
    assert set(behavior_counts.values()) == {8}
    assert {
        "patch_addition",
        "json_tool_command",
        "risky_clarifying_question",
        "plain_debugging",
        "function_completion",
        "stack_trace_diagnosis",
        "repo_context_lookup",
        "test_command",
        "command_disambiguation",
        "patch_boolean_flag",
        "code_review",
        "commit_summary",
    }.issubset(behaviors)
    assert {
        "curated_json_python_files",
        "curated_risky_question",
        "curated_is_even_completion",
    }.issubset(case_names)
    assert all("expected_behavior" in row for row in cases)
    case_by_name = {row["name"]: row for row in cases}
    assert "diff first" in case_by_name["curated_add_patch"]["prompt"]
    assert {"arithmetic.py", "sum_values", "mathlib.py", "totals.py"}.issubset(
        set(case_by_name["curated_add_patch"]["forbidden_substrings"])
    )
    assert "Do not write code" in case_by_name["curated_debugging"]["prompt"]
    assert "Find `def normalize_title`" in case_by_name["curated_repo_lookup"]["prompt"]
    assert "Start the answer with the exact requested symbol `normalize_title`" in case_by_name["curated_repo_lookup"]["prompt"]
    assert "normalize_title is implemented in titles.py" in case_by_name["curated_repo_lookup"]["required_substrings"]
    assert {"slugify", "names.py", "calc.py", "render_invoice", "invoices.py"}.issubset(
        set(case_by_name["curated_repo_lookup"]["forbidden_substrings"])
    )
    assert "calc.py" not in case_by_name["curated_repo_lookup"]["prompt"]
    repo_records = [row for row in records if row["behavior"] == "repo_context_lookup"]
    assert all("Find the matching def line" in row["system"] for row in repo_records)
    assert all("file: calc.py" in row["repo_context"] for row in repo_records)
    assert any("title_case is implemented in titles.py" in row["trace"][0]["content"] for row in repo_records)
    assert all("title_tools.py" not in row["trace"][0]["content"] for row in repo_records)
    assert any("normalize_slug is implemented in slugs.py" in row["trace"][0]["content"] for row in repo_records)
    assert all("add is implemented in calc.py" not in row["trace"][0]["content"] for row in repo_records)
    patch_records = [row for row in records if row["behavior"] == "patch_addition"]
    assert all("Copy the file path, function name, and return expression" in row["system"] for row in patch_records)
    assert any("Patch task for `calc.py`" in row["messages"][0]["content"] for row in patch_records)
    assert {"--- a/flags.py", "def is_enabled(value):", "return value == 'true'"}.issubset(
        set(case_by_name["curated_flag_patch"]["required_substrings"])
    )
    assert {"calc.py", "def add", "return a + b", "toggles.py", "cache_enabled", "== 'on'"}.issubset(
        set(case_by_name["curated_flag_patch"]["forbidden_substrings"])
    )
    assert "Boolean flag task, not arithmetic" in case_by_name["curated_flag_patch"]["prompt"]
    flag_records = [row for row in records if row["behavior"] == "patch_boolean_flag"]
    assert all("Boolean flag repair only" in row["system"] for row in flag_records)
    assert all("Test command" not in row["trace"][0]["content"] for row in flag_records)


def test_train_resume_generate_and_agentic_eval(tmp_path):
    data = tmp_path / "tiny.jsonl"
    write_tiny_agentic(data)
    tok = tmp_path / "tokenizer.json"
    packed = tmp_path / "packed"
    train = tmp_path / "train"
    eval_out = tmp_path / "agentic_eval.json"
    gen_prompt = "<|user|>\nFix add.\n<|assistant|>\n"

    subprocess.run(
        [
            sys.executable,
            "scripts/train_tokenizer.py",
            str(data),
            "--output",
            str(tok),
            "--vocab-size",
            "384",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/pack_dataset.py",
            str(data),
            "--tokenizer",
            str(tok),
            "--output-dir",
            str(packed),
            "--seq-len",
            "32",
            "--val-fraction",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
    )
    train_base_cmd = [
        sys.executable,
        "scripts/train.py",
        "--config",
        "configs/scratch/raam_agentcoder_debug.yaml",
        "--train-bin",
        str(packed / "train.bin"),
        "--val-bin",
        str(packed / "val.bin"),
        "--tokenizer",
        str(tok),
        "--output-dir",
        str(train),
    ]
    subprocess.run(
        train_base_cmd
        + [
            "--steps",
            "2",
            "--device",
            "cpu",
            "--save-best",
            "--restore-best-on-finish",
        ],
        cwd=ROOT,
        check=True,
    )
    ckpt = train / "checkpoints" / "last.pt"
    best_ckpt = train / "checkpoints" / "best.pt"
    assert ckpt.exists()
    assert best_ckpt.exists()
    manifest = json.loads((train / "manifest.json").read_text())
    assert manifest["restore_best_on_finish"] is True
    assert manifest["restored_best_on_finish"] is True
    assert manifest["final_checkpoint_step"] == manifest["best_val_step"]
    subprocess.run(
        train_base_cmd
        + [
            "--steps",
            "3",
            "--resume",
            str(ckpt),
            "--device",
            "cpu",
            "--save-best",
            "--restore-best-on-finish",
        ],
        cwd=ROOT,
        check=True,
    )
    logs = [json.loads(line) for line in (train / "train_log.jsonl").read_text().splitlines() if line.strip()]
    assert any("val_next_token_loss" in row for row in logs)
    gen = subprocess.run(
        [
            sys.executable,
            "scripts/generate.py",
            "--config",
            "configs/scratch/raam_agentcoder_debug.yaml",
            "--tokenizer",
            str(tok),
            "--checkpoint",
            str(ckpt),
            "--prompt",
            gen_prompt,
            "--device",
            "cpu",
            "--max-new-tokens",
            "4",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "generation_metrics" in gen.stdout
    subprocess.run(
        [
            sys.executable,
            "scripts/eval_agentic_coding.py",
            "--config",
            "configs/scratch/raam_agentcoder_debug.yaml",
            "--tokenizer",
            str(tok),
            "--checkpoint",
            str(ckpt),
            "--device",
            "cpu",
            "--output",
            str(eval_out),
        ],
        cwd=ROOT,
        check=True,
    )
    payload = json.loads(eval_out.read_text())
    assert len(payload["results"]) >= 8
    assert "next_token_validation_loss" in payload
