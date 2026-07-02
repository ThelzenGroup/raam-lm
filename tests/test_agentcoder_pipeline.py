from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from raam_lm.agent_data import pack_binary_shards, pack_documents, render_agent_record, write_int32_tokens
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


def test_dataset_packing(tmp_path):
    data = tmp_path / "tiny.jsonl"
    write_tiny_agentic(data)
    tokenizer = train_agent_tokenizer([data], vocab_size=384)
    manifest = pack_documents([data], tokenizer, tmp_path / "packed", seq_len=32, val_fraction=0.5)
    assert manifest["train_tokens"] > 0
    assert manifest["val_tokens"] > 0
    assert (tmp_path / "packed" / "train.bin").exists()
    assert (tmp_path / "packed" / "val.bin").exists()


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
    train_cmd = [
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
        "--steps",
        "2",
        "--device",
        "cpu",
    ]
    subprocess.run(train_cmd, cwd=ROOT, check=True)
    ckpt = train / "checkpoints" / "last.pt"
    assert ckpt.exists()
    subprocess.run(train_cmd[:-4] + ["--steps", "3", "--resume", str(ckpt), "--device", "cpu"], cwd=ROOT, check=True)
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
