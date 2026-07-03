"""Agentic coding data formatting and packed token datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import struct
from typing import Any, Iterable

import torch

from .tokenization import AgentCoderTokenizer, iter_text_files


TEXT_EXTENSIONS = {
    ".txt",
    ".md",
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".jsx",
    ".sh",
    ".yaml",
    ".yml",
    ".toml",
    ".diff",
    ".patch",
    ".log",
}


def stable_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


def render_agent_record(record: dict[str, Any]) -> str:
    """Render canonical or simple JSONL examples into one training document."""

    if "text" in record:
        return str(record["text"])
    parts: list[str] = []
    if record.get("system"):
        parts.append(f"<|system|>\n{record['system']}\n")
    if record.get("repo_context"):
        parts.append(f"<|repo_context|>\n{record['repo_context']}\n")
    for message in record.get("messages", []):
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"<|{role}|>\n{content}\n")
    for step in record.get("trace", []):
        kind = step.get("type", "assistant")
        content = step.get("content", "")
        if kind == "tool_call":
            parts.append(f"<|tool|>\n{content}\n")
        elif kind == "tool_result":
            parts.append(f"<|tool_result|>\n{content}\n")
        elif kind == "patch":
            parts.append(f"<|patch|>\n{content}\n")
        elif kind == "test_output":
            parts.append(f"<|test_output|>\n{content}\n")
        else:
            parts.append(f"<|assistant|>\n{content}\n")
    if record.get("final"):
        parts.append(f"<|final|>\n{record['final']}\n")
    return "\n".join(parts).strip() + "\n"


def load_documents(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for path in iter_text_files(paths):
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                if not line.strip():
                    continue
                record = json.loads(line)
                text = render_agent_record(record)
                docs.append({"source": f"{path}:{line_no}", "text": text, "hash": stable_hash_text(text)})
        elif suffix in TEXT_EXTENSIONS or suffix in {".json", ".jsonl"}:
            text = path.read_text(errors="replace")
            docs.append({"source": str(path), "text": text, "hash": stable_hash_text(text)})
    if not docs:
        raise ValueError("no local text/code/agent documents found")
    return docs


def write_int32_tokens(path: str | Path, tokens: list[int]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for token in tokens:
            fh.write(struct.pack("<i", int(token)))


def read_int32_tokens(path: str | Path) -> torch.Tensor:
    path = Path(path)
    size = path.stat().st_size // 4
    return torch.from_file(str(path), shared=False, size=size, dtype=torch.int32).to(torch.long)


def pack_documents(
    input_paths: Iterable[str | Path],
    tokenizer: AgentCoderTokenizer,
    output_dir: str | Path,
    seq_len: int,
    val_fraction: float = 0.1,
    seed: int = 17,
    mirror_val: bool = False,
) -> dict[str, Any]:
    docs = load_documents(input_paths)
    rng = random.Random(seed)
    rng.shuffle(docs)
    if mirror_val:
        train_docs = docs
        val_docs = docs
    else:
        val_count = max(1, int(round(len(docs) * val_fraction))) if len(docs) > 1 else 1
        val_docs = docs[:val_count]
        train_docs = docs[val_count:] or docs[:]

    def encode_docs(items: list[dict[str, Any]]) -> list[int]:
        tokens: list[int] = []
        for item in items:
            tokens.extend(tokenizer.encode(item["text"], add_bos=True, add_eos=True))
        return tokens

    train_tokens = encode_docs(train_docs)
    val_tokens = encode_docs(val_docs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_int32_tokens(output_dir / "train.bin", train_tokens)
    write_int32_tokens(output_dir / "val.bin", val_tokens)
    manifest = {
        "seq_len": seq_len,
        "seed": seed,
        "val_fraction": val_fraction,
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "train_docs": len(train_docs),
        "val_docs": len(val_docs),
        "mirror_val": mirror_val,
        "documents": [{"source": doc["source"], "hash": doc["hash"]} for doc in docs],
        "format": "int32-token-stream-v1",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def iter_binary_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*.bin")):
                if child.is_file():
                    yield child
        elif path.is_file() and path.suffix.lower() == ".bin":
            yield path


def pack_binary_shards(
    input_paths: Iterable[str | Path],
    output_dir: str | Path,
    seq_len: int,
    val_fraction: float = 0.1,
    seed: int = 17,
    mirror_val: bool = False,
) -> dict[str, Any]:
    files = list(iter_binary_files(input_paths))
    if not files:
        raise ValueError("no .bin token shards found")
    shards = [read_int32_tokens(path) for path in files]
    tokens = torch.cat(shards).tolist()
    rng = random.Random(seed)
    if len(tokens) > 1:
        # Shuffle fixed-size chunks instead of individual tokens to preserve local order.
        chunks = [tokens[i : i + seq_len] for i in range(0, len(tokens), seq_len)]
        rng.shuffle(chunks)
        tokens = [token for chunk in chunks for token in chunk]
    if mirror_val:
        train_tokens = tokens
        val_tokens = tokens
    else:
        val_count = max(1, int(round(len(tokens) * val_fraction))) if len(tokens) > 1 else 1
        val_tokens = tokens[:val_count]
        train_tokens = tokens[val_count:] or tokens[:]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_int32_tokens(output_dir / "train.bin", train_tokens)
    write_int32_tokens(output_dir / "val.bin", val_tokens)
    manifest = {
        "seq_len": seq_len,
        "seed": seed,
        "val_fraction": val_fraction,
        "train_tokens": len(train_tokens),
        "val_tokens": len(val_tokens),
        "source_shards": [str(path) for path in files],
        "format": "int32-token-stream-v1",
        "input_type": "binary_shards",
        "mirror_val": mirror_val,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


class PackedTokenDataset:
    """Random contiguous batches from an int32 token stream."""

    tokenizer_id = "agentcoder_packed_int32"

    def __init__(self, path: str | Path, seed: int = 17):
        self.tokens = read_int32_tokens(path)
        if self.tokens.numel() < 2:
            raise ValueError(f"packed dataset {path} must contain at least two tokens")
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def next_batch(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        if self.tokens.numel() >= seq_len:
            max_start = max(1, self.tokens.numel() - seq_len + 1)
            starts = torch.randint(0, max_start, (batch_size,), generator=self.generator)
            rows = [self.tokens[start : start + seq_len] for start in starts.tolist()]
            batch = torch.stack(rows)
        else:
            repeats = (seq_len + self.tokens.numel() - 1) // self.tokens.numel()
            row = self.tokens.repeat(repeats)[:seq_len]
            batch = row.unsqueeze(0).expand(batch_size, -1).clone()
        return batch.to(device=device, dtype=torch.long)
