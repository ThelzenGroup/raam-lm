"""Agentic coding data formatting and packed token datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import struct
from typing import Any, Iterable

import torch

from .tokenization import AgentCoderTokenizer, TOKEN_RE, byte_token, iter_text_files


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

ASSISTANT_LOSS_ROLES = {"assistant", "patch", "tool", "final"}


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


def _append_segment(
    segments: list[tuple[str, bool]],
    text: str,
    *,
    assistant_owned: bool,
) -> None:
    segments.append((text, assistant_owned))


def agent_record_segments(record: dict[str, Any]) -> list[tuple[str, bool]]:
    """Render a structured agent record as text segments plus SFT loss ownership."""

    if "text" in record:
        return [(str(record["text"]), True)]
    segments: list[tuple[str, bool]] = []
    if record.get("system"):
        _append_segment(segments, f"<|system|>\n{record['system']}\n", assistant_owned=False)
    if record.get("repo_context"):
        _append_segment(segments, f"<|repo_context|>\n{record['repo_context']}\n", assistant_owned=False)
    for message in record.get("messages", []):
        role = message.get("role", "user")
        content = message.get("content", "")
        _append_segment(
            segments,
            f"<|{role}|>\n{content}\n",
            assistant_owned=str(role) == "assistant",
        )
    for step in record.get("trace", []):
        kind = step.get("type", "assistant")
        content = step.get("content", "")
        if kind == "tool_call":
            _append_segment(segments, f"<|tool|>\n{content}\n", assistant_owned=True)
        elif kind == "tool_result":
            _append_segment(segments, f"<|tool_result|>\n{content}\n", assistant_owned=False)
        elif kind == "patch":
            _append_segment(segments, f"<|patch|>\n{content}\n", assistant_owned=True)
        elif kind == "test_output":
            _append_segment(segments, f"<|test_output|>\n{content}\n", assistant_owned=False)
        else:
            _append_segment(segments, f"<|assistant|>\n{content}\n", assistant_owned=True)
    if record.get("final"):
        _append_segment(segments, f"<|final|>\n{record['final']}\n", assistant_owned=True)
    return segments


def _render_segments_with_char_mask(segments: list[tuple[str, bool]]) -> tuple[str, list[int]]:
    chars: list[str] = []
    mask: list[int] = []
    for index, (text, assistant_owned) in enumerate(segments):
        if index:
            chars.append("\n")
            mask.append(0)
        chars.extend(text)
        mask.extend([1 if assistant_owned else 0] * len(text))
    start = 0
    end = len(chars)
    while start < end and chars[start].isspace():
        start += 1
    while end > start and chars[end - 1].isspace():
        end -= 1
    text = "".join(chars[start:end]) + "\n"
    char_mask = mask[start:end] + ([mask[end - 1]] if end > start else [0])
    return text, char_mask


def encode_text_with_loss_mask(
    text: str,
    char_mask: list[int],
    tokenizer: AgentCoderTokenizer,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> tuple[list[int], list[int]]:
    ids: list[int] = []
    loss_mask: list[int] = []
    if add_bos:
        ids.append(tokenizer.bos_token_id)
        loss_mask.append(0)
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        score = 1 if any(char_mask[match.start() : match.end()]) else 0
        if token in tokenizer.vocab:
            ids.append(tokenizer.vocab[token])
            loss_mask.append(score)
            continue
        for byte in token.encode("utf-8", errors="replace"):
            ids.append(tokenizer.vocab.get(byte_token(byte), tokenizer.unk_token_id))
            loss_mask.append(score)
    if add_eos:
        ids.append(tokenizer.eos_token_id)
        loss_mask.append(1 if any(loss_mask) else 0)
    return ids, loss_mask


def encode_agent_record_with_loss_mask(
    record: dict[str, Any],
    tokenizer: AgentCoderTokenizer,
    *,
    add_bos: bool = False,
    add_eos: bool = False,
) -> tuple[list[int], list[int]]:
    text, char_mask = _render_segments_with_char_mask(agent_record_segments(record))
    return encode_text_with_loss_mask(text, char_mask, tokenizer, add_bos=add_bos, add_eos=add_eos)


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
                docs.append({"source": f"{path}:{line_no}", "text": text, "hash": stable_hash_text(text), "record": record})
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
    assistant_loss_only: bool = False,
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

    def encode_docs(items: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
        tokens: list[int] = []
        loss_mask: list[int] = []
        for item in items:
            if assistant_loss_only and "record" in item:
                item_tokens, item_mask = encode_agent_record_with_loss_mask(
                    item["record"],
                    tokenizer,
                    add_bos=True,
                    add_eos=True,
                )
            else:
                item_tokens = tokenizer.encode(item["text"], add_bos=True, add_eos=True)
                item_mask = [1] * len(item_tokens)
                if item_mask:
                    item_mask[0] = 0
            tokens.extend(item_tokens)
            loss_mask.extend(item_mask)
        return tokens, loss_mask

    train_tokens, train_loss_mask = encode_docs(train_docs)
    val_tokens, val_loss_mask = encode_docs(val_docs)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_int32_tokens(output_dir / "train.bin", train_tokens)
    write_int32_tokens(output_dir / "val.bin", val_tokens)
    if assistant_loss_only:
        write_int32_tokens(output_dir / "train_loss_mask.bin", train_loss_mask)
        write_int32_tokens(output_dir / "val_loss_mask.bin", val_loss_mask)
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
        "assistant_loss_only": assistant_loss_only,
        "train_loss_tokens": int(sum(train_loss_mask)) if assistant_loss_only else None,
        "val_loss_tokens": int(sum(val_loss_mask)) if assistant_loss_only else None,
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

    def __init__(self, path: str | Path, seed: int = 17, loss_mask_path: str | Path | None = None):
        self.tokens = read_int32_tokens(path)
        if self.tokens.numel() < 2:
            raise ValueError(f"packed dataset {path} must contain at least two tokens")
        self.loss_mask = read_int32_tokens(loss_mask_path) if loss_mask_path is not None else None
        if self.loss_mask is not None and self.loss_mask.numel() != self.tokens.numel():
            raise ValueError(
                f"loss mask {loss_mask_path} length {self.loss_mask.numel()} does not match token length {self.tokens.numel()}"
            )
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)

    def _batch_from_tensor(
        self,
        tensor: torch.Tensor,
        starts: torch.Tensor | None,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        if starts is not None:
            rows = [tensor[start : start + seq_len] for start in starts.tolist()]
            return torch.stack(rows)
        repeats = (seq_len + tensor.numel() - 1) // tensor.numel()
        row = tensor.repeat(repeats)[:seq_len]
        return row.unsqueeze(0).expand(batch_size, -1).clone()

    def next_batch(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        batch, _ = self.next_batch_with_loss_mask(batch_size, seq_len, device)
        return batch

    def next_batch_with_loss_mask(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.tokens.numel() >= seq_len:
            max_start = max(1, self.tokens.numel() - seq_len + 1)
            starts = torch.randint(0, max_start, (batch_size,), generator=self.generator)
            batch = self._batch_from_tensor(self.tokens, starts, batch_size, seq_len)
            mask = (
                self._batch_from_tensor(self.loss_mask, starts, batch_size, seq_len)
                if self.loss_mask is not None
                else None
            )
        else:
            batch = self._batch_from_tensor(self.tokens, None, batch_size, seq_len)
            mask = (
                self._batch_from_tensor(self.loss_mask, None, batch_size, seq_len)
                if self.loss_mask is not None
                else None
            )
        batch = batch.to(device=device, dtype=torch.long)
        if mask is not None:
            mask = mask.to(device=device, dtype=torch.float32)
        return batch, mask
