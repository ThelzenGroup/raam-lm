"""Code/chat-oriented tokenizer with byte fallback.

This is intentionally small and dependency-free. It is not a production BPE
implementation, but it gives scratch experiments a trained vocabulary, stable
special tokens, and guaranteed byte-level coverage.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<|system|>",
    "<|user|>",
    "<|assistant|>",
    "<|tool|>",
    "<|tool_result|>",
    "<|repo_context|>",
    "<|patch|>",
    "<|test_output|>",
    "<|final|>",
]

BYTE_PREFIX = "<0x"
TOKEN_RE = re.compile(
    r"<\|[a-zA-Z0-9_]+\|>|"
    r"[A-Za-z_][A-Za-z0-9_]*|"
    r"\d+\.\d+|\d+|"
    r"==|!=|<=|>=|->|=>|::|//|/\*|\*/|\.\.\.|"
    r"```|@@|---|\+\+\+|"
    r"\s+|"
    r".",
    re.DOTALL,
)


def byte_token(byte_value: int) -> str:
    return f"{BYTE_PREFIX}{byte_value:02X}>"


def iter_text_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    allowed = {
        ".txt",
        ".md",
        ".jsonl",
        ".json",
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
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and child.name != "manifest.json" and child.suffix.lower() in allowed:
                    yield child
        elif path.is_file() and path.name != "manifest.json":
            yield path


@dataclass
class TokenizerManifest:
    tokenizer_type: str
    vocab_size: int
    special_tokens: list[str]
    byte_fallback: bool


class AgentCoderTokenizer:
    """Greedy token tokenizer with guaranteed byte fallback."""

    def __init__(self, vocab: dict[str, int]):
        self.vocab = dict(vocab)
        self.id_to_token = {idx: token for token, idx in self.vocab.items()}
        self.special_tokens = [token for token in SPECIAL_TOKENS if token in self.vocab]
        self.pad_token_id = self.vocab["<pad>"]
        self.bos_token_id = self.vocab["<bos>"]
        self.eos_token_id = self.vocab["<eos>"]
        self.unk_token_id = self.vocab["<unk>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids: list[int] = []
        if add_bos:
            ids.append(self.bos_token_id)
        for match in TOKEN_RE.finditer(text):
            token = match.group(0)
            if token in self.vocab:
                ids.append(self.vocab[token])
                continue
            for byte in token.encode("utf-8", errors="replace"):
                ids.append(self.vocab.get(byte_token(byte), self.unk_token_id))
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = False) -> str:
        parts: list[str] = []
        byte_buffer = bytearray()

        def flush_bytes() -> None:
            if byte_buffer:
                parts.append(bytes(byte_buffer).decode("utf-8", errors="replace"))
                byte_buffer.clear()

        for idx in ids:
            token = self.id_to_token.get(int(idx), "<unk>")
            if token.startswith(BYTE_PREFIX) and token.endswith(">"):
                try:
                    byte_buffer.append(int(token[3:5], 16))
                    continue
                except ValueError:
                    pass
            flush_bytes()
            if skip_special and token in self.special_tokens:
                continue
            parts.append(token)
        flush_bytes()
        return "".join(parts)

    def generation_suppressed_token_ids(self) -> list[int]:
        """Special/control ids that should not be sampled inside assistant text."""

        return [
            self.vocab[token]
            for token in self.special_tokens
            if token != "<eos>" and token in self.vocab
        ]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "manifest": TokenizerManifest(
                tokenizer_type="agentcoder_greedy_byte_fallback",
                vocab_size=self.vocab_size,
                special_tokens=self.special_tokens,
                byte_fallback=True,
            ).__dict__,
            "vocab": self.vocab,
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "AgentCoderTokenizer":
        payload = json.loads(Path(path).read_text())
        return cls({str(k): int(v) for k, v in payload["vocab"].items()})


def train_agent_tokenizer(
    input_paths: Iterable[str | Path],
    vocab_size: int = 512,
    min_frequency: int = 1,
) -> AgentCoderTokenizer:
    vocab: dict[str, int] = {}
    for token in SPECIAL_TOKENS:
        vocab[token] = len(vocab)
    for value in range(256):
        vocab[byte_token(value)] = len(vocab)

    counter: Counter[str] = Counter()
    for path in iter_text_files(input_paths):
        text = path.read_text(errors="replace")
        counter.update(match.group(0) for match in TOKEN_RE.finditer(text))

    for token, count in counter.most_common():
        if count < min_frequency or token in vocab:
            continue
        if len(vocab) >= vocab_size:
            break
        # Keep learned tokens focused on useful multi-byte structure.
        if token.strip() == "" and "\n" not in token and len(token) > 8:
            continue
        vocab[token] = len(vocab)
    return AgentCoderTokenizer(vocab)
