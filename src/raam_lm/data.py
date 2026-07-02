"""Minimal deterministic data sources and byte tokenizer fallback."""

from __future__ import annotations

from pathlib import Path
import torch


class ByteTokenizer:
    name = "byte_fallback"

    def __init__(self, vocab_size: int = 256):
        self.vocab_size = max(vocab_size, 256)

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor(list(text.encode("utf-8")), dtype=torch.long) % self.vocab_size

    def decode(self, tokens: torch.Tensor) -> str:
        values = [int(t) % 256 for t in tokens.detach().cpu().tolist()]
        return bytes(values).decode("utf-8", errors="replace")


class GeneratedTinyDataset:
    """Deterministic pseudo-language for CPU smoke tests."""

    tokenizer_id = "generated_tiny_integer_v1"

    def __init__(self, vocab_size: int, seed: int = 17):
        self.vocab_size = vocab_size
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.counter = 0

    def next_batch(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        offsets = torch.arange(batch_size).unsqueeze(1) * 13 + self.counter
        positions = torch.arange(seq_len).unsqueeze(0)
        periodic = (positions * 7 + offsets) % self.vocab_size
        noise = torch.randint(
            0,
            max(self.vocab_size // 16, 2),
            (batch_size, seq_len),
            generator=self.generator,
        )
        batch = (periodic + noise) % self.vocab_size
        if seq_len >= 16:
            span = min(8, seq_len // 4)
            batch[:, seq_len // 2 : seq_len // 2 + span] = batch[:, :span]
        self.counter += 1
        return batch.to(device=device, dtype=torch.long)


class TextFileDataset:
    tokenizer_id = "byte_fallback_text_file"

    def __init__(self, path: str | Path, vocab_size: int):
        self.tokenizer = ByteTokenizer(vocab_size)
        self.tokens = self.tokenizer.encode(Path(path).read_text())
        if self.tokens.numel() < 2:
            raise ValueError("text file must contain at least two byte tokens")
        self.offset = 0

    def next_batch(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        total = batch_size * seq_len
        idx = (torch.arange(total) + self.offset) % self.tokens.numel()
        self.offset = (self.offset + total) % self.tokens.numel()
        return self.tokens[idx].view(batch_size, seq_len).to(device)


class BinaryTokenDataset:
    tokenizer_id = "binary_token_dataset"

    def __init__(self, path: str | Path, dtype: torch.dtype = torch.int64):
        raw = torch.from_file(str(path), shared=False, dtype=dtype)
        if raw.numel() < 2:
            raise ValueError("binary token file must contain at least two tokens")
        self.tokens = raw.to(torch.long)
        self.offset = 0

    def next_batch(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        total = batch_size * seq_len
        idx = (torch.arange(total) + self.offset) % self.tokens.numel()
        self.offset = (self.offset + total) % self.tokens.numel()
        return self.tokens[idx].view(batch_size, seq_len).to(device)


def dataset_identity(dataset) -> str:
    return getattr(dataset, "tokenizer_id", dataset.__class__.__name__)

