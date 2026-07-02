from __future__ import annotations

import torch


def make_copy_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    base = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    batch = (base * 5 + torch.arange(batch_size, device=device).unsqueeze(1) * 11) % vocab_size
    span = max(2, min(seq_len // 4, 12))
    batch[:, -span:] = batch[:, :span]
    return batch.long()

