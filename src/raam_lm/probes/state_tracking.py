from __future__ import annotations

import torch


def make_state_tracking_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    batch = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
    for b in range(batch_size):
        value = 10 + b
        cursor = 0
        while cursor + 3 < seq_len:
            batch[b, cursor] = 4
            batch[b, cursor + 1] = 5
            value = (value + 3) % vocab_size
            batch[b, cursor + 2] = value
            cursor += 3
        if cursor < seq_len:
            batch[b, cursor:] = value
    return batch

