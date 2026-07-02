from __future__ import annotations

import torch


def make_passkey_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    base = (torch.arange(seq_len, device=device).unsqueeze(0) * 17 + 9) % vocab_size
    batch = base.expand(batch_size, -1).clone()
    key_token = min(vocab_size - 1, 7)
    for b in range(batch_size):
        value = (100 + b * 13) % vocab_size
        pos = max(2, seq_len // 4)
        batch[b, pos] = key_token
        batch[b, pos + 1] = value
        batch[b, -3:] = torch.tensor([3, key_token, value], device=device)
    return batch.long()

