from __future__ import annotations

import torch


def make_assoc_recall_batch(batch_size: int, seq_len: int, vocab_size: int, device: torch.device) -> torch.Tensor:
    batch = torch.zeros(batch_size, seq_len, device=device, dtype=torch.long)
    pair_count = max(2, min((seq_len - 4) // 3, 8))
    for b in range(batch_size):
        keys = (torch.arange(pair_count, device=device) * 7 + 20 + b) % vocab_size
        vals = (keys * 3 + 5) % vocab_size
        cursor = 0
        for k, v in zip(keys, vals):
            batch[b, cursor] = k
            batch[b, cursor + 1] = v
            batch[b, cursor + 2] = 1
            cursor += 3
        query_idx = pair_count // 2
        batch[b, cursor : cursor + 3] = torch.tensor([2, keys[query_idx], vals[query_idx]], device=device)
        if cursor + 3 < seq_len:
            filler = torch.arange(seq_len - cursor - 3, device=device) + 31 + b
            batch[b, cursor + 3 :] = filler % vocab_size
    return batch

