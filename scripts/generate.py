#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raam_lm.config import load_config
from raam_lm.registry import build_model
from raam_lm.tokenization import AgentCoderTokenizer
from raam_lm.train_utils import resolve_device, seed_all


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate from a RAAM-AgentCoder checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    seed_all(args.seed)
    tokenizer = AgentCoderTokenizer.load(args.tokenizer)
    config = load_config(args.config)
    config.vocab_size = tokenizer.vocab_size
    device = resolve_device(args.device)
    model = build_model(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    ids = tokenizer.encode(args.prompt, add_bos=True, add_eos=False)
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = ids[-config.max_seq_len :]
            input_ids = torch.tensor([context], device=device, dtype=torch.long)
            logits = model(input_ids)["logits"][0, -1] / max(args.temperature, 1e-6)
            if args.top_k > 0:
                values, indices = torch.topk(logits, k=min(args.top_k, logits.numel()))
                probs = torch.softmax(values, dim=-1)
                next_id = indices[torch.multinomial(probs, num_samples=1)].item()
            else:
                probs = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1).item()
            ids.append(int(next_id))
            if next_id == tokenizer.eos_token_id:
                break
    elapsed = time.perf_counter() - start
    text = tokenizer.decode(ids, skip_special=False)
    print(text)
    print(f"\n--- generation_metrics tokens={len(ids)} elapsed_sec={elapsed:.4f} tokens_per_sec={len(ids)/max(elapsed,1e-9):.2f}")


if __name__ == "__main__":
    main()

