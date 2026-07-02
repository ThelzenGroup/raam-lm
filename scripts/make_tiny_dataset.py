#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raam_lm.data import GeneratedTinyDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/tiny_tokens.txt")
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=512)
    args = parser.parse_args()
    ds = GeneratedTinyDataset(args.vocab_size)
    batch = ds.next_batch(1, args.tokens, device="cpu").squeeze(0).tolist()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(" ".join(str(t) for t in batch) + "\n")
    print(f"wrote {args.tokens} tokens to {path}")


if __name__ == "__main__":
    main()

