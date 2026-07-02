#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raam_lm.tokenization import train_agent_tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny dependency-free AgentCoder tokenizer.")
    parser.add_argument("inputs", nargs="+", help="Local text/code/jsonl files or folders.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--min-frequency", type=int, default=1)
    args = parser.parse_args()
    tokenizer = train_agent_tokenizer(args.inputs, vocab_size=args.vocab_size, min_frequency=args.min_frequency)
    tokenizer.save(args.output)
    print(f"saved tokenizer vocab_size={tokenizer.vocab_size} path={args.output}")


if __name__ == "__main__":
    main()

