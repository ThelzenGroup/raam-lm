#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raam_lm.agent_data import pack_binary_shards, pack_documents
from raam_lm.tokenization import AgentCoderTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Pack local chat/code/agent data into int32 token streams.")
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--mirror-val",
        action="store_true",
        help="Use the same tokens for train and validation; intended for tiny overfit sanity checks.",
    )
    parser.add_argument(
        "--assistant-loss-only",
        action="store_true",
        help="For structured JSONL records, train only on assistant-owned output tokens.",
    )
    args = parser.parse_args()
    if all(str(path).endswith(".bin") for path in args.inputs):
        manifest = pack_binary_shards(
            args.inputs,
            output_dir=args.output_dir,
            seq_len=args.seq_len,
            val_fraction=args.val_fraction,
            seed=args.seed,
            mirror_val=args.mirror_val,
        )
    else:
        if args.tokenizer is None:
            parser.error("--tokenizer is required for text/code/jsonl inputs")
        tokenizer = AgentCoderTokenizer.load(args.tokenizer)
        manifest = pack_documents(
            args.inputs,
            tokenizer=tokenizer,
            output_dir=args.output_dir,
            seq_len=args.seq_len,
            val_fraction=args.val_fraction,
            seed=args.seed,
            mirror_val=args.mirror_val,
            assistant_loss_only=args.assistant_loss_only,
        )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
