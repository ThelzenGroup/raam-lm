#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch


def convert_tensor(tensor: torch.Tensor, dtype: str) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor
    if dtype == "fp16":
        return tensor.half()
    if dtype == "bf16":
        return tensor.bfloat16()
    return tensor.float()


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a smaller model-only checkpoint from a training checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_state = {
        key: convert_tensor(value, args.dtype) if torch.is_tensor(value) else value
        for key, value in checkpoint["model_state"].items()
    }
    payload = {
        "model_state": model_state,
        "step": checkpoint.get("step"),
        "config": checkpoint.get("config"),
        "config_hash": checkpoint.get("config_hash"),
        "tokenizer_path": checkpoint.get("tokenizer_path"),
        "export_dtype": args.dtype,
        "export_note": "model weights only; not optimizer-resumable",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"exported_model_checkpoint path={output} bytes={output.stat().st_size} dtype={args.dtype}")


if __name__ == "__main__":
    main()
