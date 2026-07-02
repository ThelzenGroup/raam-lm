#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raam_lm.config import load_config
from raam_lm.probes import PROBE_BUILDERS
from raam_lm.registry import available_models, build_model
from raam_lm.train_utils import resolve_device, seed_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", default="runs/probe_results.json")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.device:
        config.train.device = args.device
    seed_all(config.train.seed)
    device = resolve_device(config.train.device)
    results = []
    with torch.no_grad():
        for model_name in available_models():
            model_config = copy.deepcopy(config)
            model_config.model_name = model_name
            if model_name != "raam":
                model_config.compression.enabled = False
                model_config.use_dynamic_hourglass_compression = False
                model_config.use_anchor_preserved_local_global = False
                model_config.use_attention_islands = model_name == "transformer"
            model = build_model(model_config).to(device).eval()
            for name, builder in PROBE_BUILDERS.items():
                batch = builder(model_config.train.batch_size, model_config.train.seq_len, model_config.vocab_size, device)
                out = model(batch, labels=batch, global_step=model_config.train.steps)
                logits = out["logits"]
                pred = logits[:, :-1].argmax(dim=-1)
                target = batch[:, 1:]
                acc = (pred == target).float().mean().item()
                results.append(
                    {
                        "probe": name,
                        "model_name": model_name,
                        "loss": float(out["next_token_loss"].detach().cpu()),
                        "next_token_accuracy": acc,
                        "batch_size": model_config.train.batch_size,
                        "seq_len": model_config.train.seq_len,
                    }
                )
    output = {"config": args.config, "device": str(device), "results": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
