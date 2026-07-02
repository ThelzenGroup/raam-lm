#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raam_lm.config import load_config
from raam_lm.profiling import profile_training_step


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", default="runs/profile_manifest.json")
    args = parser.parse_args()
    config = load_config(args.config)
    manifest = profile_training_step(
        config,
        config_path=args.config,
        device_override=args.device,
        steps=args.steps,
        output_path=args.output,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

