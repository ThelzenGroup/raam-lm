#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raam_lm.config import load_config
from raam_lm.train_utils import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_training(config, steps=args.steps, device_override=args.device, log_path=args.log_path)
    print(f"smoke_train_complete log_path={result['log_path']} final_loss={result['last_metrics']['train_loss']:.6f}")


if __name__ == "__main__":
    main()

