#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raam_lm.config import config_hash, load_config
from raam_lm.flops import estimate_flops_per_token


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "config_path": args.config,
                "config_hash": config_hash(config),
                "model_name": config.model_name,
                "estimated_flops_per_token": estimate_flops_per_token(config),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

