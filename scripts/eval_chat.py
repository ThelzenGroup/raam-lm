#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


PROMPTS = [
    "<|user|>\nExplain what a failing unit test means in plain English.\n<|assistant|>\n",
    "<|user|>\nAsk one clarifying question before editing a risky production file.\n<|assistant|>\n",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny chat behavior smoke eval.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="runs/chat_eval.json")
    args = parser.parse_args()
    results = []
    for prompt in PROMPTS:
        cmd = [
            sys.executable,
            "scripts/generate.py",
            "--config",
            args.config,
            "--tokenizer",
            args.tokenizer,
            "--checkpoint",
            args.checkpoint,
            "--prompt",
            prompt,
            "--device",
            args.device,
            "--max-new-tokens",
            "24",
        ]
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
        text = proc.stdout
        results.append(
            {
                "prompt": prompt,
                "output": text,
                "response_length": len(text),
                "mentions_assistant_marker": "<|assistant|>" in text,
            }
        )
    payload = {"results": results}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

