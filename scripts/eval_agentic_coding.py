#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time


TASKS = [
    {
        "name": "chat_helpfulness",
        "prompt": "<|user|>\nExplain how you would approach a risky refactor.\n<|assistant|>\n",
    },
    {
        "name": "code_completion",
        "prompt": "<|user|>\nComplete this Python function:\n```python\ndef is_even(n):\n```\n<|assistant|>\n",
    },
    {
        "name": "bug_fix_patch",
        "prompt": "<|user|>\nFix this Python bug and provide a patch:\n```python\ndef add(a,b):\n    return a-b\n```\n<|assistant|>\n",
    },
    {
        "name": "stack_trace_diagnosis",
        "prompt": "<|user|>\nDiagnose this stack trace:\nValueError: invalid literal for int() with base 10: 'abc'\n<|assistant|>\n",
    },
    {
        "name": "tool_call_format",
        "prompt": "<|user|>\nInspect tests, then propose a shell command as JSON.\n<|assistant|>\n",
    },
    {
        "name": "unit_test_repair",
        "prompt": "<|user|>\nA test says assert add(2, 3) == 5 but got -1. Propose the repair and test command.\n<|assistant|>\n",
    },
    {
        "name": "repo_question_answering",
        "prompt": "<|repo_context|>\nfile: app.py\n```python\nfrom calc import add\n```\nfile: calc.py\n```python\ndef add(a,b): return a+b\n```\n<|user|>\nWhere is add implemented?\n<|assistant|>\n",
    },
    {
        "name": "code_review_comments",
        "prompt": "<|user|>\nReview this code:\n```python\ndef parse_port(x):\n    return int(x)\n```\n<|assistant|>\n",
    },
]


def last_validation_loss_from_checkpoint(checkpoint: str) -> float | None:
    run_dir = Path(checkpoint).resolve().parent.parent
    log_path = run_dir / "train_log.jsonl"
    if not log_path.exists():
        return None
    last_value = None
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "val_next_token_loss" in row:
            last_value = float(row["val_next_token_loss"])
    return last_value


def syntax_valid_python(text: str) -> bool:
    fence = "```python"
    if fence in text:
        snippet = text.split(fence, 1)[1].split("```", 1)[0]
    else:
        snippet = text
    try:
        ast.parse(snippet)
        return True
    except SyntaxError:
        return False


def has_valid_json(text: str) -> bool:
    for start in [idx for idx, ch in enumerate(text) if ch == "{"]:
        for end in range(len(text), start, -1):
            candidate = text[start:end]
            try:
                json.loads(candidate)
                return True
            except Exception:
                continue
    return False


def patch_apply_rate(text: str) -> float:
    if "diff --git" not in text and "---" not in text and "+++" not in text:
        return 0.0
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "file.py").write_text("def add(a,b):\n    return a-b\n")
        patch = root / "change.patch"
        patch.write_text(text)
        proc = subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, capture_output=True)
        return 1.0 if proc.returncode == 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny agentic coding eval smoke test.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="runs/agentic_eval.json")
    args = parser.parse_args()
    results = []
    start_all = time.perf_counter()
    for task in TASKS:
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
            task["prompt"],
            "--device",
            args.device,
            "--max-new-tokens",
            "32",
        ]
        start = time.perf_counter()
        proc = subprocess.run(cmd, text=True, capture_output=True, check=True)
        elapsed = time.perf_counter() - start
        output = proc.stdout
        results.append(
            {
                "task": task["name"],
                "response_length": len(output),
                "latency_sec": elapsed,
                "syntax_valid": syntax_valid_python(output)
                if task["name"] in {"bug_fix_patch", "code_completion"}
                else None,
                "json_valid": has_valid_json(output),
                "exact_patch_apply_rate": patch_apply_rate(output),
                "unit_test_pass_rate": None,
                "output": output,
            }
        )
    payload = {
        "results": results,
        "latency_sec_total": time.perf_counter() - start_all,
        "next_token_validation_loss": last_validation_loss_from_checkpoint(args.checkpoint),
        "mean_patch_apply_rate": sum(r["exact_patch_apply_rate"] for r in results) / len(results),
        "json_tool_call_validity": sum(1 for r in results if r["json_valid"]) / len(results),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
