#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raam_lm.config import config_hash, load_config, to_dict
from raam_lm.registry import build_model
from raam_lm.tokenization import AgentCoderTokenizer
from raam_lm.train_utils import resolve_device, seed_all


DEFAULT_PROMPTS = [
    {
        "name": "plain_chat_debugging",
        "category": "chat",
        "prompt": (
            "<|user|>\n"
            "A Python unit test is failing. Explain in plain English how you would debug it before editing code.\n"
            "<|assistant|>\n"
        ),
    },
    {
        "name": "clarifying_question",
        "category": "chat",
        "prompt": (
            "<|user|>\n"
            "Before changing a risky production file, ask one concise clarifying question.\n"
            "<|assistant|>\n"
        ),
    },
    {
        "name": "function_completion",
        "category": "coding",
        "prompt": (
            "<|user|>\n"
            "Complete this Python function:\n"
            "```python\n"
            "def is_even(n):\n"
            "```\n"
            "<|assistant|>\n"
        ),
    },
    {
        "name": "bug_fix_patch",
        "category": "coding",
        "prompt": (
            "<|user|>\n"
            "Fix the bug and provide a minimal patch:\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a - b\n"
            "```\n"
            "<|assistant|>\n"
        ),
    },
    {
        "name": "test_command",
        "category": "software_engineering",
        "prompt": (
            "<|user|>\n"
            "You changed a Python package. Name the safest test command to run before committing.\n"
            "<|assistant|>\n"
        ),
    },
    {
        "name": "stack_trace",
        "category": "software_engineering",
        "prompt": (
            "<|user|>\n"
            "Diagnose this stack trace:\n"
            "ValueError: invalid literal for int() with base 10: 'abc'\n"
            "<|assistant|>\n"
        ),
    },
    {
        "name": "repo_context",
        "category": "agentic_coding",
        "prompt": (
            "<|repo_context|>\n"
            "file: app.py\n"
            "```python\n"
            "from calc import add\n"
            "print(add(2, 3))\n"
            "```\n"
            "file: calc.py\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n"
            "<|user|>\n"
            "Where is add implemented?\n"
            "<|assistant|>\n"
        ),
    },
    {
        "name": "json_tool_call",
        "category": "agentic_coding",
        "prompt": (
            "<|user|>\n"
            "Return one JSON object with a shell command that lists Python files.\n"
            "<|assistant|>\n"
        ),
    },
]


def parse_seed_list(raw: str) -> list[int]:
    seeds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def load_prompts(path: str | None) -> list[dict[str, str]]:
    if path is None:
        return DEFAULT_PROMPTS
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("prompts", [])
    prompts = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or "prompt" not in item:
            raise ValueError(f"prompt entry {index} must be an object with a prompt field")
        prompts.append(
            {
                "name": str(item.get("name", f"prompt_{index:02d}")),
                "category": str(item.get("category", "custom")),
                "prompt": str(item["prompt"]),
            }
        )
    if not prompts:
        raise ValueError("prompt file did not contain any prompts")
    return prompts


def git_sha() -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def has_valid_json(text: str) -> bool:
    for start in [idx for idx, ch in enumerate(text) if ch == "{"]:
        for end in range(len(text), start, -1):
            try:
                json.loads(text[start:end])
                return True
            except Exception:
                continue
    return False


def lexical_flags(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "non_whitespace_chars": len(text.strip()),
        "assistant_marker_count": text.count("<|assistant|>"),
        "contains_python_fence": "```python" in lowered,
        "contains_diff_marker": "diff --git" in lowered or "---" in text or "+++" in text,
        "contains_json_object": has_valid_json(text),
        "mentions_test_command": any(
            marker in lowered
            for marker in [
                "pytest",
                "python -m pytest",
                "npm test",
                "cargo test",
                "go test",
                "pnpm test",
            ]
        ),
    }


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0.0:
        return int(torch.argmax(logits).item())
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, num_samples=1)].item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_one(
    model: torch.nn.Module,
    tokenizer: AgentCoderTokenizer,
    prompt: str,
    *,
    device: torch.device,
    max_seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    ids = list(prompt_ids)
    generated_ids: list[int] = []
    eos_generated = False
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-max_seq_len:]
            input_ids = torch.tensor([context], device=device, dtype=torch.long)
            logits = model(input_ids)["logits"][0, -1]
            next_id = sample_next_token(logits, temperature=temperature, top_k=top_k)
            ids.append(next_id)
            generated_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                eos_generated = True
                break
    elapsed = time.perf_counter() - start
    return {
        "prompt_tokens": len(prompt_ids),
        "generated_tokens": len(generated_ids),
        "elapsed_sec": elapsed,
        "tokens_per_sec": len(generated_ids) / max(elapsed, 1e-9),
        "eos_generated": eos_generated,
        "completion": tokenizer.decode(generated_ids, skip_special=False),
        "full_text": tokenizer.decode(ids, skip_special=False),
    }


def fence(text: str) -> str:
    marker = "```"
    while marker in text:
        marker += "`"
    return f"{marker}\n{text}\n{marker}"


def write_markdown(payload: dict[str, Any], path: Path) -> None:
    lines = [
        "# Qualitative Checkpoint Inspection",
        "",
        "This artifact contains raw generations for inspection. It is not a benchmark score.",
        "",
        "## Metadata",
        "",
    ]
    metadata = payload["metadata"]
    for key in [
        "checkpoint",
        "checkpoint_step",
        "config",
        "tokenizer",
        "device",
        "temperature",
        "top_k",
        "max_new_tokens",
        "git_sha",
    ]:
        lines.append(f"- `{key}`: `{metadata.get(key)}`")
    lines.extend(["", "## Samples", ""])
    for result in payload["results"]:
        lines.extend(
            [
                f"### {result['prompt_name']} ({result['category']}, seed {result['seed']})",
                "",
                f"- generated tokens: `{result['generated_tokens']}`",
                f"- tokens/sec: `{result['tokens_per_sec']:.2f}`",
                f"- eos generated: `{result['eos_generated']}`",
                f"- lexical flags: `{json.dumps(result['flags'], sort_keys=True)}`",
                "",
                "**Prompt**",
                "",
                fence(result["prompt"]),
                "",
                "**Completion**",
                "",
                fence(result["completion"]),
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write qualitative chat/coding generations from a checkpoint.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompts-json", default=None)
    parser.add_argument("--output-json", default="runs/qualitative_samples.json")
    parser.add_argument("--output-md", default="runs/qualitative_samples.md")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--seeds", default="17")
    args = parser.parse_args()

    prompts = load_prompts(args.prompts_json)
    seeds = parse_seed_list(args.seeds)
    device = resolve_device(args.device)
    tokenizer = AgentCoderTokenizer.load(args.tokenizer)
    config = load_config(args.config)
    config.vocab_size = tokenizer.vocab_size

    seed_all(seeds[0])
    model = build_model(config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    results = []
    for seed in seeds:
        for item in prompts:
            seed_all(seed)
            sample = generate_one(
                model,
                tokenizer,
                item["prompt"],
                device=device,
                max_seq_len=config.max_seq_len,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            completion = sample["completion"]
            results.append(
                {
                    "prompt_name": item["name"],
                    "category": item["category"],
                    "seed": seed,
                    "prompt": item["prompt"],
                    **sample,
                    "flags": lexical_flags(completion),
                }
            )

    metadata = {
        "script": "scripts/qualitative_checkpoint_inspect.py",
        "git_sha": git_sha(),
        "config": str(args.config),
        "config_hash": config_hash(config),
        "config_summary": to_dict(config),
        "tokenizer": str(args.tokenizer),
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "checkpoint": str(args.checkpoint),
        "checkpoint_size_bytes": Path(args.checkpoint).stat().st_size,
        "checkpoint_keys": sorted(str(key) for key in checkpoint.keys()),
        "checkpoint_step": checkpoint.get("step"),
        "checkpoint_config_hash": checkpoint.get("config_hash"),
        "checkpoint_export_dtype": checkpoint.get("export_dtype"),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seeds": seeds,
        "prompt_count": len(prompts),
        "sample_count": len(results),
    }
    payload = {
        "metadata": metadata,
        "results": results,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(payload, Path(args.output_md))
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_md": args.output_md,
                "checkpoint_step": metadata["checkpoint_step"],
                "device": metadata["device"],
                "sample_count": metadata["sample_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
