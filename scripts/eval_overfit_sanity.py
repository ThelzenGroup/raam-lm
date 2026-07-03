#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from raam_lm.config import load_config
from raam_lm.registry import build_model
from raam_lm.tokenization import AgentCoderTokenizer
from raam_lm.train_utils import resolve_device, seed_all


CASES = [
    {
        "name": "fix_add_patch",
        "prompt": (
            "<|system|>\n"
            "You are RAAM-AgentCoder, a concise software-engineering assistant. Answer directly and preserve exact code when asked.\n"
            "\n"
            "<|repo_context|>\n"
            "file: calc.py\n"
            "```python\n"
            "def add(a, b):\n"
            "    return a - b\n"
            "```\n"
            "file: tests/test_calc.py\n"
            "```python\n"
            "from calc import add\n"
            "\n"
            "def test_add():\n"
            "    assert add(2, 3) == 5\n"
            "```\n"
            "\n"
            "<|user|>\n"
            "Fix the bug and provide a minimal patch plus the test command.\n"
            "\n"
            "<|assistant|>\n"
        ),
        "required_substrings": ["return a + b", "pytest tests/test_calc.py -q"],
        "expected_behavior": "patch_addition",
    },
    {
        "name": "json_tool_call",
        "prompt": (
            "<|system|>\n"
            "You are RAAM-AgentCoder, a concise software-engineering assistant. Return valid JSON when the user asks for JSON.\n"
            "\n"
            "<|user|>\n"
            "Return one JSON object with a shell command that lists Python files.\n"
            "\n"
            "<|assistant|>\n"
        ),
        "required_substrings": ["find . -name '*.py' -type f"],
        "expected_json": {"cmd": "find . -name '*.py' -type f"},
        "expected_behavior": "json_tool_command",
    },
    {
        "name": "clarifying_question",
        "prompt": (
            "<|system|>\n"
            "You are RAAM-AgentCoder, a concise software-engineering assistant. Ask before risky edits.\n"
            "\n"
            "<|user|>\n"
            "Before changing a risky production file, ask one concise clarifying question.\n"
            "\n"
            "<|assistant|>\n"
        ),
        "required_substrings": ["Which production file", "rollback", "test command"],
        "expected_behavior": "risky_clarifying_question",
    },
    {
        "name": "plain_debugging",
        "prompt": (
            "<|system|>\n"
            "You are RAAM-AgentCoder, a concise software-engineering assistant. Explain debugging steps plainly.\n"
            "\n"
            "<|user|>\n"
            "A Python unit test is failing. Explain in plain English how you would debug it before editing code.\n"
            "\n"
            "<|assistant|>\n"
        ),
        "required_substrings": ["failing assertion", "reproduce", "smallest code change"],
        "expected_behavior": "plain_debugging",
    },
    {
        "name": "function_completion",
        "prompt": (
            "<|system|>\n"
            "You are RAAM-AgentCoder, a concise software-engineering assistant. Complete code safely.\n"
            "\n"
            "<|user|>\n"
            "Complete this Python function:\n"
            "```python\n"
            "def is_even(n):\n"
            "```\n"
            "\n"
            "<|assistant|>\n"
        ),
        "required_substrings": ["def is_even(n):", "return n % 2 == 0"],
        "expected_behavior": "function_completion",
    },
    {
        "name": "stack_trace",
        "prompt": (
            "<|system|>\n"
            "You are RAAM-AgentCoder, a concise software-engineering assistant. Diagnose stack traces from the boundary inward.\n"
            "\n"
            "<|user|>\n"
            "Diagnose this stack trace:\n"
            "ValueError: invalid literal for int() with base 10: 'abc'\n"
            "\n"
            "<|assistant|>\n"
        ),
        "required_substrings": ["string 'abc'", "int()", "validate before conversion"],
        "expected_behavior": "stack_trace_diagnosis",
    },
    {
        "name": "repo_context",
        "prompt": (
            "<|system|>\n"
            "You are RAAM-AgentCoder, a concise software-engineering assistant. Use repo context when it is provided.\n"
            "\n"
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
            "\n"
            "<|user|>\n"
            "Where is add implemented?\n"
            "\n"
            "<|assistant|>\n"
        ),
        "required_substrings": ["add is implemented in calc.py"],
        "expected_behavior": "repo_context_lookup",
    },
    {
        "name": "test_command",
        "prompt": (
            "<|system|>\n"
            "You are RAAM-AgentCoder, a concise software-engineering assistant. Recommend verification commands before commits.\n"
            "\n"
            "<|user|>\n"
            "You changed a Python package. Name the safest test command to run before committing.\n"
            "\n"
            "<|assistant|>\n"
        ),
        "required_substrings": ["python -m pytest -q"],
        "expected_behavior": "test_command",
    },
]


def load_cases(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return CASES
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("cases", [])
    cases = []
    for index, case in enumerate(payload):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        if "prompt" not in case:
            raise ValueError(f"case {index} is missing prompt")
        required = case.get("required_substrings", [])
        if not isinstance(required, list):
            raise ValueError(f"case {index} required_substrings must be a list")
        forbidden = case.get("forbidden_substrings", [])
        if not isinstance(forbidden, list):
            raise ValueError(f"case {index} forbidden_substrings must be a list")
        item = {
            "name": str(case.get("name", f"case_{index:02d}")),
            "prompt": str(case["prompt"]),
            "required_substrings": [str(value) for value in required],
            "forbidden_substrings": [str(value) for value in forbidden],
        }
        if "slot_family" in case:
            item["slot_family"] = str(case["slot_family"])
        if "expected_slots" in case:
            item["expected_slots"] = case["expected_slots"]
        if "expected_json" in case:
            item["expected_json"] = case["expected_json"]
        if "expected_behavior" in case:
            item["expected_behavior"] = str(case["expected_behavior"])
        cases.append(item)
    if not cases:
        raise ValueError("case file did not contain any cases")
    return cases


def first_json_object(text: str) -> Any | None:
    for start in [idx for idx, ch in enumerate(text) if ch == "{"]:
        for end in range(len(text), start, -1):
            try:
                return json.loads(text[start:end])
            except Exception:
                continue
    return None


def infer_behavior(completion: str) -> str:
    lower = completion.lower()
    parsed_json = first_json_object(completion)
    if isinstance(parsed_json, dict):
        return "json_tool_command"
    if "```diff" in lower or "--- a/" in lower or "+++ b/" in lower:
        if (
            "is_enabled" in lower
            or "feature" in lower
            or "flag" in lower
            or "== 'true'" in lower
            or '== "true"' in lower
            or "enabled value" in lower
        ):
            return "patch_boolean_flag"
        return "patch_addition"
    if "which file" in lower and ("rollback" in lower or "test command" in lower or "test" in lower):
        return "risky_clarifying_question"
    if "python -m pytest -q" in lower or ("pytest" in lower and "run" in lower):
        return "test_command"
    if "```python" in lower and "def " in lower:
        return "function_completion"
    if "def is_even" in lower or "def is_odd" in lower or "return n % 2" in lower:
        return "function_completion"
    if "implemented in" in lower:
        return "repo_context_lookup"
    if "reproduce" in lower and ("assertion" in lower or "smallest" in lower):
        return "plain_debugging"
    if "65535" in lower or ("numeric" in lower and "port" in lower):
        return "code_review"
    if (
        "valueerror" in lower
        or "int()" in lower
        or "validate before conversion" in lower
        or "keyerror" in lower
        or "filenotfounderror" in lower
    ):
        return "stack_trace_diagnosis"
    if "commit" in lower and ("summary" in lower or "message" in lower):
        return "commit_summary"
    return "unknown"


def build_behavior_confusion(results: list[dict[str, Any]]) -> dict[str, Any]:
    matrix: dict[str, dict[str, int]] = {}
    labeled = 0
    correct = 0
    for row in results:
        expected = row.get("expected_behavior")
        predicted = row.get("predicted_behavior")
        if expected is None or predicted is None:
            continue
        labeled += 1
        if expected == predicted:
            correct += 1
        matrix.setdefault(str(expected), {})
        matrix[str(expected)][str(predicted)] = matrix[str(expected)].get(str(predicted), 0) + 1
    return {
        "matrix": {expected: dict(sorted(predicted.items())) for expected, predicted in sorted(matrix.items())},
        "labeled_cases": labeled,
        "correct_count": correct,
        "accuracy": correct / labeled if labeled else None,
    }


def sample_next_token(logits: torch.Tensor, temperature: float, top_k: int) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, num_samples=1)].item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate(
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
    generated: list[int] = []
    start = time.perf_counter()
    eos_generated = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-max_seq_len:]
            input_ids = torch.tensor([context], device=device, dtype=torch.long)
            logits = model(input_ids)["logits"][0, -1]
            next_id = sample_next_token(logits, temperature=temperature, top_k=top_k)
            ids.append(next_id)
            generated.append(next_id)
            if next_id == tokenizer.eos_token_id:
                eos_generated = True
                break
    elapsed = time.perf_counter() - start
    return {
        "completion": tokenizer.decode(generated, skip_special=False),
        "generated_tokens": len(generated),
        "elapsed_sec": elapsed,
        "tokens_per_sec": len(generated) / max(elapsed, 1e-9),
        "eos_generated": eos_generated,
    }


def score_case(case: dict[str, Any], completion: str) -> dict[str, Any]:
    missing = [needle for needle in case["required_substrings"] if needle not in completion]
    present_forbidden = [needle for needle in case.get("forbidden_substrings", []) if needle in completion]
    expected_json = case.get("expected_json")
    parsed_json = first_json_object(completion) if expected_json is not None else None
    json_ok = parsed_json == expected_json if expected_json is not None else None
    expected_behavior = case.get("expected_behavior")
    predicted_behavior = infer_behavior(completion)
    behavior_correct = predicted_behavior == expected_behavior if expected_behavior is not None else None
    passed = not missing and not present_forbidden and (json_ok is not False)
    slot_error = bool(behavior_correct and (missing or present_forbidden))
    return {
        "missing_required_substrings": missing,
        "forbidden_substrings": case.get("forbidden_substrings", []),
        "present_forbidden_substrings": present_forbidden,
        "expected_json": expected_json,
        "parsed_json": parsed_json,
        "json_ok": json_ok,
        "expected_behavior": expected_behavior,
        "predicted_behavior": predicted_behavior,
        "behavior_correct": behavior_correct,
        "slot_error": slot_error,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a checkpoint overfit the curated AgentCoder sanity set.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cases-json", default=None)
    parser.add_argument("--output", default="runs/agentcoder_overfit_eval.json")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-pass-rate", type=float, default=1.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    seed_all(args.seed)
    device = resolve_device(args.device)
    tokenizer = AgentCoderTokenizer.load(args.tokenizer)
    config = load_config(args.config)
    config.vocab_size = tokenizer.vocab_size
    model = build_model(config).to(device).eval()
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    cases = load_cases(args.cases_json)

    results = []
    for case in cases:
        generated = generate(
            model,
            tokenizer,
            case["prompt"],
            device=device,
            max_seq_len=config.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        score = score_case(case, generated["completion"])
        case_metadata = {
            key: case[key]
            for key in ["slot_family", "expected_slots"]
            if key in case
        }
        results.append(
            {
                "name": case["name"],
                "prompt": case["prompt"],
                **case_metadata,
                **generated,
                **score,
            }
        )

    passed = sum(1 for row in results if row["passed"])
    behavior_confusion = build_behavior_confusion(results)
    payload = {
        "metadata": {
            "config": args.config,
            "tokenizer": args.tokenizer,
            "checkpoint": args.checkpoint,
            "checkpoint_step": checkpoint.get("step"),
            "device": str(device),
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "min_pass_rate": args.min_pass_rate,
            "cases_json": args.cases_json,
        },
        "pass_count": passed,
        "case_count": len(results),
        "pass_rate": passed / len(results),
        "behavior_confusion": behavior_confusion["matrix"],
        "behavior_labeled_cases": behavior_confusion["labeled_cases"],
        "behavior_correct_count": behavior_confusion["correct_count"],
        "behavior_accuracy": behavior_confusion["accuracy"],
        "results": results,
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {k: payload[k] for k in ["pass_count", "case_count", "pass_rate", "behavior_accuracy"]},
            indent=2,
        )
    )
    if not args.no_fail and payload["pass_rate"] < args.min_pass_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
