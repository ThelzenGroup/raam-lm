#!/usr/bin/env python
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BAD_PHRASES = [
    "abyssal",
    "ai language model",
    "cahoon",
    "campfire",
    "carrots",
    "colonial",
    "communicationservice",
    "compromising",
    "nanana",
    "now, we can",
    "postwrest",
    "stamina",
    "subservient",
]


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("cases", [])
    if not isinstance(payload, list) or not payload:
        raise ValueError("case file must contain a non-empty cases list")
    cases: list[dict[str, Any]] = []
    for index, case in enumerate(payload):
        if not isinstance(case, dict):
            raise ValueError(f"case {index} must be an object")
        if "prompt" not in case:
            raise ValueError(f"case {index} is missing prompt")
        item = dict(case)
        item.setdefault("name", f"case_{index:02d}")
        item.setdefault("required_substrings", [])
        item.setdefault("forbidden_substrings", [])
        cases.append(item)
    return cases


def filter_cases(
    cases: list[dict[str, Any]],
    *,
    expected_behaviors: list[str] | None = None,
    topic_contains: list[str] | None = None,
) -> list[dict[str, Any]]:
    behaviors = {value for value in (expected_behaviors or []) if value}
    topic_needles = [value for value in (topic_contains or []) if value]
    filtered = cases
    if behaviors:
        filtered = [case for case in filtered if str(case.get("expected_behavior", "")) in behaviors]
    if topic_needles:
        filtered = [
            case
            for case in filtered
            if any(needle in str(case.get("topic", "")) for needle in topic_needles)
        ]
    if not filtered:
        raise ValueError("case filters removed all eval cases")
    return filtered


def strip_special(text: str) -> str:
    cleaned = text.replace("<eos>", "").replace("<bos>", "")
    return cleaned.strip()


def first_json_object(text: str) -> Any | None:
    stripped = strip_special(text)
    try:
        return json.loads(stripped)
    except Exception:
        pass
    for start in [idx for idx, ch in enumerate(stripped) if ch == "{"]:
        for end in range(len(stripped), start, -1):
            try:
                return json.loads(stripped[start:end])
            except Exception:
                continue
    return None


def extract_python_code(text: str) -> tuple[str, str]:
    cleaned = strip_special(text)
    marker = "```"
    python_marker = "```python"
    if python_marker in cleaned:
        after_marker = cleaned.split(python_marker, 1)[1]
        code, _, tail = after_marker.partition(marker)
        return code.strip(), tail.strip()
    if marker in cleaned:
        after_marker = cleaned.split(marker, 1)[1]
        code, _, tail = after_marker.partition(marker)
        return code.strip(), tail.strip()
    if "def " in cleaned:
        return cleaned[cleaned.index("def ") :].strip(), ""
    return cleaned, ""


def syntax_errors_for_code(code: str) -> list[str]:
    try:
        ast.parse(code)
    except SyntaxError as exc:
        return [f"{exc.__class__.__name__}: {exc.msg}"]
    return []


def safe_function_env() -> dict[str, Any]:
    safe_builtins = {
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "Exception": Exception,
        "float": float,
        "abs": abs,
        "all": all,
        "any": any,
        "int": int,
        "isinstance": isinstance,
        "len": len,
        "list": list,
        "max": max,
        "min": min,
        "range": range,
        "reversed": reversed,
        "set": set,
        "sorted": sorted,
        "str": str,
        "sum": sum,
        "tuple": tuple,
        "TypeError": TypeError,
        "ValueError": ValueError,
    }
    return {"__builtins__": safe_builtins}


def run_function_tests(code: str, spec: dict[str, Any]) -> tuple[bool, list[str]]:
    errors = syntax_errors_for_code(code)
    if errors:
        return False, errors
    env = safe_function_env()
    try:
        exec(compile(code, "<generated>", "exec"), env, env)
    except Exception as exc:
        return False, [f"exec failed: {exc.__class__.__name__}: {exc}"]
    name = str(spec["name"])
    func = env.get(name)
    if not callable(func):
        return False, [f"function {name!r} was not defined"]
    failures: list[str] = []
    for index, test in enumerate(spec.get("tests", [])):
        args = test.get("args", [])
        kwargs = test.get("kwargs", {})
        expected = test.get("expected")
        expected_raise = test.get("raises")
        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            if expected_raise and exc.__class__.__name__ == expected_raise:
                continue
            failures.append(f"test {index} raised {exc.__class__.__name__}: {exc}")
            continue
        if expected_raise:
            failures.append(f"test {index} expected {expected_raise}, got return value {result!r}")
        elif result != expected:
            failures.append(f"test {index} expected {expected!r}, got {result!r}")
    return not failures, failures


def run_python_assert_tests(code: str, tests: list[str]) -> tuple[bool, list[str]]:
    errors = syntax_errors_for_code(code)
    if errors:
        return False, errors
    env = safe_function_env()
    try:
        exec(compile(code, "<generated>", "exec"), env, env)
    except Exception as exc:
        return False, [f"exec failed: {exc.__class__.__name__}: {exc}"]
    failures: list[str] = []
    for index, test in enumerate(tests):
        snippet = str(test).strip()
        if not snippet:
            continue
        try:
            exec(compile(snippet, f"<assert_test_{index}>", "exec"), env, env)
        except Exception as exc:
            failures.append(f"assert test {index} failed: {exc.__class__.__name__}: {exc}")
    return not failures, failures


def score_patch(completion: str, expected: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    required = [
        f"--- a/{expected['path']}",
        f"+++ b/{expected['path']}",
    ]
    required.extend(f"-{line}" for line in str(expected["removed"]).splitlines())
    required.extend(f"+{line}" for line in str(expected["added"]).splitlines())
    for needle in required:
        if needle not in completion:
            failures.append(f"missing patch marker {needle!r}")
    if "```diff" not in completion:
        failures.append("missing diff fence")
    if "@@" not in completion:
        failures.append("missing hunk marker")
    return not failures, failures


def infer_behavior(completion: str) -> str:
    lower = completion.lower()
    if first_json_object(completion) is not None:
        return "json_tool_command"
    if "```diff" in lower or "--- a/" in lower or "+++ b/" in lower:
        if "safe_int" in lower or "parsers.py" in lower:
            return "patch_safe_int_default"
        if "parse_port" in lower or "config.py" in lower:
            return "patch_parse_port_range"
        if "limit + 1" in lower or "first_items" in lower:
            return "patch_off_by_one"
        if "== 'true'" in lower or "is_enabled" in lower:
            return "patch_boolean_flag"
        return "patch_addition"
    if "def test_" in lower and "assert " in lower:
        return "pytest_generation"
    if "def " in lower:
        return "function_completion"
    return "unknown"


def has_nonsense(completion: str) -> tuple[bool, list[str]]:
    lower = strip_special(completion).lower()
    hits = [phrase for phrase in BAD_PHRASES if phrase in lower]
    if "<|user|>" in lower or "<|assistant|>" in lower or "<|system|>" in lower:
        hits.append("chat marker repetition")
    return bool(hits), hits


def score_case(case: dict[str, Any], completion: str) -> dict[str, Any]:
    required = [str(value) for value in case.get("required_substrings", [])]
    forbidden = [str(value) for value in case.get("forbidden_substrings", [])]
    missing_required = [needle for needle in required if needle not in completion]
    present_forbidden = [needle for needle in forbidden if needle in completion]
    max_completion_chars = int(case.get("max_completion_chars", 0) or 0)
    too_long = max_completion_chars > 0 and len(strip_special(completion)) > max_completion_chars
    nonsense_present, nonsense_hits = has_nonsense(completion)

    code = None
    trailing_text = ""
    syntax_ok: bool | None = None
    function_ok: bool | None = None
    function_failures: list[str] = []
    assert_tests_ok: bool | None = None
    assert_test_failures: list[str] = []
    if case.get("python_function") or case.get("python_assert_tests") or case.get("python_syntax"):
        code, trailing_text = extract_python_code(completion)
        syntax_failures = syntax_errors_for_code(code)
        syntax_ok = not syntax_failures
        function_failures.extend(syntax_failures)
        if case.get("python_function"):
            function_ok, function_failures = run_function_tests(code, case["python_function"])
        if case.get("python_assert_tests"):
            assert_tests_ok, assert_test_failures = run_python_assert_tests(code, case["python_assert_tests"])

    json_ok: bool | None = None
    parsed_json = None
    if "expected_json" in case:
        parsed_json = first_json_object(completion)
        json_ok = parsed_json == case["expected_json"]
        if case.get("strict_json"):
            try:
                json_ok = json.loads(strip_special(completion)) == case["expected_json"]
            except Exception:
                json_ok = False

    patch_ok: bool | None = None
    patch_failures: list[str] = []
    if "expected_patch" in case:
        patch_ok, patch_failures = score_patch(completion, case["expected_patch"])

    no_trailing_ok: bool | None = None
    if case.get("no_trailing_text"):
        no_trailing_ok = not trailing_text

    no_nonsense_ok = True
    if case.get("no_nonsense", True):
        no_nonsense_ok = not nonsense_present

    expected_behavior = case.get("expected_behavior")
    predicted_behavior = infer_behavior(completion)
    behavior_correct = predicted_behavior == expected_behavior if expected_behavior else None

    passed = not missing_required and not present_forbidden and not too_long and no_nonsense_ok
    if syntax_ok is not None:
        passed = passed and syntax_ok
    if function_ok is not None:
        passed = passed and function_ok
    if assert_tests_ok is not None:
        passed = passed and assert_tests_ok
    if json_ok is not None:
        passed = passed and json_ok
    if patch_ok is not None:
        passed = passed and patch_ok
    if no_trailing_ok is not None:
        passed = passed and no_trailing_ok
    if behavior_correct is not None:
        passed = passed and behavior_correct

    return {
        "missing_required_substrings": missing_required,
        "present_forbidden_substrings": present_forbidden,
        "too_long": too_long,
        "max_completion_chars": max_completion_chars or None,
        "nonsense_present": nonsense_present,
        "nonsense_hits": nonsense_hits,
        "python_code": code,
        "python_syntax_ok": syntax_ok,
        "function_tests_ok": function_ok,
        "function_failures": function_failures,
        "assert_tests_ok": assert_tests_ok,
        "assert_test_failures": assert_test_failures,
        "trailing_text_after_code_block": trailing_text,
        "no_trailing_text_ok": no_trailing_ok,
        "expected_json": case.get("expected_json"),
        "parsed_json": parsed_json,
        "json_ok": json_ok,
        "patch_ok": patch_ok,
        "patch_failures": patch_failures,
        "expected_behavior": expected_behavior,
        "predicted_behavior": predicted_behavior,
        "behavior_correct": behavior_correct,
        "passed": passed,
    }


def sample_next_token(logits: Any, temperature: float, top_k: int) -> int:
    import torch

    if temperature <= 0:
        return int(torch.argmax(logits).item())
    logits = logits / max(temperature, 1e-6)
    if top_k > 0:
        values, indices = torch.topk(logits, k=min(top_k, logits.numel()))
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, num_samples=1)].item())
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_completion(
    model: Any,
    tokenizer: Any,
    prompt: str,
    *,
    device: Any,
    max_seq_len: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
) -> dict[str, Any]:
    import torch

    prompt_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    ids = list(prompt_ids)
    generated: list[int] = []
    suppressed_token_ids = tokenizer.generation_suppressed_token_ids()
    start = time.perf_counter()
    eos_generated = False
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = ids[-max_seq_len:]
            input_ids = torch.tensor([context], device=device, dtype=torch.long)
            logits = model(input_ids)["logits"][0, -1]
            if suppressed_token_ids:
                logits[suppressed_token_ids] = -1.0e9
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


def load_model_stack(config_path: str, tokenizer_path: str, checkpoint_path: str, device_name: str, seed: int) -> tuple[Any, Any, Any, Any]:
    sys.path.insert(0, str(ROOT / "src"))
    import torch

    from raam_lm.config import load_config, resolve_copy_head_token_ids
    from raam_lm.registry import build_model
    from raam_lm.tokenization import AgentCoderTokenizer
    from raam_lm.train_utils import resolve_device, seed_all

    seed_all(seed)
    tokenizer = AgentCoderTokenizer.load(tokenizer_path)
    config = load_config(config_path)
    config.vocab_size = tokenizer.vocab_size
    resolve_copy_head_token_ids(config, tokenizer)
    device = resolve_device(device_name)
    model = build_model(config).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return model, tokenizer, config, device


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(1 for row in results if row["passed"])
    topic_counts: dict[str, int] = {}
    topic_passed: dict[str, int] = {}
    for row in results:
        topic = str(row.get("topic", "unknown"))
        topic_counts[topic] = topic_counts.get(topic, 0) + 1
        topic_passed[topic] = topic_passed.get(topic, 0) + int(bool(row["passed"]))
    return {
        "pass_count": passed,
        "case_count": len(results),
        "pass_rate": passed / len(results) if results else 0.0,
        "topic_pass_counts": dict(sorted(topic_passed.items())),
        "topic_case_counts": dict(sorted(topic_counts.items())),
        "passed_cases": [row["name"] for row in results if row["passed"]],
        "failed_cases": [row["name"] for row in results if not row["passed"]],
        "nonsense_fail_count": sum(1 for row in results if row.get("nonsense_present")),
        "function_pass_count": sum(1 for row in results if row.get("function_tests_ok") is True),
        "json_pass_count": sum(1 for row in results if row.get("json_ok") is True),
        "patch_pass_count": sum(1 for row in results if row.get("patch_ok") is True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict held-out coding-ladder eval for RAAM-AgentCoder.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases-json", required=True)
    parser.add_argument("--output", default="runs/coding_ladder_eval.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--expected-behavior",
        action="append",
        default=[],
        help="Keep only cases with this expected_behavior. Repeat for multiple behaviors.",
    )
    parser.add_argument(
        "--topic-contains",
        action="append",
        default=[],
        help="Keep only cases whose topic contains this substring. Repeat for multiple substrings.",
    )
    parser.add_argument("--min-pass-rate", type=float, default=0.0)
    parser.add_argument("--no-fail", action="store_true")
    args = parser.parse_args()

    cases = filter_cases(
        load_cases(args.cases_json),
        expected_behaviors=args.expected_behavior,
        topic_contains=args.topic_contains,
    )
    model, tokenizer, config, device = load_model_stack(
        args.config,
        args.tokenizer,
        args.checkpoint,
        args.device,
        args.seed,
    )
    import torch

    checkpoint_meta = torch.load(args.checkpoint, map_location="cpu")
    results: list[dict[str, Any]] = []
    start_all = time.perf_counter()
    for case in cases:
        generated = generate_completion(
            model,
            tokenizer,
            str(case["prompt"]),
            device=device,
            max_seq_len=config.max_seq_len,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        score = score_case(case, generated["completion"])
        results.append(
            {
                "name": str(case["name"]),
                "topic": str(case.get("topic", "unknown")),
                "prompt": str(case["prompt"]),
                **generated,
                **score,
            }
        )
    summary = summarize_results(results)
    payload = {
        "metadata": {
            "config": args.config,
            "tokenizer": args.tokenizer,
            "checkpoint": args.checkpoint,
            "checkpoint_step": checkpoint_meta.get("step"),
            "device": str(device),
            "seed": args.seed,
            "temperature": args.temperature,
            "top_k": args.top_k,
            "max_new_tokens": args.max_new_tokens,
            "cases_json": args.cases_json,
            "expected_behavior_filters": args.expected_behavior,
            "topic_contains_filters": args.topic_contains,
            "min_pass_rate": args.min_pass_rate,
            "latency_sec_total": time.perf_counter() - start_all,
        },
        **summary,
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: payload[key] for key in ["pass_count", "case_count", "pass_rate", "failed_cases"]}, indent=2))
    if not args.no_fail and payload["pass_rate"] < args.min_pass_rate:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
