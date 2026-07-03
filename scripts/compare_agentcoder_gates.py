#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CURATED_CASE_BEHAVIORS = {
    "curated_add_patch": "patch_addition",
    "curated_json_python_files": "json_tool_command",
    "curated_risky_question": "risky_clarifying_question",
    "curated_debugging": "plain_debugging",
    "curated_is_even_completion": "function_completion",
    "curated_stack_valueerror": "stack_trace_diagnosis",
    "curated_repo_lookup": "repo_context_lookup",
    "curated_test_command": "test_command",
    "curated_parse_port_review": "code_review",
    "curated_flag_patch": "patch_boolean_flag",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


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


def behavior_confusion(results: list[dict[str, Any]], *, prefer_stored_predictions: bool = False) -> dict[str, Any]:
    matrix: dict[str, dict[str, int]] = {}
    labeled = 0
    correct = 0
    for row in results:
        expected = row.get("expected_behavior") or CURATED_CASE_BEHAVIORS.get(str(row.get("name")))
        if expected is None:
            continue
        if prefer_stored_predictions:
            predicted = row.get("predicted_behavior") or infer_behavior(str(row.get("completion", "")))
        else:
            predicted = infer_behavior(str(row.get("completion", "")))
        labeled += 1
        correct += int(expected == predicted)
        matrix.setdefault(expected, {})
        matrix[expected][predicted] = matrix[expected].get(predicted, 0) + 1
    return {
        "matrix": {expected: dict(sorted(predicted.items())) for expected, predicted in sorted(matrix.items())},
        "labeled_cases": labeled,
        "correct_count": correct,
        "accuracy": correct / labeled if labeled else None,
    }


def preview(text: str, limit: int = 180) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


def summarize_run(run_dir: Path, *, prefer_stored_behavior: bool = False) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    eval_path = run_dir / "curated_eval.json"
    eval_payload = read_json(eval_path)
    results = list(eval_payload.get("results", []))
    if prefer_stored_behavior:
        confusion = {
            "matrix": summary.get("behavior_confusion") or eval_payload.get("behavior_confusion"),
            "accuracy": summary.get("behavior_accuracy") or eval_payload.get("behavior_accuracy"),
            "correct_count": summary.get("behavior_correct_count") or eval_payload.get("behavior_correct_count"),
            "labeled_cases": summary.get("behavior_labeled_cases") or eval_payload.get("behavior_labeled_cases"),
        }
        if confusion["matrix"] is None and results:
            confusion = behavior_confusion(results, prefer_stored_predictions=True)
    else:
        confusion = behavior_confusion(results) if results else {
            "matrix": summary.get("behavior_confusion") or eval_payload.get("behavior_confusion"),
            "accuracy": summary.get("behavior_accuracy") or eval_payload.get("behavior_accuracy"),
            "correct_count": summary.get("behavior_correct_count") or eval_payload.get("behavior_correct_count"),
            "labeled_cases": summary.get("behavior_labeled_cases") or eval_payload.get("behavior_labeled_cases"),
        }

    failed_cases = []
    for row in results:
        if row.get("passed"):
            continue
        predicted_behavior = (
            row.get("predicted_behavior")
            if prefer_stored_behavior and row.get("predicted_behavior")
            else infer_behavior(str(row.get("completion", "")))
        )
        expected_behavior = row.get("expected_behavior") or CURATED_CASE_BEHAVIORS.get(str(row.get("name")))
        missing = row.get("missing_required_substrings", [])
        present_forbidden = row.get("present_forbidden_substrings", [])
        slot_error = bool(row.get("slot_error"))
        if not slot_error and expected_behavior == predicted_behavior and (missing or present_forbidden):
            slot_error = True
        failed_cases.append(
            {
                "name": row.get("name"),
                "expected_behavior": expected_behavior,
                "predicted_behavior": predicted_behavior,
                "missing_required_substrings": missing,
                "forbidden_substrings": row.get("forbidden_substrings", []),
                "present_forbidden_substrings": present_forbidden,
                "slot_error": slot_error,
                "json_ok": row.get("json_ok"),
                "completion_preview": preview(str(row.get("completion", ""))),
            }
        )

    last_train_row = summary.get("last_train_row", {})
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "config": summary.get("config"),
        "pass_count": summary.get("pass_count", eval_payload.get("pass_count")),
        "case_count": summary.get("case_count", eval_payload.get("case_count")),
        "pass_rate": summary.get("pass_rate", eval_payload.get("pass_rate")),
        "behavior_accuracy": confusion.get("accuracy"),
        "behavior_correct_count": confusion.get("correct_count"),
        "behavior_labeled_cases": confusion.get("labeled_cases"),
        "behavior_confusion": confusion.get("matrix"),
        "failed_cases": failed_cases,
        "train_records": summary.get("train_records"),
        "eval_cases": summary.get("eval_cases"),
        "train_tokens": summary.get("train_tokens"),
        "val_tokens": summary.get("val_tokens"),
        "final_train_loss": last_train_row.get("train_loss"),
        "final_val_loss": last_train_row.get("val_next_token_loss"),
        "final_tokens_per_sec": last_train_row.get("tokens_per_sec"),
        "param_count_non_embedding": summary.get("param_count_non_embedding"),
        "estimated_flops_per_token": summary.get("estimated_flops_per_token"),
        "behavior_counts": summary.get("behavior_counts", {}),
    }


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], output: Path) -> None:
    headers = [
        "run_name",
        "pass_count",
        "case_count",
        "pass_rate",
        "behavior_accuracy",
        "train_records",
        "train_tokens",
        "val_tokens",
        "final_val_loss",
        "final_tokens_per_sec",
    ]
    lines = [
        "# AgentCoder Gate Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_value(row.get(key)) for key in headers) + " |")

    for row in rows:
        lines.extend(["", f"## {row['run_name']}", ""])
        failed = row.get("failed_cases", [])
        if not failed:
            lines.append("No exact gate failures.")
        else:
            lines.append("| case | expected behavior | predicted behavior | missing | forbidden | slot error | JSON ok | completion preview |")
            lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
            for case in failed:
                missing = ", ".join(str(value) for value in case.get("missing_required_substrings", []))
                forbidden = ", ".join(str(value) for value in case.get("present_forbidden_substrings", []))
                lines.append(
                    "| "
                    + " | ".join(
                        [
                            format_value(case.get("name")),
                            format_value(case.get("expected_behavior")),
                            format_value(case.get("predicted_behavior")),
                            missing,
                            forbidden,
                            format_value(case.get("slot_error")),
                            format_value(case.get("json_ok")),
                            format_value(case.get("completion_preview")).replace("|", "\\|"),
                        ]
                    )
                    + " |"
                )
        confusion = row.get("behavior_confusion")
        if confusion:
            lines.extend(["", "Behavior confusion:"])
            for expected, predicted in sorted(confusion.items()):
                parts = ", ".join(f"{name}: {count}" for name, count in sorted(predicted.items()))
                lines.append(f"- {expected} -> {parts}")
    output.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare curated RAAM-AgentCoder gate artifacts.")
    parser.add_argument("run_dirs", nargs="+", help="Run directories containing summary.json and curated_eval.json.")
    parser.add_argument("--output-json", default="runs/agentcoder_gate_comparison.json")
    parser.add_argument("--output-md", default="runs/agentcoder_gate_comparison.md")
    parser.add_argument(
        "--prefer-stored-behavior",
        action="store_true",
        help="Use behavior labels saved in artifacts instead of recomputing with the current heuristic.",
    )
    args = parser.parse_args()

    rows = [summarize_run(Path(path), prefer_stored_behavior=args.prefer_stored_behavior) for path in args.run_dirs]
    payload = {"runs": rows}

    json_path = Path(args.output_json)
    md_path = Path(args.output_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_markdown(rows, md_path)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
