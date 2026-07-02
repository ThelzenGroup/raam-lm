#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Iterator


DEFAULT_AGENT_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "go",
    "rust",
    "java",
    "c",
    "cpp",
    "c++",
    "c#",
    "csharp",
    "php",
}


def coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value


def normalize_role(role: Any) -> str:
    normalized = str(role or "user").strip().lower()
    if normalized in {"prompter", "human"}:
        return "user"
    if normalized in {"system", "user", "assistant"}:
        return normalized
    if normalized in {"tool", "observation", "environment", "function", "tool_result"}:
        return "tool_result"
    return "assistant"


def normalize_messages(raw_messages: Any) -> list[dict[str, str]]:
    raw_messages = maybe_json(raw_messages)
    if not isinstance(raw_messages, list):
        return []
    messages: list[dict[str, str]] = []
    for item in raw_messages:
        if not isinstance(item, dict):
            continue
        role = normalize_role(item.get("role") or item.get("type"))
        content = coerce_text(item.get("content") or item.get("text") or item.get("observation"))
        if content.strip():
            messages.append({"role": role, "content": content})
    return messages


def openhands_row_to_record(
    row: dict[str, Any],
    *,
    source: str,
    resolved_only: bool,
    languages: set[str],
) -> dict[str, Any] | None:
    language = str(row.get("language") or "").strip().lower()
    if languages and language and language not in languages:
        return None
    if resolved_only and row.get("resolved") not in {1, True, "1", "true", "True"}:
        return None

    messages = normalize_messages(row.get("trajectory"))
    if not messages:
        return None

    system = ""
    if messages and messages[0]["role"] == "system":
        system = messages.pop(0)["content"]

    repo_bits = [
        f"source: {source}",
        f"instance_id: {row.get('instance_id', '')}",
        f"repo: {row.get('repo', '')}",
        f"license: {row.get('license', '')}",
        f"language: {row.get('language', '')}",
    ]
    if row.get("dataset"):
        repo_bits.append(f"dataset: {row.get('dataset')}")

    trace: list[dict[str, str]] = []
    model_patch = coerce_text(row.get("model_patch")).strip()
    if model_patch:
        trace.append({"type": "patch", "content": model_patch})

    metadata = maybe_json(row.get("metadata"))
    if isinstance(metadata, dict):
        reference_patch = metadata.get("reference_patch")
        if isinstance(reference_patch, dict) and reference_patch.get("patch"):
            trace.append({"type": "patch", "content": coerce_text(reference_patch["patch"])})

    return {
        "system": system,
        "repo_context": "\n".join(bit for bit in repo_bits if not bit.endswith(": ")),
        "messages": messages,
        "trace": trace,
        "final": "",
    }


def wildchat_row_to_record(row: dict[str, Any], *, english_only: bool) -> dict[str, Any] | None:
    if row.get("toxic") is True:
        return None
    language = str(row.get("language") or "").strip().lower()
    if english_only and language and language not in {"en", "english"}:
        return None
    messages = normalize_messages(row.get("conversation"))
    messages = [msg for msg in messages if msg["content"].strip()]
    if not messages:
        return None
    if messages[0]["role"] == "user" and not messages[0]["content"].strip():
        return None
    return {
        "system": "General chat transcript. Answer helpfully and concisely.",
        "messages": messages,
        "trace": [],
        "final": "",
    }


def oasst_rows_to_records(rows: Iterable[dict[str, Any]], *, english_only: bool) -> Iterator[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("deleted") is True:
            continue
        if row.get("review_result") is False:
            continue
        language = str(row.get("lang") or "").strip().lower()
        if english_only and language and language != "en":
            continue
        message_id = row.get("message_id")
        if message_id:
            seen[str(message_id)] = row
        if normalize_role(row.get("role")) != "assistant":
            continue
        parent_id = row.get("parent_id")
        parent = seen.get(str(parent_id)) if parent_id else None
        if not parent or normalize_role(parent.get("role")) != "user":
            continue
        user_text = coerce_text(parent.get("text")).strip()
        assistant_text = coerce_text(row.get("text")).strip()
        if not user_text or not assistant_text:
            continue
        yield {
            "system": "General assistant conversation.",
            "messages": [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_text},
            ],
            "trace": [],
            "final": "",
        }


def text_row_to_record(row: dict[str, Any], *, source: str) -> dict[str, str] | None:
    content = coerce_text(row.get("content") or row.get("text")).strip()
    if not content:
        return None
    return {"text": f"# source: {source}\n\n{content}\n"}


def iter_limited(records: Iterable[dict[str, Any] | None], limit: int) -> Iterator[dict[str, Any]]:
    emitted = 0
    for record in records:
        if record is None:
            continue
        yield record
        emitted += 1
        if emitted >= limit:
            break


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
            count += 1
    return count


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def parse_languages(raw: str) -> set[str]:
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def parse_subset_limits(values: list[str]) -> list[tuple[str, int]]:
    parsed: list[tuple[str, int]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected SUBSET=LIMIT, got {value!r}")
        subset, limit = value.split("=", 1)
        parsed.append((subset.strip(), int(limit.strip())))
    return parsed


def require_datasets():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "This script needs the optional Hugging Face datasets package. "
            "Install it with: python -m pip install datasets"
        ) from exc
    return load_dataset


def load_hf_dataset(name: str, *args: str, split: str, streaming: bool):
    load_dataset = require_datasets()
    return load_dataset(name, *args, split=split, streaming=streaming)


def dataset_config_arg(config_name: str) -> tuple[str, ...]:
    return () if config_name in {"", "default"} else (config_name,)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert recommended research datasets into RAAM-AgentCoder canonical JSONL."
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--streaming", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-source-error", action="store_true")
    parser.add_argument("--english-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--languages", default=",".join(sorted(DEFAULT_AGENT_LANGUAGES)))
    parser.add_argument("--resolved-open-swe-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-open-swe", type=int, default=0)
    parser.add_argument("--max-swe-zero", type=int, default=0)
    parser.add_argument("--max-wildchat", type=int, default=0)
    parser.add_argument("--max-oasst", type=int, default=0)
    parser.add_argument("--open-swe-config", default="openhands", choices=["openhands", "sweagent"])
    parser.add_argument("--open-swe-split", default="minimax_m25")
    parser.add_argument("--swe-zero-config", default="default")
    parser.add_argument("--swe-zero-split", default="train")
    parser.add_argument("--wildchat-config", default="default")
    parser.add_argument("--wildchat-split", default="train")
    parser.add_argument("--oasst-config", default="default")
    parser.add_argument("--oasst-split", default="train")
    parser.add_argument(
        "--starcoder2-extras",
        nargs="*",
        default=[],
        metavar="SUBSET=LIMIT",
        help="Example: documentation=50000 issues=50000 stackoverflow=50000",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    languages = parse_languages(args.languages)
    manifest: dict[str, Any] = {
        "format": "raam-agentcoder-research-jsonl-v1",
        "output_dir": str(out),
        "streaming": args.streaming,
        "reuse_existing": args.reuse_existing,
        "english_only": args.english_only,
        "languages": sorted(languages),
        "sources": {},
    }

    def reuse_source(key: str, path: Path, metadata: dict[str, Any]) -> bool:
        count = count_jsonl(path)
        if not args.reuse_existing or count <= 0:
            return False
        manifest["sources"][key] = {**metadata, "records": count, "reused": True}
        return True

    def record_source_error(key: str, error: Exception, metadata: dict[str, Any]) -> None:
        manifest["sources"][key] = {**metadata, "records": 0, "error": f"{type(error).__name__}: {error}"}
        if not args.continue_on_source_error:
            raise error

    if args.max_open_swe > 0:
        key = "nvidia/Open-SWE-Traces"
        path = out / "agent_traces" / "open_swe_traces.jsonl"
        metadata = {"config": args.open_swe_config, "split": args.open_swe_split}
        if not reuse_source(key, path, metadata):
            try:
                ds = load_hf_dataset(
                    key,
                    *dataset_config_arg(args.open_swe_config),
                    split=args.open_swe_split,
                    streaming=args.streaming,
                )
                records = (
                    openhands_row_to_record(
                        row,
                        source=key,
                        resolved_only=args.resolved_open_swe_only,
                        languages=languages,
                    )
                    for row in ds
                )
                count = write_jsonl(path, iter_limited(records, args.max_open_swe))
                manifest["sources"][key] = {**metadata, "records": count}
            except Exception as exc:
                record_source_error(key, exc, metadata)

    if args.max_swe_zero > 0:
        key = "nvidia/SWE-Zero-openhands-trajectories"
        path = out / "agent_traces" / "swe_zero_openhands.jsonl"
        metadata = {"config": args.swe_zero_config, "split": args.swe_zero_split}
        if not reuse_source(key, path, metadata):
            try:
                ds = load_hf_dataset(
                    key,
                    *dataset_config_arg(args.swe_zero_config),
                    split=args.swe_zero_split,
                    streaming=args.streaming,
                )
                records = (
                    openhands_row_to_record(
                        row,
                        source=key,
                        resolved_only=False,
                        languages=languages,
                    )
                    for row in ds
                )
                count = write_jsonl(path, iter_limited(records, args.max_swe_zero))
                manifest["sources"][key] = {**metadata, "records": count}
            except Exception as exc:
                record_source_error(key, exc, metadata)

    if args.max_wildchat > 0:
        key = "allenai/WildChat"
        path = out / "chat" / "wildchat.jsonl"
        metadata = {"config": args.wildchat_config, "split": args.wildchat_split}
        if not reuse_source(key, path, metadata):
            try:
                ds = load_hf_dataset(
                    key,
                    *dataset_config_arg(args.wildchat_config),
                    split=args.wildchat_split,
                    streaming=args.streaming,
                )
                records = (wildchat_row_to_record(row, english_only=args.english_only) for row in ds)
                count = write_jsonl(path, iter_limited(records, args.max_wildchat))
                manifest["sources"][key] = {**metadata, "records": count}
            except Exception as exc:
                record_source_error(key, exc, metadata)

    if args.max_oasst > 0:
        key = "OpenAssistant/oasst1"
        path = out / "chat" / "oasst1_pairs.jsonl"
        metadata = {"config": args.oasst_config, "split": args.oasst_split}
        if not reuse_source(key, path, metadata):
            try:
                ds = load_hf_dataset(
                    key,
                    *dataset_config_arg(args.oasst_config),
                    split=args.oasst_split,
                    streaming=args.streaming,
                )
                records = oasst_rows_to_records(ds, english_only=args.english_only)
                count = write_jsonl(path, iter_limited(records, args.max_oasst))
                manifest["sources"][key] = {**metadata, "records": count}
            except Exception as exc:
                record_source_error(key, exc, metadata)

    for subset, limit in parse_subset_limits(args.starcoder2_extras):
        key = f"bigcode/starcoder2data-extras/{subset}"
        path = out / "base_code_docs" / f"starcoder2_{subset}.jsonl"
        metadata = {"subset": subset, "split": "train"}
        if reuse_source(key, path, metadata):
            continue
        try:
            ds = load_hf_dataset("bigcode/starcoder2data-extras", subset, split="train", streaming=args.streaming)
            records = (text_row_to_record(row, source=key) for row in ds)
            count = write_jsonl(path, iter_limited(records, limit))
            manifest["sources"][key] = {**metadata, "records": count}
        except Exception as exc:
            record_source_error(key, exc, metadata)

    if not manifest["sources"]:
        raise SystemExit("No sources requested. Set at least one --max-* option or --starcoder2-extras SUBSET=LIMIT.")

    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
