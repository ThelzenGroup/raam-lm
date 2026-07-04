#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Iterable

from scripts.make_agentcoder_curated_sft import build_train_records


def is_structured_agent_record(record: dict[str, Any]) -> bool:
    if "text" in record:
        return False
    return any(key in record for key in ("messages", "trace", "final", "repo_context", "system"))


def render_for_length(record: dict[str, Any]) -> str:
    parts: list[str] = []
    if record.get("system"):
        parts.append(str(record["system"]))
    if record.get("repo_context"):
        parts.append(str(record["repo_context"]))
    for message in record.get("messages", []):
        parts.append(str(message.get("content", "")))
    for step in record.get("trace", []):
        parts.append(str(step.get("content", "")))
    if record.get("final"):
        parts.append(str(record["final"]))
    return "\n".join(parts)


def iter_jsonl_files(paths: Iterable[str | Path]) -> Iterable[Path]:
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            yield from sorted(child for child in path.rglob("*.jsonl") if child.is_file())
        elif path.is_file() and path.suffix.lower() == ".jsonl":
            yield path


def load_real_records(
    paths: Iterable[str | Path],
    *,
    max_document_chars: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats = {
        "jsonl_files": 0,
        "lines_seen": 0,
        "json_errors": 0,
        "plain_text_records": 0,
        "structured_records": 0,
        "skipped_long_records": 0,
    }
    for path in iter_jsonl_files(paths):
        stats["jsonl_files"] += 1
        for line_no, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            if not line.strip():
                continue
            stats["lines_seen"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["json_errors"] += 1
                continue
            if not isinstance(record, dict) or not is_structured_agent_record(record):
                stats["plain_text_records"] += 1
                continue
            stats["structured_records"] += 1
            if max_document_chars > 0 and len(render_for_length(record)) > max_document_chars:
                stats["skipped_long_records"] += 1
                continue
            record = dict(record)
            record["_curriculum_source"] = f"{path}:{line_no}"
            records.append(record)
    return records, stats


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, sort_keys=True) + "\n")


def build_mixed_records(
    raw_inputs: list[str],
    *,
    max_real_records: int,
    max_real_document_chars: int,
    curated_repeats: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    real_records, real_stats = load_real_records(raw_inputs, max_document_chars=max_real_document_chars)
    rng.shuffle(real_records)
    selected_real = real_records[:max_real_records] if max_real_records > 0 else real_records

    curated_base = build_train_records()
    curated_records: list[dict[str, Any]] = []
    for repeat in range(max(0, curated_repeats)):
        for index, record in enumerate(curated_base):
            row = dict(record)
            row["_curriculum_source"] = f"curated:{repeat}:{index}"
            curated_records.append(row)

    records = selected_real + curated_records
    rng.shuffle(records)
    manifest = {
        "format": "agentcoder-mixed-curriculum-v1",
        "seed": seed,
        "raw_inputs": raw_inputs,
        "max_real_records": max_real_records,
        "max_real_document_chars": max_real_document_chars,
        "curated_repeats": curated_repeats,
        "curated_base_records": len(curated_base),
        "curated_records": len(curated_records),
        "real_records_available_after_filter": len(real_records),
        "real_records_selected": len(selected_real),
        "total_records": len(records),
        "real_stats": real_stats,
    }
    return records, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a broad+curated AgentCoder SFT curriculum JSONL.")
    parser.add_argument("--raw-input", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--max-real-records", type=int, default=2000)
    parser.add_argument("--max-real-document-chars", type=int, default=12000)
    parser.add_argument("--curated-repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    records, manifest = build_mixed_records(
        args.raw_input,
        max_real_records=args.max_real_records,
        max_real_document_chars=args.max_real_document_chars,
        curated_repeats=args.curated_repeats,
        seed=args.seed,
    )
    output = Path(args.output)
    manifest_output = Path(args.manifest_output)
    write_jsonl(output, records)
    manifest["output"] = str(output)
    manifest["manifest_output"] = str(manifest_output)
    manifest_output.parent.mkdir(parents=True, exist_ok=True)
    manifest_output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
