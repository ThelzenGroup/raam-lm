#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from raam_lm.mlops import (
    add_artifact_reference,
    finish_run,
    load_json,
    now_iso,
    save_json,
    stable_run_id,
    start_run,
    tracker_root,
)


ARTIFACT_NAMES = {
    "agentic_eval.json",
    "atomic_eval.json",
    "chat_eval.json",
    "config.yaml",
    "generation.txt",
    "manifest.json",
    "qualitative_samples.json",
    "summary.json",
    "summary.md",
    "tokenizer.json",
    "train_log.jsonl",
}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def numeric_metrics(row: dict[str, Any]) -> dict[str, float]:
    out: dict[str, float] = {}
    for key, value in row.items():
        if key in {"step", "global_step"}:
            continue
        if isinstance(value, bool):
            out[key] = float(value)
        elif isinstance(value, (int, float)):
            out[key] = float(value)
    return out


def flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), nested, out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value


def find_run_dirs(root: Path) -> list[Path]:
    candidates: set[Path] = set()
    for train_log in root.rglob("train_log.jsonl"):
        candidates.add(train_log.parent)
        if train_log.parent.name == "train":
            candidates.add(train_log.parent.parent)
    for summary in root.rglob("summary.json"):
        candidates.add(summary.parent)
    for manifest in root.rglob("manifest.json"):
        if manifest.parent.name == "train":
            candidates.add(manifest.parent.parent)
        else:
            candidates.add(manifest.parent)
    return sorted(path for path in candidates if path.is_dir())


def infer_model_name(path: Path, params: dict[str, Any]) -> str:
    explicit = params.get("model_name") or params.get("manifest.model_name")
    if explicit:
        return str(explicit)
    name = path.name.lower()
    if "transformer" in name:
        return "transformer"
    if "mamba" in name:
        return "pure_mamba_like"
    if "raam" in name:
        return "raam"
    return "unknown"


def artifact_candidates(run_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    for base in (run_dir, run_dir / "train", run_dir / "packed", run_dir / "generated"):
        if not base.exists() or not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_file():
                candidates.append(child)
    for ckpt_dir in (run_dir / "checkpoints", run_dir / "train" / "checkpoints"):
        if not ckpt_dir.exists() or not ckpt_dir.is_dir():
            continue
        candidates.extend(path for path in ckpt_dir.iterdir() if path.is_file())
    return sorted(set(candidates))


def backfill_one(project_path: Path, run_dir: Path, source_root: Path, copy_artifacts: bool) -> str:
    run_id = stable_run_id(str(run_dir), prefix="backfill")
    params: dict[str, Any] = {
        "source_run_dir": str(run_dir),
        "source_root": str(source_root),
        "backfill_kind": "historical_artifact",
    }

    for manifest_path in [run_dir / "manifest.json", run_dir / "train" / "manifest.json", run_dir / "packed" / "manifest.json"]:
        manifest = load_json(manifest_path, {})
        if isinstance(manifest, dict):
            flatten(manifest_path.parent.name if manifest_path.parent != run_dir else "manifest", manifest, params)

    summary = load_json(run_dir / "summary.json", {})
    if isinstance(summary, dict):
        flatten("summary", summary, params)

    params["model_name"] = infer_model_name(run_dir, params)

    run_path = start_run(project_path, run_id, params=params, overwrite=True)

    train_log = run_dir / "train_log.jsonl"
    if not train_log.exists():
        train_log = run_dir / "train" / "train_log.jsonl"
    rows = iter_jsonl(train_log)
    metric_history: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        step = row.get("global_step", row.get("step", idx))
        metric_history.append({"timestamp": now_iso(), "step": int(step), **numeric_metrics(row)})

    for eval_name in ("agentic_eval.json", "atomic_eval.json", "chat_eval.json", "qualitative_samples.json"):
        eval_path = run_dir / eval_name
        data = load_json(eval_path, {})
        if isinstance(data, dict):
            metrics: dict[str, float] = {}
            for key, value in data.items():
                if isinstance(value, bool):
                    metrics[f"eval.{key}"] = float(value)
                elif isinstance(value, (int, float)):
                    metrics[f"eval.{key}"] = float(value)
            if metrics:
                metric_history.append({"timestamp": now_iso(), **metrics})

    save_json(run_path / "metrics.json", metric_history)

    for artifact in artifact_candidates(run_dir):
        if artifact.name in ARTIFACT_NAMES or artifact.suffix in {".yaml", ".md"}:
            add_artifact_reference(project_path, run_id, artifact, kind="evidence", copy=copy_artifacts)
        elif artifact.suffix in {".pt", ".pth", ".ckpt", ".safetensors"}:
            add_artifact_reference(project_path, run_id, artifact, kind="checkpoint_reference", copy=False)

    finish_run(project_path, run_id, status="success")
    return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill pulled RAAM run artifacts into .mlops/experiments.")
    parser.add_argument("--project-path", default=".", help="RAAM repo path that will receive .mlops/experiments.")
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help="Directory to scan for pulled run artifacts. Can be passed more than once.",
    )
    parser.add_argument("--copy-artifacts", action="store_true", help="Copy small evidence files into .mlops runs.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum run dirs to import.")
    args = parser.parse_args()

    project_path = Path(args.project_path).expanduser().resolve()
    imported: list[str] = []
    seen: set[Path] = set()
    for source in args.source_root:
        source_root = Path(source).expanduser().resolve()
        for run_dir in find_run_dirs(source_root):
            if run_dir in seen:
                continue
            seen.add(run_dir)
            imported.append(backfill_one(project_path, run_dir, source_root, copy_artifacts=args.copy_artifacts))
            if args.limit and len(imported) >= args.limit:
                break
        if args.limit and len(imported) >= args.limit:
            break

    print(
        json.dumps(
            {
                "project_path": str(project_path),
                "tracker_root": str(tracker_root(project_path)),
                "imported_count": len(imported),
                "run_ids": imported,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
