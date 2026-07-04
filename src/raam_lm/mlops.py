"""Dependency-free writer for the mlops-mcp-server native experiment store."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def tracker_root(project_path: str | Path) -> Path:
    return Path(project_path).expanduser().resolve() / ".mlops" / "experiments"


def stable_run_id(source: str | Path, prefix: str = "raam") -> str:
    text = str(Path(source).expanduser().resolve() if isinstance(source, Path) else source)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    stem = Path(text).name if text else "run"
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in stem).strip("-")
    slug = "-".join(part for part in slug.split("-") if part)[:60] or "run"
    return f"{prefix}-{slug}-{digest}"


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _run_dir(project_path: str | Path, run_id: str) -> Path:
    return tracker_root(project_path) / run_id


def _registry_path(project_path: str | Path) -> Path:
    return tracker_root(project_path) / "registry.json"


def init_tracker(project_path: str | Path) -> Path:
    root = tracker_root(project_path)
    root.mkdir(parents=True, exist_ok=True)
    registry = _registry_path(project_path)
    if not registry.exists():
        save_json(registry, {"runs": []})
    return root


def _update_registry(project_path: str | Path, record: dict[str, Any]) -> None:
    init_tracker(project_path)
    registry_path = _registry_path(project_path)
    registry = load_json(registry_path, {"runs": []})
    rows = [row for row in registry.get("runs", []) if row.get("run_id") != record["run_id"]]
    rows.append(record)
    rows.sort(key=lambda row: row.get("ended_at") or row.get("started_at") or "", reverse=True)
    registry["runs"] = rows
    save_json(registry_path, registry)


def start_run(
    project_path: str | Path,
    run_id: str,
    params: dict[str, Any] | None = None,
    started_at: str | None = None,
    overwrite: bool = False,
) -> Path:
    init_tracker(project_path)
    run_path = _run_dir(project_path, run_id)
    if overwrite and run_path.exists():
        shutil.rmtree(run_path)
    run_path.mkdir(parents=True, exist_ok=True)
    save_json(run_path / "params.json", params or {})
    if not (run_path / "metrics.json").exists():
        save_json(run_path / "metrics.json", [])
    if not (run_path / "artifacts.json").exists():
        save_json(run_path / "artifacts.json", [])
    start = started_at or now_iso()
    save_json(run_path / "status.json", {"status": "running", "started_at": start})
    _update_registry(project_path, {"run_id": run_id, "status": "running", "started_at": start})
    return run_path


def update_params(project_path: str | Path, run_id: str, params: dict[str, Any]) -> None:
    path = _run_dir(project_path, run_id) / "params.json"
    existing = load_json(path, {})
    existing.update(params)
    save_json(path, existing)


def append_metrics(project_path: str | Path, run_id: str, metrics: dict[str, Any], step: int | None = None) -> None:
    path = _run_dir(project_path, run_id) / "metrics.json"
    history = load_json(path, [])
    entry = {"timestamp": now_iso(), **metrics}
    if step is not None:
        entry["step"] = int(step)
    elif "global_step" in metrics:
        entry["step"] = int(metrics["global_step"])
    history.append(entry)
    save_json(path, history)


def add_artifact_reference(
    project_path: str | Path,
    run_id: str,
    artifact_path: str | Path,
    kind: str = "artifact",
    copy: bool = False,
) -> None:
    run_path = _run_dir(project_path, run_id)
    source = Path(artifact_path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        return
    artifacts_path = run_path / "artifacts.json"
    manifest = load_json(artifacts_path, [])
    entry: dict[str, Any] = {
        "kind": kind,
        "source": str(source),
        "size": source.stat().st_size,
        "logged_at": now_iso(),
    }
    if copy:
        target = run_path / "artifacts" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        entry["path"] = str(target.relative_to(run_path))
    manifest.append(entry)
    save_json(artifacts_path, manifest)


def finish_run(project_path: str | Path, run_id: str, status: str = "success") -> None:
    run_path = _run_dir(project_path, run_id)
    status_path = run_path / "status.json"
    state = load_json(status_path, {})
    started_at = state.get("started_at")
    ended_at = now_iso()
    duration_seconds = None
    if started_at:
        try:
            duration_seconds = max(
                0.0,
                (datetime.fromisoformat(ended_at) - datetime.fromisoformat(started_at)).total_seconds(),
            )
        except ValueError:
            duration_seconds = None
    payload = {
        "status": status,
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_seconds": duration_seconds,
    }
    save_json(status_path, payload)
    _update_registry(
        project_path,
        {
            "run_id": run_id,
            "status": status,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": duration_seconds,
        },
    )
