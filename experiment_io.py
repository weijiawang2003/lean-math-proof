"""Helpers for standardized experiment output layout and logging."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def make_run_dir(base_dir: str, method: str, run_id: str | None = None) -> Path:
    rid = run_id or f"{method}-{_utc_stamp()}-{uuid.uuid4().hex[:6]}"
    run_dir = Path(base_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def init_run_artifacts(base_dir: str, method: str, config: dict[str, Any], run_id: str | None = None) -> dict[str, str]:
    run_dir = make_run_dir(base_dir=base_dir, method=method, run_id=run_id)
    traces_path = run_dir / "traces.jsonl"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    return {
        "run_dir": str(run_dir),
        "traces_path": str(traces_path),
        "metrics_path": str(metrics_path),
        "config_path": str(config_path),
    }


def write_metrics(path: str, metrics: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
