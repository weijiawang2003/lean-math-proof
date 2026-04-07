"""Aggregate multiple run metrics files under runs/*/metrics.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick(primary: dict, *keys, default=None):
    cur = primary
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _normalize_row(run_id: str, m: dict) -> dict:
    method = m.get("method") or _pick(m, "run_summary", "method", default=None)
    if method is None:
        if run_id.startswith("search-"):
            method = "beam_search"
        elif run_id.startswith("rollout-"):
            method = "policy_rollout"
        elif run_id.startswith("collect-"):
            method = "scripted_collect"
        else:
            method = "unknown"

    episodes = _pick(m, "episode_metrics", "episodes", default=m.get("episodes_total"))
    success_rate = _pick(m, "episode_metrics", "success_rate", default=None)
    finished = _pick(m, "episode_metrics", "finished_episodes", default=m.get("episodes_finished"))
    error_episodes = _pick(m, "episode_metrics", "error_episodes", default=m.get("episodes_error"))
    rows = _pick(m, "transition_metrics", "rows", default=None)

    # Normalize lightweight rollout metrics into comparable columns.
    if episodes is None and "finished" in m:
        episodes = 1
        finished = int(bool(m.get("finished")))
        error_episodes = int(bool(m.get("has_error")))
        success_rate = float(bool(m.get("finished")))
        rows = m.get("num_steps")

    return {
        "run_id": run_id,
        "method": method,
        "episodes": episodes,
        "success_rate": success_rate,
        "finished": finished,
        "error_episodes": error_episodes,
        "rows": rows,
        "trace_file": m.get("trace_file") or m.get("trace_path") or m.get("traces_path") or "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare metrics across run directories.")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    metrics_files = sorted(runs_dir.glob("*/metrics.json"))

    if not metrics_files:
        print(f"No metrics files found under: {runs_dir}")
        return

    rows = [_normalize_row(mf.parent.name, _load_metrics(mf)) for mf in metrics_files]
    rows = rows[-args.limit :]

    print("run_id | method | episodes | success_rate | finished | error_episodes | rows | trace_file")
    print("-" * 110)
    for r in rows:
        sr = "" if r["success_rate"] is None else f"{r['success_rate']:.3f}"
        print(
            f"{r['run_id']} | {r['method']} | {r['episodes']} | {sr} | "
            f"{r['finished']} | {r['error_episodes']} | {r['rows']} | {r['trace_file']}"
        )


if __name__ == "__main__":
    main()
