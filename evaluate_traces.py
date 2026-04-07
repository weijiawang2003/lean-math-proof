"""Lightweight evaluation for rollout/search trace JSONL outputs."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from typing import Any

from experiment_io import write_metrics
from trace_io import iter_jsonl


def _safe_episode_key(row: dict[str, Any]) -> tuple[str, str]:
    rid = row.get("run_id") or "na_run"
    eid = row.get("episode_id") or f"{row.get('file_path', '')}:{row.get('full_name', '')}"
    return rid, eid


def evaluate(path: str) -> dict[str, Any]:
    transitions_total = 0
    transitions_finished = 0
    transitions_error = 0

    theorem_steps = Counter()
    theorem_finished_rows = Counter()
    method_counts = Counter()

    episodes = defaultdict(lambda: {
        "run_id": "",
        "episode_id": "",
        "file_path": "",
        "full_name": "",
        "method": "unknown",
        "num_steps": 0,
        "finished": False,
        "has_error": False,
        "max_step": None,
    })

    for row in iter_jsonl(path) or []:
        transitions_total += 1

        theorem_key = (row.get("file_path", ""), row.get("full_name", ""))
        theorem_steps[theorem_key] += 1

        if row.get("proof_finished", False):
            transitions_finished += 1
            theorem_finished_rows[theorem_key] += 1
        if row.get("error_message"):
            transitions_error += 1

        method_counts[row.get("method", "unknown")] += 1

        ep_key = _safe_episode_key(row)
        ep = episodes[ep_key]
        ep["run_id"], ep["episode_id"] = ep_key
        ep["file_path"] = row.get("file_path", "")
        ep["full_name"] = row.get("full_name", "")
        ep["method"] = row.get("method", "unknown")
        ep["num_steps"] += 1
        ep["finished"] = ep["finished"] or bool(row.get("proof_finished", False))
        ep["has_error"] = ep["has_error"] or bool(row.get("error_message"))
        step = row.get("step")
        if isinstance(step, int):
            ep["max_step"] = max(step, ep["max_step"] or step)

    episode_list = list(episodes.values())
    episode_total = len(episode_list)
    episode_finished = sum(1 for e in episode_list if e["finished"])
    episode_error = sum(1 for e in episode_list if e["has_error"])
    avg_steps = (sum(e["num_steps"] for e in episode_list) / episode_total) if episode_total else 0.0

    run_counts = Counter(e["run_id"] for e in episode_list)
    theorem_episode_success = defaultdict(lambda: {"episodes": 0, "finished": 0})
    for e in episode_list:
        tkey = (e["file_path"], e["full_name"])
        theorem_episode_success[tkey]["episodes"] += 1
        theorem_episode_success[tkey]["finished"] += int(e["finished"])

    metrics = {
        "trace_file": path,
        "transition_metrics": {
            "rows": transitions_total,
            "proof_finished_rows": transitions_finished,
            "error_rows": transitions_error,
            "distinct_theorem_targets": len(theorem_steps),
        },
        "episode_metrics": {
            "episodes": episode_total,
            "finished_episodes": episode_finished,
            "error_episodes": episode_error,
            "success_rate": (episode_finished / episode_total) if episode_total else 0.0,
            "avg_steps_per_episode": avg_steps,
        },
        "run_metrics": {
            "runs": len(run_counts),
            "episodes_per_run": dict(run_counts),
        },
        "rows_by_method": dict(method_counts),
        "per_theorem_transition_summary": {
            f"{name}|{path}": {
                "steps": theorem_steps[(path, name)],
                "finished_rows": theorem_finished_rows[(path, name)],
            }
            for (path, name) in theorem_steps
        },
        "per_theorem_episode_summary": {
            f"{name}|{path}": {
                "episodes": v["episodes"],
                "finished_episodes": v["finished"],
                "success_rate": (v["finished"] / v["episodes"]) if v["episodes"] else 0.0,
            }
            for (path, name), v in theorem_episode_success.items()
        },
    }
    return metrics


def print_metrics(metrics: dict[str, Any]) -> None:
    t = metrics["transition_metrics"]
    e = metrics["episode_metrics"]
    r = metrics["run_metrics"]

    print(f"Trace file: {metrics['trace_file']}")
    print(f"transition rows: {t['rows']}")
    print(f"finished rows: {t['proof_finished_rows']}")
    print(f"error rows: {t['error_rows']}")
    print(f"distinct theorem targets: {t['distinct_theorem_targets']}")

    print("\nEpisode-level:")
    print(f"episodes: {e['episodes']}")
    print(f"finished episodes: {e['finished_episodes']}")
    print(f"error episodes: {e['error_episodes']}")
    print(f"episode success rate: {e['success_rate']:.3f}")
    print(f"avg steps/episode: {e['avg_steps_per_episode']:.2f}")

    print("\nRun-level:")
    print(f"runs: {r['runs']}")
    print(f"episodes per run: {r['episodes_per_run']}")

    print("\nrows by method:")
    for method, count in sorted(metrics["rows_by_method"].items(), key=lambda kv: kv[1], reverse=True):
        print(f"  - {method}: {count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trace JSONL outputs.")
    parser.add_argument("--in", dest="in_path", default="traces_from_search.jsonl")
    parser.add_argument("--out-metrics", default="")
    args = parser.parse_args()

    metrics = evaluate(args.in_path)
    print_metrics(metrics)
    if args.out_metrics:
        write_metrics(args.out_metrics, metrics)
        print(f"\nWrote metrics JSON to {args.out_metrics}")
