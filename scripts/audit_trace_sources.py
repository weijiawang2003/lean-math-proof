"""NS11 Stage 1 — audit all available trace sources.

Walks every traces.jsonl under project/evolve and produces a per-run
table of: theorem_set / # episodes / close transitions / advance
transitions / origin histogram / whether skeleton metadata is present.

Output: prints to stdout, also writes
project/evolve/reports/ns11_trace_source_audit.md
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path


def _classify_set(path: Path) -> str:
    """Heuristic: extract theorem set from path components."""
    parts = [p.lower() for p in path.parts]
    for p in parts:
        if "large" in p and "v5" in p:
            return "large_v5"
        if "large" in p:
            return "large?"
        if "medium" in p:
            return "nat_defs_medium"
        if "demo_v1" in p or "demo-v1" in p:
            return "demo_v1"
        if "curriculum" in p:
            return "curriculum"
        if "nat_defs_subset" in p:
            return "nat_defs_subset"
    return "?"


def _is_close(r: dict) -> bool:
    return bool(r.get("proof_finished"))


def _is_advance(r: dict) -> bool:
    if r.get("proof_finished"):
        return False
    kind = r.get("result_kind") or ""
    if kind == "LeanError":
        return False
    if kind in {"SkippedBloatingApply", "SkippedKnownError"}:
        return False
    if r.get("loop_detected") or r.get("bloat_rejected"):
        return False
    return r.get("state_hash_after") is not None


def audit_trace(p: Path) -> dict:
    """Stream one traces.jsonl and return summary."""
    episodes: set[str] = set()
    n_close = 0
    n_adv = 0
    n_lean_err = 0
    n_rows = 0
    origins: dict[str, int] = defaultdict(int)
    has_skeleton = False
    try:
        text = p.read_text(encoding="utf-8")
    except Exception as e:
        return {"path": str(p), "error": str(e)}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        n_rows += 1
        if r.get("episode_id"):
            episodes.add(r["episode_id"])
        if _is_close(r):
            n_close += 1
            origins[r.get("tactic_origin") or "?"] += 1
        elif _is_advance(r):
            n_adv += 1
            origins[r.get("tactic_origin") or "?"] += 1
        elif r.get("result_kind") == "LeanError":
            n_lean_err += 1
        if "skeleton_stable_id" in r:
            has_skeleton = True
    return {
        "path": str(p),
        "n_rows": n_rows,
        "n_episodes": len(episodes),
        "n_close": n_close,
        "n_advance": n_adv,
        "n_lean_err": n_lean_err,
        "origins": dict(origins),
        "has_skeleton": has_skeleton,
    }


def main() -> None:
    root = Path("project/evolve")
    traces = sorted(root.rglob("traces.jsonl"))

    by_dir_close: dict[str, int] = defaultdict(int)
    by_dir_adv: dict[str, int] = defaultdict(int)
    by_dir_eps: dict[str, int] = defaultdict(int)
    by_dir_files: dict[str, int] = defaultdict(int)
    by_dir_set: dict[str, set[str]] = defaultdict(set)
    by_dir_origins: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_dir_skel: dict[str, bool] = defaultdict(bool)

    grand_close = 0
    grand_adv = 0
    grand_eps = 0
    for p in traces:
        info = audit_trace(p)
        if "error" in info:
            continue
        # Bucket by 3rd-level dir under project/evolve, e.g.
        # project/evolve/ns9_runs/ns8-2026...   →  ns9_runs/ns8-2026...
        rel = p.relative_to(root)
        # Use 2 components as the run group.
        if len(rel.parts) >= 2:
            group = "/".join(rel.parts[:2])
        else:
            group = str(rel)
        by_dir_close[group] += info["n_close"]
        by_dir_adv[group] += info["n_advance"]
        by_dir_eps[group] += info["n_episodes"]
        by_dir_files[group] += 1
        by_dir_set[group].add(_classify_set(p))
        for k, v in info["origins"].items():
            by_dir_origins[group][k] += v
        if info["has_skeleton"]:
            by_dir_skel[group] = True
        grand_close += info["n_close"]
        grand_adv += info["n_advance"]
        grand_eps += info["n_episodes"]

    rows = sorted(by_dir_close.keys(), key=lambda k: -by_dir_close[k])

    lines: list[str] = []
    lines.append("# NS11 trace source audit\n")
    lines.append(f"- total traces.jsonl files: **{len(traces)}**")
    lines.append(f"- total episodes (across files): **{grand_eps}**")
    lines.append(f"- total close transitions:    **{grand_close}**")
    lines.append(f"- total advance transitions:  **{grand_adv}**")
    lines.append("")
    lines.append("| run group | files | episodes | close | advance | "
                 "theorem sets | skel meta | origins (close+adv) |")
    lines.append("|---|---:|---:|---:|---:|---|---|---|")
    for g in rows:
        sets = ", ".join(sorted(by_dir_set[g]))
        orig = ", ".join(
            f"{k}:{v}" for k, v in sorted(
                by_dir_origins[g].items(), key=lambda kv: -kv[1]
            )[:5]
        )
        skel = "yes" if by_dir_skel[g] else "no"
        lines.append(
            f"| `{g}` | {by_dir_files[g]} | {by_dir_eps[g]} | "
            f"{by_dir_close[g]} | {by_dir_adv[g]} | {sets} | {skel} | {orig} |"
        )

    txt = "\n".join(lines) + "\n"
    out = Path("project/evolve/reports/ns11_trace_source_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    print(txt)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
