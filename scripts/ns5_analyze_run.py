"""NS5 — analyze a skeleton-evolution run directory and emit summary.

Usage:

    python scripts/ns5_analyze_run.py project/evolve/skeleton_runs/<run_id>/

Reads `scoreboard.jsonl`, `mutation_log.md`, `best_candidate.json`, and
the archive at `project/evolve/archive/skeletons.jsonl` to produce a
ready-to-paste fill for the NS5 report (sections 3-12). The script is
read-only — it never mutates the archive or the run directory.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evolve.skeleton_archive import (
    aggregate,
    dead_skeletons,
    load_archive,
    top_skeletons_by_wins,
)


def _load_scoreboard(p: Path) -> list[dict]:
    rows: list[dict] = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def analyze(run_dir: Path) -> str:
    scoreboard = _load_scoreboard(run_dir / "scoreboard.jsonl")
    best_path = run_dir / "best_candidate.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else {}
    archive_path = REPO_ROOT / "project/evolve/archive/skeletons.jsonl"
    rows = load_archive(archive_path)
    stats = aggregate(rows)

    # Per-operator stats.
    op_counts: Counter[str] = Counter()
    op_promotions: Counter[str] = Counter()
    op_accepted: Counter[str] = Counter()
    promotions: list[dict] = []
    regressions: list[dict] = []
    archive_seed_runs: list[tuple[int, int, int | None]] = []
    medium_results: list[int] = []
    large_results: list[int] = []
    total_runtime = 0.0
    cycles_run = len(scoreboard)

    for r in scoreboard:
        op = r.get("operator", "?")
        op_counts[op] += 1
        if r.get("accepted"):
            op_accepted[op] += 1
        if r.get("promoted_to_best"):
            op_promotions[op] += 1
            promotions.append(r)
        if r.get("newly_lost"):
            regressions.append(r)
        if op == "archive_seed":
            kw = r.get("operator_kwargs") or {}
            archive_seed_runs.append((
                int(kw.get("top_n") or 0),
                int(r.get("enabled_skeletons_count") or 0),
                r.get("proved_medium"),
            ))
        pm = r.get("proved_medium")
        if pm is not None:
            medium_results.append(pm)
        pl = r.get("proved_large")
        if pl is not None:
            large_results.append(pl)
        total_runtime += float(r.get("runtime_seconds") or 0.0)

    # Compute per-theorem diffs union.
    all_newly_proved: dict[str, int] = defaultdict(int)
    all_newly_lost: dict[str, int] = defaultdict(int)
    for r in scoreboard:
        for t in r.get("newly_proved") or []:
            all_newly_proved[t] += 1
        for t in r.get("newly_lost") or []:
            all_newly_lost[t] += 1

    # Build sections.
    lines: list[str] = []
    lines.append(f"## run summary")
    lines.append(f"- run dir: `{run_dir}`")
    lines.append(f"- cycles: {cycles_run}")
    lines.append(f"- runtime (sum of cycles): {total_runtime/3600:.2f}h")
    lines.append(
        f"- best medium: {best.get('proved_medium')}  "
        f"best large: {best.get('proved_large')}"
    )
    lines.append(
        f"- enabled skeletons in best: {best.get('enabled_skeletons')}"
    )
    lines.append("")
    lines.append("## per-operator")
    lines.append("| operator | cycles | accepted | promotions |")
    lines.append("|----------|------:|--------:|-----------:|")
    for op in sorted(op_counts.keys()):
        lines.append(
            f"| {op} | {op_counts[op]} | {op_accepted[op]} | {op_promotions[op]} |"
        )
    lines.append("")
    lines.append("## promotions")
    if not promotions:
        lines.append("(none)")
    else:
        for p in promotions:
            lines.append(
                f"- cycle {p['cycle']} {p['name']}: "
                f"medium={p.get('proved_medium')} large={p.get('proved_large')} "
                f"enabled={p.get('enabled_skeletons_count')}  ({p.get('notes')})"
            )
    lines.append("")
    lines.append("## regressions")
    if not regressions:
        lines.append("(none — no candidate lost a theorem)")
    else:
        for r in regressions:
            lines.append(
                f"- cycle {r['cycle']} {r['name']}: lost={r['newly_lost']}"
            )
    lines.append("")
    lines.append("## compact-genome experiment (archive_seed cycles)")
    if not archive_seed_runs:
        lines.append("(none recorded)")
    else:
        lines.append("| top_n | enabled_skeletons | proved_medium |")
        lines.append("|------:|------------------:|--------------:|")
        for top_n, enabled, pm in archive_seed_runs:
            lines.append(
                f"| {top_n} | {enabled} | {pm if pm is not None else '—'} |"
            )
    lines.append("")
    lines.append("## medium series")
    lines.append(f"distinct medium results across cycles: "
                 f"{sorted(set(medium_results))}")
    lines.append("")
    lines.append("## archive top-15 by wins (post-run)")
    for s in top_skeletons_by_wins(stats, n=15):
        lines.append(
            f"- `{s.skeleton_name}` wins={s.wins} adv={s.advances} att={s.attempts} "
            f"shape={s.skeleton_shape} family={s.skeleton_family} origin={s.origin}"
        )
    lines.append("")
    lines.append("## archive dead-skeletons (attempts>=10, wins==0)")
    for s in dead_skeletons(stats, min_attempts=10)[:30]:
        lines.append(
            f"- `{s.skeleton_name}` att={s.attempts} "
            f"shape={s.skeleton_shape} family={s.skeleton_family} "
            f"origin={s.origin}"
        )
    lines.append("")
    # Skeleton coverage: what % of cycles' proved theorems came from the top-k
    # archived skeletons.
    lines.append("## skeleton coverage (top-N wins as fraction of total)")
    win_counts = [s.wins for s in stats.values() if s.wins > 0]
    if win_counts:
        win_counts.sort(reverse=True)
        total = sum(win_counts)
        for n in (1, 3, 5, 10, 15):
            head = sum(win_counts[:n])
            pct = 100.0 * head / total
            lines.append(
                f"- top {n:>2} skeleton(s) account for {head:>3}/{total} wins "
                f"({pct:.1f}%)"
            )
    else:
        lines.append("- no winners in archive yet")
    lines.append("")
    lines.append("## theorem-level diffs (union across cycles)")
    if all_newly_proved:
        lines.append("newly proved (at least once):")
        for t, c in sorted(all_newly_proved.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{t}` (x{c})")
    else:
        lines.append("(no theorem newly proved in any cycle)")
    if all_newly_lost:
        lines.append("\nlost (at least once):")
        for t, c in sorted(all_newly_lost.items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{t}` (x{c})")
    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"no such dir: {run_dir}")
        sys.exit(1)
    print(analyze(run_dir))


if __name__ == "__main__":
    main()
