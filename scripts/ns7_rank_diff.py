"""NS7 — rank-diff diagnostic.

Compute the per-shape skeleton-emit-rank difference between a baseline
genome and a mutated genome, intersected with a protected-skeleton set.

Output: markdown report listing every protected skeleton whose
position in the bag's deterministic emit order differs between the two
genomes. Useful for understanding *why* a mutation got rejected by the
pre-flight detector, and for surfacing rank-coupling effects before a
sweep starts.

Usage:
    python scripts/ns7_rank_diff.py \\
        --baseline-genome project/evolve/ns7_runs/baseline/genome.json \\
        --mutated-genome path/to/mutated.json \\
        --protected project/evolve/archive/protected_skeletons.json \\
        --out project/evolve/reports/ns7_rank_diff_diagnostics.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evolve.rank_coupling import (
    check_rank_coupling,
    summarize_violations,
    _enabled_skeletons_by_shape,
    _rank_by_stable_id,
)
from evolve.skeleton_mutator import genome_to_bag


def _shape_breakdown(genome: dict[str, Any], shapes: list[str]) -> dict[str, list[dict]]:
    bag = genome_to_bag(genome)
    out: dict[str, list[dict]] = {}
    for sh in shapes:
        skels = _enabled_skeletons_by_shape(bag, sh)
        out[sh] = [
            {
                "rank": i, "name": s.name, "stable_id": s.stable_id,
                "origin": s.origin, "shape": s.shape, "family": s.family,
            }
            for i, s in enumerate(skels)
        ]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-genome", type=Path, required=True)
    ap.add_argument("--mutated-genome", type=Path, required=True)
    ap.add_argument("--protected", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rank-slack", type=int, default=0)
    args = ap.parse_args()

    baseline = json.loads(args.baseline_genome.read_text(encoding="utf-8"))
    if "genome" in baseline and isinstance(baseline["genome"], dict):
        baseline = baseline["genome"]
    mutated = json.loads(args.mutated_genome.read_text(encoding="utf-8"))
    if "genome" in mutated and isinstance(mutated["genome"], dict):
        mutated = mutated["genome"]
    protected = json.loads(args.protected.read_text(encoding="utf-8"))

    violations = check_rank_coupling(
        baseline, mutated, protected["entries"], rank_slack=args.rank_slack,
    )
    summary = summarize_violations(violations)

    # Build per-shape rank tables for human inspection.
    shapes = sorted({e.get("shape") or "any" for e in protected["entries"]})
    base_break = _shape_breakdown(baseline, shapes)
    mut_break = _shape_breakdown(mutated, shapes)

    lines: list[str] = []
    lines.append("# NS7 — rank-diff diagnostic\n")
    lines.append(f"- baseline genome: `{args.baseline_genome}`")
    lines.append(f"- mutated  genome: `{args.mutated_genome}`")
    lines.append(f"- protected set : `{args.protected}` ({protected.get('skeleton_count')} skeletons, {protected.get('entry_count')} entries)\n")
    lines.append("## Summary\n")
    lines.append(f"- total violations: **{summary['total']}**")
    lines.append(f"- by kind: `{summary['by_kind']}`")
    lines.append(f"- by reason: `{summary['by_reason']}`")
    lines.append(f"- affected theorems: {len(summary['affected_theorems'])}")
    if summary["affected_theorems"][:5]:
        lines.append("    e.g. " + ", ".join(summary["affected_theorems"][:5]))
    lines.append("")

    if violations:
        lines.append("## Violations\n")
        lines.append("| stable_id | name | shape | reason | base_rank | mut_rank | kind |")
        lines.append("|---|---|---|---|---:|---:|---|")
        for v in violations:
            lines.append(
                f"| {v.skeleton_stable_id} | {v.skeleton_name or '-'} | "
                f"{(v.notes and v.notes.split()[0]) or '-'} | {v.reason} | "
                f"{v.baseline_rank} | {v.mutated_rank if v.mutated_rank is not None else '∅'} | {v.kind} |"
            )
        lines.append("")

    for sh in shapes:
        lines.append(f"## Skeleton-emit rank — shape={sh}\n")
        rows_base = base_break.get(sh, [])
        rows_mut = mut_break.get(sh, [])
        max_len = max(len(rows_base), len(rows_mut))
        lines.append("| rank | baseline (stable_id, name) | mutated (stable_id, name) |")
        lines.append("|---:|---|---|")
        for i in range(max_len):
            b = rows_base[i] if i < len(rows_base) else None
            m = rows_mut[i] if i < len(rows_mut) else None
            b_str = f"`{b['stable_id']}` {b['name']}" if b else "—"
            m_str = f"`{m['stable_id']}` {m['name']}" if m else "—"
            same = b and m and b["stable_id"] == m["stable_id"]
            marker = "" if same else "  ←"
            lines.append(f"| {i} | {b_str} | {m_str}{marker} |")
        lines.append("")

    report = "\n".join(lines)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(report)


if __name__ == "__main__":
    main()
