"""NS5 — inspect a skeleton-bag genome and emit a readable summary.

Usage:

    python scripts/ns5_skeleton_inspect.py path/to/genome.json

Reads a legacy strategy-config dict (as written by `write_strategy_config`),
converts it to a SkeletonBag, and prints a per-shape / per-family
breakdown including which skeletons would emit for typical Nat-goal
shapes.

Useful for the final NS5 report when we want to show the *shape* of the
best-evolved genome (e.g., "the compact core has only 22 skeletons").
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from evolve.skeleton_bag import SkeletonBag


def inspect(genome_path: Path) -> str:
    genome = json.loads(genome_path.read_text(encoding="utf-8"))
    bag = SkeletonBag.from_legacy_strategy_config(genome)

    lines: list[str] = []
    lines.append(f"# Skeleton inspection — {genome_path}\n")
    lines.append(f"- total skeletons: {len(bag)}")
    lines.append(f"- enabled: {sum(1 for s in bag.all_skeletons() if s.enabled)}")
    counts = bag.count_by_origin()
    lines.append(f"- by origin: {dict(counts)}")
    lines.append(f"- shape slots: {sorted(bag.skeletons.keys())}")
    lines.append(f"- family slots: {sorted(bag.families.keys())}")
    lines.append("")

    lines.append("## Per-shape, per-origin")
    for shape in sorted(bag.skeletons.keys()):
        ss = bag.skeletons[shape]
        per_origin: Counter[str] = Counter()
        for s in ss:
            per_origin[s.origin] += 1
        lines.append(f"### shape={shape}  total={len(ss)}")
        lines.append(f"by origin: {dict(per_origin)}")
        for s in sorted(ss, key=lambda s: (s.priority, s.specificity)):
            flag = "" if s.enabled else " [disabled]"
            tpl = s.template if len(s.template) <= 70 else s.template[:67] + "..."
            lines.append(
                f"  · {s.name:20s} pri={s.priority:>2} spec={s.specificity} "
                f"family={s.family}  origin={s.origin}{flag}"
            )
            lines.append(f"      template: `{tpl}`")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print(f"no such file: {p}")
        sys.exit(1)
    # Allow pointing at a run dir; resolve to best_candidate.json.
    if p.is_dir():
        best = p / "best_candidate.json"
        if best.exists():
            data = json.loads(best.read_text(encoding="utf-8"))
            # The best file wraps the genome under "genome" — extract it.
            genome = data.get("genome") if "genome" in data else data
            tmp = p / "_inspect_genome.json"
            tmp.write_text(
                json.dumps(genome, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            p = tmp
        else:
            print(f"no best_candidate.json under {p}")
            sys.exit(1)
    print(inspect(p))


if __name__ == "__main__":
    main()
