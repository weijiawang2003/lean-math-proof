"""NS4 helper: compare two metrics.json files theorem-by-theorem.

Usage:
    python scripts/ns4_compare_metrics.py <metrics_legacy> <metrics_bag>

Prints:
  - proved counts side by side
  - theorems that diverge (proved on one path but not the other)
  - origin/family_source mismatches on shared proofs
  - unknown-constant counts and total errored counts
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def summarize(tag: str, m: dict) -> None:
    print(
        f"[{tag}] proved={m.get('proved')}/{m.get('available')}  "
        f"errored={m.get('errored')}  exhausted={m.get('exhausted')}  "
        f"skipped={m.get('skipped')}"
    )
    pbo = m.get("proved_by_origin") or {}
    print(f"  proved_by_origin: {pbo}")
    fcs = m.get("family_proved_counts") or {}
    print(f"  family_proved_counts: {fcs}")


def per_theorem_index(m: dict) -> dict[str, dict]:
    return {row["full_name"]: row for row in m.get("per_theorem") or []}


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    legacy = load(sys.argv[1])
    bag = load(sys.argv[2])

    print("=" * 70)
    summarize("legacy", legacy)
    summarize("bag   ", bag)
    print("=" * 70)

    L = per_theorem_index(legacy)
    B = per_theorem_index(bag)
    only_names = set(L.keys()) ^ set(B.keys())
    if only_names:
        print(f"  WARNING: per-theorem coverage differs by {only_names}")

    proved_diff: list[tuple[str, bool, bool]] = []
    origin_diff: list[tuple[str, str, str]] = []
    winning_diff: list[tuple[str, str, str]] = []
    fam_diff: list[tuple[str, str, str]] = []

    for name in sorted(set(L.keys()) & set(B.keys())):
        l = L[name]
        b = B[name]
        lp = bool(l.get("finished") and not l.get("has_error"))
        bp = bool(b.get("finished") and not b.get("has_error"))
        if lp != bp:
            proved_diff.append((name, lp, bp))
            continue
        if not lp:
            continue
        lo = l.get("winning_tactic_origin")
        bo = b.get("winning_tactic_origin")
        if lo != bo:
            origin_diff.append((name, lo or "", bo or ""))
        lt = (l.get("winning_tactic") or "")
        bt = (b.get("winning_tactic") or "")
        if lt != bt:
            winning_diff.append((name, lt[:80], bt[:80]))
        lf = l.get("winning_tactic_family_source")
        bf = b.get("winning_tactic_family_source")
        if lf != bf:
            fam_diff.append((name, str(lf), str(bf)))

    print()
    print(f"theorems proved_diff (legacy_proved, bag_proved): {len(proved_diff)}")
    for row in proved_diff:
        print(f"  - {row[0]}: legacy={row[1]}, bag={row[2]}")
    print(f"theorems with origin diff on shared proof: {len(origin_diff)}")
    for row in origin_diff[:10]:
        print(f"  - {row[0]}: legacy={row[1]}  bag={row[2]}")
    print(f"theorems with winning_tactic diff on shared proof: {len(winning_diff)}")
    for row in winning_diff[:10]:
        print(f"  - {row[0]}:")
        print(f"      legacy: {row[1]}")
        print(f"      bag:    {row[2]}")
    print(f"theorems with family_source diff on shared proof: {len(fam_diff)}")
    for row in fam_diff[:10]:
        print(f"  - {row[0]}: legacy={row[1]}  bag={row[2]}")

    parity = (not proved_diff) and (not origin_diff) and (not winning_diff)
    print()
    print("PARITY:", "OK" if parity else "FAIL")
    return 0 if parity else 1


if __name__ == "__main__":
    sys.exit(main())
