"""MX2 Stage 1 — inventory Set theorems an `aesop` fallback could capture.

MX1 found two NEW wins beyond the production wrapper on the Set frontier
(`Set.Finite.toFinset_insert`, `Set.Finite.toFinset_offDiag`) that the strict
live relabel classified as `over_attributed_raw`: a plain `aesop` closes them.
Production missed them only because `aesop` is not in the Set route's emission
battery (the Set route is the gen_v5_ns12_balanced generative policy, no aesop
fallback — unlike Finset, which got NS21's aesop). This stage inventories the
known aesop-misses plus similar Set lemmas an `aesop` fallback might also catch.

Reads (no Lean run here):
  - project/data/mx1_minimal_symbolic_frontier_labels.json  (the 2 aesop closers)
  - project/data/mx1_live_mining_probe_meta.json            (per-set wins)
  - project/data/mx1_symbolic_frontier_audit_meta.json      (full Set frontier)
  - project/evolve/routing/mx1_theorem_sets.json            (already-mined Set)

Outputs:
  project/data/mx2_set_aesop_candidate_meta.json
  project/evolve/reports/mx2_set_aesop_candidate_inventory.md
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "project/data/mx1_minimal_symbolic_frontier_labels.json"
PROBE = ROOT / "project/data/mx1_live_mining_probe_meta.json"
AUDIT = ROOT / "project/data/mx1_symbolic_frontier_audit_meta.json"
MX1SETS = ROOT / "project/evolve/routing/mx1_theorem_sets.json"
OUT_META = ROOT / "project/data/mx2_set_aesop_candidate_meta.json"
OUT_MD = ROOT / "project/evolve/reports/mx2_set_aesop_candidate_inventory.md"

# name-prefix buckets we consider aesop-amenable on the Set surface
BUCKETS = ["Set.Finite.toFinset", "Set.toFinset", "Set.Finite.",
           "Set.image", "Set.preimage", "Set.insert", "Set.offDiag"]


def bucket_of(fn: str) -> str | None:
    for b in BUCKETS:
        if fn.startswith(b):
            return b
    return None


def main() -> None:
    # 1. the known aesop closers from the MX1 live relabel
    known = []
    if LABELS.exists():
        for r in json.loads(LABELS.read_text()).get("labels", []):
            if r["namespace"] == "Set" and \
                    r.get("classification") == "over_attributed_raw" and \
                    (r.get("minimal_closer") or "").strip() == "aesop":
                known.append(r["theorem"])
    # 2. all MX1 new-Set-wins beyond production (for completeness)
    new_set_wins = []
    if PROBE.exists():
        for r in json.loads(PROBE.read_text()).get("new_win_records", []):
            if r.get("namespace") == "Set":
                new_set_wins.append(r["theorem"])

    # 3. file_path map + already-mined Set set
    audit = {c["full_name"]: c
             for c in json.loads(AUDIT.read_text())["candidates"]
             if c["namespace"] == "Set"}
    mx1_set = {it["full_name"]
               for it in json.loads(MX1SETS.read_text())["mx1_set_ext_frontier"]}

    # 4. bucket the full fresh Set frontier
    by_bucket = defaultdict(list)
    for fn, c in audit.items():
        b = bucket_of(fn)
        if b:
            by_bucket[b].append({"full_name": fn, "file_path": c["file_path"],
                                 "already_mined_mx1": fn in mx1_set,
                                 "difficulty": c.get("difficulty", "?")})

    meta = {
        "description": "MX2 Stage 1 — Set theorems an `aesop` fallback could "
                       "capture (the MX1 aesop-misses + similar names). No Lean "
                       "run here; availability confirmed at eval time.",
        "known_aesop_wins": known,
        "mx1_new_set_wins_beyond_production": new_set_wins,
        "buckets": {b: {"count": len(v), "theorems": v}
                    for b, v in by_bucket.items()},
        "bucket_counts": {b: len(v) for b, v in by_bucket.items()},
        "total_candidates": sum(len(v) for v in by_bucket.values()),
        "note": ("The Set route (gen_v5_ns12_balanced) has no aesop fallback, "
                 "unlike Finset (NS21). MX2 adds a Set-gated aesop fallback "
                 "mirroring NS19's finset_aesop_only and tests whether it "
                 "captures these without regressions."),
    }
    OUT_META.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                        encoding="utf-8")

    lines = ["# MX2 Set-aesop candidate inventory\n",
             "MX1 found 2 new Set wins beyond production, both "
             "`over_attributed_raw` (a plain `aesop` closes them). The Set route "
             "carries no aesop fallback (unlike Finset/NS21). This inventories "
             "the misses + similar Set lemmas a Set-gated aesop fallback might "
             "also catch.\n",
             f"## Known aesop wins ({len(known)})\n"]
    for fn in known:
        lines.append(f"- `{fn}`")
    lines += ["", "## Candidate buckets (fresh Set frontier)\n",
              "| name prefix | count | already mined (MX1) |", "|---|---|---|"]
    for b in BUCKETS:
        v = by_bucket.get(b, [])
        if v:
            mined = sum(1 for x in v if x["already_mined_mx1"])
            lines.append(f"| `{b}*` | {len(v)} | {mined} |")
    lines += ["", f"**Total candidates: {meta['total_candidates']}** across "
              f"{len([b for b in by_bucket if by_bucket[b]])} buckets.",
              "", meta["note"]]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {OUT_META.relative_to(ROOT)}")
    print(f"wrote {OUT_MD.relative_to(ROOT)}")
    print(f"known aesop wins: {known}")
    print(f"bucket_counts: {meta['bucket_counts']} total={meta['total_candidates']}")


if __name__ == "__main__":
    main()
