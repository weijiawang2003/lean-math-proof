"""AX4 Stage 2 — carve disjoint mining theorem sets from the Stage-1 frontier.

Reads project/data/ax4_multiset_catalog_expand_meta.json (313 frontier
candidates, bucketed by induction-likelihood + cross_surface flag) and writes
disjoint theorem sets to project/evolve/routing/ax4_theorem_sets.json, consumed
by tasks._load_ax4_sets().

Disjointness is by full_name; every candidate lands in at most one set. A
deterministic reserved held-out slice is carved FIRST from the strongest
induction candidates so it stays untouched by the training pool (its labels
are excluded from training in Stage 5; it is still mined with the oracle in
Stage 3 to measure held-out theorem-level wins).

Sets:
  ax4_multiset_induction_high_confidence   strongest induction_on+simp_all
  ax4_multiset_induction_medium_confidence  plausible, often simp-closable
  ax4_multiset_induction_hard               induction-shaped but hard
  ax4_multiset_cross_surface                non-Basic files (barely mined)
  ax4_multiset_negative_control             ext/count/Rel — expected NULL
  ax4_multiset_induction_heldout            reserved eval (untouched by train)

Caps keep the total mine load bounded (~200 + ~45 held-out); availability is
unconfirmed so we oversample the high/medium induction surface where labels
actually come from.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "project/data/ax4_multiset_catalog_expand_meta.json"
OUT = ROOT / "project/evolve/routing/ax4_theorem_sets.json"

# caps per set (availability attrition expected; oversample label-yielding ones)
CAP_HELDOUT = 45
CAP_HIGH = 55
CAP_MEDIUM = 55
CAP_HARD = 28
CAP_CROSS = 45
CAP_NEG = 28


def entry(c: dict) -> dict:
    return {"file_path": c["file"], "full_name": c["full_name"]}


def main() -> None:
    cands = json.loads(META.read_text(encoding="utf-8"))["frontier_candidates"]
    # stable order
    cands.sort(key=lambda c: c["full_name"])

    used: set[str] = set()

    def take(pool, cap):
        out = []
        for c in pool:
            if len(out) >= cap:
                break
            if c["full_name"] in used:
                continue
            used.add(c["full_name"])
            out.append(entry(c))
        return out

    high = [c for c in cands if c["bucket"] == "high"]
    medium = [c for c in cands if c["bucket"] == "medium"]
    hard = [c for c in cands if c["bucket"] == "hard"]
    neg = [c for c in cands if c["bucket"] == "negative"]
    cross = [c for c in cands if c["cross_surface"]
             and c["bucket"] in ("high", "medium")]

    # ---- reserve held-out FIRST, from the strongest Basic induction names ----
    # interleave high & medium Basic candidates deterministically for a mixed,
    # representative held-out surface; cross-surface stays out of held-out.
    held_pool = [c for c in (high + medium) if not c["cross_surface"]]
    held_pool.sort(key=lambda c: (c["bucket"] != "high", c["full_name"]))
    heldout = take(held_pool, CAP_HELDOUT)
    # carve cross-surface before high/medium so it isn't starved
    cross_set = take(cross, CAP_CROSS)

    sets = {
        "ax4_multiset_induction_heldout": heldout,
        "ax4_multiset_cross_surface": cross_set,
        "ax4_multiset_induction_high_confidence": take(high, CAP_HIGH),
        "ax4_multiset_induction_medium_confidence": take(medium, CAP_MEDIUM),
        "ax4_multiset_induction_hard": take(hard, CAP_HARD),
        "ax4_multiset_negative_control": take(neg, CAP_NEG),
    }

    OUT.write_text(json.dumps(sets, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    total = 0
    for k, v in sets.items():
        total += len(v)
        print(f"  {k:44s} {len(v)}")
    print(f"  {'TOTAL':44s} {total}")
    # disjointness assertion
    allnames = [t["full_name"] for v in sets.values() for t in v]
    assert len(allnames) == len(set(allnames)), "sets not disjoint!"
    print("  disjoint: OK")


if __name__ == "__main__":
    main()
