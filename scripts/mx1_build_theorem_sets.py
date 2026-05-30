"""MX1 Stage 2 — carve disjoint live-mining theorem sets from the frontier.

Reads the Stage-1 audit (project/data/mx1_symbolic_frontier_audit_meta.json) and
writes disjoint sets to project/evolve/routing/mx1_theorem_sets.json, consumed by
tasks._load_mx1_sets(). Availability-screened candidates are preferred (they were
confirmed loadable by the cx1 probe); the rest are availability-unconfirmed and
confirmed at mine time (timeout-guarded), exactly as AX4 did.

Sets:
  mx1_multiset_frontier          remaining fresh Multiset (induction/ext)
  mx1_finset_symbolic_frontier   Finset ext/cases (the largest fresh symbolic surface)
  mx1_list_frontier              remaining fresh List (cases/induction)
  mx1_set_ext_frontier           Set extensionality candidates
  mx1_mixed_symbolic_frontier    mixed hard frontier for stress testing

Caps keep the live-mine load bounded; the full frontier is far larger (logged in
the audit) — what is NOT carved here is recorded so coverage is explicit.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "project/data/mx1_symbolic_frontier_audit_meta.json"
OUT = ROOT / "project/evolve/routing/mx1_theorem_sets.json"

# per-set caps (bounded for live mining; frontier is much larger)
CAP_MULTISET = 18
CAP_FINSET = 40
CAP_LIST = 20
CAP_SET = 40
CAP_MIXED = 20


# Known-bad theorems to exclude from live mining. Multiset.eq_of_mem_map_const
# hangs the Lean REPL indefinitely past per-tactic timeouts (observed in AX3).
KNOWN_BAD = {"Multiset.eq_of_mem_map_const"}


def entry(c):
    return {"file_path": c["file_path"], "full_name": c["full_name"]}


def main() -> None:
    cand = json.load(open(AUDIT))["candidates"]
    used = set()

    def take(pred, cap, prefer_screened=True):
        pool = [c for c in cand if c["full_name"] not in used
                and c["full_name"] not in KNOWN_BAD and pred(c)]
        # prefer availability-screened, then easier difficulty, stable by name
        diff_rank = {"easy": 0, "medium": 1, "hard": 2, "?": 3}
        pool.sort(key=lambda c: (
            0 if c.get("availability") == "screened" else 1,
            diff_rank.get(c.get("difficulty"), 3), c["full_name"]))
        out = []
        for c in pool[:cap]:
            used.add(c["full_name"])
            out.append(entry(c))
        return out

    multiset = take(lambda c: c["namespace"] == "Multiset", CAP_MULTISET)
    finset = take(lambda c: c["namespace"] == "Finset"
                  and c["likely_family"] in ("finset_ext_simp",
                                             "finset_cases_simp"), CAP_FINSET)
    lst = take(lambda c: c["namespace"] == "List", CAP_LIST)
    setext = take(lambda c: c["namespace"] == "Set"
                  and c["likely_family"] == "set_ext_simp", CAP_SET)
    # mixed hard stress: hardest remaining across all priority namespaces
    mixed = take(lambda c: c.get("difficulty") in ("hard", "medium"), CAP_MIXED)

    sets = {
        "mx1_multiset_frontier": multiset,
        "mx1_finset_symbolic_frontier": finset,
        "mx1_list_frontier": lst,
        "mx1_set_ext_frontier": setext,
        "mx1_mixed_symbolic_frontier": mixed,
    }
    OUT.write_text(json.dumps(sets, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    total = sum(len(v) for v in sets.values())
    # coverage note: how much of the frontier we carved vs left
    from collections import Counter
    frontier_by_ns = Counter(c["namespace"] for c in cand)
    carved_by_ns = Counter()
    for v in sets.values():
        for it in v:
            carved_by_ns[it["full_name"].split(".", 1)[0]] += 1
    print(f"wrote {OUT.relative_to(ROOT)}  (total {total} theorems carved)")
    for k, v in sets.items():
        scr = sum(1 for it in v if any(
            c["full_name"] == it["full_name"] and c.get("availability") ==
            "screened" for c in cand))
        print(f"  {k:32s} {len(v):3d}  (screened={scr})")
    print("coverage carved/frontier by ns:")
    for ns in ("Multiset", "Finset", "List", "Set", "Option"):
        print(f"  {ns}: {carved_by_ns.get(ns,0)}/{frontier_by_ns.get(ns,0)}")


if __name__ == "__main__":
    main()
