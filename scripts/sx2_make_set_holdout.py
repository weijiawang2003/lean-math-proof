#!/usr/bin/env python3
"""SX2 Part 5 — build a fresh Set holdout from the SF1 frontier.

Selects Set-namespace theorems with resolved file paths from the SF1
frontier_with_paths.jsonl, EXCLUDING:
  * the 12 SF2 deep-dive selected cases (--exclude selected_cases.json),
  * non-Set namespaces / junk / unresolved-path rows,
  * (flagged, not dropped) the 3 SF2 *deferred* near-duplicates, so overfit risk
    is transparent rather than hidden.

The holdout deliberately retains the fresh `Set.ite_*` / `Set.mem_dite_*` theorems
that were NOT probed in the deep dive — these are the real generalization test for
the SET_ITE_SIMP gate. Each row is tagged `ite_shaped` and `set2_gate_likely` from
the theorem name so the eval can be read against expectation.

Outputs:
  set2_holdout_cases.json
"""
from __future__ import annotations

import argparse
import json
import os

# the 3 SF2-deferred near-duplicates (kept but flagged)
_DEFERRED = {
    "Set.ite_eq_of_subset_right", "Set.ite_inter_compl_self",
    "Set.not_monotoneOn_not_antitoneOn_iff_exists_le_le",
}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--frontier",
                   default="project/evolve/experiments/sf1/out/real/frontier_with_paths.jsonl")
    p.add_argument("--exclude",
                   default="project/evolve/experiments/sf2/out/set_cluster_deep_dive/selected_cases.json")
    p.add_argument("--out",
                   default="project/evolve/experiments/sx2/out/set2_holdout_cases.json")
    p.add_argument("--max-cases", type=int, default=20)
    args = p.parse_args(argv)

    excl = set()
    if args.exclude and os.path.exists(args.exclude):
        eobj = json.load(open(args.exclude))
        for r in eobj.get("selected", []) + eobj.get("deferred", []):
            excl.add(r["full_name"])

    rows = [json.loads(l) for l in open(args.frontier) if l.strip()]
    cases, skipped = [], []
    for r in rows:
        name = r.get("name")
        ns = r.get("namespace") or ""
        fp = r.get("file_path")
        # Set surface only: namespace Set AND name starts with Set.
        is_set = (ns == "Set") and str(name).startswith("Set.")
        if not is_set:
            skipped.append({"name": name, "why": f"non-Set namespace ({ns})"})
            continue
        if name in excl:
            skipped.append({"name": name, "why": "in deep-dive selected/deferred exclude set"})
            continue
        if not fp:
            skipped.append({"name": name, "why": "no resolved file_path"})
            continue
        lname = name.split(".")[-1]
        ite_shaped = ("ite" in lname) or ("dite" in lname)
        cases.append({
            "full_name": name,
            "file_path": fp,
            "namespace": ns,
            "primary_goal_shape": "unknown_pre_live",
            "ite_shaped": ite_shaped,
            "set2_gate_likely": "SET_ITE_SIMP" if ite_shaped else (
                "SET_IFF_CONSTRUCTOR" if "iff" in lname else "SET_EXT_SIMP/ANTISYMM"),
            "deferred_near_duplicate": name in _DEFERRED,
            "selection_reason": "fresh SF1 Set frontier theorem with resolved path, "
                                "not among the 12 deep-dive cases",
        })

    # prioritise: fresh ite-shaped first (the real SET_ITE_SIMP test), then others;
    # within each, non-deferred before deferred.
    cases.sort(key=lambda c: (not c["ite_shaped"], c["deferred_near_duplicate"],
                              c["full_name"]))
    cases = cases[:args.max_cases]

    out = {
        "source_frontier": args.frontier,
        "excluded_count": len(excl),
        "num_cases": len(cases),
        "num_ite_shaped": sum(1 for c in cases if c["ite_shaped"]),
        "num_deferred_near_duplicates": sum(1 for c in cases if c["deferred_near_duplicate"]),
        "note": "Fresh Set holdout for SET2. ite-shaped theorems are the SET_ITE_SIMP "
                "generalization test. Goal shapes resolved live at eval time. "
                "deferred_near_duplicate rows are SF2 deferrals (kept, flagged for "
                "overfit transparency).",
        "cases": cases,
        "skipped_sample": skipped[:25],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[sx2:holdout] cases={out['num_cases']} ite_shaped={out['num_ite_shaped']} "
          f"deferred_dup={out['num_deferred_near_duplicates']} excluded={len(excl)} "
          f"-> {args.out}")
    for c in cases:
        print(f"   {c['full_name']:48s} ite={c['ite_shaped']} "
              f"gate~{c['set2_gate_likely']} dup={c['deferred_near_duplicate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
