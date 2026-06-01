#!/usr/bin/env python3
"""RC5H Part 10 — RC2 vs RC4 static vs RC5H B5/B10/B20 comparison.

RC5H solved(B) = RC4 static solved OR a TRUE_HYBRID_DELTA dynamic win at budget ≤ B. Reports
solved counts, new wins over RC2 / over RC4, regressions (additive ⇒ 0), probes run, dynamic
probes per additional win, floors, and by namespace / benchmark set. Identifies the best budget.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLOOR_SETS = ("canonical_floors",)


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--static-results", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--dynamic-b5", required=True)
    ap.add_argument("--dynamic-b10")
    ap.add_argument("--dynamic-b20")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2_results)))["results"]}
    static = {r["full_name"]: r for r in json.load(open(_p(args.static_results)))["results"]}
    attr = {r["full_name"]: r for r in json.load(open(_p(args.attribution)))["records"]}
    sets_of = {fn: r.get("sets", []) for fn, r in static.items()}

    def _load(path):
        return json.load(open(_p(path))) if path and os.path.exists(_p(path)) else None
    stages = {"B5": _load(args.dynamic_b5), "B10": _load(args.dynamic_b10), "B20": _load(args.dynamic_b20)}
    # budget at which each true-delta theorem was won
    true_delta_budget = {fn: r.get("budget") for fn, r in attr.items()
                         if r["classification"] == "TRUE_HYBRID_DELTA"}

    universe = set(static) | set(rc2)
    rc2_solved = {fn for fn in universe if rc2.get(fn, {}).get("status") == "solved"}
    rc4_solved = {fn for fn in universe if static.get(fn, {}).get("status") == "solved"}

    def rc5h_solved(maxb):
        s = set(rc4_solved)
        for fn, b in true_delta_budget.items():
            if b is not None and b <= maxb:
                s.add(fn)
        return s

    # probe counts
    def stage_probes(stage):
        if not stage:
            return 0
        return sum(r.get("programs_attempted", 0) for r in stage.get("results", []))
    probes = {b: stage_probes(stages[b]) for b in ("B5", "B10", "B20")}
    cum_probes = {"B5": probes["B5"], "B10": probes["B5"] + probes["B10"],
                  "B20": probes["B5"] + probes["B10"] + probes["B20"]}

    rows = {}
    for label, solved in (("RC2", rc2_solved), ("RC4_static", rc4_solved),
                          ("RC5H_B5", rc5h_solved(5)), ("RC5H_B10", rc5h_solved(10)),
                          ("RC5H_B20", rc5h_solved(20))):
        rows[label] = {
            "solved": len(solved),
            "new_over_rc2": len(solved - rc2_solved),
            "new_over_rc4": len(solved - rc4_solved),
            "regressions": len(rc4_solved - solved),  # additive => 0
        }
    # marginal dynamic cost
    base_new = rows["RC4_static"]["new_over_rc2"]
    for b, cum in (("RC5H_B5", cum_probes["B5"]), ("RC5H_B10", cum_probes["B10"]), ("RC5H_B20", cum_probes["B20"])):
        add_wins = rows[b]["new_over_rc4"]
        rows[b]["dynamic_probes"] = cum
        rows[b]["dynamic_probes_per_additional_win"] = round(cum / add_wins, 1) if add_wins else None

    # floors
    floor_names = [fn for fn, ss in sets_of.items() if any(s in FLOOR_SETS for s in ss)]
    floors = {"n": len(floor_names),
              "rc2": sum(1 for fn in floor_names if fn in rc2_solved),
              "rc4": sum(1 for fn in floor_names if fn in rc4_solved),
              "rc5h": sum(1 for fn in floor_names if fn in rc5h_solved(20))}

    # by namespace (new over rc4 at B20)
    new_b20 = rc5h_solved(20) - rc4_solved
    by_ns = Counter(static.get(fn, rc2.get(fn, {})).get("namespace") for fn in new_b20)
    by_set = Counter(s for fn in new_b20 for s in sets_of.get(fn, []))

    # best budget: most new-over-rc4 at lowest marginal cost
    best = max(("B5", "B10", "B20"),
               key=lambda b: (rows["RC5H_" + b]["new_over_rc4"], -cum_probes[b]))

    out = {
        "generated_by": "scripts/rc5h_compare_systems.py",
        "systems": rows, "cumulative_dynamic_probes": cum_probes,
        "floors": floors, "new_over_rc4_by_namespace": dict(by_ns),
        "new_over_rc4_by_set": dict(by_set),
        "rc4_static_contribution": base_new,
        "dynamic_stage_contribution": rows["RC5H_B20"]["new_over_rc4"],
        "hybrid_net_delta_over_rc2": rows["RC5H_B20"]["new_over_rc2"],
        "best_budget": best,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5H system comparison", "",
          "| system | solved | new/RC2 | new/RC4 | regr | dyn probes | probes/win |",
          "|---|---|---|---|---|---|---|"]
    for label in ("RC2", "RC4_static", "RC5H_B5", "RC5H_B10", "RC5H_B20"):
        r = rows[label]
        md.append(f"| {label} | {r['solved']} | {r['new_over_rc2']} | {r['new_over_rc4']} | "
                  f"{r['regressions']} | {r.get('dynamic_probes','-')} | "
                  f"{r.get('dynamic_probes_per_additional_win','-')} |")
    md += ["", f"- RC4 static contribution (new/RC2): **{base_new}**",
           f"- dynamic stage contribution (new/RC4 @B20): **{out['dynamic_stage_contribution']}**",
           f"- hybrid net delta over RC2 @B20: **{out['hybrid_net_delta_over_rc2']}**",
           f"- floors (n={floors['n']}): RC2 {floors['rc2']} / RC4 {floors['rc4']} / RC5H {floors['rc5h']}",
           f"- new/RC4 by namespace: {dict(by_ns)} | by set: {dict(by_set)}",
           f"- **best budget: {best}**"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-compare] RC2={rows['RC2']['solved']} RC4={rows['RC4_static']['solved']} "
          f"RC5H_B20={rows['RC5H_B20']['solved']} dyn_contrib={out['dynamic_stage_contribution']} best={best}")


if __name__ == "__main__":
    main()
