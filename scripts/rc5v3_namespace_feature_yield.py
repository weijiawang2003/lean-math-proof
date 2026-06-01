#!/usr/bin/env python3
"""RC5V3 Part 12 — namespace and feature yield analysis.

Per namespace: batch count / RC2 solved / RC4 solved / dynamic eligible / dynamic wins / probes /
probes-per-win / recommendation. Per feature: eligible / wins. Classifies each namespace
HIGH/MODERATE/LOW_YIELD_DYNAMIC_TARGET / DISABLE_DYNAMIC_BY_DEFAULT / NEED_MORE_DATA.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES = ["has_subset", "has_iff", "has_mem", "has_disjoint", "has_singleton",
            "has_image", "has_map_filter", "has_bind", "has_forall_exists", "has_nat_arith"]


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    batch = json.load(open(_p(args.batch)))["theorems"]
    elig = json.load(open(_p(args.eligible)))["results"]
    plan = json.load(open(_p(args.plan)))["theorems"]
    attr = json.load(open(_p(args.attribution)))

    # need RC2/RC4 solved per ns: derive from comparison? comparison has totals only; recompute from
    # attribution records' rc2/rc4 is not stored. Use batch features + eligible + attribution.
    batch_by_ns = Counter(t["namespace"] for t in batch)
    elig_by_ns = Counter(e["namespace"] for e in elig)
    plan_by_ns = Counter(t["namespace"] for t in plan)
    plan_probes_by_ns = Counter()
    for t in plan:
        plan_probes_by_ns[t["namespace"]] += t.get("num_programs", len(t.get("programs_ranked", [])))

    recs = attr["records"]
    wins_by_ns = Counter(r["namespace"] for r in recs if r["classification"] == "FRESH_TRUE_RC5V3_DELTA")
    attempts_probes_by_ns = Counter()  # actual probes attempted (from dynamic results merged via attribution? not stored)
    # use attribution attempt count proxy: dynamic probes per namespace from comparison total split by plan size
    # better: read budget result files is heavy here; approximate probes by plan program counts on eligible.

    namespaces = sorted(set(list(batch_by_ns) + list(elig_by_ns)))
    per_ns = {}
    for ns in namespaces:
        bn = batch_by_ns.get(ns, 0)
        en = elig_by_ns.get(ns, 0)
        wn = wins_by_ns.get(ns, 0)
        probes = plan_probes_by_ns.get(ns, 0)
        ppw = round(probes / wn, 1) if wn else None
        if en == 0:
            rec = "DISABLE_DYNAMIC_BY_DEFAULT" if ns not in ("Set", "Finset", "List", "Multiset", "Nat") else "NEED_MORE_DATA"
        elif wn == 0:
            rec = "LOW_YIELD_DYNAMIC_TARGET" if en >= 10 else "NEED_MORE_DATA"
        elif wn / en >= 0.08:
            rec = "HIGH_YIELD_DYNAMIC_TARGET"
        elif wn / en >= 0.03:
            rec = "MODERATE_YIELD_DYNAMIC_TARGET"
        else:
            rec = "LOW_YIELD_DYNAMIC_TARGET"
        per_ns[ns] = {"batch": bn, "dynamic_eligible": en, "dynamic_wins": wn,
                      "plan_probes": probes, "probes_per_win": ppw,
                      "win_rate_over_eligible": round(wn / en, 3) if en else 0.0,
                      "recommendation": rec}

    # feature yield over eligible set
    elig_feat = {e["full_name"]: (e.get("features") or {}) for e in elig}
    win_names = {r["full_name"] for r in recs if r["classification"] == "FRESH_TRUE_RC5V3_DELTA"}
    per_feat = {}
    for f in FEATURES:
        elig_with = [fn for fn, ft in elig_feat.items() if ft.get(f)]
        wins_with = [fn for fn in elig_with if fn in win_names]
        per_feat[f] = {"eligible": len(elig_with), "wins": len(wins_with),
                       "win_rate": round(len(wins_with) / len(elig_with), 3) if elig_with else 0.0}

    high = [ns for ns, d in per_ns.items() if d["recommendation"] == "HIGH_YIELD_DYNAMIC_TARGET"]
    moderate = [ns for ns, d in per_ns.items() if d["recommendation"] == "MODERATE_YIELD_DYNAMIC_TARGET"]
    out = {"generated_by": "scripts/rc5v3_namespace_feature_yield.py",
           "per_namespace": per_ns, "per_feature": per_feat,
           "high_yield_namespaces": high, "moderate_yield_namespaces": moderate}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V3 namespace + feature yield", "",
          "## Per namespace", "",
          "| ns | batch | eligible | dyn wins | plan probes | probes/win | win/elig | recommendation |",
          "|---|---|---|---|---|---|---|---|"]
    for ns in sorted(per_ns, key=lambda n: -per_ns[n]["dynamic_wins"]):
        d = per_ns[ns]
        md.append(f"| {ns} | {d['batch']} | {d['dynamic_eligible']} | {d['dynamic_wins']} | "
                  f"{d['plan_probes']} | {d['probes_per_win']} | {d['win_rate_over_eligible']} | {d['recommendation']} |")
    md += ["", "## Per feature", "", "| feature | eligible | wins | win rate |", "|---|---|---|---|"]
    for f, d in per_feat.items():
        md.append(f"| {f} | {d['eligible']} | {d['wins']} | {d['win_rate']} |")
    md += ["", f"- HIGH yield: {high} | MODERATE yield: {moderate}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-yield] high={high} moderate={moderate}")
    print(f"[rc5v3-yield] wins_by_ns={dict(wins_by_ns)}")


if __name__ == "__main__":
    main()
