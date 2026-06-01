#!/usr/bin/env python3
"""TR4 Part 6 — probe-budget simulation.

Simulates running only the top-B programs per theorem (by each ranker) over the TR3
program lists, counting successes/credited recovered and probes run vs the full
4,377-program TR3 search. Decision: RANKER_USEFUL_FOR_PROBE_REDUCTION if the best model
recovers ≥70% of successes using ≤20% of probes.
"""
from __future__ import annotations

import argparse
import json
import os
import math

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranking-eval", required=True)
    ap.add_argument("--examples", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    re_ = json.load(open(_p(args.ranking_eval)))
    pt = re_["per_theorem"]
    models = [m for m in next(iter(pt))["best_success_rank"].keys()
              if m not in ("random_expected",)] if pt else []

    total_programs = sum(t["num_programs"] for t in pt)
    total_succ = sum(1 for t in pt if t["num_successes"] > 0)
    total_cred = sum(1 for t in pt if t["num_credited"] > 0)

    abs_budgets = [1, 3, 5, 10, 20]
    pct_budgets = [0.05, 0.10, 0.20]

    def _sim(model, B=None, pct=None):
        succ = cred = progs = 0
        for t in pt:
            n = t["num_programs"]
            b = B if B is not None else max(1, math.ceil(pct * n))
            progs += min(b, n)
            sr = t["best_success_rank"].get(model)
            cr = t["best_credit_rank"].get(model)
            if sr is not None and sr <= b:
                succ += 1
            if cr is not None and cr <= b:
                cred += 1
        return {"successes_recovered": succ, "credited_recovered": cred,
                "programs_run": progs,
                "probe_reduction_pct": round(100 * (1 - progs / max(1, total_programs)), 1),
                "success_recovery_frac": round(succ / max(1, total_succ), 3),
                "credit_recovery_frac": round(cred / max(1, total_cred), 3)}

    sims = {}
    for m in models:
        sims[m] = {f"B={b}": _sim(m, B=b) for b in abs_budgets}
        sims[m].update({f"top{int(p*100)}%": _sim(m, pct=p) for p in pct_budgets})

    # decision: best model recovers >=70% successes with <=20% probes
    best_model, best = None, None
    for m in models:
        if m in ("original_order",):
            continue
        s = sims[m]["top20%"]
        if s["success_recovery_frac"] >= 0.70:
            if best is None or s["success_recovery_frac"] > best:
                best, best_model = s["success_recovery_frac"], m
    # also check small absolute budget B=5
    useful = best_model is not None
    decision = "RANKER_USEFUL_FOR_PROBE_REDUCTION" if useful else "RANKER_NOT_YET_USEFUL"

    out = {"generated_by": "scripts/tr4_probe_budget_simulation.py",
           "total_programs": total_programs, "total_theorems": len(pt),
           "total_theorems_with_success": total_succ,
           "total_theorems_with_credit": total_cred,
           "simulations": sims, "best_model_top20pct": best_model,
           "decision": decision}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR4 probe-budget simulation", "",
          f"- total programs (TR3): {total_programs} | theorems: {len(pt)} | "
          f"with success: {total_succ}",
          f"- **decision: {decision}** (best @top20%: {best_model})", ""]
    for m in models:
        if m == "original_order":
            continue
        md.append(f"## {m}")
        md.append("| budget | successes | success_frac | credited | programs_run | probe_reduction |")
        md.append("|---|---|---|---|---|---|")
        for b, s in sims[m].items():
            md.append(f"| {b} | {s['successes_recovered']}/{total_succ} | "
                      f"{s['success_recovery_frac']} | {s['credited_recovered']}/{total_cred} | "
                      f"{s['programs_run']} | {s['probe_reduction_pct']}% |")
        md.append("")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr4-budget] decision={decision} best={best_model}; "
          f"hgb@B5={sims.get('hgb',{}).get('B=5')}")


if __name__ == "__main__":
    main()
