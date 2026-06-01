#!/usr/bin/env python3
"""RC5V3 Part 11 — RC2 vs RC4 vs RC5V3-B1/B3/B5 comparison + B1/B3/B5 cost curve.

RC5V3-Bk solved = RC4_solved ∪ {fresh true deltas achieved within budget k}. Cost curve reports
cumulative probes + cumulative/marginal fresh wins + marginal probes/win for each budget, then
recommends a budget (B1_ONLY / B3_RECOMMENDED / B5_RECOMMENDED / B5_TOO_EXPENSIVE).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _solved(path):
    return {r["full_name"] for r in json.load(open(_p(path)))["results"] if r.get("status") == "solved"}


def _probes(path):
    return json.load(open(_p(path))).get("programs_attempted", 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--static-results", required=True)
    ap.add_argument("--dynamic-b1", required=True)
    ap.add_argument("--dynamic-b3", required=True)
    ap.add_argument("--dynamic-b5", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out-comparison-json", required=True)
    ap.add_argument("--out-cost-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc2_solved = _solved(args.rc2_results)
    rc4_solved = _solved(args.static_results)
    attr = json.load(open(_p(args.attribution)))
    fresh = [r for r in attr["records"] if r["classification"] == "FRESH_TRUE_RC5V3_DELTA"]
    fresh_b1 = {r["full_name"] for r in fresh if r["budget_solved"] == "B1"}
    fresh_b3 = {r["full_name"] for r in fresh if r["budget_solved"] == "B3"}
    fresh_b5 = {r["full_name"] for r in fresh if r["budget_solved"] == "B5"}

    p_b1 = _probes(args.dynamic_b1)
    p_b3 = _probes(args.dynamic_b3)
    p_b5 = _probes(args.dynamic_b5)

    rc5_b1 = rc4_solved | fresh_b1
    rc5_b3 = rc5_b1 | fresh_b3
    rc5_b5 = rc5_b3 | fresh_b5

    systems = {
        "RC2": {"solved": len(rc2_solved)},
        "RC4_static": {"solved": len(rc4_solved), "delta_over_rc2": len(rc4_solved - rc2_solved)},
        "RC5V3_B1": {"solved": len(rc5_b1), "delta_over_rc4": len(rc5_b1 - rc4_solved),
                     "delta_over_rc2": len(rc5_b1 - rc2_solved),
                     "regressions": len(rc4_solved - rc5_b1)},
        "RC5V3_B3": {"solved": len(rc5_b3), "delta_over_rc4": len(rc5_b3 - rc4_solved),
                     "delta_over_rc2": len(rc5_b3 - rc2_solved),
                     "regressions": len(rc4_solved - rc5_b3)},
        "RC5V3_B5": {"solved": len(rc5_b5), "delta_over_rc4": len(rc5_b5 - rc4_solved),
                     "delta_over_rc2": len(rc5_b5 - rc2_solved),
                     "regressions": len(rc4_solved - rc5_b5)},
    }

    def ppw(probes, wins):
        return round(probes / wins, 1) if wins else None

    w_b1, w_b3, w_b5 = len(fresh_b1), len(fresh_b3), len(fresh_b5)
    cum_p1, cum_p3, cum_p5 = p_b1, p_b1 + p_b3, p_b1 + p_b3 + p_b5
    cum_w1, cum_w3, cum_w5 = w_b1, w_b1 + w_b3, w_b1 + w_b3 + w_b5
    cost_curve = {
        "B1": {"probes": p_b1, "marginal_wins": w_b1, "cumulative_probes": cum_p1,
               "cumulative_wins": cum_w1, "marginal_probes_per_win": ppw(p_b1, w_b1),
               "cumulative_probes_per_win": ppw(cum_p1, cum_w1)},
        "B3": {"probes": p_b3, "marginal_wins": w_b3, "cumulative_probes": cum_p3,
               "cumulative_wins": cum_w3, "marginal_probes_per_win": ppw(p_b3, w_b3),
               "cumulative_probes_per_win": ppw(cum_p3, cum_w3)},
        "B5": {"probes": p_b5, "marginal_wins": w_b5, "cumulative_probes": cum_p5,
               "cumulative_wins": cum_w5, "marginal_probes_per_win": ppw(p_b5, w_b5),
               "cumulative_probes_per_win": ppw(cum_p5, cum_w5)},
    }

    # budget recommendation
    EXPENSIVE = 250  # marginal probes/win threshold above which a budget tier is "too expensive"
    if cum_w5 == 0:
        rec = "B5_TOO_EXPENSIVE"
    elif w_b3 == 0 and w_b5 == 0:
        rec = "B1_ONLY"
    elif w_b5 > 0 and (ppw(p_b5, w_b5) or 1e9) <= EXPENSIVE:
        rec = "B5_RECOMMENDED"
    elif w_b3 > 0 and (ppw(p_b3, w_b3) or 1e9) <= EXPENSIVE:
        rec = "B3_RECOMMENDED"
    elif w_b1 > 0:
        rec = "B1_ONLY"
    else:
        rec = "B5_TOO_EXPENSIVE"

    by_ns = Counter(r["namespace"] for r in fresh)
    comparison = {"generated_by": "scripts/rc5v3_compare_systems_and_cost.py",
                  "systems": systems,
                  "fresh_dynamic_delta_over_rc4": len(fresh),
                  "fresh_delta_by_budget": {"B1": w_b1, "B3": w_b3, "B5": w_b5},
                  "fresh_delta_by_namespace": dict(by_ns),
                  "total_dynamic_probes": cum_p5,
                  "total_probes_per_fresh_win": ppw(cum_p5, cum_w5),
                  "rc4_remains_static_core": True,
                  "safe_dynamic_gives_fresh_gain": len(fresh) > 0,
                  "recommended_budget": rec}
    json.dump(comparison, open(_p(args.out_comparison_json), "w"), ensure_ascii=False, indent=2)
    json.dump({"generated_by": "scripts/rc5v3_compare_systems_and_cost.py",
               "cost_curve": cost_curve, "recommended_budget": rec,
               "total_probes_per_fresh_win": ppw(cum_p5, cum_w5)},
              open(_p(args.out_cost_json), "w"), ensure_ascii=False, indent=2)

    md = ["# RC5V3 system comparison + cost curve", "",
          "| system | solved | Δ/RC2 | Δ/RC4 | regr |", "|---|---|---|---|---|",
          f"| RC2 | {systems['RC2']['solved']} | — | — | — |",
          f"| RC4 static | {systems['RC4_static']['solved']} | {systems['RC4_static']['delta_over_rc2']} | 0 | 0 |",
          f"| RC5V3-B1 | {systems['RC5V3_B1']['solved']} | {systems['RC5V3_B1']['delta_over_rc2']} | "
          f"**{systems['RC5V3_B1']['delta_over_rc4']}** | {systems['RC5V3_B1']['regressions']} |",
          f"| RC5V3-B3 | {systems['RC5V3_B3']['solved']} | {systems['RC5V3_B3']['delta_over_rc2']} | "
          f"**{systems['RC5V3_B3']['delta_over_rc4']}** | {systems['RC5V3_B3']['regressions']} |",
          f"| RC5V3-B5 | {systems['RC5V3_B5']['solved']} | {systems['RC5V3_B5']['delta_over_rc2']} | "
          f"**{systems['RC5V3_B5']['delta_over_rc4']}** | {systems['RC5V3_B5']['regressions']} |", "",
          "## Cost curve", "",
          "| budget | probes | marginal wins | cum probes | cum wins | marginal probes/win | cum probes/win |",
          "|---|---|---|---|---|---|---|"]
    for b in ("B1", "B3", "B5"):
        c = cost_curve[b]
        md.append(f"| {b} | {c['probes']} | {c['marginal_wins']} | {c['cumulative_probes']} | "
                  f"{c['cumulative_wins']} | {c['marginal_probes_per_win']} | {c['cumulative_probes_per_win']} |")
    md += ["", f"- **fresh delta over RC4 (B5): {len(fresh)}** | by namespace: {dict(by_ns)}",
           f"- total dynamic probes: {cum_p5} | total probes/fresh win: {ppw(cum_p5, cum_w5)}",
           f"- **recommended budget: {rec}**"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-compare] RC2={systems['RC2']['solved']} RC4={systems['RC4_static']['solved']} "
          f"RC5V3-B5={systems['RC5V3_B5']['solved']} fresh_delta={len(fresh)} "
          f"by_budget=B1:{w_b1}/B3:{w_b3}/B5:{w_b5} rec={rec}")


if __name__ == "__main__":
    main()
