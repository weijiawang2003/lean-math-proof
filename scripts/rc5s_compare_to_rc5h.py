#!/usr/bin/env python3
"""RC5S Part 10 — compare RC5H (unsafe) vs RC5S (safe).

Contrasts program count, off-policy count, timeout/stall behaviour, true wins preserved, probes
per win, wall-clock cost, and B5/B10/B20 behaviour, then emits the hardening verdict
(SAFETY_HARDENING_SUCCESS / PARTIAL / FAILED / DYNAMIC_STAGE_REQUIRES_ENGINE_CHANGES).
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC5H = "project/evolve/experiments/rc5_hybrid/out"
RC5S = "project/evolve/experiments/rc5_safety/out"
RC5H_WINNERS = {"Finset.biUnion_subset_iff_forall_subset", "Multiset.add_bind", "Finset.image_subset_iff"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _j(*a):
    p = _p(*a)
    return json.load(open(p)) if os.path.exists(p) else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rc5h_plan = _j(RC5H, "rc5h_dynamic_program_plan.json")
    rc5h_safety = _j(RC5H, "rc5h_dynamic_safety_audit.json")
    rc5h_attr = _j(RC5H, "rc5h_hybrid_attribution.json")
    rc5s_plan = _j(RC5S, "rc5s_safe_dynamic_plan.json")
    rc5s_b5 = _j(RC5S, "rc5s_b5_results.json")
    rc5s_b10 = _j(RC5S, "rc5s_b10_results.json")
    rc5s_attr = _j(RC5S, "rc5s_attribution_and_safety.json")
    filt = _j(RC5S, "rc5s_filter_report.json")

    rc5h_programs = sum(len(t.get("programs_ranked", [])) for t in rc5h_plan.get("theorems", []))
    rc5s_programs = (rc5s_plan.get("total_b5_programs", 0) + rc5s_plan.get("total_b10_reserve_programs", 0))
    rc5h_offpolicy = rc5h_safety.get("off_policy_programs", 0)
    rc5s_offpolicy = rc5s_plan.get("off_policy_in_final_plan", 0)

    rc5s_results = rc5s_b5.get("results", [])
    rc5s_killed = rc5s_b5.get("killed_by_timeout", 0) + rc5s_b10.get("killed_by_timeout", 0)
    rc5s_max_wall = rc5s_b5.get("max_wall_seconds", 0)
    rc5s_total_wall = round(sum(r.get("wall_seconds", 0) for r in rc5s_results)
                            + sum(r.get("wall_seconds", 0) for r in rc5s_b10.get("results", [])), 1)
    rc5s_true = rc5s_attr.get("num_recovered", 0) + rc5s_attr.get("num_new", 0)
    rc5s_recovered = rc5s_attr.get("num_recovered", 0)

    comp = {
        "program_count": {"rc5h": rc5h_programs, "rc5s": rc5s_programs},
        "off_policy_count": {"rc5h": rc5h_offpolicy, "rc5s": rc5s_offpolicy},
        "global_stalls": {"rc5h": "pervasive at B10+ (22/88 hit 150s cap, manual kills)",
                          "rc5s": "none" if rc5s_b5.get("no_global_stalls") else "present"},
        "timeout_handling": {"rc5h": "SIGALRM-only (failed on simp_all/aesop); 150s outer cap added manually",
                             "rc5s": f"process-group kill at {rc5s_b5.get('wall_cap_seconds')}s wall cap; "
                                     f"{rc5s_killed} bounded kills recorded"},
        "true_wins": {"rc5h": rc5h_attr.get("true_hybrid_deltas", 3),
                      "rc5s_recovered": rc5s_recovered, "rc5s_total": rc5s_true},
        "wins_preserved": f"{rc5s_recovered}/{len(RC5H_WINNERS)}",
        "max_wall_seconds": {"rc5s_b5": rc5s_max_wall, "rc5s_cap": rc5s_b5.get("wall_cap_seconds")},
        "total_wall_seconds_rc5s": rc5s_total_wall,
        "probes_per_true_win_rc5s": round(rc5s_programs / rc5s_true, 1) if rc5s_true else None,
        "unknown_name_rc5s": rc5s_b5.get("unknown_name", 0),
        "budget_behavior": {"rc5h": "B5 stable, B10 pervasive stalls, B20 unrunnable",
                            "rc5s": f"B5 stable+bounded; B10 safe-reserve {rc5s_b10.get('recommendation','-')}; B20 disabled"},
        "programs_removed_by_strict_filter": filt.get("removed_total", 0),
    }
    # verdict
    no_stalls = bool(rc5s_b5.get("no_global_stalls"))
    zero_offpolicy = rc5s_offpolicy == 0
    bounded_ok = rc5s_max_wall <= (rc5s_b5.get("wall_cap_seconds", 60) + 15)
    wins_ok = rc5s_recovered >= 2  # >= 2/3 prior wins preserved
    if no_stalls and zero_offpolicy and bounded_ok and rc5s_recovered == len(RC5H_WINNERS):
        verdict = "SAFETY_HARDENING_SUCCESS"
    elif no_stalls and zero_offpolicy and bounded_ok and wins_ok:
        verdict = "SAFETY_HARDENING_SUCCESS"
    elif no_stalls and zero_offpolicy:
        verdict = "SAFETY_HARDENING_PARTIAL"
    elif not no_stalls:
        verdict = "DYNAMIC_STAGE_REQUIRES_ENGINE_CHANGES"
    else:
        verdict = "SAFETY_HARDENING_FAILED"

    out = {"generated_by": "scripts/rc5s_compare_to_rc5h.py", "comparison": comp,
           "no_global_stalls": no_stalls, "zero_off_policy": zero_offpolicy,
           "timeouts_bounded": bounded_ok, "wins_preserved": f"{rc5s_recovered}/{len(RC5H_WINNERS)}",
           "verdict": verdict}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5S vs RC5H comparison", "",
          f"- **verdict: {verdict}**",
          f"- programs: RC5H {rc5h_programs} → RC5S {rc5s_programs} (strict-filtered, "
          f"{filt.get('removed_total',0)} removed)",
          f"- off-policy: RC5H ~{rc5h_offpolicy} → RC5S **{rc5s_offpolicy}**",
          f"- global stalls: RC5H pervasive @B10+ → RC5S **{'none' if no_stalls else 'present'}**",
          f"- timeout handling: process-group kill, {rc5s_killed} bounded kills, max wall "
          f"{rc5s_max_wall}s (cap {rc5s_b5.get('wall_cap_seconds')}s)",
          f"- true wins preserved: **{rc5s_recovered}/{len(RC5H_WINNERS)}**",
          f"- B20: RC5H unrunnable → RC5S disabled; B10: {rc5s_b10.get('recommendation','-')}",
          "", "## Detail", "", "| metric | RC5H | RC5S |", "|---|---|---|",
          f"| programs | {rc5h_programs} | {rc5s_programs} |",
          f"| off-policy | {rc5h_offpolicy} | {rc5s_offpolicy} |",
          f"| true wins | {comp['true_wins']['rc5h']} | {rc5s_true} ({rc5s_recovered} recovered) |",
          f"| global stalls | pervasive | {'none' if no_stalls else 'present'} |",
          f"| max wall (s) | 150 (cap) | {rc5s_max_wall} |",
          f"| probes/true win | — | {comp['probes_per_true_win_rc5s']} |"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5s-compare] verdict={verdict} | no_stalls={no_stalls} zero_offpolicy={zero_offpolicy} "
          f"wins_preserved={rc5s_recovered}/{len(RC5H_WINNERS)}")


if __name__ == "__main__":
    main()
