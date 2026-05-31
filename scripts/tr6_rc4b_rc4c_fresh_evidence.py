#!/usr/bin/env python3
"""TR6 Part 11 — RC4B / RC4C fresh-holdout evidence.

Aggregates fresh live wins into RC4B (Set.disjoint_left bridge) and RC4C (d2_simp_aesop)
evidence: fresh true wins, fresh failures where the gate fired but failed (off-gate /
miss signal), namespace coverage, source-specific risk, and a validation-readiness
decision. These are FRESH holdouts (excluded from TR3/TR5), so a positive decision is
fresh-supported, not reproduction-only. Does NOT create any RC4B/RC4C artifact.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    fp = _p(path) if not os.path.isabs(path) else path
    return json.load(open(fp)) if os.path.exists(fp) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--ranked-plan", required=True)
    ap.add_argument("--b5", default="project/evolve/experiments/tr6/out/tr6_b5_live_results.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    attr = {r["full_name"]: r for r in _load(args.attribution)["records"]}
    plan = {t["full_name"]: t for t in _load(args.ranked_plan)["theorems"]}
    b5 = _load(args.b5)

    # gate-fire counts in top-5: how often each candidate appeared and whether it closed
    def fire_stats(pred_lemma=None, pred_family=None):
        fired_top5 = closed = fired_but_failed = 0
        for fn, t in plan.items():
            top5 = [p for p in t.get("programs_ranked", []) if p["rank"] <= 5]
            present = False
            for p in top5:
                lemmas = p.get("used_lemmas") or p.get("lemmas") or []
                ok = (pred_lemma and any(pred_lemma in (L or "") for L in lemmas)) or \
                     (pred_family and p.get("family") == pred_family)
                if ok:
                    present = True
            if present:
                fired_top5 += 1
                a = attr.get(fn, {})
                wp = a.get("winning_program") if a.get("credited") else None
                wl = (wp.get("used_lemmas") or wp.get("lemmas") or []) if wp else []
                won = wp and ((pred_lemma and any(pred_lemma in (L or "") for L in wl)) or
                              (pred_family and wp.get("family") == pred_family))
                if won:
                    closed += 1
                else:
                    fired_but_failed += 1
        return {"fired_in_top5": fired_top5, "closed_as_credited": closed,
                "fired_but_not_winning": fired_but_failed}

    rc4b_wins = [r for r in attr.values() if r.get("rc4b_evidence")]
    rc4c_wins = [r for r in attr.values() if r.get("rc4c_evidence")]
    rc4b_fire = fire_stats(pred_lemma="disjoint_left")
    rc4c_fire = fire_stats(pred_family="d2_simp_aesop")

    rc4b = {
        "candidate": "RC4B_set_disjoint_left",
        "fresh_true_wins": len(rc4b_wins),
        "fresh_win_targets": [r["full_name"] for r in rc4b_wins],
        "win_namespaces": dict(Counter(r["namespace"] for r in rc4b_wins)),
        "fire_stats": rc4b_fire,
        "off_gate_risk": ("low — single named-lemma rewrite gated to disjoint-shaped goals; "
                          f"fired in top-5 on {rc4b_fire['fired_in_top5']} theorems, "
                          f"closed {rc4b_fire['closed_as_credited']}, "
                          f"fired-but-failed {rc4b_fire['fired_but_not_winning']}"),
        "validation_set_suggestion": "fresh Set/Finset/List disjoint_* wins + held-out disjoint goals",
    }
    rc4c = {
        "candidate": "RC4C_d2_simp_aesop",
        "fresh_true_wins": len(rc4c_wins),
        "fresh_win_targets": [r["full_name"] for r in rc4c_wins],
        "win_namespaces": dict(Counter(r["namespace"] for r in rc4c_wins)),
        "fire_stats": rc4c_fire,
        "overlap_with_rc4b": [r["full_name"] for r in rc4c_wins if r.get("rc4b_evidence")],
        "source_specific_risk": ("medium — credit is the simp[L] enabling step; SX4 "
                                 "PRODUCTION_SUBSUMED guard applied (RC2 confirmed-failure)"),
    }

    def decide(c):
        n = c["fresh_true_wins"]
        fresh_nonrepro = n  # all TR6 wins are fresh by registry construction
        if n == 0:
            return "REJECT" if c["fire_stats"]["fired_in_top5"] == 0 else "NEED_MORE_FRESH_EVIDENCE"
        if n >= 2:
            return "READY_FOR_LITERAL_VALIDATION_WITH_FRESH_SUPPORT"
        return "NEED_MORE_FRESH_EVIDENCE"

    rc4b["decision"] = decide(rc4b)
    rc4c["decision"] = decide(rc4c)

    out = {"generated_by": "scripts/tr6_rc4b_rc4c_fresh_evidence.py", "rc4b": rc4b, "rc4c": rc4c}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 RC4B / RC4C fresh-holdout evidence", "",
          "## RC4B — `Set.disjoint_left` bridge",
          f"- decision: **{rc4b['decision']}**",
          f"- fresh true wins: {rc4b['fresh_true_wins']} → {rc4b['fresh_win_targets']}",
          f"- win namespaces: {rc4b['win_namespaces']}",
          f"- fire stats (top-5): {rc4b['fire_stats']}",
          f"- off-gate risk: {rc4b['off_gate_risk']}", "",
          "## RC4C — `d2_simp_aesop`",
          f"- decision: **{rc4c['decision']}**",
          f"- fresh true wins: {rc4c['fresh_true_wins']} → {rc4c['fresh_win_targets']}",
          f"- win namespaces: {rc4c['win_namespaces']}",
          f"- fire stats (top-5): {rc4c['fire_stats']}",
          f"- overlap with RC4B: {rc4c['overlap_with_rc4b']}",
          f"- source-specific risk: {rc4c['source_specific_risk']}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-rc4bc] RC4B={rc4b['decision']}({rc4b['fresh_true_wins']}) "
          f"RC4C={rc4c['decision']}({rc4c['fresh_true_wins']})")


if __name__ == "__main__":
    main()
