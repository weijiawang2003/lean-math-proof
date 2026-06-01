#!/usr/bin/env python3
"""TR5 Part 9 — RC4B / RC4C candidate evidence analysis.

Aggregates the live-confirmed wins into evidence for the two next candidate families:
RC4B (Set.disjoint_left bridge) and RC4C (d2_simp_aesop = retrieval-depth simp[L]<;>aesop).
For each: count true wins, reproduced-TR3 vs fresh, false-positive / off-gate risk, and
recommend whether a separate literal-RC2⊕candidate validation is warranted. Does NOT
create any RC4B/RC4C validation artifact.
"""
from __future__ import annotations

import argparse
import json
import os

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
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    attr = _load(args.attribution)
    records = {r["full_name"]: r for r in attr["records"]}
    plan = {t["full_name"]: t for t in _load(args.ranked_plan)["theorems"]}
    tr3_winners = {r["full_name"]: r["best_win"] for r in
                   (_load("project/evolve/experiments/tr3/out/tr3_depth_program_results.json") or {}).get("results", [])
                   if r.get("best_win")}

    # how often each candidate-family program FIRED (was attempted) but FAILED — off-gate proxy
    def family_fire_stats(predicate):
        fired = closed = 0
        for fn, r in records.items():
            wp = r.get("winning_program")
            if wp and predicate(wp) and r.get("credited"):
                closed += 1
        # attempted-but-not-winning count from plan top-5 (proxy for over-firing)
        attempted = 0
        for fn, t in plan.items():
            for p in t.get("programs_ranked", []):
                if p["rank"] <= 5 and predicate(p):
                    attempted += 1
        return {"attempted_in_top5": attempted, "closed_as_credited": closed}

    def is_disjoint_left(p):
        return any("disjoint_left" in (L or "") for L in p.get("used_lemmas", []))

    def is_d2_aesop(p):
        return p.get("family") == "d2_simp_aesop"

    # RC4B
    rc4b_wins = [r for r in records.values() if r.get("rc4b_evidence")]
    rc4b_repro = [r for r in rc4b_wins if r.get("reproduces_tr3_win")]
    rc4b_fresh = [r for r in rc4b_wins if not r.get("reproduces_tr3_win")]
    rc4b_fire = family_fire_stats(is_disjoint_left)
    rc4b = {
        "candidate": "RC4B_set_disjoint_left",
        "true_wins": len(rc4b_wins),
        "win_targets": [r["full_name"] for r in rc4b_wins],
        "reproduced_tr3": [r["full_name"] for r in rc4b_repro],
        "fresh_wins": [r["full_name"] for r in rc4b_fresh],
        "fire_stats": rc4b_fire,
        "off_gate_risk": ("low — simp[Set.disjoint_left] is a single named-lemma rewrite "
                          "gated to Set goals; only fires when the lemma is retrieved"),
        "candidate_policy_suggestion": ("narrow allowlist gate: add `simp [Set.disjoint_left]` "
                                        "(and the d2 `simp [Set.disjoint_left] <;> aesop`) to the "
                                        "Set route battery, gated to goals mentioning Disjoint; "
                                        "off-by-default, additive over RC2 (SET_ITE_SIMP / RC4A pattern)"),
        "validation_set_suggestion": ("the live Set.disjoint_* wins + held-out Disjoint/disjoint_left "
                                       "Set theorems from the discovered catalog as fresh holdouts"),
    }
    # RC4C
    rc4c_wins = [r for r in records.values() if r.get("rc4c_evidence")]
    rc4c_repro = [r for r in rc4c_wins if r.get("reproduces_tr3_win")]
    rc4c_fresh = [r for r in rc4c_wins if not r.get("reproduces_tr3_win")]
    rc4c_fire = family_fire_stats(is_d2_aesop)
    # false positives = d2_simp_aesop rank<=5 programs that were attempted live and failed
    b5 = _load("project/evolve/experiments/tr5/out/tr5_b5_live_results.json")
    rc4c_fp = 0
    if b5:
        for r in b5["results"]:
            for f in r.get("failures", []):
                if f.get("family") == "d2_simp_aesop":
                    rc4c_fp += 1
    rc4c = {
        "candidate": "RC4C_d2_simp_aesop",
        "true_wins": len(rc4c_wins),
        "win_targets": [r["full_name"] for r in rc4c_wins],
        "reproduced_tr3": [r["full_name"] for r in rc4c_repro],
        "fresh_wins": [r["full_name"] for r in rc4c_fresh],
        "fire_stats": rc4c_fire,
        "false_positives_live_b5": rc4c_fp,
        "source_specific_risk": ("medium — the win depends on the retrieved lemma L being the "
                                 "right bridge; aesop after simp[L] can also close goals where "
                                 "plain aesop times out, so the credit is the simp[L] enabling step "
                                 "(SX4 PRODUCTION_SUBSUMED guard already applied — RC2's best-first "
                                 "search does NOT reach the simp[L]-advanced state)"),
        "separate_validation_warranted": len(rc4c_wins) >= 2,
    }

    def decide(cand, fresh, fp=0):
        n = cand["true_wins"]
        if n == 0:
            return "REJECT"
        if n >= 2 and (fresh or n >= 3):
            return "READY_FOR_RC4B_VALIDATION" if "RC4B" in cand["candidate"] else "READY_FOR_RC4C_VALIDATION"
        if n >= 1:
            return "NEED_MORE_EVIDENCE"
        return "TRAINING_ONLY"

    rc4b["decision"] = decide(rc4b, rc4b["fresh_wins"])
    rc4c["decision"] = decide(rc4c, rc4c["fresh_wins"], rc4c["false_positives_live_b5"])

    out = {"generated_by": "scripts/tr5_candidate_evidence_analysis.py",
           "rc4b": rc4b, "rc4c": rc4c}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR5 RC4B / RC4C evidence", "",
          "## RC4B — `Set.disjoint_left` bridge",
          f"- decision: **{rc4b['decision']}**",
          f"- true wins: {rc4b['true_wins']} → {rc4b['win_targets']}",
          f"- reproduced TR3: {rc4b['reproduced_tr3']} | fresh: {rc4b['fresh_wins']}",
          f"- fire stats (top-5): {rc4b['fire_stats']}",
          f"- off-gate risk: {rc4b['off_gate_risk']}",
          f"- candidate policy: {rc4b['candidate_policy_suggestion']}",
          f"- validation set: {rc4b['validation_set_suggestion']}", "",
          "## RC4C — `d2_simp_aesop` (`simp [L] <;> aesop`)",
          f"- decision: **{rc4c['decision']}**",
          f"- true wins: {rc4c['true_wins']} → {rc4c['win_targets']}",
          f"- reproduced TR3: {rc4c['reproduced_tr3']} | fresh: {rc4c['fresh_wins']}",
          f"- fire stats (top-5): {rc4c['fire_stats']}",
          f"- false positives (live B5 d2_simp_aesop fails): {rc4c['false_positives_live_b5']}",
          f"- source-specific risk: {rc4c['source_specific_risk']}",
          f"- separate validation warranted: {rc4c['separate_validation_warranted']}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr5-evidence] RC4B={rc4b['decision']}({rc4b['true_wins']}) "
          f"RC4C={rc4c['decision']}({rc4c['true_wins']})")


if __name__ == "__main__":
    main()
