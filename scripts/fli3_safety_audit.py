#!/usr/bin/env python3
"""FLI3 Part 8a — safety audit: regressions / offgate / vacuity over the candidate eval."""
from __future__ import annotations
import argparse, json, os
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _p(*a): return os.path.join(_REPO, *a)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True); ap.add_argument("--candidate-results", required=True)
    ap.add_argument("--attribution", required=True); ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True); a = ap.parse_args()
    cand = json.load(open(_p(a.candidate_results)))
    attr = json.load(open(_p(a.attribution)))
    res = cand["results"]
    # offgate emissions: offgate_negative / floor / regression items where gate fired
    offgate = [r for r in res if r["set"] in ("offgate_negative", "canonical_floor", "regression_guard")
               and r["gate"]]
    # regressions: a floor/regression theorem that the candidate would break — impossible since gate
    # does not fire on them (additive design: candidate ≡ RC2). Confirm none fired.
    floor_fires = [r for r in res if r["set"] in ("canonical_floor", "regression_guard") and r["gate"]]
    # vacuity: any win where deployed lemma == theorem
    vacuous = [r for r in res if r.get("candidate_win") and r.get("lemma") == r["theorem"]]
    # gate fired on RC2-solved (would-be offgate-on-solved)
    fired_on_solved = [r for r in res if r["gate"] and r.get("rc2_status") == "solved"]
    true_delta = attr["true_fli3_delta"]
    verdict = "FLI3_SAFE" if (not offgate and not floor_fires and not vacuous and not fired_on_solved
                              and cand["regressions"] == 0) else "FLI3_SAFETY_ISSUE"
    out = {"generated_by": "scripts/fli3_safety_audit.py",
           "offgate_emissions": len(offgate), "offgate_emission_theorems": [r["theorem"] for r in offgate],
           "floor_regression_gate_fires": len(floor_fires),
           "regressions": cand["regressions"], "vacuous_wins": len(vacuous),
           "gate_fired_on_rc2_solved": len(fired_on_solved),
           "true_fli3_delta": true_delta,
           "protected_files_note": "checked separately via git diff (must be empty)",
           "verdict": verdict}
    json.dump(out, open(_p(a.out_json), "w"), indent=2)
    open(_p(a.out_md), "w").write(
        f"# FLI3 safety audit\n\n- **verdict: {verdict}**\n- offgate emissions: {len(offgate)}\n"
        f"- floor/regression gate-fires: {len(floor_fires)} | regressions: {cand['regressions']}\n"
        f"- vacuous wins: {len(vacuous)} | gate fired on RC2-solved: {len(fired_on_solved)}\n"
        f"- TRUE_FLI3_DELTA: {true_delta}\n")
    print(f"[fli3-safety] verdict={verdict} offgate={len(offgate)} regr={cand['regressions']} "
          f"vacuous={len(vacuous)} fired_on_solved={len(fired_on_solved)}")
if __name__ == "__main__": main()
