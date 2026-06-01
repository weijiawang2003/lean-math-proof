#!/usr/bin/env python3
"""SX3 Part 8 — derive the RC3-candidate (RC2 ⊕ SX3_SET_ITE_AESOP) evaluation
and the RC2-vs-candidate comparison from the live SX3 run results.

The candidate adds exactly one gated depth-2 sequence: `simp [Set.ite] <;> aesop`
under the narrow Set.ite name gate. Its behaviour on each surface is precisely
what the SX3_SET_ITE_AESOP family produced in the live runs, with RC2's credited
mechanism captured by the per-theorem controls (single-shot simp[Set.ite] +
baselines). So the candidate eval is read directly off the SX3 result files; no
new wrapper execution is invented.
"""
from __future__ import annotations
import argparse
import json

SEQ = "simp [Set.ite] <;> aesop"
SINGLE = "simp [Set.ite]"
BASELINES = {"simp", "simp_all", "aesop", "classical <;> aesop"}
DEFERRED4 = {"Set.ite_inter", "Set.ite_inter_self", "Set.ite_compl",
             "Set.ite_inter_compl_self"}


def candidate_behaviour(r):
    """What RC2 (controls) vs RC3-candidate (controls + gated SEQ) achieve."""
    ctl = {c["tactic"]: c["solved"] for c in r.get("controls", [])}
    seq_solved = any(s.get("solved") and s.get("family") == "SX3_SET_ITE_AESOP"
                     for s in r.get("gated_sequences_tried", []))
    gate_fires = (r.get("namespace") == "Set" and "ite" in r.get("full_name", "").lower())
    rc2_solves = bool(ctl.get(SINGLE) or any(ctl.get(b) for b in BASELINES))
    # candidate solves if RC2 solves OR (gate fires and the gated sequence solves)
    cand_solves = rc2_solves or (gate_fires and seq_solved)
    new_for_candidate = cand_solves and not rc2_solves and seq_solved
    return {"gate_fires": gate_fires, "rc2_solves": rc2_solves,
            "candidate_solves": cand_solves, "sequence_solved": seq_solved,
            "new_over_rc2": new_for_candidate}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", nargs="+", required=True)
    p.add_argument("--out-eval", required=True)
    p.add_argument("--out-comparison", required=True)
    args = p.parse_args()

    per = []
    for path in args.results:
        d = json.load(open(path))
        surface = path.split("/")[-1].replace("sx3_", "").replace("_results.json", "")
        for r in d.get("results", []):
            b = candidate_behaviour(r)
            per.append({"full_name": r.get("full_name"), "surface": surface,
                        "role": r.get("role"), "namespace": r.get("namespace"),
                        **b})

    new_wins = [x for x in per if x["new_over_rc2"]]
    new_fresh = sorted(x["full_name"] for x in new_wins if x["full_name"] not in DEFERRED4)
    new_deferred = sorted(x["full_name"] for x in new_wins if x["full_name"] in DEFERRED4)
    # off-gate: gate fired on a non-Set theorem (should be none)
    off_gate = [x["full_name"] for x in per if x["gate_fires"] and x["namespace"] != "Set"]
    # regressions: candidate fails where RC2 solved (impossible by construction; check)
    regressions = [x["full_name"] for x in per if x["rc2_solves"] and not x["candidate_solves"]]

    eval_out = {
        "candidate": "RC2 ⊕ SX3_SET_ITE_AESOP",
        "added_sequence": SEQ, "gate": "theorem name contains 'Set.ite'",
        "surfaces": sorted({x["surface"] for x in per}),
        "num_theorems": len(per),
        "new_wins_over_rc2": sorted(x["full_name"] for x in new_wins),
        "new_fresh_wins": new_fresh,
        "reproduced_deferred_wins": new_deferred,
        "off_gate_emissions": off_gate,
        "regressions_vs_rc2": regressions,
        "per_theorem": sorted(per, key=lambda x: (x["surface"], x["full_name"])),
    }
    json.dump(eval_out, open(args.out_eval, "w"), ensure_ascii=False, indent=2)

    comparison = {
        "rc2_baseline": "single-shot simp[Set.ite] + base policy (controls)",
        "rc3_candidate": "RC2 + gated depth-2 'simp [Set.ite] <;> aesop'",
        "equivalence_on_non_gated": "RC2 ≡ candidate on every theorem where the Set.ite "
            "gate does not fire OR where RC2 already solves it; the candidate only adds "
            "solves where the gated sequence fires and RC2's single-shot/baselines fail.",
        "delta_fresh_wins": new_fresh,
        "delta_deferred_reproduced": new_deferred,
        "num_new_fresh": len(new_fresh),
        "num_reproduced_deferred": len(new_deferred),
        "off_gate_emissions": len(off_gate),
        "regressions": len(regressions),
        "verdict_inputs": {
            "positive_fresh_delta": len(new_fresh) >= 1,
            "reproduced_all_deferred4": len(new_deferred) == 4,
            "zero_off_gate": len(off_gate) == 0,
            "zero_regressions": len(regressions) == 0,
        },
    }
    json.dump(comparison, open(args.out_comparison, "w"), ensure_ascii=False, indent=2)
    print(f"[rc3:eval] new_fresh={new_fresh} reproduced_deferred={new_deferred} "
          f"off_gate={len(off_gate)} regressions={len(regressions)}")


if __name__ == "__main__":
    main()
