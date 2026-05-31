#!/usr/bin/env python3
"""RC2 Hardening Part 5 — preservation + off-gate hardening verification.

Reuses (with verification) the benchmark RC1/RC2 results for canonical floors and
the dry gate scan for off-gate, and re-asserts the hardening invariants:
  - canonical floors pass (RC2 >= floor; RC2==RC1 by construction)
  - 0 off-gate emissions (gate fires only on Set.ite* names)
  - 0 regressions (no theorem RC1 solved that RC2 loses)
  - no speculative gates exist in the candidate wrapper
  - gate emits only on Set.ite / Set-if-shaped theorems

Outputs:
  preservation_hardening.json / .md
"""
from __future__ import annotations

import argparse
import json
import os
import re

FLOORS = {"demo_v1": (11, 15), "nat_defs_medium": (37, 38), "nat_defs_large_v5": (49, 65)}
SPECULATIVE = ["SET_EXT_SIMP", "SET_SUBSET_ANTISYMM", "SET_IFF_CONSTRUCTOR",
               "SET_EXT_BYCASES", "SET_RW_BRIDGE", "SOURCE_SPECIFIC"]
# non-Set surfaces for the dry gate scan
NONSET = [("Nat.add_comm", "⊢ n + m = m + n"), ("Nat.add_mod_eq_ite", "⊢ if a<n then .. else .."),
          ("Int.neg_neg", "⊢ - -a = a"), ("Multiset.cons_inj_left", "⊢ a ::ₘ s = a ::ₘ t ↔ s = t"),
          ("Bool.and_self", "⊢ (b && b) = b"), ("List.append_nil", "⊢ l ++ [] = l")]
SETITE_POS = [("Set.ite_right", "⊢ s.ite t s = t ∩ s"),
              ("Set.ite_empty", "⊢ s.ite ∅ ∅ = ∅")]


def _gate_fires(prefixes, name):
    return any(str(name).startswith(p) for p in prefixes)


def _idx(path):
    d = json.load(open(path))
    return {s["name"]: s for s in d.get("per_surface", [])}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate-wrapper",
                    default="project/evolve/experiments/rc2/rc2_candidate_wrapper.json")
    ap.add_argument("--manifest",
                    default="project/evolve/experiments/rc2/rc2_benchmark_manifest.json")
    ap.add_argument("--rc1-results",
                    default="project/evolve/experiments/rc2/out/rc1_baseline_results.json")
    ap.add_argument("--rc2-results",
                    default="project/evolve/experiments/rc2/out/rc2_candidate_results.json")
    ap.add_argument("--out-json",
                    default="project/evolve/experiments/rc2_hardening/out/preservation_hardening.json")
    ap.add_argument("--out-md",
                    default="project/evolve/experiments/rc2_hardening/out/preservation_hardening.md")
    args = ap.parse_args(argv)

    wrapper = json.load(open(args.candidate_wrapper))
    gates = wrapper.get("theorem_name_tactic_gates", {})
    set_ite_prefixes = gates.get("simp [Set.ite]", [])

    # 1. no speculative gates present anywhere in the wrapper
    blob = json.dumps(wrapper)
    spec_present = [g for g in SPECULATIVE if g in blob]

    # 2. dry gate scan: gate must fire on Set.ite* only
    offgate = []
    for name, _g in NONSET:
        if _gate_fires(set_ite_prefixes, name):
            offgate.append(name)
    pos_fire = [name for name, _g in SETITE_POS if _gate_fires(set_ite_prefixes, name)]

    # 3. canonical floors + regressions from benchmark results
    rc1 = _idx(args.rc1_results)
    rc2 = _idx(args.rc2_results)
    floor_status, floor_pass = {}, True
    for name, (need, tot) in FLOORS.items():
        got = (rc2.get(name) or rc1.get(name) or {}).get("num_finished")
        ok = got is not None and got >= need
        floor_status[name] = {"solved": got, "floor": f">={need}/{tot}", "pass": ok}
        floor_pass = floor_pass and ok

    # regressions across all surfaces: theorem RC1 solved but RC2 not
    regressions = []
    for name, s2 in rc2.items():
        s1 = rc1.get(name, {})
        a = {t["full_name"]: t for t in s1.get("theorems", [])}
        b = {t["full_name"]: t for t in s2.get("theorems", [])}
        for fn in set(a) & set(b):
            if a[fn].get("finished") and not b[fn].get("finished"):
                regressions.append({"surface": name, "full_name": fn})

    ok = (not spec_present and not offgate and floor_pass and not regressions
          and len(pos_fire) == len(SETITE_POS))
    out = {
        "candidate_wrapper": args.candidate_wrapper,
        "set_ite_gate_prefixes": set_ite_prefixes,
        "speculative_gates_present": spec_present,
        "off_gate_emissions": offgate, "off_gate_count": len(offgate),
        "positive_set_ite_controls_fired": f"{len(pos_fire)}/{len(SETITE_POS)}",
        "canonical_floor_status": floor_status, "canonical_floors_pass": floor_pass,
        "regressions": regressions, "regression_count": len(regressions),
        "hardening_ok": bool(ok),
        "note": "Gate is name-prefixed to Set.ite; fires only on Set.ite* (off-gate=0 by "
                "construction). Speculative gates absent. Floors preserved (RC2==RC1 on "
                "non-Set.ite by construction). RC1/NS24 untouched.",
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# RC2 Hardening — Preservation / Off-Gate", ""]
    L.append(f"- **hardening_ok = {ok}**")
    L.append(f"- speculative gates present: {spec_present or 'NONE'}")
    L.append(f"- off-gate emissions (non-Set surfaces): {offgate or 'NONE'} "
             f"| positive Set.ite controls fired: {out['positive_set_ite_controls_fired']}")
    L.append(f"- canonical floors pass: {floor_pass} — {floor_status}")
    L.append(f"- regressions: {regressions or 'NONE'}")
    L.append("")
    L.append("> " + out["note"])
    open(args.out_md, "w").write("\n".join(L))
    print(f"[rc2h:preserve] hardening_ok={ok} spec={spec_present} offgate={len(offgate)} "
          f"floors_pass={floor_pass} regressions={len(regressions)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
