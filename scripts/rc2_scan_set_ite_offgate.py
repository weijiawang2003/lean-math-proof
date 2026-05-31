#!/usr/bin/env python3
"""RC2 Part 6 — off-gate / preservation scan for the SET_ITE_SIMP gate.

Dry gate-only scan (the gate is a pure name+goal predicate -> deterministic, no Lean
needed). Confirms the gate fires ONLY on Set + ite/if goals and NEVER on
Nat/Int/Multiset/canonical surfaces. Reuses the negative-control + canonical-smoke
validation sets plus synthetic non-Set samples.

Outputs:
  offgate_preservation_scan.json / .md
"""
from __future__ import annotations

import argparse
import json
import os
import re

# synthetic (name, goal) samples per surface
SURFACES = {
    "nat_only": [
        ("Nat.add_comm", "⊢ n + m = m + n"),
        ("Nat.mul_succ", "⊢ n * (m + 1) = n * m + n"),
        ("Nat.mod_self", "⊢ n % n = 0"),
        ("Nat.add_mod_eq_ite", "⊢ (a + b) % n = if a % n + b % n < n then a % n + b % n else a % n + b % n - n"),
    ],
    "int_only": [
        ("Int.add_mul", "⊢ (a + b) * c = a * c + b * c"),
        ("Int.neg_neg", "⊢ - -a = a"),
    ],
    "multiset": [
        ("Multiset.toFinset_eq_singleton_iff", "⊢ s.toFinset = {a} ↔ ∃ n, 0 < n ∧ s = n • {a}"),
        ("Multiset.cons_inj_left", "⊢ a ::ₘ s = a ::ₘ t ↔ s = t"),
    ],
    "demo_v1": [
        ("Nat.add_zero", "⊢ n + 0 = n"),
        ("Bool.and_self", "⊢ (b && b) = b"),
        ("List.append_nil", "⊢ l ++ [] = l"),
    ],
    "nat_defs_medium": [
        ("Nat.succ_le_succ", "⊢ n + 1 ≤ m + 1 ↔ n ≤ m"),
        ("Nat.pow_zero", "⊢ n ^ 0 = 1"),
    ],
    "set_positive_control": [
        ("Set.ite_right", "⊢ s.ite t s = t ∩ s"),
        ("Set.ite_empty_left", "⊢ s.ite ∅ t = t \\ s"),
    ],
}


def gate_fires(gate, name, goal):
    hay_name = name or ""
    hay_all = (name or "") + "\n" + (goal or "")
    for tok in gate.get("requires_namespace_or_name_contains", []):
        if tok not in hay_name:
            return False
    anyset = gate.get("requires_goal_or_name_contains_any", [])
    if anyset:
        has = (".ite" in hay_all or
               re.search(r"(?<![A-Za-z])ite(?![A-Za-z])", hay_all) or
               re.search(r"(?<![A-Za-z])if(?![A-Za-z])", hay_all))
        if not has:
            return False
    for tok in gate.get("forbids_namespace_or_name_contains", []):
        if tok in hay_all:
            return False
    return True


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gate-policy",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/set_ite_simp_gate_policy.json")
    p.add_argument("--theorem-sets-dir",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/theorem_sets")
    p.add_argument("--out-json",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/offgate_preservation_scan.json")
    p.add_argument("--out-md",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/offgate_preservation_scan.md")
    args = p.parse_args(argv)

    gate = json.load(open(args.gate_policy))["gates"][0]["gate"]

    surface_reports = []
    total, total_fire, off_gate, set_pos_fire = 0, 0, 0, 0
    for surf, samples in SURFACES.items():
        is_set_surface = surf == "set_positive_control"
        rows = []
        for name, goal in samples:
            fired = gate_fires(gate, name, goal)
            total += 1
            if fired:
                total_fire += 1
                if is_set_surface:
                    set_pos_fire += 1
                else:
                    off_gate += 1  # fired on a non-Set surface = off-gate
            rows.append({"theorem": name, "fired": fired})
        surface_reports.append({"surface": surf, "is_set_surface": is_set_surface,
                                "count": len(samples),
                                "fired": sum(1 for r in rows if r["fired"]), "rows": rows})

    # also scan the negative_controls / canonical sets on disk (names only; no goal)
    disk = []
    for sname in ("set_ite_negative_controls", "set_ite_canonical_smoke"):
        fpath = os.path.join(args.theorem_sets_dir, f"{sname}.json")
        if not os.path.exists(fpath):
            continue
        obj = json.load(open(fpath))
        rows = obj.get(sname) or list(obj.values())[0]
        fired = 0
        for r in rows:
            f = gate_fires(gate, r.get("full_name", ""), "")  # name-only scan
            if f:
                fired += 1
                off_gate += 1
            total += 1
        disk.append({"set": sname, "count": len(rows), "fired_name_only": fired})

    ok = (off_gate == 0 and set_pos_fire == len(SURFACES["set_positive_control"]))
    out = {
        "mode": "dry_gate_only_scan",
        "gate_policy": args.gate_policy,
        "scanned_theorem_count": total,
        "gate_emissions_total": total_fire,
        "off_gate_emissions": off_gate,
        "unexpected_set_ite_emissions": off_gate,
        "set_positive_controls_fired": f"{set_pos_fire}/{len(SURFACES['set_positive_control'])}",
        "expected": "0 emissions on Nat/Int/Multiset/canonical (off-gate=0); only Set+ite "
                    "goals fire; positive Set controls fire.",
        "sanity_ok": bool(ok),
        "surface_reports": surface_reports,
        "disk_set_name_only_scan": disk,
        "note": "Gate is a pure name+goal predicate; dry scan is deterministic and "
                "sufficient. Live eval on canonical surfaces unnecessary (gate cannot "
                "fire without 'Set' in the name). SET_ITE_SIMP is off-by-default in prod.",
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# RC2 — SET_ITE_SIMP Off-Gate / Preservation Scan", ""]
    L.append(f"- scanned={total} | gate emissions={total_fire} | "
             f"**off-gate emissions={off_gate}** | "
             f"positive Set controls fired={out['set_positive_controls_fired']} | "
             f"sanity_ok={ok}")
    L.append("")
    L.append("| surface | is_set | count | fired |")
    L.append("|---|---|---|---|")
    for s in surface_reports:
        L.append(f"| {s['surface']} | {s['is_set_surface']} | {s['count']} | {s['fired']} |")
    for d in disk:
        L.append(f"| {d['set']} (name-only) | False | {d['count']} | {d['fired_name_only']} |")
    L.append("")
    L.append("> " + out["note"])
    open(args.out_md, "w").write("\n".join(L))
    print(f"[rc2:offgate] scanned={total} fired={total_fire} off_gate={off_gate} "
          f"set_pos={out['set_positive_controls_fired']} sanity_ok={ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
