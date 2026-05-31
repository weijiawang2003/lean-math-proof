#!/usr/bin/env python3
"""SX2 Part 7 — off-gate sanity check for SET2 (dry, gate-only; no Lean).

Confirms the SET2 gates do NOT fire on irrelevant non-Set surfaces. Live eval on
canonical surfaces would be expensive and unnecessary: the gates are pure
predicates over (theorem name, goal pp), so a gate-only scan is sufficient and
deterministic. Gates are evaluated with force_enable=True (the strongest test);
the production default (global_enabled=false) emits nothing regardless.

Surfaces (Nat/Multiset/canonical): synthetic name+goal samples that mimic each
named surface. We assert ZERO emissions on every non-Set sample, plus a couple of
positive Set controls to prove the gate logic is live (would-emit only when forced).

Outputs:
  set2_sanity_check.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sx2_set2_wrapper as setw  # noqa: E402

# Representative non-Set surfaces — name + a plausible goal pretty-print.
SURFACE_SAMPLES = {
    "demo_v1": [
        ("Nat.add_comm", "⊢ n + m = m + n"),
        ("Nat.succ_le_succ", "⊢ n + 1 ≤ m + 1 ↔ n ≤ m"),
        ("Bool.and_self", "⊢ (b && b) = b"),
        ("List.append_nil", "⊢ l ++ [] = l"),
    ],
    "nat_defs_medium": [
        ("Nat.mul_succ", "⊢ n * (m + 1) = n * m + n"),
        ("Nat.lt_irrefl", "⊢ ¬ n < n"),
        ("Nat.mod_self", "⊢ n % n = 0"),
        ("Nat.pow_zero", "⊢ n ^ 0 = 1"),
        ("Int.add_mul", "⊢ (a + b) * c = a * c + b * c"),
    ],
    "multiset_preservation": [
        ("Multiset.toFinset_eq_singleton_iff", "⊢ s.toFinset = {a} ↔ ∃ n, 0 < n ∧ s = n • {a}"),
        ("Multiset.cons_inj_left", "⊢ a ::ₘ s = a ::ₘ t ↔ s = t"),
        ("Multiset.map_id", "⊢ Multiset.map id s = s"),
    ],
}

# Positive controls: real Set surfaces that SHOULD fire a gate when forced
# (proves the scan is live, not vacuously zero).
POSITIVE_CONTROLS = [
    ("Set.ite_right", "⊢ s.ite t s = t ∩ s", "SET_ITE_SIMP"),
    ("Set.union_empty_iff", "⊢ s ∪ t = ∅ ↔ s = ∅ ∧ t = ∅", "SET_IFF_CONSTRUCTOR"),
]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--gate-policy",
                   default="project/evolve/experiments/sx2/set2_gate_policy.json")
    p.add_argument("--surfaces", default="demo_v1,nat_defs_medium")
    p.add_argument("--out",
                   default="project/evolve/experiments/sx2/out/set2_sanity_check.json")
    args = p.parse_args(argv)

    policy = setw.load_policy(args.gate_policy)
    requested = [s.strip() for s in args.surfaces.split(",") if s.strip()]
    # always include multiset_preservation as an extra safety surface
    surfaces = list(dict.fromkeys(requested + ["multiset_preservation"]))

    surface_reports = []
    total_thms = 0
    total_default_emit = 0
    total_forced_emit = 0
    total_off_gate = 0
    for surf in surfaces:
        samples = SURFACE_SAMPLES.get(surf)
        if samples is None:
            surface_reports.append({"surface": surf, "available": False,
                                    "note": "no synthetic sample bank; skipped"})
            continue
        rows = []
        for name, goal in samples:
            default_em = setw.eval_gates(policy, name, goal)            # prod default
            forced_em = setw.eval_gates(policy, name, goal, force_enable=True)
            off = [e for e in forced_em if e.get("off_gate")]
            total_thms += 1
            total_default_emit += len(default_em)
            total_forced_emit += len(forced_em)
            total_off_gate += len(off)
            rows.append({"theorem": name, "default_emissions": len(default_em),
                         "forced_emissions": [e["gate_id"] for e in forced_em],
                         "off_gate": len(off)})
        surface_reports.append({"surface": surf, "available": True,
                                "theorem_count": len(samples), "rows": rows,
                                "forced_emissions_total": sum(len(r["forced_emissions"]) for r in rows)})

    # positive controls
    pos = []
    for name, goal, expect in POSITIVE_CONTROLS:
        forced = [e["gate_id"] for e in setw.eval_gates(policy, name, goal, force_enable=True)]
        pos.append({"theorem": name, "expected_gate": expect,
                    "forced_emissions": forced, "fired_expected": expect in forced,
                    "default_emissions": len(setw.eval_gates(policy, name, goal))})

    ok = (total_default_emit == 0 and total_forced_emit == 0 and total_off_gate == 0
          and all(c["fired_expected"] for c in pos)
          and all(c["default_emissions"] == 0 for c in pos))
    out = {
        "gate_policy": args.gate_policy,
        "mode": "dry_gate_only_scan",
        "surfaces_checked": surfaces,
        "theorem_count": total_thms,
        "set2_emissions_production_default": total_default_emit,
        "set2_emissions_forced": total_forced_emit,
        "off_gate_emissions": total_off_gate,
        "expected": "0 SET2 emissions on Nat/Multiset/canonical surfaces (both prod-default "
                    "AND forced); positive Set controls fire only when forced.",
        "sanity_ok": bool(ok),
        "surface_reports": surface_reports,
        "positive_controls": pos,
        "note": "Gates are pure name+goal predicates; a dry scan is deterministic and "
                "sufficient. Live eval on demo_v1/nat_defs_medium is unnecessary and "
                "was not run (would be expensive). SET2 is off-by-default in production.",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
    print(f"[sx2:sanity] surfaces={surfaces} thms={total_thms} "
          f"forced_emit_on_nonset={total_forced_emit} off_gate={total_off_gate} "
          f"sanity_ok={ok}")
    for c in pos:
        print(f"   positive control {c['theorem']:30s} expect={c['expected_gate']} "
              f"fired={c['fired_expected']} default_emit={c['default_emissions']}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
