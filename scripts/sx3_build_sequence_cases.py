#!/usr/bin/env python3
"""SX3 Part 5 — build the SX3 sequence-search case sets + manifest.

Deterministically derives five case files under
project/evolve/experiments/sx3/cases/ from existing artifacts:
  1. sx3_deferred_set_ite_cases.json      — the +4 RC2-hardening deferred candidates
  2. sx3_set_ite_fresh_holdout.json        — Set.ite/dite theorems NOT in credited+5
                                             or deferred+4 (genuine holdout)
  3. sx3_set_failure_cluster_cases.json    — general (non-ite) Set failures from SF2 +
                                             frontier (iff / subset / diff / equality)
  4. sx3_negative_controls.json            — Nat/Int/Multiset/List (must emit 0 Set seqs)
  5. sx3_canonical_smoke.json              — small sampled canonical Nat/Bool/List cases
plus sx3_case_manifest.json documenting sizes, exclusions and leakage risk.

Sources (read-only): rc2_delta_ledger, sf1 catalog/frontier_with_paths, sf2 selected
cases, the rc2 candidate negative-control & canonical-smoke theorem sets.
"""
from __future__ import annotations
import json
import os

SET_BASIC = "Mathlib/Data/Set/Basic.lean"
OUT = "project/evolve/experiments/sx3/cases"

CREDITED5 = ["Set.ite_empty", "Set.ite_empty_left", "Set.ite_empty_right",
             "Set.ite_left", "Set.ite_right"]
DEFERRED4 = ["Set.ite_inter", "Set.ite_inter_self", "Set.ite_compl",
             "Set.ite_inter_compl_self"]

# Fresh Set.ite/dite holdout (from sf1 catalog, all resolve to Set/Basic.lean,
# excludes credited5 + deferred4). Shapes hand-classified from source statements.
FRESH_HOLDOUT = [
    ("Set.ite_eq_of_subset_left", "set_equality",
     "source: ext+by_cases (depth>2); subset hyp"),
    ("Set.ite_eq_of_subset_right", "set_equality", "subset hyp"),
    ("Set.ite_inter_inter", "set_equality", "pure ite equality"),
    ("Set.ite_inter_of_inter_eq", "set_equality",
     "source: rw bridge (rw [<- ite_inter, <- h, ite_same]) -> source-specific risk"),
    ("Set.ite_univ", "set_equality",
     "source: simp [Set.ite] SINGLE-SHOT -> expected RC2 single_step_duplicate"),
    ("Set.mem_dite", "set_membership_iff", "dite membership iff"),
    ("Set.mem_dite_empty_left", "set_membership_iff", "dite membership iff"),
    ("Set.mem_dite_empty_right", "set_membership_iff", "dite membership iff"),
    ("Set.mem_dite_univ_left", "set_membership_iff", "dite membership iff"),
    ("Set.mem_dite_univ_right", "set_membership_iff", "dite membership iff"),
    ("Set.mem_ite_empty_left", "set_membership_iff", "ite membership iff"),
    ("Set.mem_ite_empty_right", "set_membership_iff", "ite membership iff"),
    ("Set.subset_ite", "set_subset_iff", "subset iff"),
]

# General (non-ite) Set failure cluster: SF2 selected (general) + frontier equality.
CLUSTER = [
    ("Set.diff_singleton_subset_iff", "set_subset_iff", "sf2", "rw-bridge source"),
    ("Set.pair_eq_pair_iff", "set_iff", "sf2", "sf2 win: simp[...]<;>aesop (depth-2)"),
    ("Set.subset_insert_iff", "set_iff", "sf2", "sf2 UNSOLVED"),
    ("Set.subset_singleton_iff_eq", "set_iff", "sf2", "sf2 UNSOLVED"),
    ("Set.union_empty_iff", "set_iff", "sf2", "rw-bridge source"),
    ("Set.antitoneOn_iff_antitone", "set_iff", "sf2", "simp[Antitone,AntitoneOn] source"),
    ("Set.ssubset_singleton_iff", "set_iff", "sf2", "rw-bridge source"),
    ("Set.diff_union_inter", "set_equality", "frontier", "set equality"),
    ("Set.insert_diff_of_mem", "set_equality", "frontier", "set equality, hyp"),
    ("Set.insert_diff_eq_singleton", "set_equality", "frontier", "set equality, hyp"),
    ("Set.pair_diff_left", "set_equality", "frontier", "set equality, hyp"),
    ("Set.powerset_singleton", "set_equality", "frontier", "set equality"),
]

# Negative controls (non-Set): must produce 0 Set sequence emissions. file_path null
# entries still get a valid gate decision (computed before Dojo open).
NEGATIVES = [
    ("Nat.add_comm", "Nat", None),
    ("Nat.mul_succ", "Nat", None),
    ("Int.add_mul", "Int", None),
    ("Multiset.toFinset_eq_singleton_iff", "Multiset", "Mathlib/Data/Finset/Basic.lean"),
    ("Multiset.cons_inj_left", "Multiset", None),
    ("List.append_nil", "List", None),
]

# Canonical smoke (regression-floor sample): non-Set, must not emit Set sequences.
SMOKE = [
    ("Nat.add_zero", "Nat", "demo_v1"),
    ("Nat.zero_add", "Nat", "demo_v1"),
    ("Nat.succ_le_succ", "Nat", "nat_defs_medium"),
    ("Bool.and_self", "Bool", "demo_v1"),
    ("List.append_nil", "List", "demo_v1"),
]


def dump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    json.dump(obj, open(path, "w"), ensure_ascii=False, indent=2)
    n = len(obj.get("cases", []))
    print(f"  wrote {path} ({n} cases)")
    return n


def main():
    # 1. deferred (already authored by hand; regenerate identically for provenance)
    deferred = {"name": "sx3_deferred_set_ite_cases",
                "description": "The +4 RC2-hardening deferred SX3 depth-2 sequence candidates.",
                "provenance": "rc2_hardening/out/rc2_delta_ledger.json deferred_sx3",
                "cases": [{"full_name": n, "file_path": SET_BASIC, "namespace": "Set",
                           "source": "rc2_deferred", "role": "deferred_known",
                           "shape": "set_equality", "known_rc2_status_finished": False,
                           "exclude_reason": None} for n in DEFERRED4]}
    n_def = dump("sx3_deferred_set_ite_cases.json", deferred)

    # 2. fresh holdout
    fresh = {"name": "sx3_set_ite_fresh_holdout",
             "description": "Set.ite/dite theorems NOT in RC2 credited+5 or deferred+4.",
             "provenance": "sf1 catalog (Set.*ite*), file paths confirmed in traced Set/Basic.lean",
             "excluded_credited5": CREDITED5, "excluded_deferred4": DEFERRED4,
             "cases": [{"full_name": n, "file_path": SET_BASIC, "namespace": "Set",
                        "source": "fresh_holdout_catalog", "role": "fresh_holdout",
                        "shape": shape, "known_rc2_status_finished": None,
                        "note": note, "exclude_reason": None}
                       for (n, shape, note) in FRESH_HOLDOUT]}
    n_fresh = dump("sx3_set_ite_fresh_holdout.json", fresh)

    # 3. cluster
    cluster = {"name": "sx3_set_failure_cluster_cases",
               "description": "General (non-ite) Set failures: SF2 selected + frontier equality.",
               "provenance": "sf2 selected_cases (general) + sf1 frontier_with_paths",
               "cases": [{"full_name": n, "file_path": SET_BASIC, "namespace": "Set",
                          "source": f"cluster_{src}", "role": "set_cluster_failure",
                          "shape": shape, "known_rc2_status_finished": None,
                          "note": note, "exclude_reason": None}
                         for (n, shape, src, note) in CLUSTER]}
    n_cluster = dump("sx3_set_failure_cluster_cases.json", cluster)

    # 4. negatives
    neg = {"name": "sx3_negative_controls",
           "description": "Non-Set cases; MUST produce 0 Set sequence emissions (gate test).",
           "provenance": "rc2_candidates set_ite_negative_controls",
           "cases": [{"full_name": n, "file_path": fp, "namespace": ns,
                      "source": "negative_control", "role": "negative_control",
                      "shape": "non_set", "known_rc2_status_finished": None,
                      "exclude_reason": None} for (n, ns, fp) in NEGATIVES]}
    n_neg = dump("sx3_negative_controls.json", neg)

    # 5. smoke
    smoke = {"name": "sx3_canonical_smoke",
             "description": "Small canonical regression-floor sample (non-Set).",
             "provenance": "rc2_candidates set_ite_canonical_smoke",
             "cases": [{"full_name": n, "file_path": None, "namespace": ns,
                        "source": surf, "role": "canonical_smoke", "shape": "non_set",
                        "known_rc2_status_finished": None, "exclude_reason": None}
                       for (n, ns, surf) in SMOKE]}
    n_smoke = dump("sx3_canonical_smoke.json", smoke)

    manifest = {
        "name": "sx3_case_manifest",
        "sizes": {"deferred": n_def, "fresh_holdout": n_fresh, "cluster": n_cluster,
                  "negative_controls": n_neg, "canonical_smoke": n_smoke},
        "roles": {
            "deferred_known": "the +4 RC2-deferred candidates; reproduction target, NOT fresh delta",
            "fresh_holdout": "Set.ite/dite never credited or deferred; the genuine fresh-delta surface",
            "set_cluster_failure": "general Set failures to test ext/iff/subset families",
            "negative_control": "non-Set; expect 0 Set sequence emissions (off-gate guard)",
            "canonical_smoke": "non-Set canonical floor sample; expect 0 Set emissions",
        },
        "exclusions": {
            "credited5_excluded_everywhere": CREDITED5,
            "deferred4_in_deferred_set_only": DEFERRED4,
            "fresh_holdout_excludes": "credited5 + deferred4",
        },
        "leakage_risk": [
            "Set.ite_univ (fresh) source proof is single-shot simp[Set.ite] -> expected "
            "single_step_duplicate, NOT a depth-2 win (attribution must catch it).",
            "Set.ite_inter_of_inter_eq (fresh) source uses an rw bridge -> source-specific risk.",
            "Some fresh-holdout names overlap RC2's own fresh_holdout set; that is acceptable "
            "because SX3 freshness is defined relative to credited5+deferred4, and attribution "
            "is recomputed live here.",
            "Deferred4 are 'known' wins (reproduction), so they are reported separately from "
            "fresh delta and never counted toward the RC3 fresh-delta gate.",
        ],
        "note": "Every claimed new win is compared against RC2's deterministic credited "
                "mechanism (single-shot simp[Set.ite] + bare baselines), which the runner "
                "executes as controls per theorem.",
    }
    dump("sx3_case_manifest.json", manifest)
    print("[sx3:cases] built all case sets.")


if __name__ == "__main__":
    main()
