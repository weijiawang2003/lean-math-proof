"""Lightweight theorem/task configuration registry.

Keep this file easy to edit during experiments.

Availability notes (mathlib4 commit 29dcec07):
- Mathlib/Data/Nat/Defs.lean: confirmed available (Nat.mul_add_mod')
- Mathlib/Data/Nat/Basic.lean: MISSING trace artifacts
- Mathlib/Data/Set/Basic.lean: confirmed available
- Mathlib/Data/Finset/Basic.lean: confirmed available
- Simple Nat lemmas (zero_add, sub_self, etc.) live in Init, not Mathlib — unavailable.
"""

from __future__ import annotations

from core_types import TheoremConfig

THEOREM_SETS: dict[str, list[TheoremConfig]] = {
    # ---- Minimal / smoke test ----
    "nat_single": [
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_add_mod'"),
    ],

    # ---- DEMO SET: all from confirmed-available files ----
    # Three domains: Nat (Defs), Set (Basic), Finset (Basic)
    "demo_v1": [
        # Nat/Defs — confirmed available
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_add_mod'"),
        # Set/Basic — confirmed available
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.ite_univ"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_comm"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_comm"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.mem_union"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.mem_inter_iff"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.subset_univ"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.empty_subset"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_empty"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.empty_union"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_univ"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.univ_inter"),
        # Finset/Basic — confirmed available
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.disjoint_insert_right"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_insert"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_singleton"),
    ],

    # ================================================================
    # CURRICULUM LEARNING TIERS
    # ================================================================
    # Tier 1 (easy): Theorems we already prove reliably — bootstrap set.
    # All 1-step simp/ext proofs from Set/Basic + Nat/Defs.
    "curriculum_tier1": [
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_comm"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_comm"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.mem_union"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.mem_inter_iff"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.subset_univ"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.empty_subset"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_empty"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.empty_union"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_univ"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.univ_inter"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.ite_univ"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean",  full_name="Nat.mul_add_mod'"),
    ],

    # Tier 2 (medium): Associativity, idempotence, subset directions,
    # absorption.  Require similar tactics but in less obvious combos.
    # Used to test zero-shot transfer from tier1 training.
    "curriculum_tier2": [
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_assoc"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_assoc"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_self"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_self"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_univ"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_empty"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.subset_union_left"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.subset_union_right"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_subset_left"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_subset_right"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_insert"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_singleton"),
    ],

    # Tier 3 (hard): Distributivity, diff, complement, Finset harder.
    # Stretch goals — success here shows real generalization.
    "curriculum_tier3": [
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_inter_distrib_left"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_union_distrib_left"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.diff_self"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.diff_empty"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.empty_diff"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.disjoint_insert_right"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.insert_comm"),
    ],

    # Combined: all curriculum theorems for final aggregate eval
    "curriculum_all": [
        # Tier 1
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_comm"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_comm"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.mem_union"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.mem_inter_iff"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.subset_univ"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.empty_subset"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_empty"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.empty_union"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_univ"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.univ_inter"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.ite_univ"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean",  full_name="Nat.mul_add_mod'"),
        # Tier 2
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_assoc"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_assoc"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_self"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_self"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_univ"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_empty"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.subset_union_left"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.subset_union_right"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_subset_left"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_subset_right"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_insert"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_singleton"),
        # Tier 3
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.union_inter_distrib_left"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.inter_union_distrib_left"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.diff_self"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.diff_empty"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.empty_diff"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.disjoint_insert_right"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.insert_comm"),
    ],

    # ---- Legacy sets kept for compatibility ----
    "toy_search": [
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_add_mod'"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.add_mod"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.mul_mod"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.mod_add_mod"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.ite_univ"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.disjoint_insert_right"),
    ],
    "nat_more": [
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.add_mod"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.mul_mod"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.mod_add_mod"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_add_mod'"),
    ],
    "set_small": [
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.ite_univ"),
    ],
    "finset_small": [
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.disjoint_insert_right"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_insert"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_singleton"),
    ],
    "mixed_easy_v2": [
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_add_mod'"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.add_mod"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.mul_mod"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.ite_univ"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.disjoint_insert_right"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_insert"),
    ],
}


def list_theorem_sets() -> list[str]:
    return sorted(THEOREM_SETS)


def get_theorems(set_name: str) -> list[TheoremConfig]:
    if set_name not in THEOREM_SETS:
        known = ", ".join(list_theorem_sets())
        raise ValueError(f"Unknown theorem set '{set_name}'. Available: {known}")
    return THEOREM_SETS[set_name]
