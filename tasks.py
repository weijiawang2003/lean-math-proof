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

    # ---- nat_defs_subset: 15-theorem evolve target from Mathlib/Data/Nat/Defs.lean ----
    # Built from project/discovered_theorems.json filtered to Nat/Defs and
    # cross-referenced with project/project_state.json. Skewed easy/medium so
    # gen_v5 has a non-zero baseline; 7 of 15 are unsolved-or-unsearched to
    # leave headroom for an evolved wrapper to climb.
    # Composition: 5 easy-proved, 5 easy-failed, 2 medium-proved,
    #              1 medium-unsearched, 1 hard-proved, 1 hard-unsearched.
    "nat_defs_subset": [
        # easy, proved by some prior method (mostly `omega`)
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_max_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_min_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_right"),
        # easy, searched-and-failed by prior runs (real challenge)
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_mod_eq_add_mod_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_mod_eq_add_mod_right"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_le_div_right"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_lt_iff_lt_mul'"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_lt_one_iff"),
        # medium
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.half_le_of_sub_le_half"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_and_le_add_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.AM_GM"),
        # hard
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_or_le_of_add_eq_add_pred"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_mod_eq_ite"),
    ],

    # ---- nat_defs_medium: superset of nat_defs_subset for generalization ----
    # v3.5 scale-out test set: nat_defs_subset (15) + 22 additional theorems
    # drawn from Mathlib/Data/Nat/Defs.lean with varied name prefixes
    # (add, mul, lt, le, eq, sub, mod, div, one, succ, pred, sqrt, pow, dvd, two).
    # Total 37 theorems. All marked "easy" in discovered_theorems.json — the
    # point is breadth of statement shape, not difficulty.
    "nat_defs_medium": [
        # === Inherits all 15 from nat_defs_subset ===
        # easy, proved by some prior method (mostly `omega`)
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_max_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_min_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_right"),
        # easy, searched-and-failed by prior runs
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_mod_eq_add_mod_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_mod_eq_add_mod_right"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_le_div_right"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_lt_iff_lt_mul'"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_lt_one_iff"),
        # medium
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.half_le_of_sub_le_half"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_and_le_add_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.AM_GM"),
        # hard
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_or_le_of_add_eq_add_pred"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_mod_eq_ite"),
        # === New (v3.5) breadth additions ===
        # add / mul
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_zero"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_pos_iff_pos_or_pos"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_eq_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_eq_right"),
        # lt / le / one
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.lt_iff_add_one_le"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.lt_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_one_iff_eq_zero_or_eq_one"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_add_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.one_add_le_iff"),
        # eq / sub
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.eq_one_of_mul_eq_one_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.eq_zero_of_double_le"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.sub_lt_iff_lt_add"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.sub_lt_iff_lt_add'"),
        # mod / div
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mod_two_ne_one"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mod_two_ne_zero"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_pos"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_pos_iff"),
        # succ / pred / sqrt / pow / dvd / two
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.succ_succ_ne_one"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.pred_eq_of_eq_succ"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.sqrt_lt"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.pow_lt_pow_iff_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.dvd_iff_div_mul_eq"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.two_mul_ne_two_mul_add_one"),
    ],

    # ---- nat_defs_large_v5: 38 medium + 30 new for generalization study ----
    # Used by Direction D of v5_research_plan to test whether v5 candidate
    # improvements generalize beyond the medium set. New theorems drawn
    # from discovered_theorems.json easy / medium across diverse name
    # prefix buckets (div, dvd, mul, mod, pow, sqrt, succ, sub, two, max,
    # min, lt, le, eq, find, etc.).
    "nat_defs_large_v5": [
        # First 38 are nat_defs_medium verbatim
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_max_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_min_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_right"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_mod_eq_add_mod_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_mod_eq_add_mod_right"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_le_div_right"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_lt_iff_lt_mul'"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_lt_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.half_le_of_sub_le_half"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_and_le_add_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.AM_GM"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_or_le_of_add_eq_add_pred"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_mod_eq_ite"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_zero"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_pos_iff_pos_or_pos"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_eq_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_eq_right"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.lt_iff_add_one_le"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.lt_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_one_iff_eq_zero_or_eq_one"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_add_one_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.one_add_le_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.eq_one_of_mul_eq_one_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.eq_zero_of_double_le"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.sub_lt_iff_lt_add"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.sub_lt_iff_lt_add'"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mod_two_ne_one"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mod_two_ne_zero"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_pos"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_pos_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.succ_succ_ne_one"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.pred_eq_of_eq_succ"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.sqrt_lt"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.pow_lt_pow_iff_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.dvd_iff_div_mul_eq"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.two_mul_ne_two_mul_add_one"),
        # New for generalization (drawn from discovered_theorems.json)
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_two_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.add_eq_three_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.div_ne_zero_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.dvd_right_iff_eq"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.dvd_left_iff_eq"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.eq_div_of_mul_eq_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.eq_mul_of_div_eq_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.find_eq_zero"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.forall_lt_succ"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_add_pred_of_pos"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.le_of_mul_le_mul_right"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.lt_one_add_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.max_eq_zero_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.min_eq_zero_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mod_mul_mod"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mod_eq_iff_lt"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.one_le_div_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.one_le_pow"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.pow_le_pow_iff_left"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.pred_eq_self_iff"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.pred_one_add"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.self_add_sub_one"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.sqrt_lt'"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.sub_eq_of_eq_add'"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.succ_add_sub_one"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.two_pow_succ"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.zero_eq_mul"),
    ],

    # ---- Frontier set: theorems unsolved by every checkpoint to date ----
    # Used by experiments/SEARCH_FRONTIER_BRIEF.md to ask whether wide
    # beam search can find proofs without retraining.
    "frontier_v1": [
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_insert"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.mem_singleton"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.disjoint_insert_right"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.insert_comm"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean",     full_name="Nat.mul_add_mod'"),
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


# ---- NS14: load the wider theorem sets from JSON if present ----
# The four ns14_* sets are emitted by scripts/build_ns14_theorem_sets.py.
# We load them lazily so this module still imports if the file is absent.
def _load_ns14_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/ns14_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_ns14_sets()


# ---- NS16: load the expanded Nat theorem sets from JSON if present ----
# The four ns16_nat_* sets are emitted by
# scripts/build_ns16_theorem_sets.py. Loaded lazily so the module
# still imports if the file is absent.
def _load_ns16_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/ns16_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_ns16_sets()


# ---- NS17: load the family-mining theorem sets from JSON if present ----
def _load_ns17_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/ns17_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_ns17_sets()


# ---- NS19: load the targeted family-mining theorem sets from JSON ----
def _load_ns19_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/ns19_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_ns19_sets()


# ---- NS20: load the final Finset/aesop mining theorem sets ----
def _load_ns20_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/ns20_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_ns20_sets()


# ---- CX1: load the Mathlib-catalog-extension theorem sets ----
def _load_cx1_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/cx1_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_cx1_sets()


# ---- CX2: load the Int iff_omega mining theorem sets ----
def _load_cx2_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/cx2_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_cx2_sets()


def _load_cx3_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/cx3_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_cx3_sets()


def _load_wx2_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/wx2_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_wx2_sets()


def _load_ax2_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/ax2_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_ax2_sets()


def _load_wx3_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/wx3_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_wx3_sets()


def _load_ax3_sets() -> None:
    import json as _json
    from pathlib import Path as _Path
    p = _Path("project/evolve/routing/ax3_theorem_sets.json")
    if not p.exists():
        return
    try:
        data = _json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return
    for name, items in data.items():
        THEOREM_SETS[name] = [
            TheoremConfig(file_path=t["file_path"], full_name=t["full_name"])
            for t in items
        ]


_load_ax3_sets()


def list_theorem_sets() -> list[str]:
    return sorted(THEOREM_SETS)


def get_theorems(set_name: str) -> list[TheoremConfig]:
    if set_name not in THEOREM_SETS:
        known = ", ".join(list_theorem_sets())
        raise ValueError(f"Unknown theorem set '{set_name}'. Available: {known}")
    return THEOREM_SETS[set_name]
