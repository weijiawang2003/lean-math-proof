"""Lightweight theorem/task configuration registry.

Keep this file easy to edit during experiments.
"""

from __future__ import annotations

from core_types import TheoremConfig

THEOREM_SETS: dict[str, list[TheoremConfig]] = {
    "nat_single": [
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_add_mod'"),
    ],
    "toy_search": [
        TheoremConfig(file_path="Mathlib/Data/Nat/Defs.lean", full_name="Nat.mul_add_mod'"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.add_mod"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.mul_mod"),
        TheoremConfig(file_path="Mathlib/Data/Nat/Basic.lean", full_name="Nat.mod_add_mod"),
        TheoremConfig(file_path="Mathlib/Data/Set/Basic.lean", full_name="Set.ite_univ"),
        TheoremConfig(file_path="Mathlib/Data/Finset/Basic.lean", full_name="Finset.disjoint_insert_right"),
    ],
}


def get_theorems(set_name: str) -> list[TheoremConfig]:
    if set_name not in THEOREM_SETS:
        known = ", ".join(sorted(THEOREM_SETS))
        raise ValueError(f"Unknown theorem set '{set_name}'. Available: {known}")
    return THEOREM_SETS[set_name]
