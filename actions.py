"""Action space registry for search/rollout experiments.

`ACTIONS` is kept as the classifier-compatible default label set.
Use `get_action_space(...)` to opt into larger tactic spaces for search.
"""

from __future__ import annotations

from collections import OrderedDict

# Stable, classifier-compatible action list used by policy.py label indices.
CORE_ACTIONS_V1: list[str] = [
    "intro",
    "refl",
    "rfl",
    "simp",
    "assumption",
    "cases a",
    "induction a",
    "rw [Nat.mul_add_mod]",
    "rw [Nat.mul_comm]",
    "rw [Nat.mul_comm, Nat.mul_add_mod]",
    "rw [Nat.mul_add_mod, Nat.mul_comm]",
    "simp [Nat.mul_add_mod, Nat.mul_comm]",
]

# Broader search-oriented pool (kept deterministic and duplicate-free).
EXPANDED_SEARCH_ACTIONS_V2: list[str] = list(
    OrderedDict.fromkeys(
        CORE_ACTIONS_V1
        + [
            # Generic structural steps
            "constructor",
            "exact?",
            "aesop",
            "omega",
            "linarith",
            "ring",
            "norm_num",
            "decide",
            "tauto",
            # Rewriting/simplification variants
            "simp_all",
            "simpa",
            "simp [Nat.add_comm, Nat.add_left_comm, Nat.add_assoc]",
            "simp [Nat.mul_comm, Nat.mul_left_comm, Nat.mul_assoc]",
            "rw [Nat.add_comm]",
            "rw [Nat.add_left_comm]",
            "rw [Nat.add_assoc]",
            "rw [Nat.mul_assoc]",
            "rw [Nat.mod_eq_of_lt]",
            "rw [Nat.add_mod]",
            "rw [Nat.mul_mod]",
            # Case/intro families
            "intro h",
            "intro x",
            "cases h",
            "cases n",
            "induction n with | zero => simp | succ n ih => simp [ih]",
            # Set / Finset flavored moves for mixed theorem sets
            "ext x <;> simp",
            "ext x <;> constructor <;> intro hx <;> simpa using hx",
            "simp [Set.mem_univ]",
            "simp [Finset.mem_insert, Finset.mem_singleton]",
            "rw [Finset.disjoint_left]",
        ]
    )
)

ACTION_SPACES: dict[str, list[str]] = {
    "core_v1": CORE_ACTIONS_V1,
    "search_v2": EXPANDED_SEARCH_ACTIONS_V2,
}

# Backward-compatible alias used by the classifier policy head.
ACTIONS = CORE_ACTIONS_V1


def list_action_spaces() -> list[str]:
    return sorted(ACTION_SPACES)


def get_action_space(name: str) -> list[str]:
    if name not in ACTION_SPACES:
        known = ", ".join(list_action_spaces())
        raise ValueError(f"Unknown action space '{name}'. Available: {known}")
    return ACTION_SPACES[name]
