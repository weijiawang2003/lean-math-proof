"""Action space registry for search/rollout experiments.

`ACTIONS` is kept as the classifier-compatible default label set.
Use `get_action_space(...)` to opt into larger tactic spaces for search.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

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

# v3: substantially broader action space for demo / multi-domain coverage.
# Includes: generic power tactics, self-referential simp lemmas, Set/Finset/List
# closers, Nat/Int arithmetic, logic, and multi-step combinators.
EXPANDED_SEARCH_ACTIONS_V3: list[str] = list(
    OrderedDict.fromkeys(
        EXPANDED_SEARCH_ACTIONS_V2
        + [
            # ---- Power tactics / automation ----
            "trivial",
            "contradiction",
            "exact rfl",
            "apply?",
            "simp only []",
            "simp_all only []",
            "push_neg",
            "push_neg at *",

            # ---- Logic / propositional ----
            "intro a",
            "intro b",
            "intro ha",
            "intro hb",
            "intro hab",
            "intros",
            "rintro ⟨h1, h2⟩",
            "obtain ⟨h1, h2⟩ := h",
            "left",
            "right",
            "exfalso",
            "by_contra h",
            "exact h",
            "exact ha",
            "exact hb",
            "apply h",
            "apply And.intro",
            "cases h with | inl h => exact h | inr h => exact h",
            "rcases h with h1 | h2",

            # ---- Nat arithmetic ----
            "simp [Nat.succ_eq_add_one]",
            "simp [Nat.add_sub_cancel]",
            "simp [Nat.sub_self]",
            "simp [Nat.zero_add]",
            "simp [Nat.add_zero]",
            "simp [Nat.mul_zero]",
            "simp [Nat.zero_mul]",
            "simp [Nat.mul_one]",
            "simp [Nat.one_mul]",
            "rw [Nat.succ_eq_add_one]",
            "rw [Nat.sub_self]",
            "rw [Nat.add_sub_cancel]",
            "rw [Nat.zero_add]",
            "rw [Nat.add_zero]",
            "rw [Nat.mul_zero]",
            "rw [Nat.zero_mul]",
            "rw [Nat.mul_one]",
            "rw [Nat.one_mul]",
            "induction n",
            "induction m",
            "induction k",
            "simp [*]",
            "simp_arith",
            "omega",

            # ---- Set operations ----
            "ext",
            "ext x",
            "funext x",
            "simp [Set.ite]",
            "simp [Set.mem_ite]",
            "simp [Set.mem_setOf_eq]",
            "simp [Set.mem_union]",
            "simp [Set.mem_inter_iff]",
            "simp [Set.mem_diff]",
            "simp [Set.mem_compl_iff]",
            "simp [Set.subset_def]",
            "simp [Set.ext_iff]",
            "simp [Set.univ_eq_empty_iff]",
            "rfl",

            # ---- Finset operations ----
            "simp [Finset.mem_insert]",
            "simp [Finset.mem_union]",
            "simp [Finset.mem_filter]",
            "simp [Finset.mem_sdiff]",
            "simp [Finset.disjoint_left]",
            "simp [Finset.disjoint_insert_right]",
            "simp [Finset.disjoint_iff_ne]",
            "rw [Finset.disjoint_insert_right]",
            "rw [Finset.mem_insert]",

            # ---- List operations ----
            "simp [List.mem_cons_iff]",
            "simp [List.length_cons]",
            "simp [List.append_nil]",
            "simp [List.nil_append]",
            "simp [List.map]",
            "simp [List.filter]",

            # ---- Bool ----
            "simp [Bool.and_comm]",
            "simp [Bool.or_comm]",
            "decide",

            # ---- Iff splits ----
            "constructor <;> intro h <;> simp_all",
            "constructor <;> intro h <;> exact h",
            "constructor <;> simp",
            "refine ⟨?_, ?_⟩",
            "iff_of_true trivial trivial",

            # ---- Cleanup / finishing ----
            "assumption",
            "exact?",
            "simp at *",
            "simp_all [*]",
        ]
    )
)


# v4: curriculum-oriented — adds tactics for associativity, distributivity,
# diff/compl, subset relations, and multi-step combinators.
EXPANDED_SEARCH_ACTIONS_V4: list[str] = list(
    OrderedDict.fromkeys(
        EXPANDED_SEARCH_ACTIONS_V3
        + [
            # ---- Associativity / commutativity via logic ----
            "ext x; simp [or_assoc]",
            "ext x; simp [and_assoc]",
            "ext x; simp [or_comm]",
            "ext x; simp [and_comm]",
            "ext x; simp [or_self]",
            "ext x; simp [and_self]",
            "simp [Set.union_assoc]",
            "simp [Set.inter_assoc]",

            # ---- Idempotence / absorption ----
            "simp [Set.union_self]",
            "simp [Set.inter_self]",
            "simp [Set.union_univ]",
            "simp [Set.inter_empty]",
            "simp [Set.empty_inter]",
            "simp [Set.univ_union]",

            # ---- Subset proofs ----
            "intro x hx",
            "intro x ⟨hx1, hx2⟩",
            "exact Set.mem_union_left _ hx",
            "exact Set.mem_union_right _ hx",
            "exact hx.1",
            "exact hx.2",
            "exact ⟨hx, trivial⟩",
            "exact Set.subset_union_left hx",
            "exact Set.subset_union_right hx",
            "exact Set.inter_subset_left hx",
            "exact Set.inter_subset_right hx",
            "apply Set.subset_union_left",
            "apply Set.subset_union_right",
            "apply Set.inter_subset_left",
            "apply Set.inter_subset_right",

            # ---- Diff / complement ----
            "simp [Set.diff_self]",
            "simp [Set.diff_empty]",
            "simp [Set.empty_diff]",
            "simp [Set.diff_eq]",
            "simp [Set.mem_diff]",
            "simp [Set.diff_union_self]",
            "ext x; simp [Set.mem_diff]",

            # ---- Distributivity ----
            "simp [Set.union_inter_distrib_left]",
            "simp [Set.inter_union_distrib_left]",
            "ext x; simp [and_or_left]",
            "ext x; simp [or_and_left]",

            # ---- Finset harder ----
            "simp [Finset.insert_comm]",
            "rw [Finset.insert_comm]",
            "simp [Finset.disjoint_iff_inter_eq_empty]",
            "simp [Finset.inter_comm]",
            "rw [Finset.disjoint_insert_right]",
            "constructor <;> intro h <;> simp_all [Finset.mem_insert]",

            # ---- Multi-step combinators ----
            "ext x; constructor <;> intro hx <;> simp_all",
            "ext x; simp only [Set.mem_union, Set.mem_inter_iff]",
            "ext x; simp only [Set.mem_inter_iff, Set.mem_union]",
            "ext x; constructor <;> simp [or_assoc]",
            "ext x; constructor <;> simp [and_assoc]",
            "refine Set.eq_of_subset_of_subset ?_ ?_",
            "apply Set.Subset.antisymm",
            "apply le_antisymm",
        ]
    )
)

# Remove tactics known to crash the Lean REPL (exhaustive search / OOM).
# exact? and apply? trigger unbounded elaboration that can kill the process.
_CRASH_PRONE_TACTICS = {"exact?", "apply?"}
EXPANDED_SEARCH_ACTIONS_V4 = [t for t in EXPANDED_SEARCH_ACTIONS_V4 if t not in _CRASH_PRONE_TACTICS]


# v5: Strategic / backward-reasoning action space.
# Captures the collaborator's key insight: instead of only forward search
# (trying tactics one step at a time), include tactics that:
#   1. Propose intermediate sub-goals (`have`, `suffices`)
#   2. Reduce to simpler equivalent statements (`calc`, `show`)
#   3. Translate between mathematical "fields" (Set <-> logic, Finset <-> Set)
#
# These are "strategic" moves — they don't immediately close a goal but
# restructure the proof into a form where domain-specific tools can finish it.
STRATEGIC_ACTIONS_V5: list[str] = list(
    OrderedDict.fromkeys(
        EXPANDED_SEARCH_ACTIONS_V4
        + [
            # ──── BACKWARD REASONING: suffices / have ────
            # "suffices" introduces a sub-goal and proves the main goal from it
            "suffices h : True by trivial",  # template — real instances use retriever
            "suffices h : _ by assumption",
            "suffices h : _ by exact h",

            # "have" introduces a helper lemma mid-proof
            "have h := rfl",
            "have h : _ := by simp",
            "have h : _ := by aesop",
            "have h : _ := by omega",
            "have h : _ := by ring",
            "have h : _ := by tauto",
            "have h : _ := by exact?",

            # ──── SHOW: re-state goal in equivalent form ────
            # "show" clarifies the goal type — useful when Lean's display is opaque
            "show _",
            "change _",

            # ──── CALC: multi-step equational reasoning ────
            # calc blocks express chains of equalities/inequalities
            "calc _ = _ := by simp\n  _ = _ := by ring",

            # ──── DOMAIN TRANSLATION: Set <-> Logic ────
            # These "bridge" between set theory and propositional logic
            # (the collaborator's cross-field translation insight)
            "simp only [Set.ext_iff]",     # reduce set equality to ∀ x, membership
            "simp only [Set.subset_def]",  # reduce ⊆ to ∀ x, x ∈ ... → x ∈ ...
            "simp only [Set.mem_setOf_eq]",
            "ext x; simp only [Set.mem_union, Set.mem_inter_iff, Set.mem_diff]",
            "ext x; push_neg",             # negate universally → find counterexample structure

            # ──── DOMAIN TRANSLATION: Finset <-> Multiset <-> Set ────
            "simp only [Finset.mem_coe]",
            "rw [Finset.subset_iff]",
            "rw [Finset.ext_iff]",
            "intro x; simp only [Finset.mem_union, Finset.mem_inter, Finset.mem_sdiff]",

            # ──── DOMAIN TRANSLATION: Nat <-> general algebra ────
            "simp only [Nat.add_comm, Nat.add_assoc, Nat.add_left_comm]",
            "simp only [Nat.mul_comm, Nat.mul_assoc, Nat.mul_left_comm]",
            "rw [show _ = _ from by ring]",

            # ──── STRUCTURAL: convert between Iff <-> implications ────
            "constructor",                 # split ↔ into → and ←
            "intro h; constructor",
            "obtain ⟨h1, h2⟩ := h",       # destruct ∧
            "obtain h | h := h",           # destruct ∨
            "rcases h with ⟨h1, h2⟩ | ⟨h3, h4⟩",  # complex destruct

            # ──── REDUCTION: simplify before applying specific lemmas ────
            "simp only []",                # normalize without closing
            "norm_num",
            "ring_nf",                     # normalize ring expression
            "push_neg; simp",              # negate then simplify

            # ──── WITNESS CONSTRUCTION ────
            "use 0",
            "use 1",
            "refine ⟨_, ?_⟩",
            "refine ⟨?_, ?_⟩",

            # ──── SYMMETRY / TRANSITIVITY ────
            "symm",
            "trans",
            "congr 1",
            "congr",
            "apply Eq.symm",
            "apply le_trans",
        ]
    )
)

# Remove crash-prone from v5 as well
STRATEGIC_ACTIONS_V5 = [t for t in STRATEGIC_ACTIONS_V5 if t not in _CRASH_PRONE_TACTICS]


ACTION_SPACES: dict[str, list[str]] = {
    "core_v1": CORE_ACTIONS_V1,
    "search_v2": EXPANDED_SEARCH_ACTIONS_V2,
    "search_v3": EXPANDED_SEARCH_ACTIONS_V3,
    "search_v4": EXPANDED_SEARCH_ACTIONS_V4,
    "strategic_v5": STRATEGIC_ACTIONS_V5,
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


def save_action_space(path: str, actions: list[str]) -> None:
    payload = {"actions": actions}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_action_space(path: str) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    actions = payload.get("actions")
    if not isinstance(actions, list) or not all(isinstance(x, str) for x in actions):
        raise ValueError(f"Invalid action space file: {path}")
    return actions
