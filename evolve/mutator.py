"""Mutator — produces a child SearchCandidate by local mutation.

Version 1 is fully deterministic (no LLM): given the same parent, generation
and index, mutate_candidate always returns the same child. The RNG is seeded
from those three values so an evolve run is reproducible.

A later version can swap this for an LLM-driven mutator that proposes new
fallback orders / templates / prompts from the current best candidates — that
is the "LLM Mutator" box in the AlphaEvolve loop. The function signature is
designed to stay the same.
"""

from __future__ import annotations

import random

from evolve.candidate import SearchCandidate

# Generic fallback tactics, ordered roughly cheap -> heavy.
FALLBACK_POOL: list[str] = [
    "simp",
    "aesop",
    "omega",
    "linarith",
    "ring",
    "norm_num",
    "constructor",
    "intro",
    "rfl",
]

# Nat-specific tactic templates. {var} is a placeholder a future strategy
# wrapper would fill with the induction/cases variable. They are NOT required
# to be valid for every theorem — they are candidate templates.
NAT_TEMPLATES: list[str] = [
    "omega",
    "simp",
    "norm_num",
    "induction {var} with | zero => simp | succ n ih => simp [ih]",
    "cases {var} <;> simp",
    "simp [Nat.add_comm, Nat.add_assoc, Nat.left_comm]",
    "simp [Nat.mul_comm, Nat.mul_assoc, Nat.left_comm]",
    "try omega",
]

TOP_K_CHOICES: list[int] = [4, 8, 12, 16]
MAX_STEPS_CHOICES: list[int] = [6, 8, 10, 12]
TIMEOUT_CHOICES: list[int] = [10, 20, 30]
# v4.7: retrieval-side mutation knobs.
RETRIEVAL_TOP_K_CHOICES: list[int] = [4, 6, 8, 10, 12]
RETRIEVAL_FORM_POOL: list[str] = ["rw", "simp", "apply", "exact"]
# Per-family budget bump (in slot count). The seed div family is 8;
# mutator can shift it within ±4 to test budget sensitivity without
# blowing past the per-state cap.
FAMILY_BUDGET_DELTAS: list[int] = [-4, -2, +2, +4]

_MUTATION_OPS = [
    "top_k",
    "max_steps",
    "reorder_fallback",
    "add_fallback",
    "add_nat_template",
    "timeout",
    "reorder_family_tactics",
    "retrieval_top_k",
    "reorder_retrieval_forms",
    "family_budget_delta",
]


def mutate_candidate(
    parent: SearchCandidate, generation: int, index: int
) -> SearchCandidate:
    """Return a mutated child of `parent`.

    1-3 local mutations are applied. Deterministic in (parent.name, generation,
    index). The child's name encodes its slot so names are unique within a run.
    """
    rng = random.Random(f"{parent.name}|{generation}|{index}")
    child = parent.copy()
    applied: list[str] = []

    n_mut = rng.randint(1, 3)
    for op in rng.sample(_MUTATION_OPS, n_mut):
        if op == "top_k":
            child.top_k = rng.choice(TOP_K_CHOICES)
            applied.append(f"top_k={child.top_k}")

        elif op == "max_steps":
            child.max_steps = rng.choice(MAX_STEPS_CHOICES)
            applied.append(f"max_steps={child.max_steps}")

        elif op == "timeout":
            child.timeout_per_theorem = rng.choice(TIMEOUT_CHOICES)
            applied.append(f"timeout={child.timeout_per_theorem}")

        elif op == "reorder_fallback":
            if len(child.fallback_tactics) > 1:
                rng.shuffle(child.fallback_tactics)
                applied.append("reordered fallbacks")
            else:
                child.fallback_tactics = rng.sample(FALLBACK_POOL, k=5)
                applied.append("seeded fallbacks")

        elif op == "add_fallback":
            missing = [t for t in FALLBACK_POOL if t not in child.fallback_tactics]
            if missing:
                child.fallback_tactics.append(rng.choice(missing))
                applied.append("added fallback tactic")

        elif op == "add_nat_template":
            missing = [t for t in NAT_TEMPLATES if t not in child.tactic_templates]
            if missing:
                child.tactic_templates.append(rng.choice(missing))
                applied.append("added Nat template")

        elif op == "reorder_family_tactics":
            # v3.4: pick one family with >=2 tactics and shuffle its
            # ordering in-place. Family selection is deterministic via the
            # already-seeded RNG, so the mutation is reproducible.
            fam_dict = getattr(child, "theorem_family_tactics", None) or {}
            reorderable = [k for k, v in fam_dict.items() if len(v) > 1]
            if reorderable:
                fam = rng.choice(sorted(reorderable))
                tactics = list(fam_dict[fam])
                rng.shuffle(tactics)
                fam_dict[fam] = tactics
                child.theorem_family_tactics = fam_dict
                applied.append(f"reordered family[{fam}]")

        elif op == "retrieval_top_k":
            # v4.7: pick a different retrieval_top_k from the choice set.
            # No-op if retrieval is disabled on the parent.
            if getattr(child, "retrieval_enabled", False):
                current = getattr(child, "retrieval_top_k", 0) or 0
                choices = [k for k in RETRIEVAL_TOP_K_CHOICES if k != current]
                if choices:
                    child.retrieval_top_k = rng.choice(choices)
                    applied.append(f"retrieval_top_k={child.retrieval_top_k}")

        elif op == "reorder_retrieval_forms":
            # v4.7: shuffle the configured retrieval_tactic_forms. Order
            # affects which forms get emitted first under shape filtering
            # and the per-state budget cap.
            forms = list(getattr(child, "retrieval_tactic_forms", []) or [])
            if len(forms) > 1:
                rng.shuffle(forms)
                child.retrieval_tactic_forms = forms
                applied.append("reordered retrieval forms")

        elif op == "family_budget_delta":
            # v4.7: bump one family's budget by a small delta. Tests
            # whether the constructor seed's div_budget=8 is optimal.
            budgets = dict(getattr(child, "family_budgets", {}) or {})
            if budgets:
                fam = rng.choice(sorted(budgets))
                delta = rng.choice(FAMILY_BUDGET_DELTAS)
                new_budget = max(1, budgets[fam] + delta)
                budgets[fam] = new_budget
                child.family_budgets = budgets
                applied.append(f"family_budget[{fam}]={new_budget}")

    # Guarantee the child differs from the parent in at least one knob.
    if not applied:
        child.top_k = rng.choice([k for k in TOP_K_CHOICES if k != parent.top_k])
        applied.append(f"top_k={child.top_k}")

    child.name = f"g{generation}-i{index}-tk{child.top_k}-ms{child.max_steps}"
    child.description = (
        f"gen-{generation} mutation of '{parent.name}': " + ", ".join(applied)
    )
    child.metadata = dict(child.metadata)
    child.metadata.update(
        {
            "parent": parent.name,
            "generation": generation,
            "index": index,
            "mutations": applied,
        }
    )
    return child
