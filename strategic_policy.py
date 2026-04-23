"""Strategic policy: backward reasoning + premise retrieval + cross-field translation.

This policy layer implements the collaborator's key insights:

1. BACKWARD REASONING ("Sufficient Statement" thinking):
   Instead of only searching forward one tactic at a time, generate
   intermediate sub-goals that would make the proof easy if established.
   In Lean terms: `have h : <useful_fact> := by ...` and `suffices h : ...`.

2. PREMISE-AUGMENTED GENERATION (from ReProver paper):
   Retrieve relevant lemma names before generating tactics, so the model
   can produce specific `rw [Lemma.name]` instead of hoping to hallucinate
   the right identifier from memory.

3. CROSS-FIELD TRANSLATION (collaborator's "mirroring" insight):
   Generate tactics that re-state the problem in a different mathematical
   "language" — e.g., converting a Set problem into pure logic via ext_iff,
   or converting a Finset problem into Set membership.  These correspond to
   the "high-value translations" between mathematical fields.

Architecture:
  StrategicPolicy wraps an existing generative/hybrid policy and augments it:
  - Retrieves premises relevant to the current state
  - Generates "strategic" tactics (suffices, have, domain translations)
  - Combines with the base policy's forward-search tactics
  - Returns a unified ranked list

This is ADDITIVE: it doesn't replace the existing pipeline, it enriches it.

Usage:
    from strategic_policy import StrategicPolicy
    pol = StrategicPolicy(base_policy="hybrid", gen_ckpt="project/gen_ckpt_v5")
    tactics = pol.rank_tactics(state_pp, full_name, k=16)
"""

from __future__ import annotations

import re
from typing import Optional

from actions import get_action_space
from premise_retriever import PremiseRetriever, _detect_domains, extract_premises_from_tactic


# ── Backward-reasoning tactic templates ─────────────────────────────

def _generate_have_tactics(premises: list[str], state_pp: str) -> list[str]:
    """Generate `have` tactics that introduce useful intermediate facts.

    This implements the "sufficient statement" idea: propose a fact that,
    if true, makes the main goal straightforward to prove.

    Strategy: For each retrieved premise, generate a `have` that applies it.
    """
    tactics = []

    for premise in premises[:5]:  # Top-5 premises
        # Pattern: have h := <premise> applied to relevant args
        # These are templates — Lean will elaborate them
        tactics.append(f"have h := @{premise}")
        # Also try using the premise in a rewrite
        tactics.append(f"rw [{premise}]")
        tactics.append(f"simp [{premise}]")

    return tactics


def _generate_suffices_tactics(state_pp: str, domains: list[str]) -> list[str]:
    """Generate `suffices` tactics based on detected structure.

    The idea: propose a simpler statement that implies the goal,
    effectively working backward from the desired conclusion.
    """
    tactics = []

    # If goal is an equality (⊢ X = Y), suffice to show a simpler equality
    if " = " in state_pp:
        tactics.append("suffices h : _ by exact h")
        tactics.append("suffices h : _ by rw [h]")
        tactics.append("symm")
        # Convert to pointwise: often easier
        if "Set" in " ".join(domains):
            tactics.append("suffices ∀ x, x ∈ _ ↔ x ∈ _ from Set.ext this")

    # If goal is a subset (⊆), reduce to element-level
    if "⊆" in state_pp or "Subset" in state_pp:
        tactics.append("intro x hx")
        tactics.append("suffices h : _ by exact Set.mem_of_mem_of_subset hx h")

    # If goal is ↔, split into two directions
    if "↔" in state_pp:
        tactics.append("constructor")
        tactics.append("suffices (→) : _ by exact ⟨this, ?_⟩")

    # If goal involves ∀, introduce the variable
    if "∀" in state_pp:
        tactics.append("intro x")
        tactics.append("intro x hx")

    return tactics


def _generate_translation_tactics(state_pp: str, domains: list[str]) -> list[str]:
    """Generate cross-field translation tactics.

    This is the collaborator's core insight: problems become easier when
    restated in the "right" mathematical language.  These tactics don't
    close the goal but restructure it into a form where other tools work.

    Key translations:
      Set equality  →  ∀ x, membership ↔     (via ext_iff)
      Set subset    →  ∀ x, membership →      (via subset_def)
      Finset props  →  Set coercion props     (via mem_coe)
      Nat equations →  ring/omega-friendly    (via ring_nf)
    """
    tactics = []

    if "Set" in " ".join(domains):
        # Set → Logic translation (the most powerful bridge)
        if " = " in state_pp and ("∪" in state_pp or "∩" in state_pp or
                                   "Set" in state_pp):
            tactics.extend([
                "ext x",                          # pointwise
                "simp only [Set.ext_iff]",       # explicit
                "ext x; simp [Set.mem_union, Set.mem_inter_iff, Set.mem_diff]",
            ])
        if "⊆" in state_pp:
            tactics.extend([
                "simp only [Set.subset_def]",
                "intro x hx; simp at hx ⊢",
            ])
        if "∅" in state_pp or "empty" in state_pp.lower():
            tactics.extend([
                "simp [Set.eq_empty_iff_forall_not_mem]",
                "ext x; simp",
            ])

    if "Finset" in " ".join(domains):
        # Finset → element-level translation
        if " = " in state_pp:
            tactics.extend([
                "ext x",
                "rw [Finset.ext_iff]",
                "simp [Finset.mem_insert, Finset.mem_union, Finset.mem_inter]",
            ])
        if "disjoint" in state_pp.lower() or "Disjoint" in state_pp:
            tactics.extend([
                "rw [Finset.disjoint_left]",
                "simp [Finset.disjoint_iff_ne]",
            ])

    if "Nat" in " ".join(domains):
        # Nat → linear/ring arithmetic translation
        if "%" in state_pp or "mod" in state_pp.lower():
            tactics.extend([
                "omega",
                "simp [Nat.mod_def]",
            ])
        if "+" in state_pp or "*" in state_pp:
            tactics.extend([
                "ring",
                "ring_nf",
                "omega",
            ])

    return tactics


# ── Main Strategic Policy ───────────────────────────────────────────

class StrategicPolicy:
    """Policy that combines backward reasoning + premise retrieval + base policy.

    This wraps an existing policy (generative, hybrid, or action-space)
    and enriches its output with strategically-generated tactics.

    The key architectural choice: strategic tactics go FIRST in the ranking
    because they represent "big moves" (field translations, sub-goal generation)
    that restructure the proof.  Base policy tactics follow as "execution" moves
    that close restructured goals.
    """

    def __init__(
        self,
        base_policy: str = "hybrid",
        gen_ckpt_dir: str = "gen_ckpt",
        action_space: str = "strategic_v5",
        traces_path: str = "project/all_traces.jsonl",
        premise_index_path: str = "project/premise_index.json",
        strategic_weight: float = 0.4,  # fraction of output dedicated to strategic tactics
        use_premise_augmented_gen: bool = True,
    ):
        self._base_policy_type = base_policy
        self._gen_ckpt_dir = gen_ckpt_dir
        self._action_space_name = action_space
        self._traces_path = traces_path
        self._premise_index_path = premise_index_path
        self._strategic_weight = strategic_weight
        self._use_premise_augmented_gen = use_premise_augmented_gen

        # Lazy-loaded
        self._base_policy = None
        self._retriever: Optional[PremiseRetriever] = None

    def _ensure_loaded(self) -> None:
        if self._base_policy is not None:
            return

        # Load base policy
        if self._base_policy_type == "hybrid":
            from hybrid_policy import HybridPolicy
            self._base_policy = HybridPolicy(
                gen_ckpt_dir=self._gen_ckpt_dir,
                action_space=self._action_space_name,
            )
        elif self._base_policy_type == "generative":
            from generative_policy import GenerativePolicy
            self._base_policy = GenerativePolicy(ckpt_dir=self._gen_ckpt_dir)
        elif self._base_policy_type == "action_space":
            # Pure action space — no model needed
            self._base_policy = _ActionSpacePolicy(self._action_space_name)
        else:
            raise ValueError(f"Unknown base policy type: {self._base_policy_type}")

        # Load premise retriever
        self._retriever = PremiseRetriever()
        from pathlib import Path
        if Path(self._premise_index_path).exists():
            self._retriever.load_index(self._premise_index_path)
            print(f"[StrategicPolicy] Loaded premise index from {self._premise_index_path}")
        else:
            self._retriever.build_index_from_traces(self._traces_path)

        print(f"[StrategicPolicy] Ready: base={self._base_policy_type}, "
              f"strategic_weight={self._strategic_weight}")

    @property
    def model_type(self) -> str:
        return "strategic"

    def rank_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        k: int = 16,
    ) -> list[str]:
        """Generate top-k tactics combining strategic + base policy.

        Architecture:
          1. Retrieve relevant premises for this state
          2. Generate strategic tactics (backward, translation, premise-based)
          3. Generate base policy tactics (forward search)
          4. Interleave: strategic first, then base (deduplicated)
        """
        self._ensure_loaded()

        # Step 1: Retrieve premises
        premises = self._retriever.retrieve(state_pp, k=15)
        domains = _detect_domains(state_pp)

        # Step 2: Generate strategic tactics
        strategic_tactics = []

        # 2a: Backward reasoning (suffices, have)
        strategic_tactics.extend(_generate_suffices_tactics(state_pp, domains))

        # 2b: Cross-field translation
        strategic_tactics.extend(_generate_translation_tactics(state_pp, domains))

        # 2c: Premise-based tactics (have h := @Premise, rw [Premise], simp [Premise])
        strategic_tactics.extend(_generate_have_tactics(premises, state_pp))

        # Step 3: Base policy tactics
        k_base = max(k // 2, k - len(strategic_tactics))
        base_tactics = self._base_policy.rank_tactics(state_pp, full_name, k=k_base)

        # Step 4: Combine — strategic first, then base, deduplicated
        k_strategic = min(
            int(k * self._strategic_weight + 0.5),
            len(strategic_tactics),
        )

        result = []
        seen = set()

        # Strategic tactics first (up to weight allocation)
        for tac in strategic_tactics[:k_strategic]:
            if tac not in seen:
                seen.add(tac)
                result.append(tac)

        # Base tactics fill the rest
        for tac in base_tactics:
            if tac not in seen:
                seen.add(tac)
                result.append(tac)

        # If we still have room, add remaining strategic tactics
        if len(result) < k:
            for tac in strategic_tactics[k_strategic:]:
                if tac not in seen:
                    seen.add(tac)
                    result.append(tac)
                if len(result) >= k:
                    break

        return result[:k]

    def choose_tactic(self, state_pp: str, full_name: str = "") -> str:
        """Return the single best tactic."""
        tactics = self.rank_tactics(state_pp, full_name, k=1)
        return tactics[0] if tactics else "sorry"

    def get_premises(self, state_pp: str, k: int = 10) -> list[str]:
        """Expose retrieved premises for debugging/inspection."""
        self._ensure_loaded()
        return self._retriever.retrieve(state_pp, k=k)


class _ActionSpacePolicy:
    """Minimal wrapper to use a fixed action space as a 'policy'."""

    def __init__(self, action_space: str):
        self._actions = get_action_space(action_space)

    @property
    def model_type(self) -> str:
        return "action_space"

    def rank_tactics(self, state_pp: str, full_name: str = "", k: int = 10) -> list[str]:
        return self._actions[:k]

    def choose_tactic(self, state_pp: str, full_name: str = "") -> str:
        return self._actions[0] if self._actions else "sorry"
