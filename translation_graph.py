"""Translation Graph: the algorithmic core of cross-field mathematical reasoning.

This module encodes the collaborator's key insight as a computable data structure:
mathematical proofs often succeed by translating the problem from one "domain"
(field of math) to another where the available tools are more powerful.

The Translation Graph has:
  - NODES: mathematical domains (Set, Finset, Nat, logic, arithmetic, ...)
  - EDGES: Lean tactics that translate between domains
  - WEIGHTS: learned success rates for each translation

The graph answers three questions algorithmically:
  1. DETECT: What domain is the current proof state in?
  2. PLAN:   What translations are available, and which are most likely to help?
  3. ACT:    What specific Lean tactics implement the chosen translation?

This is the bridge between the abstract idea ("think like Grothendieck —
find the right category to work in") and concrete tactic generation.

Usage:
    graph = TranslationGraph()
    graph.learn_from_proofs(project_state_path)

    state_pp = "⊢ s ∪ t = t ∪ s"
    plan = graph.plan_translations(state_pp)
    # => [('Set', 'logic', 0.87, ['ext x', 'simp [or_comm]']),
    #     ('Set', 'automation', 0.65, ['aesop']),
    #     ...]
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ══════════════════════════════════════════════════════════════════════
#  DOMAIN CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════

# Each domain represents a "mathematical world" with its own language,
# concepts, and proof tools.  The key insight: the SAME mathematical
# fact can often be stated in multiple domains, and some domains make
# the proof trivial.

DOMAINS = [
    "Set",          # Set theory: ∪, ∩, ⊆, ∅, univ
    "Finset",       # Finite sets: insert, mem, disjoint, card
    "Nat",          # Natural numbers: +, *, mod, div, succ
    "Int",          # Integers
    "List",         # Lists: cons, append, map, filter
    "Multiset",     # Multisets (bridge between List and Finset)
    "Bool",         # Boolean algebra
    "logic",        # Propositional logic: ∨, ∧, ¬, →, ↔
    "arithmetic",   # Abstract arithmetic: ring, omega, linarith
    "order",        # Partial orders: ≤, <, sup, inf
    "membership",   # Element-level: x ∈ S, ∀ x, ∃ x
    "equality",     # Equational reasoning: rfl, congr, symm, trans
    "automation",   # Domain-agnostic automation: aesop, simp, tauto
]


def detect_domain_from_name(theorem_name: str, file_path: str = "") -> str:
    """Classify a theorem into its primary domain by name/file."""
    name = theorem_name
    if "Finset" in name or "Finset/" in file_path:
        return "Finset"
    if "Set." in name or "Set/" in file_path:
        return "Set"
    if "Nat." in name or "Nat/" in file_path:
        return "Nat"
    if "Int." in name or "Int/" in file_path:
        return "Int"
    if "List" in name or "List/" in file_path:
        return "List"
    if "Multiset" in name:
        return "Multiset"
    if "Bool" in name:
        return "Bool"
    return "logic"


def detect_domain_from_state(state_pp: str) -> list[str]:
    """Detect which domains appear in a proof state.

    Returns a ranked list: primary domain first, then secondary.
    A proof state can involve multiple domains simultaneously.
    """
    domains = []
    scores: dict[str, float] = {}

    # Set indicators
    set_score = 0
    if "∪" in state_pp or "∩" in state_pp:
        set_score += 2
    if "⊆" in state_pp:
        set_score += 2
    if "Set " in state_pp or "Set." in state_pp:
        set_score += 1
    if "∅" in state_pp or "univ" in state_pp:
        set_score += 1
    if set_score > 0:
        scores["Set"] = set_score

    # Finset indicators
    finset_score = 0
    if "Finset" in state_pp:
        finset_score += 3
    if "insert" in state_pp and "Finset" in state_pp:
        finset_score += 1
    if "Disjoint" in state_pp:
        finset_score += 1
    if finset_score > 0:
        scores["Finset"] = finset_score

    # Nat indicators
    nat_score = 0
    if "ℕ" in state_pp or "Nat" in state_pp:
        nat_score += 2
    if re.search(r"\b\d+\b", state_pp):
        nat_score += 1
    if "mod" in state_pp or "div" in state_pp or "%" in state_pp:
        nat_score += 2
    if "succ" in state_pp:
        nat_score += 1
    if nat_score > 0:
        scores["Nat"] = nat_score

    # Logic indicators
    logic_score = 0
    if "∨" in state_pp or "∧" in state_pp:
        logic_score += 2
    if "¬" in state_pp:
        logic_score += 1
    if "↔" in state_pp:
        logic_score += 2
    if "→" in state_pp:
        logic_score += 1
    if "True" in state_pp or "False" in state_pp:
        logic_score += 1
    if logic_score > 0:
        scores["logic"] = logic_score

    # Membership indicators
    if "∈" in state_pp or "∉" in state_pp:
        scores["membership"] = 2

    # Equality indicators
    if " = " in state_pp:
        scores["equality"] = 1

    # Order indicators
    if "≤" in state_pp or "≥" in state_pp or " < " in state_pp or " > " in state_pp:
        scores["order"] = 2

    # Sort by score
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [d for d, _ in ranked] if ranked else ["logic"]


def classify_tactic_domain(tactic: str) -> str:
    """Classify what domain a tactic operates in / translates to.

    This is the critical function: it identifies WHAT WORLD the tactic
    takes us to, which may differ from the world the problem started in.
    """
    # Pure automation — transcends domains
    if tactic in ("aesop", "tauto", "trivial", "assumption", "decide"):
        return "automation"

    # Arithmetic world
    if tactic in ("omega", "ring", "ring_nf", "norm_num", "linarith", "simp_arith"):
        return "arithmetic"
    if "Nat." in tactic or "Int." in tactic:
        return "arithmetic"

    # Translation TO logic (the most common cross-field move)
    if "ext_iff" in tactic or "subset_def" in tactic:
        return "logic"      # Set equality → ∀ x, ↔ | Set subset → ∀ x, →
    if "or_comm" in tactic or "and_comm" in tactic or "or_assoc" in tactic:
        return "logic"
    if tactic.startswith("push_neg"):
        return "logic"
    if tactic in ("constructor", "left", "right", "exfalso", "by_contra h"):
        return "logic"

    # Translation TO membership (element-level reasoning)
    if "mem_union" in tactic or "mem_inter" in tactic or "mem_diff" in tactic:
        return "membership"
    if "mem_insert" in tactic or "mem_singleton" in tactic or "mem_filter" in tactic:
        return "membership"
    if "mem_setOf" in tactic:
        return "membership"

    # Translation TO pointwise (ext tactic)
    if tactic.startswith("ext"):
        return "membership"  # equality → pointwise membership

    # Set-internal
    if "Set." in tactic:
        return "Set"

    # Finset-internal
    if "Finset." in tactic:
        return "Finset"

    # List-internal
    if "List." in tactic:
        return "List"

    # Intro/intros → going to element-level
    if tactic.startswith("intro"):
        return "membership"

    # Generic simp
    if "simp" in tactic:
        return "automation"

    return "other"


# ══════════════════════════════════════════════════════════════════════
#  TRANSLATION EDGE
# ══════════════════════════════════════════════════════════════════════

@dataclass
class TranslationEdge:
    """An edge in the translation graph: a way to move from one domain to another.

    Each edge represents a "mathematical insight" — the recognition that
    a problem in domain A can be restated in domain B where it becomes easier.
    """
    source: str           # domain the problem is stated in
    target: str           # domain we translate to
    tactics: list[str]    # Lean tactics that implement this translation
    success_count: int    # how many times this translation led to a proof
    attempt_count: int    # how many times this translation was tried
    example_theorems: list[str]  # theorems proved via this edge

    @property
    def success_rate(self) -> float:
        return self.success_count / self.attempt_count if self.attempt_count > 0 else 0.0

    @property
    def is_cross_field(self) -> bool:
        """True if this edge represents a genuine cross-field translation."""
        return self.source != self.target and self.target != "automation"


# ══════════════════════════════════════════════════════════════════════
#  TRANSLATION GRAPH
# ══════════════════════════════════════════════════════════════════════

class TranslationGraph:
    """The mathematical domain translation graph.

    This is the algorithmic representation of the collaborator's insight:
    genius mathematicians solve problems by finding the right "field"
    to work in.  The graph encodes which fields exist, how to move
    between them, and how likely each translation is to succeed.

    Three-phase algorithm:
      1. DETECT: identify the domain(s) of the current proof state
      2. PLAN:   rank available translations by expected success rate
      3. ACT:    return the Lean tactics that implement the best translation

    The graph is learned from proof data — every proved theorem teaches
    us which translations work.
    """

    def __init__(self):
        self.edges: dict[tuple[str, str], TranslationEdge] = {}
        self._learned = False

    def _get_or_create_edge(self, source: str, target: str) -> TranslationEdge:
        key = (source, target)
        if key not in self.edges:
            self.edges[key] = TranslationEdge(
                source=source, target=target,
                tactics=[], success_count=0, attempt_count=0,
                example_theorems=[],
            )
        return self.edges[key]

    def learn_from_proofs(self, project_state_path: str) -> None:
        """Learn translation edges from proved theorems.

        For each proved theorem:
          - Determine its problem domain (from theorem name/file)
          - Determine its solution domain (from winning tactic)
          - Record this as a successful translation edge
        """
        data = json.loads(Path(project_state_path).read_text(encoding="utf-8"))

        for name, t in data["theorems"].items():
            if not t.get("proved"):
                continue

            tactic = t.get("proof_tactics", "")
            if not tactic:
                continue

            prob_domain = detect_domain_from_name(name, t.get("file_path", ""))
            sol_domain = classify_tactic_domain(tactic)

            edge = self._get_or_create_edge(prob_domain, sol_domain)
            edge.success_count += 1
            edge.attempt_count += 1  # we only see successes from project state

            if tactic not in edge.tactics:
                edge.tactics.append(tactic)
            if len(edge.example_theorems) < 10:
                edge.example_theorems.append(name)

        # Also learn from failures (searched but unproved)
        for name, t in data["theorems"].items():
            if t.get("searched") and not t.get("proved"):
                prob_domain = detect_domain_from_name(name, t.get("file_path", ""))
                # We don't know which translations were attempted,
                # but we know the common ones failed
                for target in ["automation", "logic", "arithmetic", "membership"]:
                    edge = self._get_or_create_edge(prob_domain, target)
                    edge.attempt_count += 1

        self._learned = True

    def plan_translations(
        self,
        state_pp: str,
        theorem_name: str = "",
        file_path: str = "",
        k: int = 5,
    ) -> list[dict]:
        """Plan which translations to try for a given proof state.

        Returns ranked list of translation plans:
          [{"source": "Set", "target": "logic", "confidence": 0.87,
            "tactics": ["ext x", "simp [or_comm]"], "reason": "..."},
           ...]

        This is the CORE ALGORITHM — it decides how the model should
        "think about" the problem before generating tactics.
        """
        # Phase 1: DETECT domains
        if theorem_name:
            primary = detect_domain_from_name(theorem_name, file_path)
        else:
            detected = detect_domain_from_state(state_pp)
            primary = detected[0] if detected else "logic"

        state_domains = detect_domain_from_state(state_pp)

        # Phase 2: RANK translations
        candidates = []

        for (src, tgt), edge in self.edges.items():
            # Only consider edges from the problem's domain
            if src != primary:
                continue

            confidence = edge.success_rate

            # Boost cross-field translations (the interesting ones)
            boost = 1.0
            if edge.is_cross_field:
                boost = 1.2  # prefer cross-field over same-domain

            # Boost if the target domain's "language" appears in the state
            # (suggests the translation is partially done already)
            if tgt in state_domains:
                boost *= 1.1

            score = confidence * boost

            reason = self._explain_translation(src, tgt, edge)

            candidates.append({
                "source": src,
                "target": tgt,
                "confidence": round(confidence, 3),
                "score": round(score, 3),
                "tactics": edge.tactics[:5],
                "is_cross_field": edge.is_cross_field,
                "success_count": edge.success_count,
                "reason": reason,
            })

        # Sort by score
        candidates.sort(key=lambda x: -x["score"])
        return candidates[:k]

    def get_translation_tactics(
        self,
        source_domain: str,
        target_domain: str,
    ) -> list[str]:
        """Get the tactics that implement a specific translation."""
        key = (source_domain, target_domain)
        if key in self.edges:
            return self.edges[key].tactics
        return []

    def get_all_tactics_for_domain(self, domain: str) -> list[str]:
        """Get all tactics that translate FROM a given domain."""
        tactics = []
        for (src, _), edge in self.edges.items():
            if src == domain:
                tactics.extend(edge.tactics)
        return list(dict.fromkeys(tactics))  # dedupe

    def _explain_translation(self, src: str, tgt: str, edge: TranslationEdge) -> str:
        """Generate a human-readable explanation of why this translation works."""
        explanations = {
            ("Set", "logic"): "Reduce set operations to propositional logic (∪→∨, ∩→∧, ⊆→→)",
            ("Set", "membership"): "Reason about individual elements instead of whole sets",
            ("Set", "automation"): "Use automated Set decision procedures",
            ("Finset", "automation"): "Use automated Finset decision procedures",
            ("Finset", "Set"): "Treat finite sets as sets (coercion), use Set tools",
            ("Finset", "membership"): "Reason about Finset membership element-by-element",
            ("Nat", "arithmetic"): "Use arithmetic solvers (omega, ring) instead of structural induction",
            ("Nat", "logic"): "Reduce number theory to logical conditions",
        }
        return explanations.get((src, tgt), f"Translate {src} problem into {tgt} language")

    def summary(self) -> str:
        """Print a summary of the translation graph."""
        lines = []
        lines.append("Translation Graph Summary:")
        lines.append(f"  Edges: {len(self.edges)}")

        cross = [e for e in self.edges.values() if e.is_cross_field]
        lines.append(f"  Cross-field edges: {len(cross)}")
        lines.append("")

        # Group by source
        by_source: dict[str, list] = defaultdict(list)
        for (src, tgt), edge in self.edges.items():
            by_source[src].append((tgt, edge))

        for src in sorted(by_source):
            targets = by_source[src]
            total_success = sum(e.success_count for _, e in targets)
            lines.append(f"  {src} ({total_success} proved):")
            for tgt, edge in sorted(targets, key=lambda x: -x[1].success_count):
                cf = " ★" if edge.is_cross_field else ""
                lines.append(
                    f"    → {tgt:<15s} : {edge.success_count:3d} proved, "
                    f"rate={edge.success_rate:.0%}{cf}"
                )

        return "\n".join(lines)

    def save(self, path: str) -> None:
        """Save graph to JSON."""
        data = {}
        for (src, tgt), edge in self.edges.items():
            data[f"{src}->{tgt}"] = {
                "source": src, "target": tgt,
                "tactics": edge.tactics,
                "success_count": edge.success_count,
                "attempt_count": edge.attempt_count,
                "success_rate": edge.success_rate,
                "is_cross_field": edge.is_cross_field,
                "example_theorems": edge.example_theorems,
            }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load(self, path: str) -> None:
        """Load graph from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        for key, info in data.items():
            edge = self._get_or_create_edge(info["source"], info["target"])
            edge.tactics = info["tactics"]
            edge.success_count = info["success_count"]
            edge.attempt_count = info["attempt_count"]
            edge.example_theorems = info.get("example_theorems", [])
        self._learned = True


# ══════════════════════════════════════════════════════════════════════
#  TRANSLATION-GUIDED POLICY
# ══════════════════════════════════════════════════════════════════════

class TranslationGuidedPolicy:
    """A policy that uses the translation graph to guide tactic selection.

    This is the algorithmic realization of the collaborator's vision:
    instead of blindly trying tactics, first ASK "what domain should I
    translate this problem to?" and then generate tactics for that domain.

    Algorithm:
      1. Detect the problem's domain from the proof state
      2. Query the translation graph for the best translation edges
      3. For each promising translation: generate the implementing tactics
      4. Combine with base policy tactics (for execution after translation)
    """

    def __init__(
        self,
        graph: TranslationGraph,
        base_policy=None,
        max_translations: int = 3,
    ):
        self.graph = graph
        self._base_policy = base_policy
        self._max_translations = max_translations

    @property
    def model_type(self) -> str:
        return "translation_guided"

    def rank_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        k: int = 12,
    ) -> list[str]:
        """Generate tactics guided by the translation graph.

        Strategy:
          1. Ask the graph: "what translations should I try?"
          2. For each translation: add its implementing tactics
          3. Fill remaining slots with base policy tactics
        """
        # Step 1: Plan translations
        plans = self.graph.plan_translations(
            state_pp, theorem_name=full_name, k=self._max_translations
        )

        # Step 2: Collect translation tactics (ordered by confidence)
        translation_tactics = []
        for plan in plans:
            for tac in plan["tactics"]:
                if tac not in translation_tactics:
                    translation_tactics.append(tac)

        # Step 3: Base policy tactics
        base_tactics = []
        if self._base_policy is not None:
            base_tactics = self._base_policy.rank_tactics(state_pp, full_name, k=k)

        # Step 4: Interleave — translations first, then base
        result = []
        seen = set()

        # Translation tactics get priority
        for tac in translation_tactics:
            if tac not in seen and len(result) < k:
                seen.add(tac)
                result.append(tac)

        # Base policy fills the rest
        for tac in base_tactics:
            if tac not in seen and len(result) < k:
                seen.add(tac)
                result.append(tac)

        return result[:k]

    def choose_tactic(self, state_pp: str, full_name: str = "") -> str:
        tactics = self.rank_tactics(state_pp, full_name, k=1)
        return tactics[0] if tactics else "sorry"


# ══════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build and query the translation graph.")
    parser.add_argument("--project-state", default="project/project_state.json")
    parser.add_argument("--out", default="project/translation_graph.json")
    parser.add_argument("--test-state", default="",
                        help="Test translation planning on a proof state.")
    args = parser.parse_args()

    graph = TranslationGraph()
    graph.learn_from_proofs(args.project_state)
    graph.save(args.out)

    print(graph.summary())
    print(f"\nSaved to {args.out}")

    # Test
    if args.test_state:
        test = args.test_state
    else:
        test = "α : Type u\ns t : Set α\n⊢ s ∪ t = t ∪ s"

    print(f"\n{'='*60}")
    print(f"Translation plan for: {test.splitlines()[-1]}")
    print(f"{'='*60}")

    plans = graph.plan_translations(test)
    for i, plan in enumerate(plans, 1):
        cf = " ★ CROSS-FIELD" if plan["is_cross_field"] else ""
        print(f"\n  Option {i}: {plan['source']} → {plan['target']}"
              f" (confidence={plan['confidence']:.0%}){cf}")
        print(f"  Reason: {plan['reason']}")
        print(f"  Tactics: {plan['tactics'][:3]}")


if __name__ == "__main__":
    main()
