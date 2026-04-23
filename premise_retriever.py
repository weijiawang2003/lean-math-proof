"""Lightweight premise retrieval for tactic generation.

Given a proof state, retrieves relevant lemma/theorem names that could
be used in `rw [...]`, `simp [...]`, `exact ...`, or `apply ...` tactics.

This captures the key insight from the ReProver paper (Yang et al., NeurIPS 2023):
a tactic generator performs much better when it knows *which premises* are
relevant, rather than having to hallucinate lemma names from memory.

Two retrieval strategies:
  1. Token-overlap (BM25-like): fast, no GPU, good baseline
  2. Embedding similarity: uses sentence-transformers for semantic matching
     (falls back to token-overlap if sentence-transformers not installed)

The premise index is built from:
  - Successful traces (which tactics used which lemma names)
  - A static catalog of common mathlib lemmas per domain

Usage:
    retriever = PremiseRetriever()
    retriever.build_index_from_traces("project/all_traces.jsonl")
    premises = retriever.retrieve(state_pp="⊢ a ∪ b = b ∪ a", k=10)
    # => ["Set.union_comm", "Set.ext_iff", "or_comm", ...]
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


# ── Premise extraction from tactic text ──────────────────────────────

# Patterns that reference specific lemma/theorem names in tactics
_BRACKET_ARGS = re.compile(r"\[([^\]]+)\]")          # simp [X, Y, Z]
_APPLY_EXACT = re.compile(r"(?:apply|exact)\s+(\S+)") # apply Foo / exact Foo
_RW_ARGS = re.compile(r"rw\s*\[([^\]]+)\]")           # rw [X, Y]
_HAVE_TYPE = re.compile(r"have\s+\w+\s*:\s*(.+?)\s*:=")  # have h : T := ...
_SUFFICES = re.compile(r"suffices\s+\w+\s*:\s*(.+?)\s+by") # suffices h : T by ...


def extract_premises_from_tactic(tactic: str) -> list[str]:
    """Extract premise/lemma names referenced in a tactic string.

    Examples:
        "simp [Set.ext_iff, or_comm]"  => ["Set.ext_iff", "or_comm"]
        "rw [Nat.add_comm]"            => ["Nat.add_comm"]
        "exact Set.subset_union_left"  => ["Set.subset_union_left"]
        "apply And.intro"              => ["And.intro"]
    """
    premises = []

    # Extract from bracket arguments (simp [...], rw [...], simp_all [...])
    for match in _BRACKET_ARGS.finditer(tactic):
        args_str = match.group(1)
        for arg in args_str.split(","):
            arg = arg.strip()
            # Remove leading ← (backward rewrite)
            if arg.startswith("←") or arg.startswith("<-"):
                arg = arg.lstrip("←<- ").strip()
            # Filter: must look like a name (contains ., is capitalized, or is a known pattern)
            if arg and not arg.startswith("*") and not arg.isdigit():
                # Accept: qualified names (Foo.bar), capitalized (And.intro),
                # or lowercase identifiers that look like lemmas (or_comm, add_assoc)
                if ("." in arg) or arg[0].isupper() or ("_" in arg and arg[0].isalpha()):
                    premises.append(arg)

    # Extract from apply/exact
    for match in _APPLY_EXACT.finditer(tactic):
        name = match.group(1)
        # Filter out common non-premise tokens
        if name and not name.startswith("(") and not name.startswith("⟨"):
            premises.append(name)

    # Filter out hypothesis references (h, hx, h1, hx.1, etc.)
    # These are local bindings, not reusable lemma names
    _HYPOTHESIS_PATTERN = re.compile(r'^h\w*(\.\d+)?$')
    premises = [p for p in premises if not _HYPOTHESIS_PATTERN.match(p)]

    return list(dict.fromkeys(premises))  # dedupe, preserve order


# ── Static premise catalog (common mathlib lemmas by domain) ────────

# These are high-value lemmas that appear frequently in proofs.
# Organized by the "namespace" / topic so the retriever can match
# state tokens to relevant domains.
STATIC_PREMISES: dict[str, list[str]] = {
    "Set": [
        "Set.ext_iff", "Set.subset_def", "Set.mem_union", "Set.mem_inter_iff",
        "Set.mem_diff", "Set.mem_compl_iff", "Set.mem_setOf_eq",
        "Set.union_comm", "Set.inter_comm", "Set.union_assoc", "Set.inter_assoc",
        "Set.union_empty", "Set.empty_union", "Set.inter_univ", "Set.univ_inter",
        "Set.union_self", "Set.inter_self", "Set.diff_self", "Set.diff_empty",
        "Set.subset_union_left", "Set.subset_union_right",
        "Set.inter_subset_left", "Set.inter_subset_right",
        "Set.union_inter_distrib_left", "Set.inter_union_distrib_left",
        "Set.diff_eq", "Set.compl_eq_univ_diff",
    ],
    "Finset": [
        "Finset.mem_insert", "Finset.mem_singleton", "Finset.mem_union",
        "Finset.mem_filter", "Finset.mem_sdiff", "Finset.mem_inter",
        "Finset.disjoint_left", "Finset.disjoint_insert_right",
        "Finset.insert_comm", "Finset.subset_iff",
        "Finset.card_insert_of_not_mem", "Finset.card_union_add_card_inter",
    ],
    "Nat": [
        "Nat.add_comm", "Nat.add_assoc", "Nat.add_left_comm",
        "Nat.mul_comm", "Nat.mul_assoc", "Nat.mul_left_comm",
        "Nat.add_zero", "Nat.zero_add", "Nat.mul_one", "Nat.one_mul",
        "Nat.mul_zero", "Nat.zero_mul", "Nat.succ_eq_add_one",
        "Nat.add_sub_cancel", "Nat.sub_self",
        "Nat.mul_add_mod", "Nat.add_mod", "Nat.mul_mod",
        "Nat.mod_eq_of_lt",
    ],
    "List": [
        "List.mem_cons_iff", "List.length_cons", "List.append_nil",
        "List.nil_append", "List.map_cons", "List.filter_cons",
    ],
    "logic": [
        "or_comm", "and_comm", "or_assoc", "and_assoc",
        "or_self", "and_self", "or_true", "true_or",
        "and_true", "true_and", "or_false", "false_or",
        "and_false", "false_and", "not_not",
        "and_or_left", "or_and_left",
        "Classical.em", "Classical.byContradiction",
    ],
    "Int": [
        "Int.add_comm", "Int.add_assoc", "Int.mul_comm", "Int.mul_assoc",
        "Int.add_zero", "Int.zero_add", "Int.mul_one", "Int.one_mul",
        "Int.neg_neg", "Int.sub_self",
    ],
}


# ── BM25-like token overlap scorer ──────────────────────────────────

def _tokenize_state(state_pp: str) -> list[str]:
    """Tokenize a proof state into words relevant for premise matching."""
    # Split on whitespace and common Lean separators
    tokens = re.split(r"[\s\n:→←⊢∀∃⟨⟩(){}[\],;]+", state_pp)
    # Keep meaningful tokens (identifiers, not single chars)
    return [t for t in tokens if len(t) > 1 and not t.startswith("✝")]


def _detect_domains(state_pp: str) -> list[str]:
    """Detect which mathematical domains appear in a proof state."""
    domains = []
    text = state_pp.lower()
    if "set " in text or "∪" in text or "∩" in text or "⊆" in text or "set." in state_pp:
        domains.append("Set")
    if "finset" in text or "Finset" in state_pp:
        domains.append("Finset")
    if ("nat" in text or "ℕ" in text or "Nat" in state_pp
            or re.search(r'\b\d+\b', state_pp) or "+ " in state_pp or "* " in state_pp):
        domains.append("Nat")
    if "list" in text or "List" in state_pp:
        domains.append("List")
    if "int " in text or "ℤ" in text or "Int" in state_pp:
        domains.append("Int")
    # Logic is always somewhat relevant
    if "∨" in text or "∧" in text or "¬" in text or "↔" in text:
        domains.append("logic")
    return domains if domains else ["logic"]  # default to logic


class PremiseRetriever:
    """Retrieves relevant premises for a given proof state.

    Combines:
      1. Static catalog of common mathlib lemmas (domain-matched)
      2. Premise usage patterns mined from successful traces
      3. Token-overlap scoring (BM25-like)
    """

    def __init__(self):
        # Premise -> list of states where it was used successfully
        self._premise_contexts: dict[str, list[str]] = defaultdict(list)
        # Premise -> usage count (for popularity weighting)
        self._premise_counts: Counter = Counter()
        # Total documents for IDF calculation
        self._n_docs: int = 0
        # Premise -> set of document indices containing it (for IDF)
        self._premise_doc_freq: Counter = Counter()
        # All known premise names
        self._all_premises: set[str] = set()
        # Whether index has been built
        self._indexed: bool = False

    def build_index_from_traces(self, traces_path: str, max_contexts: int = 50) -> None:
        """Build premise index from a traces JSONL file.

        Extracts which premises were used in successful tactics,
        paired with the proof state they were applied to.
        """
        path = Path(traces_path)
        if not path.exists():
            print(f"[PremiseRetriever] No traces at {traces_path}, using static catalog only")
            self._indexed = True
            return

        n_traces = 0
        n_premises_found = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Only learn from successful tactics (not errors)
                if rec.get("result_kind") == "LeanError":
                    continue

                tactic = rec.get("tactic", "")
                state = rec.get("state_pp", "")
                if not tactic or not state:
                    continue

                n_traces += 1
                premises = extract_premises_from_tactic(tactic)
                if premises:
                    for p in premises:
                        self._premise_counts[p] += 1
                        if len(self._premise_contexts[p]) < max_contexts:
                            self._premise_contexts[p].append(state)
                        self._premise_doc_freq[p] += 1
                        self._all_premises.add(p)
                        n_premises_found += 1

                self._n_docs += 1

        # Add static premises
        for domain, premises in STATIC_PREMISES.items():
            for p in premises:
                self._all_premises.add(p)

        self._indexed = True
        print(f"[PremiseRetriever] Indexed {n_traces} traces, "
              f"found {len(self._all_premises)} unique premises "
              f"({n_premises_found} total references)")

    def retrieve(
        self,
        state_pp: str,
        k: int = 15,
        include_static: bool = True,
    ) -> list[str]:
        """Retrieve the top-k most relevant premises for a proof state.

        Scoring combines:
          - Domain match: premises from detected domains get a boost
          - Token overlap: BM25-like score between state tokens and premise contexts
          - Popularity: frequently-successful premises get a small boost
        """
        if not self._indexed:
            # Auto-build from default location
            self.build_index_from_traces("project/all_traces.jsonl")

        state_tokens = set(_tokenize_state(state_pp))
        domains = _detect_domains(state_pp)

        scores: dict[str, float] = {}

        # 1. Score trace-mined premises by context similarity
        for premise, contexts in self._premise_contexts.items():
            # BM25-like: token overlap between query state and states where
            # this premise was successfully used
            best_overlap = 0.0
            for ctx in contexts:
                ctx_tokens = set(_tokenize_state(ctx))
                if not ctx_tokens:
                    continue
                overlap = len(state_tokens & ctx_tokens) / (len(ctx_tokens) ** 0.5 + 1)
                best_overlap = max(best_overlap, overlap)

            # IDF-like weighting: rare premises that match well are more valuable
            idf = 1.0
            if self._n_docs > 0:
                df = self._premise_doc_freq.get(premise, 1)
                idf = math.log(1 + self._n_docs / df)

            # Popularity boost (small): frequently-used premises are more likely correct
            pop = math.log(1 + self._premise_counts.get(premise, 0)) * 0.1

            scores[premise] = best_overlap * idf + pop

        # 2. Score static premises by domain match
        if include_static:
            for domain in domains:
                for premise in STATIC_PREMISES.get(domain, []):
                    # Domain match bonus
                    domain_bonus = 2.0

                    # Token overlap bonus: does the premise name appear in state?
                    name_parts = set(premise.replace(".", " ").split())
                    name_overlap = len(state_tokens & name_parts) * 0.5

                    static_score = domain_bonus + name_overlap
                    # Take max of static and trace-mined score
                    scores[premise] = max(scores.get(premise, 0), static_score)

            # Always-useful "cross-field" premises (logic layer)
            # These implement the collaborator's insight: logic premises
            # act as bridges between different mathematical domains
            for premise in STATIC_PREMISES.get("logic", []):
                if premise not in scores:
                    scores[premise] = 0.5  # low baseline, but always considered

        # 3. Sort by score and return top-k
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [name for name, _score in ranked[:k]]

    def format_premises_for_prompt(self, premises: list[str], max_chars: int = 300) -> str:
        """Format retrieved premises as a string to prepend to the generator input.

        This is the key connection to ReProver: we augment the proof state
        with relevant premise names so the generator can reference them.
        """
        if not premises:
            return ""
        text = "Relevant premises: " + ", ".join(premises)
        if len(text) > max_chars:
            text = text[:max_chars - 3] + "..."
        return text + "\n\n"

    def save_index(self, path: str) -> None:
        """Save the premise index to disk."""
        data = {
            "premise_counts": dict(self._premise_counts.most_common()),
            "premise_doc_freq": dict(self._premise_doc_freq),
            "n_docs": self._n_docs,
            "all_premises": sorted(self._all_premises),
        }
        Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def load_index(self, path: str) -> None:
        """Load a previously saved premise index."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self._premise_counts = Counter(data.get("premise_counts", {}))
        self._premise_doc_freq = Counter(data.get("premise_doc_freq", {}))
        self._n_docs = data.get("n_docs", 0)
        self._all_premises = set(data.get("all_premises", []))
        self._indexed = True


# ── Convenience: build and save index from CLI ──────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build premise retrieval index from traces.")
    parser.add_argument("--traces", default="project/all_traces.jsonl")
    parser.add_argument("--out", default="project/premise_index.json")
    parser.add_argument("--test-state", default="",
                        help="Test retrieval on a sample proof state.")
    args = parser.parse_args()

    retriever = PremiseRetriever()
    retriever.build_index_from_traces(args.traces)
    retriever.save_index(args.out)
    print(f"Saved premise index to {args.out}")

    # Test retrieval
    if args.test_state:
        test_state = args.test_state
    else:
        test_state = "α : Type u\ns t : Set α\n⊢ s ∪ t = t ∪ s"
    print(f"\nTest retrieval for state:\n  {test_state}")
    premises = retriever.retrieve(test_state, k=10)
    print(f"Top-10 premises:")
    for i, p in enumerate(premises, 1):
        print(f"  {i}. {p}")

    formatted = retriever.format_premises_for_prompt(premises)
    print(f"\nFormatted prompt prefix:\n  {formatted}")


if __name__ == "__main__":
    main()
