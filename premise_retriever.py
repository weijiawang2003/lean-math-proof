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
    # v4.1: div-family premises for Nat. Verified present in
    # Mathlib/Data/Nat/Defs.lean (or transitively imported there) against
    # the lean_dojo Mathlib4 cache at HEAD. Used by retrieve_for_state to
    # surface candidate `rw [..]` / `exact ..` lemmas when the wrapper's
    # div family activates on a Nat.div_* / Nat.dvd_* theorem.
    "Nat.div": [
        "Nat.div_le_div_right", "Nat.div_le_iff_le_mul",
        "Nat.div_lt_iff_lt_mul", "Nat.div_lt_iff_lt_mul'",
        "Nat.div_eq_of_lt", "Nat.div_pos", "Nat.div_pos_iff",
        "Nat.div_lt_one_iff", "Nat.div_eq_zero_iff",
        "Nat.dvd_iff_div_mul_eq",
        "Nat.mul_div_cancel", "Nat.div_mul_cancel",
        "Nat.mod_add_div", "Nat.lt_of_lt_of_le",
        "Nat.pos_of_ne_zero", "Nat.mod_eq_of_lt",
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


# ── v4.1 stateless retrieval helper ─────────────────────────────────

# Cheap family-key → catalog-bucket mapping. Used by retrieve_for_state
# to decide which static premise bucket to draw from when a strategy
# wrapper family activates. Kept narrow on purpose: v4.1 ships the div
# family only; v4.2 / v4.3 will extend this map.
_FAMILY_CATALOG_KEYS: dict[str, list[str]] = {
    "div": ["Nat.div"],
}


# v4.2 static availability denylist. Lemma names that produced `unknown
# constant` during the v4.1 eval AND would still produce it after the
# self-filter — i.e. unavailable for reasons other than self-reference.
# Two sub-classes:
#
#   1. Genuinely outside the import closure of nat_defs_medium's eval
#      environment (e.g. defined in a Mathlib file not transitively
#      imported by Mathlib/Data/Nat/Defs.lean):
#        - Nat.div_eq_zero_iff
#        - Nat.div_le_iff_le_mul
#
#   2. Forward-reference traps: target theorems of the eval set that
#      live in the same source file as other targets, so they are
#      unknown at the proof position of any target declared *earlier*
#      in the file. Confirmed in the v4.2-pre run where, after the
#      self-filter eliminated 116 attempts, 20 cross-theorem unknown_
#      constant errors remained — all pairs (target, other-target):
#        - Nat.div_le_div_right
#        - Nat.div_lt_one_iff
#        - Nat.div_pos
#        - Nat.div_pos_iff
#        - Nat.dvd_iff_div_mul_eq
#
# Listing class-2 lemmas globally is technically broader than the v4.1
# self-only behavior, but it captures the same intent (don't waste Lean
# roundtrips on lemmas we know will fail) and keeps the diagnostic
# separable from the self-filter via the filtered_self / filtered_
# unavailable counts. Class-2 lemmas can be re-added once v4.3's import-
# reachability checker can resolve their availability per proof site.
_UNAVAILABLE_LEMMAS: set[str] = {
    "Nat.div_eq_zero_iff",
    "Nat.div_le_iff_le_mul",
    "Nat.div_le_div_right",
    "Nat.div_lt_one_iff",
    "Nat.div_pos",
    "Nat.div_pos_iff",
    "Nat.dvd_iff_div_mul_eq",
}


def _name_namespace_variants(theorem_name: str) -> set[str]:
    """Return name variants for self-comparison.

    Returns the original name plus the bare unqualified name (after the
    last dot) and lowercased forms — enough to catch e.g. retrieved
    `Nat.div_pos` when the target is also `Nat.div_pos` even if the
    catalog ever ships an unqualified alias.
    """
    variants: set[str] = {theorem_name}
    if "." in theorem_name:
        variants.add(theorem_name.rsplit(".", 1)[-1])
    variants.add(theorem_name.lower())
    return variants


def retrieve_for_state(
    state_pp: str,
    theorem_name: str | None = None,
    k: int = 10,
    family_key: str | None = None,
    filter_self: bool = True,
    filter_unavailable: bool = True,
    return_diagnostics: bool = False,
):
    """Return up to `k` lemma names relevant to the given proof state.

    Stateless and deterministic — does not load traces or external models.
    Designed to be called once per state by `StrategyWrapperPolicy` when a
    theorem family activates. Returns lemma names only; the caller is
    responsible for wrapping them in `rw [..]` / `exact ..` etc.

    Args:
        state_pp: the Lean state pretty-print (full hypotheses + goal).
        theorem_name: full theorem name (e.g. `Nat.div_pos_iff`); used to
            bias scoring toward premises whose name shares tokens with it.
        k: max number of names to return.
        family_key: which family the caller has identified (e.g. "div").
            If provided, the corresponding catalog bucket is used; otherwise
            buckets are picked by substring detection on the theorem name
            and the state.

    Returns:
        By default, up to k lemma names ranked by token-overlap score
        (descending). When return_diagnostics=True, returns a tuple
        ``(premises, diag)`` where ``diag`` is a dict with counts of
        ``filtered_self`` and ``filtered_unavailable`` premises removed
        from the catalog before scoring. Order is deterministic for a
        given input.
    """
    diag = {"filtered_self": 0, "filtered_unavailable": 0}
    if k <= 0:
        return ([], diag) if return_diagnostics else []

    # Pick catalog buckets to draw from
    buckets: list[str] = []
    if family_key and family_key in _FAMILY_CATALOG_KEYS:
        buckets = list(_FAMILY_CATALOG_KEYS[family_key])
    else:
        # Heuristic fallback: substring-match on the theorem name
        name_lower = (theorem_name or "").lower()
        if "div" in name_lower or "dvd" in name_lower:
            buckets = list(_FAMILY_CATALOG_KEYS["div"])

    if not buckets:
        return ([], diag) if return_diagnostics else []

    candidates: list[str] = []
    seen: set[str] = set()
    for bucket in buckets:
        for premise in STATIC_PREMISES.get(bucket, []):
            if premise in seen:
                continue
            seen.add(premise)
            candidates.append(premise)

    # v4.2 filters: target-theorem self-retrieval and known-unavailable
    # lemmas. Applied before scoring so the diagnostic counts reflect
    # how many catalog entries each filter actually removed for this call.
    if filter_self and theorem_name:
        self_variants = _name_namespace_variants(theorem_name)
        kept: list[str] = []
        for p in candidates:
            if p in self_variants:
                diag["filtered_self"] += 1
            else:
                kept.append(p)
        candidates = kept

    if filter_unavailable:
        kept = []
        for p in candidates:
            if p in _UNAVAILABLE_LEMMAS:
                diag["filtered_unavailable"] += 1
            else:
                kept.append(p)
        candidates = kept

    if not candidates:
        return ([], diag) if return_diagnostics else []

    # Score by token overlap. The query tokens come from both the state
    # pretty-print and the theorem name (the latter is highly diagnostic
    # — e.g. `Nat.div_pos_iff` shares `div`, `pos`, `iff` with the right
    # premises in the bucket).
    state_tokens: set[str] = set(_tokenize_state(state_pp))
    theorem_tokens: set[str] = set()
    if theorem_name:
        # Split the name on dots and underscores so e.g. "Nat.div_pos_iff"
        # yields {"Nat", "div", "pos", "iff"}.
        for part in re.split(r"[._]", theorem_name):
            part = part.strip()
            if len(part) > 1:
                theorem_tokens.add(part)
    query_tokens = state_tokens | theorem_tokens

    # Family token: which top-level family this call is targeting. Used
    # as a small bonus so premises whose names contain the family token
    # (e.g. "div") rank above premises that happen to share other tokens.
    family_token = family_key.lower() if family_key else None

    scored: list[tuple[float, int, str]] = []
    for idx, premise in enumerate(candidates):
        # Split premise name into parts (Nat.div_pos_iff → div, pos, iff).
        # Drop the "Nat" namespace token — too generic to be diagnostic.
        parts = [p for p in re.split(r"[._]", premise) if p and p != "Nat"]
        parts_set = set(parts)
        overlap = sum(1 for p in parts if p in query_tokens)
        # Family bonus: lemma shares the family token with the theorem.
        family_bonus = 0.5 if (family_token and family_token in parts_set
                               and family_token in theorem_tokens) else 0.0
        score = overlap + family_bonus
        # Tiebreak by catalog order (lower idx wins) so the output is stable.
        scored.append((-score, idx, premise))

    scored.sort()
    result = [premise for _score, _idx, premise in scored[:k]]
    return (result, diag) if return_diagnostics else result


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
