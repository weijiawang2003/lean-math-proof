"""Strategy wrapper — the v3 / v3.1 hybrid_evolved policy.

Wraps a base generative policy (typically GenerativePolicy on gen_v5) and
augments each rank_tactics() output with the candidate's fallback_tactics
and tactic_templates. The wrapper exists so that those two genome fields,
which were inert in v1/v2, actually affect Lean evaluation without
retraining the model or touching its checkpoint.

Ordering at every step:
  1. base policy's top-k generative tactics (in beam-search order)
  2. candidate.fallback_tactics (in list order, applied verbatim)
  3. candidate.tactic_templates (in list order, rendered per Nat variable in
     scope at this state; templates without `{var}` are emitted once)

Duplicates are removed preserving first-occurrence order. The wrapper also
exposes `last_origins` parallel to the returned list so the eval loop can
tag the winning tactic by source, and `last_template_sources` so the trace
can also record the raw (pre-render) template string for template entries.

Read from a JSON config (so the eval subprocess can pick up the candidate's
genome without a CLI explosion):

    {
      "fallback_tactics": ["omega", "simp_arith", ...],
      "tactic_templates": ["induction {var} with | zero => simp | succ n ih => simp [ih]"]
    }
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ORIGIN_GENERATIVE = "generative_topk"
ORIGIN_FALLBACK = "fallback_tactic"
ORIGIN_TEMPLATE = "tactic_template"
ORIGIN_FAMILY = "family_tactic"
# v4.1: retrieved premise origin. Tagged on synthesized rw/simp/exact/apply
# entries whose lemma name came from premise_retriever.retrieve_for_state.
ORIGIN_RETRIEVED = "retrieved_premise"
# v5: term-mode proof skeleton origin. Tagged on candidate tactics emitted
# by the term_builder block (exact ⟨…, …⟩ / refine ⟨?_, ?_⟩ etc.). The
# block runs per goal shape so each candidate carries its own shape-gate
# in the genome rather than being broadcast against every state.
ORIGIN_TERM_BUILDER = "term_builder"

# Default tactic forms used to wrap a retrieved lemma name. `{p}` is the
# substitution placeholder. Override per-call via retrieval_tactic_forms.
_DEFAULT_RETRIEVAL_TACTIC_FORMS: list[str] = [
    "rw [{p}]",
    "simp [{p}]",
    "exact {p}",
    "apply {p}",
]

# v4.2 form-family aliases: when the candidate ships short names like
# "rw" / "simp" / "apply" / "exact" instead of full templates, expand
# them so the JSON config surface can stay terse.
_FORM_FAMILY_TEMPLATES: dict[str, str] = {
    "rw": "rw [{p}]",
    "simp": "simp [{p}]",
    "apply": "apply {p}",
    "exact": "exact {p}",
}


def _normalize_retrieval_forms(forms: list[str] | None) -> list[str]:
    """Expand short form-family names to full templates.

    Accepts a mix: "rw" → "rw [{p}]", "simp [{p}]" passes through.
    Empty / None falls back to the default form list. Deduplicates while
    preserving caller order.
    """
    if not forms:
        return list(_DEFAULT_RETRIEVAL_TACTIC_FORMS)
    out: list[str] = []
    seen: set[str] = set()
    for f in forms:
        if not f:
            continue
        expanded = _FORM_FAMILY_TEMPLATES.get(f.strip(), f)
        if expanded not in seen:
            seen.add(expanded)
            out.append(expanded)
    return out or list(_DEFAULT_RETRIEVAL_TACTIC_FORMS)


def _form_family_label(form_template: str) -> str:
    """Return the canonical form-family label for a tactic form.

    Examples:
        "rw [{p}]" → "rw"
        "simp [{p}]" → "simp"
        "exact {p}" → "exact"
        "apply {p}" → "apply"
    Anything unusual returns the first whitespace-separated token.
    """
    token = form_template.strip().split(None, 1)[0] if form_template.strip() else ""
    return token.strip("[]")


def _match_families(theorem_name: str, family_keys: list[str]) -> list[str]:
    """Return the subset of family_keys whose key appears as a substring
    in theorem_name, preserving the order of family_keys.

    Case-sensitive match on the literal substring — keeps activation
    deterministic and predictable across runs. The caller controls
    specificity ordering by the order in which they declare families
    (more-specific keys, e.g. "add_mod_eq_ite", should appear before
    less-specific keys, e.g. "mod", so their tactics queue first).
    """
    name = theorem_name or ""
    return [k for k in family_keys if k and k in name]

# Matches a Lean state line like "a b c d m n k : ℕ" (or "n : Nat").
# Function-typed hypotheses (e.g. "p q : ℕ → Prop") are intentionally
# rejected because the type doesn't terminate at the end of line.
_NAT_LINE = re.compile(
    r"^\s*([A-Za-z_][\w']*(?:\s+[A-Za-z_][\w']*)*)\s*:\s*(ℕ|Nat)\s*$"
)

# v4.5: matches a single-named hypothesis line like "h : a ≤ b" or
# "hba : b ≤ a" or "hb : 0 < b". The single-name LHS distinguishes
# these from multi-name binder lines (`a b c : ℕ`) which `_NAT_LINE`
# already catches.
_HYP_LINE = re.compile(r"^\s*([A-Za-z_][\w']*)\s*:\s*(.+)$")
_POS_PREFIX = re.compile(r"^\s*0\s*<")


def _extract_hypotheses(state_pp: str) -> dict[str, str | None]:
    """Return a map of hypothesis-shape placeholders to actual names.

    The current shapes supported:
      hyp_le:        first hypothesis of shape `x ≤ y` (any expressions)
      hyp_pos:       first hypothesis of shape `0 < x`
      hyp_ne_zero:   first hypothesis of shape `x ≠ 0`

    The first match wins (stable ordering). Hypothesis lines after the
    `⊢` goal line are ignored. Lines that aren't single-named hypotheses
    (e.g. `a b c : ℕ` binders) are silently skipped.
    """
    result: dict[str, str | None] = {
        "hyp_le": None,
        "hyp_pos": None,
        "hyp_ne_zero": None,
    }
    for line in state_pp.splitlines():
        s = line.lstrip()
        if s.startswith("⊢"):
            break
        m = _HYP_LINE.match(s)
        if not m:
            continue
        name, type_str = m.group(1), m.group(2)
        # Skip the `a b c : ℕ` binder line — it has multiple names left
        # of `:` and would be caught by `_NAT_LINE`. Defensive recheck.
        if name in ("⊢",):
            continue
        if result["hyp_le"] is None and " ≤ " in type_str:
            result["hyp_le"] = name
        if result["hyp_pos"] is None and _POS_PREFIX.match(type_str):
            result["hyp_pos"] = name
        if result["hyp_ne_zero"] is None and "≠ 0" in type_str:
            result["hyp_ne_zero"] = name
    return result


_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _extract_nat_vars(state_pp: str) -> list[str]:
    """Extract Nat-typed identifiers from a Lean state pretty-print.

    Preference: variables that actually appear in the current goal line
    (the one beginning with `⊢`). This keeps templates focused — inducting
    on a variable that never appears in the goal is wasted Lean time.
    Falls back to the full Nat-binding set when no goal var matches (rare).
    """
    nat_vars: list[str] = []
    seen: set[str] = set()
    goal_text: str | None = None

    for line in state_pp.splitlines():
        m = _NAT_LINE.match(line)
        if m:
            for tok in m.group(1).split():
                if tok and tok not in seen:
                    seen.add(tok)
                    nat_vars.append(tok)
            continue
        stripped = line.lstrip()
        if goal_text is None and stripped.startswith("⊢"):
            goal_text = stripped[1:]

    if goal_text and nat_vars:
        used = [
            v for v in nat_vars
            if re.search(rf"\b{re.escape(v)}\b", goal_text)
        ]
        if used:
            return used
    return nat_vars


def _render_template(
    template: str,
    nat_vars: list[str],
    hypotheses: dict[str, str | None] | None = None,
) -> list[str]:
    """Render a tactic template, substituting placeholders.

    Supported placeholders (any subset can appear in a template):
      `{var}`        renders once per Nat-typed identifier in scope
      `{hyp_le}`     name of a hypothesis with type `x ≤ y`
      `{hyp_pos}`    name of a hypothesis with type `0 < x`
      `{hyp_ne_zero}` name of a hypothesis with type `x ≠ 0`

    Rules:
      * No placeholders → render the template verbatim once.
      * `{var}` present, no Nat vars → skip (fictitious name would just
        error in Lean and waste a roundtrip).
      * Any `{hyp_*}` placeholder present but the corresponding
        hypothesis isn't in scope → skip the template entirely.
      * If `{var}` AND `{hyp_*}` both present, render once per Nat var
        with the same hypothesis name substituted.

    The shape-extractors run once per `rank_tactics` call and the same
    `hypotheses` dict is reused across every template.
    """
    hypotheses = hypotheses or {}
    placeholders = set(_PLACEHOLDER_RE.findall(template))
    if not placeholders:
        return [template]

    # Verify every hyp_* placeholder has a backing hypothesis name.
    for ph in placeholders:
        if ph == "var":
            continue
        if hypotheses.get(ph) is None:
            return []

    def _substitute(s: str, var: str | None = None) -> str:
        if var is not None:
            s = s.replace("{var}", var)
        for ph in placeholders:
            if ph == "var":
                continue
            name = hypotheses.get(ph)
            if name is not None:
                s = s.replace(f"{{{ph}}}", name)
        return s

    if "var" in placeholders:
        if not nat_vars:
            return []
        return [_substitute(template, var=v) for v in nat_vars]
    return [_substitute(template)]


class StrategyWrapperPolicy:
    """Augment a base generative policy with candidate-provided tactics.

    The base policy's `rank_tactics(state_pp, full_name, k)` is called
    unchanged; the wrapper then appends fallback_tactics + rendered
    tactic_templates, deduped, and returns the longer list. eval_rollout_all
    will try every tactic in this list before declaring the step a failure.

    After each `rank_tactics` call:
      - `last_ranked_tactics`     — the full (possibly-extended) list
      - `last_origins`            — parallel origin labels
      - `last_template_sources`   — parallel raw template strings (only
                                    populated for tactic_template entries;
                                    None otherwise)
      - `last_family_sources`     — parallel family-key strings (only
                                    populated for family_tactic AND
                                    retrieved_premise entries; None
                                    otherwise)
      - `last_retrieved_premises` — parallel lemma names (only populated
                                    for retrieved_premise entries; None
                                    otherwise)
      - `last_activated_families` — list of family keys that matched the
                                    current theorem name (in declaration
                                    order; used by eval to aggregate
                                    per-family counts)
      - `last_retrieval_activation` — family key that triggered v4.1
                                    premise retrieval on this call, or
                                    None if retrieval did not activate
      - `last_retrieved_lemma_set` — ordered list of lemma names returned
                                    by retrieve_for_state on this call
                                    (before tactic-form expansion)
    """

    def __init__(
        self,
        base_policy: Any,
        fallback_tactics: list[str] | None = None,
        tactic_templates: list[str] | None = None,
        max_extra_tactics_per_state: int | None = None,
        theorem_family_tactics: dict[str, list[str]] | None = None,
        family_budgets: dict[str, int] | None = None,
        theorem_tactic_denylist: dict[str, list[str]] | None = None,
        retrieval_enabled: bool = False,
        retrieval_top_k: int = 0,
        retrieval_tactic_forms: list[str] | None = None,
        retrieval_filter_self: bool = True,
        retrieval_filter_unavailable: bool = True,
        retrieval_shape_filter: bool = True,
        term_builder_templates: dict[str, list[str]] | None = None,
        term_builder_budget: int = 0,
        priority_templates: dict[str, list[str]] | None = None,
        priority_template_budget: int = 0,
    ) -> None:
        self.base_policy = base_policy
        self.fallback_tactics: list[str] = [
            t for t in (fallback_tactics or []) if t and t.strip()
        ]
        # Templates kept RAW — rendered per state inside rank_tactics.
        self.tactic_templates: list[str] = [
            t for t in (tactic_templates or []) if t and t.strip()
        ]
        # When set, cap the number of *extra* (fallback + template) tactics
        # appended after the generative top-k. None = no cap (v3.1 behavior).
        self.max_extra_tactics_per_state = max_extra_tactics_per_state
        # v3.4: theorem-name-aware tactic families. Keys are substrings to
        # match against full_name; values are ordered tactic strings (may
        # contain `{var}` placeholders rendered per Nat var in scope).
        # Insertion order controls activation priority (declare most-
        # specific keys first).
        self.theorem_family_tactics: dict[str, list[str]] = {
            k: [t for t in (v or []) if t and t.strip()]
            for k, v in (theorem_family_tactics or {}).items()
            if k
        }
        self.family_budgets: dict[str, int] = dict(family_budgets or {})
        # v3.6 per-theorem tactic deny-list. Substring match against the
        # tactic string; any match filters that tactic out of the ranked
        # list for the named theorem only.
        self.theorem_tactic_denylist: dict[str, list[str]] = {
            k: [s for s in (v or []) if s]
            for k, v in (theorem_tactic_denylist or {}).items()
            if k
        }
        # v4.1 premise-retrieval config. Off by default — only activates
        # when retrieval_enabled is True AND an activated family has a
        # matching catalog bucket in premise_retriever._FAMILY_CATALOG_KEYS.
        self.retrieval_enabled: bool = bool(retrieval_enabled)
        self.retrieval_top_k: int = max(0, int(retrieval_top_k or 0))
        # v4.2: accept either full templates ("rw [{p}]") or short
        # form-family names ("rw"). _normalize_retrieval_forms expands
        # short names to templates and dedupes.
        self.retrieval_tactic_forms: list[str] = _normalize_retrieval_forms(
            retrieval_tactic_forms
        )
        # v4.2 filter knobs. When set, the wrapper passes them to
        # retrieve_for_state which removes target-theorem self-retrievals
        # and known-unavailable lemmas before scoring.
        self.retrieval_filter_self: bool = bool(retrieval_filter_self)
        self.retrieval_filter_unavailable: bool = bool(retrieval_filter_unavailable)
        # v4.4 shape filter. When True the retriever computes goal_shape
        # from state_pp and gives matching lemma shapes a scoring bonus;
        # the wrapper restricts the emitted forms per lemma using
        # `forms_for_shape_pair`. When False the v4.3 behavior is
        # preserved (every retrieved lemma emits every configured form).
        self.retrieval_shape_filter: bool = bool(retrieval_shape_filter)
        # v5 term-mode proof skeleton block. Keys are goal-shape labels
        # ("iff", "dvd", "eq", "lt", "le", "and", "or", "unknown", or
        # the special key "any" which always matches). Values are
        # template strings rendered via _render_template, then emitted
        # with ORIGIN_TERM_BUILDER between family/retrieval and generic
        # fallbacks. Activates only when the current state's classified
        # goal_shape matches a configured key (or the "any" key is set).
        # term_builder_budget caps how many term-mode entries are
        # appended to the per-state list; 0 means unbounded within the
        # existing extras cap.
        self.term_builder_templates: dict[str, list[str]] = {
            str(k): [t for t in (v or []) if t and t.strip()]
            for k, v in (term_builder_templates or {}).items()
            if k
        }
        self.term_builder_budget: int = max(0, int(term_builder_budget or 0))
        # v5: count of term_builder entries emitted in the most recent
        # rank_tactics call (post-dedup). Reset per call.
        self.last_term_builder_attempt_count: int = 0
        self.last_term_builder_shape_key: str | None = None
        # v5: priority_templates block — same shape-keyed schema as
        # term_builder, but emitted BEFORE generative_topk. Used when
        # the family's template is more reliable than the model's
        # guess and we want to skip the model entirely on this state.
        # Set via the same load/dump_strategy_config plumbing.
        self.priority_templates: dict[str, list[str]] = {
            str(k): [t for t in (v or []) if t and t.strip()]
            for k, v in (priority_templates or {}).items()
            if k
        }
        self.priority_template_budget: int = max(0, int(priority_template_budget or 0))
        self.last_priority_template_attempt_count: int = 0
        self.last_ranked_tactics: list[str] = []
        self.last_origins: list[str] = []
        self.last_template_sources: list[str | None] = []
        self.last_family_sources: list[str | None] = []
        self.last_retrieved_premises: list[str | None] = []
        # v4.2: parallel form-family label per entry ("rw" / "simp" /
        # "apply" / "exact"); None for non-retrieved entries.
        self.last_retrieved_forms: list[str | None] = []
        # v4.4: parallel lemma-shape label per entry ("iff" / "eq" /
        # "lt" / "le" / "dvd" / "unknown"); None for non-retrieved entries.
        self.last_retrieved_shapes: list[str | None] = []
        self.last_goal_shape: str = "unknown"
        self.last_activated_families: list[str] = []
        self.last_retrieval_activation: str | None = None
        self.last_retrieved_lemma_set: list[str] = []
        # v4.2 per-call diagnostic counts from the retriever's filter step.
        self.last_retrieval_filtered_self_count: int = 0
        self.last_retrieval_filtered_unavailable_count: int = 0
        # v4.4 per-call shape-mismatch filter count (forms dropped by
        # shape_filter that would have been emitted under the configured
        # form list).
        self.last_shape_mismatch_filtered_count: int = 0
        # v3.6: number of (already-deduped) entries filtered by the deny-
        # list in the most recent rank_tactics call. Reset per call.
        self.last_denied_count: int = 0

    def rank_tactics(
        self, state_pp: str, full_name: str = "", k: int = 5
    ) -> list[str]:
        base = self.base_policy.rank_tactics(state_pp, full_name, k=k)
        nat_vars = _extract_nat_vars(state_pp)
        # v4.5: extract hypothesis-shape placeholders once per state so
        # the same dict is reused across every family / generic template
        # render. Templates that reference an absent {hyp_*} placeholder
        # are skipped silently.
        hypotheses = _extract_hypotheses(state_pp)

        # v4.4: entries are 7-tuples
        # (tactic, origin, template_source, family_source,
        #  retrieved_premise, retrieved_form, retrieved_shape).
        # retrieved_shape is the lemma shape ("iff"/"eq"/"lt"/"le"/"dvd"/
        # "unknown") for ORIGIN_RETRIEVED entries; None for every other
        # origin.
        Entry = tuple[
            str, str, str | None, str | None,
            str | None, str | None, str | None,
        ]

        # v5: priority_templates block — emitted before generative_topk
        # so that a known-good family template runs at step 1, before
        # the model's `simp [...]` advances state into a less useful
        # form. Tagged with ORIGIN_TEMPLATE (template_source carries the
        # raw key for traceability).
        seen: set[str] = set()
        priority_entries: list[Entry] = []
        pt_shape_key: str | None = None
        if self.priority_templates:
            try:
                from premise_retriever import classify_goal_shape
                pt_goal_shape = classify_goal_shape(state_pp)
            except Exception:
                pt_goal_shape = "unknown"
            if pt_goal_shape in self.priority_templates:
                pt_shape_key = pt_goal_shape
            elif "any" in self.priority_templates:
                pt_shape_key = "any"
            if pt_shape_key is not None:
                emitted = 0
                for raw in self.priority_templates[pt_shape_key]:
                    for rendered in _render_template(raw, nat_vars, hypotheses):
                        if rendered and rendered not in seen:
                            seen.add(rendered)
                            priority_entries.append(
                                (rendered, ORIGIN_TEMPLATE, raw,
                                 f"priority:{pt_shape_key}",
                                 None, None, None)
                            )
                            emitted += 1
                            if self.priority_template_budget and emitted >= self.priority_template_budget:
                                break
                    if self.priority_template_budget and emitted >= self.priority_template_budget:
                        break
        self.last_priority_template_attempt_count = len(priority_entries)

        base_entries: list[Entry] = []
        for t in base:
            if t and t not in seen:
                seen.add(t)
                base_entries.append(
                    (t, ORIGIN_GENERATIVE, None, None, None, None, None)
                )

        # v3.4: family-specific tactics first (they encode targeted
        # knowledge for the matched theorem family), then generic
        # fallbacks/templates.
        activated_families = _match_families(
            full_name, list(self.theorem_family_tactics.keys())
        )
        self.last_activated_families = list(activated_families)

        family_entries: list[Entry] = []
        for fam in activated_families:
            for raw in self.theorem_family_tactics.get(fam, []):
                for rendered in _render_template(raw, nat_vars, hypotheses):
                    if rendered and rendered not in seen:
                        seen.add(rendered)
                        family_entries.append(
                            (rendered, ORIGIN_FAMILY, None, fam, None, None, None)
                        )

        # v4.1: premise retrieval. Insert between family and generic
        # entries. Only activates when retrieval_enabled and an activated
        # family has a catalog bucket in _FAMILY_CATALOG_KEYS. For each
        # retrieved lemma name, synthesize one entry per configured
        # tactic form (rw / simp / exact / apply by default). v4.2 also
        # passes filter_self / filter_unavailable through and records
        # the per-call filter counts.
        retrieved_entries: list[Entry] = []
        retrieval_activation: str | None = None
        retrieved_lemma_set: list[str] = []
        filtered_self = 0
        filtered_unavailable = 0
        goal_shape = "unknown"
        lemma_shapes: dict[str, str] = {}
        shape_mismatch_filtered = 0
        if self.retrieval_enabled and self.retrieval_top_k > 0 and activated_families:
            from premise_retriever import (
                _FAMILY_CATALOG_KEYS,
                forms_for_shape_pair,
                retrieve_for_state,
            )
            for fam in activated_families:
                if fam in _FAMILY_CATALOG_KEYS:
                    retrieval_activation = fam
                    retrieved_lemma_set, diag = retrieve_for_state(
                        state_pp=state_pp,
                        theorem_name=full_name or None,
                        k=self.retrieval_top_k,
                        family_key=fam,
                        filter_self=self.retrieval_filter_self,
                        filter_unavailable=self.retrieval_filter_unavailable,
                        shape_aware=self.retrieval_shape_filter,
                        return_diagnostics=True,
                    )
                    filtered_self = diag.get("filtered_self", 0)
                    filtered_unavailable = diag.get("filtered_unavailable", 0)
                    goal_shape = diag.get("goal_shape", "unknown")
                    lemma_shapes = diag.get("lemma_shapes", {}) or {}
                    break
            # Configured form-family labels (e.g. ["rw","simp","apply"]),
            # used both for emission and for shape-aware filtering.
            configured_form_labels = [
                _form_family_label(t) for t in self.retrieval_tactic_forms
            ]
            label_to_template = dict(
                zip(configured_form_labels, self.retrieval_tactic_forms)
            )
            for premise in retrieved_lemma_set:
                lemma_shape = lemma_shapes.get(premise, "unknown")
                if self.retrieval_shape_filter:
                    allowed_labels = forms_for_shape_pair(
                        goal_shape, lemma_shape, configured_form_labels,
                    )
                else:
                    allowed_labels = list(configured_form_labels)
                shape_mismatch_filtered += max(
                    0, len(configured_form_labels) - len(allowed_labels)
                )
                for label in allowed_labels:
                    form = label_to_template.get(label)
                    if not form:
                        continue
                    tactic = form.replace("{p}", premise).strip()
                    if tactic and tactic not in seen:
                        seen.add(tactic)
                        retrieved_entries.append(
                            (tactic, ORIGIN_RETRIEVED, None,
                             retrieval_activation, premise, label, lemma_shape)
                        )
        self.last_retrieval_activation = retrieval_activation
        self.last_retrieved_lemma_set = list(retrieved_lemma_set)
        self.last_retrieval_filtered_self_count = filtered_self
        self.last_retrieval_filtered_unavailable_count = filtered_unavailable
        self.last_shape_mismatch_filtered_count = shape_mismatch_filtered

        # v5 term-mode proof skeleton block. If `term_builder_templates`
        # is configured, classify the current goal shape (independently
        # of the retrieval block, which may have set goal_shape only
        # when retrieval activated) and look up the matching template
        # list. The "any" key fires for every goal shape; an exact
        # shape key fires only when the goal classifies to that shape.
        # Entries are tagged ORIGIN_TERM_BUILDER and inserted between
        # the retrieval entries and the generic fallback entries — they
        # are more targeted than generic fallbacks but less targeted
        # than per-theorem family tactics, so this is the right slot.
        term_builder_entries: list[Entry] = []
        tb_shape_key: str | None = None
        if self.term_builder_templates:
            if goal_shape == "unknown":
                # Retrieval didn't classify; classify now.
                try:
                    from premise_retriever import classify_goal_shape
                    goal_shape = classify_goal_shape(state_pp)
                except Exception:
                    goal_shape = "unknown"
            # Resolve which key to pull templates from. Exact shape
            # match wins over the "any" key.
            if goal_shape in self.term_builder_templates:
                tb_shape_key = goal_shape
            elif "any" in self.term_builder_templates:
                tb_shape_key = "any"
            if tb_shape_key is not None:
                raw_templates = self.term_builder_templates[tb_shape_key]
                emitted = 0
                limit = self.term_builder_budget or len(raw_templates) * max(1, len(nat_vars) + 1)
                for raw in raw_templates:
                    for rendered in _render_template(raw, nat_vars, hypotheses):
                        if rendered and rendered not in seen:
                            seen.add(rendered)
                            term_builder_entries.append(
                                (rendered, ORIGIN_TERM_BUILDER, raw, tb_shape_key,
                                 None, None, None)
                            )
                            emitted += 1
                            if self.term_builder_budget and emitted >= self.term_builder_budget:
                                break
                    if self.term_builder_budget and emitted >= self.term_builder_budget:
                        break
        self.last_goal_shape = goal_shape
        self.last_term_builder_attempt_count = len(term_builder_entries)
        self.last_term_builder_shape_key = tb_shape_key

        # Generic fallbacks + rendered templates, deduped against base,
        # family and retrieval entries, in deterministic genome order.
        generic_entries: list[Entry] = []
        for t in self.fallback_tactics:
            if t and t not in seen:
                seen.add(t)
                generic_entries.append((t, ORIGIN_FALLBACK, None, None, None, None, None))
        for raw_template in self.tactic_templates:
            for rendered in _render_template(raw_template, nat_vars, hypotheses):
                if rendered and rendered not in seen:
                    seen.add(rendered)
                    generic_entries.append(
                        (rendered, ORIGIN_TEMPLATE, raw_template, None, None, None, None)
                    )

        extra_entries = (
            family_entries + retrieved_entries + term_builder_entries
            + generic_entries
        )

        # v3.6 per-theorem deny-list: filter entries whose tactic string
        # contains any denied substring for this theorem. Applied after
        # family/generic assembly (and dedup, which happened inline), but
        # BEFORE the per-state budget cap so denied tactics don't consume
        # cap slots. Substring match keeps the list compact.
        denied_substrings = self.theorem_tactic_denylist.get(full_name, []) if full_name else []
        denied_count = 0
        if denied_substrings:
            def _is_denied(tac: str) -> bool:
                return any(d and d in tac for d in denied_substrings)
            base_kept: list[Entry] = []
            for e in base_entries:
                if _is_denied(e[0]):
                    denied_count += 1
                else:
                    base_kept.append(e)
            base_entries = base_kept
            extra_kept: list[Entry] = []
            for e in extra_entries:
                if _is_denied(e[0]):
                    denied_count += 1
                else:
                    extra_kept.append(e)
            extra_entries = extra_kept
        self.last_denied_count = denied_count

        # Effective per-state extras cap: when any family activates, use
        # max(family_budgets[f] for f in activated). Otherwise use the
        # global max_extra_tactics_per_state (None = unbounded).
        # v4.1: when premise retrieval activates on this state, reserve
        # additional cap slots so retrieved tactics fit alongside the
        # family tactics (otherwise the cap drops them all). Slot count =
        # len(retrieval_tactic_forms) * retrieval_top_k — an upper bound
        # on retrieved_entries before dedup.
        cap = self.max_extra_tactics_per_state
        if activated_families:
            budgets = [
                self.family_budgets[f]
                for f in activated_families
                if f in self.family_budgets
            ]
            if budgets:
                cap = max(budgets)
        if retrieval_activation is not None and cap is not None:
            cap += len(self.retrieval_tactic_forms) * self.retrieval_top_k
        # v5: reserve cap slots for term_builder entries so they aren't
        # crowded out by family + retrieval entries. The exact count is
        # the number we actually emitted (already capped by the
        # term_builder_budget if set).
        if term_builder_entries and cap is not None:
            cap += len(term_builder_entries)
        if cap is not None:
            extra_entries = extra_entries[:cap]

        # v5: priority_entries go first — even before base. They are
        # not subject to the per-state extras cap because they are part
        # of a small targeted set the genome explicitly prioritized.
        all_entries = priority_entries + base_entries + extra_entries
        self.last_ranked_tactics = [e[0] for e in all_entries]
        self.last_origins = [e[1] for e in all_entries]
        self.last_template_sources = [e[2] for e in all_entries]
        self.last_family_sources = [e[3] for e in all_entries]
        self.last_retrieved_premises = [e[4] for e in all_entries]
        self.last_retrieved_forms = [e[5] for e in all_entries]
        self.last_retrieved_shapes = [e[6] for e in all_entries]
        return self.last_ranked_tactics

    def origin_of_rank(self, rank: int) -> str | None:
        if 0 <= rank < len(self.last_origins):
            return self.last_origins[rank]
        return None


def load_strategy_config(
    path: str | Path,
) -> tuple[
    list[str], list[str], int | None,
    dict[str, list[str]], dict[str, int], dict[str, list[str]],
    bool, int, list[str], bool, bool, bool, bool,
    dict[str, list[str]], int,
]:
    """Read the strategy config from JSON.

    Returns (fallback_tactics, tactic_templates, max_extra_tactics_per_state,
            theorem_family_tactics, family_budgets, theorem_tactic_denylist,
            retrieval_enabled, retrieval_top_k, retrieval_tactic_forms,
            retrieval_filter_self, retrieval_filter_unavailable,
            retrieval_skip_bloating_apply, retrieval_shape_filter,
            term_builder_templates, term_builder_budget).

    Missing keys produce safe defaults; unknown keys are ignored. All
    retrieval filter flags default to True so older configs benefit
    from the newer filters automatically. v5 term_builder fields
    default to empty / 0 so older configs are no-ops.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    fb = list(raw.get("fallback_tactics") or [])
    tmpl = list(raw.get("tactic_templates") or [])
    cap_raw = raw.get("max_extra_tactics_per_state")
    cap = int(cap_raw) if cap_raw is not None else None
    fam_raw = raw.get("theorem_family_tactics") or {}
    fam = {str(k): list(v or []) for k, v in fam_raw.items()}
    fb_raw = raw.get("family_budgets") or {}
    fam_budgets = {str(k): int(v) for k, v in fb_raw.items()}
    deny_raw = raw.get("theorem_tactic_denylist") or {}
    deny = {str(k): list(v or []) for k, v in deny_raw.items()}
    retrieval_enabled = bool(raw.get("retrieval_enabled", False))
    retrieval_top_k = int(raw.get("retrieval_top_k", 0) or 0)
    forms_raw = raw.get("retrieval_tactic_forms") or []
    retrieval_tactic_forms = [str(s) for s in forms_raw if s]
    retrieval_filter_self = bool(raw.get("retrieval_filter_self", True))
    retrieval_filter_unavailable = bool(raw.get("retrieval_filter_unavailable", True))
    retrieval_skip_bloating_apply = bool(
        raw.get("retrieval_skip_bloating_apply", True)
    )
    retrieval_shape_filter = bool(raw.get("retrieval_shape_filter", True))
    tb_raw = raw.get("term_builder_templates") or {}
    term_builder_templates = {str(k): list(v or []) for k, v in tb_raw.items()}
    term_builder_budget = int(raw.get("term_builder_budget", 0) or 0)
    pt_raw = raw.get("priority_templates") or {}
    priority_templates = {str(k): list(v or []) for k, v in pt_raw.items()}
    priority_template_budget = int(raw.get("priority_template_budget", 0) or 0)
    return (
        fb, tmpl, cap, fam, fam_budgets, deny,
        retrieval_enabled, retrieval_top_k, retrieval_tactic_forms,
        retrieval_filter_self, retrieval_filter_unavailable,
        retrieval_skip_bloating_apply, retrieval_shape_filter,
        term_builder_templates, term_builder_budget,
        priority_templates, priority_template_budget,
    )


def dump_strategy_config(
    path: str | Path,
    fallback_tactics: list[str],
    tactic_templates: list[str],
    max_extra_tactics_per_state: int | None = None,
    theorem_family_tactics: dict[str, list[str]] | None = None,
    family_budgets: dict[str, int] | None = None,
    theorem_tactic_denylist: dict[str, list[str]] | None = None,
    retrieval_enabled: bool = False,
    retrieval_top_k: int = 0,
    retrieval_tactic_forms: list[str] | None = None,
    retrieval_filter_self: bool = True,
    retrieval_filter_unavailable: bool = True,
    retrieval_skip_bloating_apply: bool = True,
    retrieval_shape_filter: bool = True,
    term_builder_templates: dict[str, list[str]] | None = None,
    term_builder_budget: int = 0,
    priority_templates: dict[str, list[str]] | None = None,
    priority_template_budget: int = 0,
) -> None:
    """Write the JSON config the subprocess will read. Parent dirs are
    created if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps(
            {
                "fallback_tactics": list(fallback_tactics),
                "tactic_templates": list(tactic_templates),
                "max_extra_tactics_per_state": max_extra_tactics_per_state,
                "theorem_family_tactics": {
                    str(k): list(v or [])
                    for k, v in (theorem_family_tactics or {}).items()
                },
                "family_budgets": {
                    str(k): int(v) for k, v in (family_budgets or {}).items()
                },
                "theorem_tactic_denylist": {
                    str(k): list(v or [])
                    for k, v in (theorem_tactic_denylist or {}).items()
                },
                "retrieval_enabled": bool(retrieval_enabled),
                "retrieval_top_k": int(retrieval_top_k or 0),
                "retrieval_tactic_forms": [
                    str(s) for s in (retrieval_tactic_forms or []) if s
                ],
                "retrieval_filter_self": bool(retrieval_filter_self),
                "retrieval_filter_unavailable": bool(retrieval_filter_unavailable),
                "retrieval_skip_bloating_apply": bool(
                    retrieval_skip_bloating_apply
                ),
                "retrieval_shape_filter": bool(retrieval_shape_filter),
                "term_builder_templates": {
                    str(k): list(v or [])
                    for k, v in (term_builder_templates or {}).items()
                },
                "term_builder_budget": int(term_builder_budget or 0),
                "priority_templates": {
                    str(k): list(v or [])
                    for k, v in (priority_templates or {}).items()
                },
                "priority_template_budget": int(priority_template_budget or 0),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
