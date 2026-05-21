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


def _render_template(template: str, nat_vars: list[str]) -> list[str]:
    """Render a tactic template once per Nat variable in scope.

    If the template has no `{var}` placeholder, render it once verbatim.
    If `{var}` is present but no Nat variables were extracted, skip it
    entirely (rendering with a fictitious name would just guarantee a
    Lean error and waste a roundtrip).
    """
    if "{var}" not in template:
        return [template]
    if not nat_vars:
        return []
    return [template.replace("{var}", v) for v in nat_vars]


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
                                    populated for family_tactic entries;
                                    None otherwise)
      - `last_activated_families` — list of family keys that matched the
                                    current theorem name (in declaration
                                    order; used by eval to aggregate
                                    per-family counts)
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
        self.last_ranked_tactics: list[str] = []
        self.last_origins: list[str] = []
        self.last_template_sources: list[str | None] = []
        self.last_family_sources: list[str | None] = []
        self.last_activated_families: list[str] = []
        # v3.6: number of (already-deduped) entries filtered by the deny-
        # list in the most recent rank_tactics call. Reset per call.
        self.last_denied_count: int = 0

    def rank_tactics(
        self, state_pp: str, full_name: str = "", k: int = 5
    ) -> list[str]:
        base = self.base_policy.rank_tactics(state_pp, full_name, k=k)
        nat_vars = _extract_nat_vars(state_pp)

        # Build base entries (deduped against themselves). Each entry:
        # (tactic, origin, template_source, family_source)
        base_entries: list[tuple[str, str, str | None, str | None]] = []
        seen: set[str] = set()
        for t in base:
            if t and t not in seen:
                seen.add(t)
                base_entries.append((t, ORIGIN_GENERATIVE, None, None))

        # v3.4: family-specific tactics first (they encode targeted
        # knowledge for the matched theorem family), then generic
        # fallbacks/templates.
        activated_families = _match_families(
            full_name, list(self.theorem_family_tactics.keys())
        )
        self.last_activated_families = list(activated_families)

        family_entries: list[tuple[str, str, str | None, str | None]] = []
        for fam in activated_families:
            for raw in self.theorem_family_tactics.get(fam, []):
                for rendered in _render_template(raw, nat_vars):
                    if rendered and rendered not in seen:
                        seen.add(rendered)
                        family_entries.append(
                            (rendered, ORIGIN_FAMILY, None, fam)
                        )

        # Generic fallbacks + rendered templates, deduped against base and
        # family entries, in deterministic genome order.
        generic_entries: list[tuple[str, str, str | None, str | None]] = []
        for t in self.fallback_tactics:
            if t and t not in seen:
                seen.add(t)
                generic_entries.append((t, ORIGIN_FALLBACK, None, None))
        for raw_template in self.tactic_templates:
            for rendered in _render_template(raw_template, nat_vars):
                if rendered and rendered not in seen:
                    seen.add(rendered)
                    generic_entries.append(
                        (rendered, ORIGIN_TEMPLATE, raw_template, None)
                    )

        extra_entries = family_entries + generic_entries

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
            base_kept: list[tuple[str, str, str | None, str | None]] = []
            for e in base_entries:
                if _is_denied(e[0]):
                    denied_count += 1
                else:
                    base_kept.append(e)
            base_entries = base_kept
            extra_kept: list[tuple[str, str, str | None, str | None]] = []
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
        cap = self.max_extra_tactics_per_state
        if activated_families:
            budgets = [
                self.family_budgets[f]
                for f in activated_families
                if f in self.family_budgets
            ]
            if budgets:
                cap = max(budgets)
        if cap is not None:
            extra_entries = extra_entries[:cap]

        all_entries = base_entries + extra_entries
        self.last_ranked_tactics = [e[0] for e in all_entries]
        self.last_origins = [e[1] for e in all_entries]
        self.last_template_sources = [e[2] for e in all_entries]
        self.last_family_sources = [e[3] for e in all_entries]
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
]:
    """Read the strategy config from JSON.

    Returns (fallback_tactics, tactic_templates, max_extra_tactics_per_state,
            theorem_family_tactics, family_budgets, theorem_tactic_denylist).

    Missing keys produce empty lists / dicts / None; unknown keys are
    ignored. Older configs (pre-v3.4 / pre-v3.6) just get empty dicts for
    the newer fields, which is a no-op in the wrapper.
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
    return fb, tmpl, cap, fam, fam_budgets, deny


def dump_strategy_config(
    path: str | Path,
    fallback_tactics: list[str],
    tactic_templates: list[str],
    max_extra_tactics_per_state: int | None = None,
    theorem_family_tactics: dict[str, list[str]] | None = None,
    family_budgets: dict[str, int] | None = None,
    theorem_tactic_denylist: dict[str, list[str]] | None = None,
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
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
