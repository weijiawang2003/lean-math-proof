"""SearchCandidate — the genome of the evolutionary loop.

A candidate is NOT a proof and NOT a trained model. It is a configuration that
describes *how* to search for proofs: which policy/checkpoint to use, the
search hyperparameters, and (for future strategy wrappers) the fallback tactic
order, prompt template, and theorem-family-specific tactic templates.

In AlphaEvolve terms this is the "program" being evolved. The evaluator turns a
candidate into a behaviour by feeding it to the rollout machinery; Lean grades
the behaviour.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Optional

# Shared prompt format mirrors core_types.build_prompt in the repo root.
DEFAULT_PROMPT_TEMPLATE = "Theorem: {full_name}\n\nProof state:\n{state_pp}\n"


@dataclass
class SearchCandidate:
    """One proof-search strategy.

    Fields consumed by the *real* evaluator today (mapped to eval_rollout_all.py
    CLI args): policy_type, ckpt_dir, top_k, max_steps.

    `timeout_per_theorem` is used to derive the subprocess-level total timeout
    in evaluator._derive_eval_timeout, but is not (yet) passed as a per-theorem
    CLI knob — eval_rollout_all.py has no such argument.

    Inert fields — fallback_tactics, prompt_template, tactic_templates,
    theorem_filters, decode_mode, temperature, seed — are part of the genome
    but NOT passed to the subprocess. They are mutated/tracked now; connecting
    them to behaviour requires a strategy-wrapper layer (see README v3).
    decode_mode/temperature/seed mirror the corresponding eval_rollout_all.py
    CLI defaults so they are safe to wire later without changing semantics.
    """

    name: str
    description: str
    policy_type: str = "generative"
    ckpt_dir: Optional[str] = None
    top_k: int = 8
    max_steps: int = 8
    timeout_per_theorem: int = 20
    fallback_tactics: list[str] = field(default_factory=list)
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE
    tactic_templates: list[str] = field(default_factory=list)
    theorem_filters: dict[str, Any] = field(default_factory=dict)
    # Inert decode-strategy knobs (mirror eval_rollout_all.py defaults).
    # Not passed to the subprocess yet; reserved for a future strategy-wrapper.
    decode_mode: str = "beam"
    temperature: float = 0.8
    seed: Optional[int] = None
    # Cap on how many *extra* tactics (fallbacks + rendered templates,
    # deduped against generative top-k) the wrapper appends per state.
    # None means no cap. Used by hybrid_evolved in v3.2+ to bound the
    # per-step Lean roundtrip count.
    max_extra_tactics_per_state: Optional[int] = None
    # v3.3: when True the eval rollout tracks per-theorem seen-state
    # hashes and prefers tactics that produce unseen states; tactics
    # known to error on the current state are skipped to save Lean
    # roundtrips. DEFAULT FALSE: in the v3.3 nat_defs_subset run, anti-
    # loop preserved the 10/15 proved count in the underlying traces
    # (17 loops detected, 145 known-error skips) but the per-step
    # exploration cost on the Nat.div_* theorems exceeded the 1005s
    # subprocess watchdog, producing 0/15 in the metrics layer. The
    # mutator can flip this to True on a candidate for A/B comparison
    # alongside a bumped timeout_per_theorem.
    enable_loop_avoidance: bool = False
    # v3.4 theorem-name-aware tactic layer. Keys are substrings matched
    # against the theorem's full_name (case-sensitive, dict-insertion
    # order); values are ordered tactic strings (may contain `{var}`
    # placeholders, rendered via the same per-state Nat-var extraction
    # as tactic_templates). All families whose key matches activate; the
    # union of their tactics is added after the generative top-k and
    # ahead of the generic fallbacks/templates, since they encode
    # targeted knowledge for that theorem family.
    theorem_family_tactics: dict[str, list[str]] = field(default_factory=dict)
    # v3.4 per-family extras budget. When at least one family activates
    # for a theorem, the effective max_extra_tactics_per_state for that
    # step becomes max(family_budgets[f] for f in activated). When no
    # family activates, max_extra_tactics_per_state is used as before.
    family_budgets: dict[str, int] = field(default_factory=dict)
    # v3.6 per-theorem tactic deny-list. Maps theorem full_name to a list
    # of substrings; any candidate tactic whose string contains a denied
    # substring is filtered out for that theorem only. Used to prevent
    # specific (theorem, tactic) pairs known to crash the Dojo REPL
    # without removing the tactic globally (it may still win on other
    # theorems). Substring match keeps the deny-list compact when the
    # same tactic is wrapped differently across families/fallbacks.
    theorem_tactic_denylist: dict[str, list[str]] = field(default_factory=dict)
    # v4.1 premise-retrieval config. When retrieval_enabled is True the
    # strategy wrapper calls premise_retriever.retrieve_for_state on every
    # state whose theorem-family key has a catalog bucket (currently only
    # "div" via _FAMILY_CATALOG_KEYS). Retrieved lemma names are wrapped
    # in retrieval_tactic_forms (rw / simp / exact / apply) and inserted
    # between family tactics and generic fallbacks. retrieval_top_k = 0
    # silently no-ops even when retrieval_enabled is True.
    retrieval_enabled: bool = False
    retrieval_top_k: int = 0
    retrieval_tactic_forms: list[str] = field(default_factory=list)
    # v4.2 filter knobs. retrieval_filter_self excludes the target
    # theorem from its own retrieved-premise set (eliminates the v4.1
    # self-reference trap). retrieval_filter_unavailable applies a static
    # denylist of lemmas observed to produce `unknown constant` in the
    # eval environment's import closure (premise_retriever._UNAVAILABLE_LEMMAS).
    # Both default True since they are net wins on the v4.1 traces.
    retrieval_filter_self: bool = True
    retrieval_filter_unavailable: bool = True
    # v4.3 goal-shape filter for retrieved-premise `apply LEMMA` tactics.
    # When True (default), the eval rollout rejects retrieved-apply
    # transitions that strictly increase the open-goal count and
    # pre-filters subsequent `apply LEMMA` candidates on the same
    # theorem. Suppresses the pathological `apply Nat.lt_of_lt_of_le`
    # 2-per-step goal-stack bloat observed in v4.2. Per-theorem only —
    # the lemma is not globally banned, only its `apply` form is
    # suppressed after one observed bloat on the theorem being proved.
    retrieval_skip_bloating_apply: bool = True
    # v4.4 shape-aware retrieval. When True, the retriever classifies
    # the current goal's head connective (eq/iff/lt/le/dvd/...) and
    # the wrapper emits only the tactic forms appropriate for the
    # (goal_shape, lemma_shape) pair (forms_for_shape_pair in
    # premise_retriever). Suppresses the v4.3 mismatch where every
    # retrieved iff lemma also emitted `apply LEMMA` against an iff
    # goal, guaranteed to fail to unify.
    retrieval_shape_filter: bool = True
    # v5 term-mode proof skeleton block (Direction A). Keyed by goal-shape
    # ("iff"/"dvd"/"eq"/"lt"/"le"/"and"/"or"/"unknown") or "any". Templates
    # support {var}/{hyp_pos}/{hyp_le}/{hyp_ne_zero} placeholders identical
    # to tactic_templates rendering. term_builder_budget caps how many
    # term-mode entries are emitted per state (0 = unbounded within
    # max_extra_tactics_per_state). See strategy_wrapper.ORIGIN_TERM_BUILDER.
    term_builder_templates: dict[str, list[str]] = field(default_factory=dict)
    term_builder_budget: int = 0
    # v5 priority_templates — same shape-keyed schema as term_builder,
    # but emitted BEFORE generative_topk so a known-good family template
    # runs at step 1 before the model's `simp [...]` can derail the
    # rewrite chain. Used surgically on theorems where the model has
    # been observed to advance state into a worse form.
    priority_templates: dict[str, list[str]] = field(default_factory=dict)
    priority_template_budget: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- serialization ----------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SearchCandidate":
        """Build a candidate from a dict, ignoring unknown keys for forward
        compatibility (older population files may lack newer fields)."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save_json(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path) -> "SearchCandidate":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def copy(self) -> "SearchCandidate":
        """Deep-ish copy via round-trip (lists/dicts are rebuilt)."""
        return SearchCandidate.from_dict(json.loads(json.dumps(self.to_dict())))
