"""NS4 / NS4.1 skeleton-bag.

A skeleton-based representation that coexists with the legacy flat-field
wrapper. Implements:

  - `Skeleton`: one emit unit, gated by shape and (optionally) family.
  - `SkeletonBag`: ordered collection keyed by shape; produces EmittedTactic
    lists in the same order today's wrapper produces them.
  - `EmittedTactic`: a rendered tactic with origin / attribution metadata.

Scope (NS4.1):
  - `priority_template`, `family_tactic`, `fallback_tactic`,
    `tactic_template`, and `term_builder` origins all flow through the
    bag when `use_skeleton_bag=True`. The retrieved_premise origin still
    runs through the legacy inline block (it has its own per-state
    classification and shape-form filtering).
  - No slot-vocabulary mutation. Templates render through
    `strategy_wrapper._render_template`.
  - No JSON schema changes beyond the existing `use_skeleton_bag` flag.

See `project/evolve/reports/ns4_skeleton_bag_design_note.md` and
`project/evolve/reports/ns4_1_skeleton_unification.md` for context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


SHAPE_ANY = "any"

# Priority bands. Lower = earlier in the emit order. The exact numbers
# don't matter — only their order. Spaced to leave room for mutator
# inserts between bands.
PRIORITY_PRIORITY_TEMPLATE = 0    # emit before model
PRIORITY_FAMILY = 10              # emit after model, before generic
PRIORITY_RETRIEVED = 12           # NS4.2: retrieval slots between family and term_builder
PRIORITY_TERM_BUILDER = 15
PRIORITY_FALLBACK = 20
PRIORITY_TACTIC_TEMPLATE = 25

# Specificity ranks. Lower emits first within a (shape, priority) slot.
# Matches `strategy_wrapper.classify_template_specificity`.
SPECIFICITY_SPECIFIC = 0
SPECIFICITY_GENERIC = 1


@dataclass
class Skeleton:
    """One emission unit in the new representation.

    Fields:
      name:        Human-readable handle (used for trace attribution and
                   future mutator references).
      shape:       Goal-shape gate; one of "iff"/"eq"/"lt"/"le"/"dvd"/
                   "and"/"or"/"unknown"/"any". "any" matches every shape.
      family:      Optional substring; if set, the skeleton fires only
                   when this substring appears in the theorem full_name.
      priority:    Lower = earlier in the emit order.
      specificity: 0 = specific, 1 = generic (NS1 sort within slot).
      template:    Raw tactic string with `{var}`/`{hyp_*}` placeholders.
      origin:      Legacy origin tag (preserved so traces stay stable).
      enabled:     Cheap toggle for mutator (disable → skip).
      tags:        Free-form labels (e.g. ["NS3.5", "shape_specific"]).
    """

    name: str
    shape: str
    template: str
    origin: str
    family: str | None = None
    priority: int = PRIORITY_PRIORITY_TEMPLATE
    specificity: int = SPECIFICITY_GENERIC
    enabled: bool = True
    tags: list[str] = field(default_factory=list)

    @property
    def stable_id(self) -> str:
        """NS7 stable identifier — invariant across mutations / rebuilds.

        `name` drifts because `from_legacy_strategy_config` re-indexes
        skeletons by insertion order (so disabling one priority_template
        renumbers all subsequent ones). The stable_id is computed from
        normalized identity fields only — origin, shape, family,
        specificity, and the canonical-form template text — so the same
        underlying skeleton keeps the same id no matter how the genome
        is rebuilt around it.
        """
        import hashlib
        canonical = "|".join((
            self.origin,
            self.shape,
            self.family or "",
            str(self.specificity),
            (self.template or "").strip(),
        ))
        return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:12]


@dataclass
class EmittedTactic:
    """A rendered tactic together with its attribution metadata.

    The `tactic` field is the post-render Lean tactic string. Everything
    else is for trace / scoring attribution.

    NS4.2 retrieved-premise emissions reuse this dataclass; the bag
    method synthesizes EmittedTactic instances per-state instead of
    pre-registering them as Skeletons. The three retrieved_* fields
    carry the per-lemma metadata the eval loop needs to populate
    `winning_tactic_retrieved_*` and the per-form / per-shape counters.
    They stay None on every other origin.
    """

    tactic: str
    origin: str
    skeleton_name: str
    shape: str
    family: str | None
    specificity: int
    priority: int
    template_source: str
    family_source: str | None = None
    retrieved_premise: str | None = None
    retrieved_form: str | None = None
    retrieved_shape: str | None = None
    skeleton_stable_id: str | None = None


class SkeletonBag:
    """Ordered collection of Skeletons, keyed by shape.

    Internal storage: `self.skeletons` is `dict[shape, list[Skeleton]]`,
    insertion-ordered per shape. The `for_state` and `emit_*` methods
    pick up the shape-matching list plus the `any` list and order them
    by `priority`, then `specificity`, then insertion order.
    """

    def __init__(self) -> None:
        self.skeletons: dict[str, list[Skeleton]] = {}
        # Secondary index by family name, preserving insertion order so
        # `_match_families` semantics carry over (declare-most-specific-first).
        self.families: dict[str, list[Skeleton]] = {}

    def add(self, skeleton: Skeleton) -> None:
        if skeleton.shape not in self.skeletons:
            self.skeletons[skeleton.shape] = []
        self.skeletons[skeleton.shape].append(skeleton)
        if skeleton.family is not None:
            if skeleton.family not in self.families:
                self.families[skeleton.family] = []
            self.families[skeleton.family].append(skeleton)

    def for_shape(self, shape: str) -> list[Skeleton]:
        """Return the shape-slot list followed by the any-slot list,
        each stable-sorted by (priority, specificity, insertion order).
        Skeletons with `enabled=False` are dropped.

        The shape-slot is emitted before the any-slot at the same
        priority — matches NS3.5's "shape first, then any as true
        fallback" semantics.
        """
        out: list[Skeleton] = []
        for slot_key in (shape, SHAPE_ANY):
            if slot_key == SHAPE_ANY and shape == SHAPE_ANY:
                # Avoid emitting the any-slot twice when caller passes "any".
                continue
            slot = self.skeletons.get(slot_key, [])
            ordered = sorted(
                [s for s in slot if s.enabled],
                key=lambda s: (s.priority, s.specificity),
            )
            out.extend(ordered)
        return out

    def for_state(
        self,
        goal_shape: str,
        active_families: list[str],
    ) -> list[Skeleton]:
        """Return all enabled skeletons that fire for this state.

        Family gate: a skeleton with `family=None` always passes; a
        skeleton with `family=X` passes only if X is in active_families.
        """
        active_set = set(active_families)
        return [
            s for s in self.for_shape(goal_shape)
            if s.family is None or s.family in active_set
        ]

    # ------------------------------------------------------------------
    # Emit (prototype: only priority_template skeletons are emitted here).
    # ------------------------------------------------------------------

    def emit_priority_tactics(
        self,
        state_pp: str,
        goal_shape: str,
        nat_vars: list[str],
        hypotheses: dict[str, str | None],
        budget: int = 0,
        already_seen: set[str] | None = None,
    ) -> list[EmittedTactic]:
        """Render every priority-template skeleton that fires for this
        state, deduped against `already_seen` if provided.

        Ordering replicates the NS3.5 wrapper block exactly:
          1. shape-slot specifics
          2. shape-slot generics
          3. any-slot specifics
          4. any-slot generics

        which is what `for_shape(goal_shape)` already produces because
        every priority_template skeleton has priority=PRIORITY_PRIORITY_TEMPLATE
        and is tagged by specificity, and shape-slot is enumerated before
        any-slot inside `for_shape`.
        """
        # Local import to avoid circular import at module load time.
        from evolve.strategy_wrapper import _render_template

        seen = already_seen if already_seen is not None else set()
        out: list[EmittedTactic] = []
        emitted = 0

        for skel in self.for_shape(goal_shape):
            if skel.origin != "priority_template":
                continue
            if budget and emitted >= budget:
                break
            for rendered in _render_template(
                skel.template, nat_vars, hypotheses
            ):
                if not rendered or rendered in seen:
                    continue
                seen.add(rendered)
                spec_label = (
                    "specific" if skel.specificity == SPECIFICITY_SPECIFIC
                    else "generic"
                )
                # Preserve the legacy family_source string the eval loop
                # expects for priority-template entries:
                #   "priority:<slot_key>:<specificity>"
                slot_key = skel.shape
                family_source = f"priority:{slot_key}:{spec_label}"
                out.append(
                    EmittedTactic(
                        tactic=rendered,
                        origin=skel.origin,
                        skeleton_name=skel.name,
                        skeleton_stable_id=skel.stable_id,
                        shape=skel.shape,
                        family=skel.family,
                        specificity=skel.specificity,
                        priority=skel.priority,
                        template_source=skel.template,
                        family_source=family_source,
                    )
                )
                emitted += 1
                if budget and emitted >= budget:
                    break
        return out

    def emit_family_tactics(
        self,
        full_name: str,
        nat_vars: list[str],
        hypotheses: dict[str, str | None],
        already_seen: set[str] | None = None,
    ) -> tuple[list[EmittedTactic], list[str]]:
        """Emit family-gated tactics for the matched theorem name.

        Families are matched via `_match_families` (substring match,
        preserving declaration order). Within each family, skeletons
        are stable-sorted by specificity (NS1 invariant lifted from
        priority_templates into the family path — see NS4.1 unification
        report for parity discussion).

        Returns `(entries, active_families)`. `family_source` on each
        EmittedTactic is the raw family key (not the `priority:...`
        format used for priority entries), preserving the legacy trace
        attribution.
        """
        from evolve.strategy_wrapper import _match_families, _render_template

        family_keys = list(self.families.keys())
        active = _match_families(full_name, family_keys)
        seen = already_seen if already_seen is not None else set()
        out: list[EmittedTactic] = []
        for fam in active:
            skels = [
                s for s in self.families.get(fam, [])
                if s.enabled and s.origin == "family_tactic"
            ]
            ordered = sorted(skels, key=lambda s: s.specificity)
            for skel in ordered:
                for rendered in _render_template(
                    skel.template, nat_vars, hypotheses
                ):
                    if not rendered or rendered in seen:
                        continue
                    seen.add(rendered)
                    out.append(EmittedTactic(
                        tactic=rendered,
                        origin=skel.origin,
                        skeleton_name=skel.name,
                        skeleton_stable_id=skel.stable_id,
                        shape=skel.shape,
                        family=skel.family,
                        specificity=skel.specificity,
                        priority=skel.priority,
                        template_source=skel.template,
                        family_source=fam,
                    ))
        return out, list(active)

    def emit_fallback_tactics(
        self,
        already_seen: set[str] | None = None,
    ) -> list[EmittedTactic]:
        """Emit fallback skeletons verbatim in insertion order.

        Fallback skeletons are literal tactic strings (no `{var}` /
        `{hyp_*}` placeholders in practice). To match the legacy
        wrapper exactly we DO NOT render them — render would split
        templated entries per-variable and the legacy code never did
        that for fallback_tactics.
        """
        seen = already_seen if already_seen is not None else set()
        out: list[EmittedTactic] = []
        for skel in self.skeletons.get(SHAPE_ANY, []):
            if not skel.enabled or skel.origin != "fallback_tactic":
                continue
            t = skel.template
            if t and t not in seen:
                seen.add(t)
                out.append(EmittedTactic(
                    tactic=t,
                    origin=skel.origin,
                    skeleton_name=skel.name,
                    skeleton_stable_id=skel.stable_id,
                    shape=skel.shape,
                    family=skel.family,
                    specificity=skel.specificity,
                    priority=skel.priority,
                    template_source=skel.template,
                    family_source=None,
                ))
        return out

    def emit_tactic_template_tactics(
        self,
        nat_vars: list[str],
        hypotheses: dict[str, str | None],
        already_seen: set[str] | None = None,
    ) -> list[EmittedTactic]:
        """Emit `tactic_template` (generic, shape=any) skeletons.

        Rendered via `_render_template` — matches the legacy generic-
        block behavior where templated strings expanded one tactic per
        Nat-var in scope. Origin is `tactic_template` (mapped to the
        legacy `ORIGIN_TEMPLATE` constant at the call site).
        """
        from evolve.strategy_wrapper import _render_template

        seen = already_seen if already_seen is not None else set()
        out: list[EmittedTactic] = []
        for skel in self.skeletons.get(SHAPE_ANY, []):
            if not skel.enabled or skel.origin != "tactic_template":
                continue
            for rendered in _render_template(
                skel.template, nat_vars, hypotheses
            ):
                if not rendered or rendered in seen:
                    continue
                seen.add(rendered)
                out.append(EmittedTactic(
                    tactic=rendered,
                    origin=skel.origin,
                    skeleton_name=skel.name,
                    skeleton_stable_id=skel.stable_id,
                    shape=skel.shape,
                    family=skel.family,
                    specificity=skel.specificity,
                    priority=skel.priority,
                    template_source=skel.template,
                    family_source=None,
                ))
        return out

    def emit_retrieved_tactics(
        self,
        state_pp: str,
        theorem_name: str | None,
        activated_families: list[str],
        retrieval_top_k: int,
        retrieval_tactic_forms: list[str],
        retrieval_filter_self: bool,
        retrieval_filter_unavailable: bool,
        retrieval_shape_filter: bool,
        already_seen: set[str] | None = None,
    ) -> tuple[list[EmittedTactic], dict[str, Any]]:
        """NS4.2: emit retrieved-premise tactics as dynamic EmittedTactic
        instances. Wraps the same `retrieve_for_state` /
        `forms_for_shape_pair` logic the legacy block uses — the only
        difference is that the output is an EmittedTactic stream rather
        than a 7-tuple stream, so the standard skeleton-attribution
        plumbing picks up retrieval entries automatically.

        Returns `(entries, diagnostics)` where `diagnostics` carries
        the per-call counters the wrapper needs to surface
        (`last_retrieval_*` fields). Each EmittedTactic carries:
          - `skeleton_name`     = `retrieved:<lemma>:<form_label>`
          - `shape`             = goal shape (from retriever diagnostics)
          - `family`            = the family key that activated retrieval
          - `specificity`       = `SPECIFICITY_SPECIFIC` (retrieved lemmas
                                  are by construction targeted)
          - `priority`          = `PRIORITY_RETRIEVED`
          - `template_source`   = the form template ("rw [{p}]", etc.)
          - `family_source`     = the activated family (preserves the
                                  legacy `last_family_sources` mapping for
                                  retrieved entries)
          - `retrieved_premise` = lemma name
          - `retrieved_form`    = short form label ("rw"/"simp"/...)
          - `retrieved_shape`   = lemma shape ("iff"/"eq"/...)
        """
        from premise_retriever import (
            _FAMILY_CATALOG_KEYS,
            forms_for_shape_pair,
            retrieve_for_state,
        )
        from evolve.strategy_wrapper import _form_family_label

        diagnostics: dict[str, Any] = {
            "activation": None,
            "retrieved_lemma_set": [],
            "filtered_self": 0,
            "filtered_unavailable": 0,
            "goal_shape": "unknown",
            "lemma_shapes": {},
            "shape_mismatch_filtered": 0,
        }

        retrieval_activation: str | None = None
        retrieved_lemma_set: list[str] = []
        goal_shape = "unknown"
        lemma_shapes: dict[str, str] = {}
        for fam in activated_families:
            if fam in _FAMILY_CATALOG_KEYS:
                retrieval_activation = fam
                retrieved_lemma_set, diag = retrieve_for_state(
                    state_pp=state_pp,
                    theorem_name=theorem_name or None,
                    k=retrieval_top_k,
                    family_key=fam,
                    filter_self=retrieval_filter_self,
                    filter_unavailable=retrieval_filter_unavailable,
                    shape_aware=retrieval_shape_filter,
                    return_diagnostics=True,
                )
                diagnostics["filtered_self"] = diag.get("filtered_self", 0)
                diagnostics["filtered_unavailable"] = diag.get(
                    "filtered_unavailable", 0
                )
                diagnostics["goal_shape"] = diag.get("goal_shape", "unknown")
                diagnostics["lemma_shapes"] = diag.get("lemma_shapes", {}) or {}
                goal_shape = diagnostics["goal_shape"]
                lemma_shapes = diagnostics["lemma_shapes"]
                break
        diagnostics["activation"] = retrieval_activation
        diagnostics["retrieved_lemma_set"] = list(retrieved_lemma_set)

        configured_form_labels = [
            _form_family_label(t) for t in retrieval_tactic_forms
        ]
        label_to_template = dict(
            zip(configured_form_labels, retrieval_tactic_forms)
        )
        seen = already_seen if already_seen is not None else set()
        out: list[EmittedTactic] = []
        shape_mismatch_filtered = 0
        for premise in retrieved_lemma_set:
            lemma_shape = lemma_shapes.get(premise, "unknown")
            if retrieval_shape_filter:
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
                    import hashlib as _hl
                    _canonical = "|".join((
                        "retrieved_premise",
                        goal_shape,
                        retrieval_activation or "",
                        str(SPECIFICITY_SPECIFIC),
                        form.strip(),
                        premise,
                        label,
                    ))
                    _stable = _hl.sha1(_canonical.encode("utf-8")).hexdigest()[:12]
                    out.append(EmittedTactic(
                        tactic=tactic,
                        origin="retrieved_premise",
                        skeleton_name=f"retrieved:{premise}:{label}",
                        skeleton_stable_id=_stable,
                        shape=goal_shape,
                        family=retrieval_activation,
                        specificity=SPECIFICITY_SPECIFIC,
                        priority=PRIORITY_RETRIEVED,
                        template_source=form,
                        family_source=retrieval_activation,
                        retrieved_premise=premise,
                        retrieved_form=label,
                        retrieved_shape=lemma_shape,
                    ))
        diagnostics["shape_mismatch_filtered"] = shape_mismatch_filtered
        return out, diagnostics

    def emit_term_builder_tactics(
        self,
        goal_shape: str,
        nat_vars: list[str],
        hypotheses: dict[str, str | None],
        budget: int = 0,
        already_seen: set[str] | None = None,
    ) -> tuple[list[EmittedTactic], str | None]:
        """Emit term_builder skeletons.

        IMPORTANT: term_builder uses LEGACY "shape XOR any" semantics
        (the older path that priority_templates moved away from in
        NS3.5). Exact shape match wins; if no skeleton matches the
        current goal_shape, fall back to the `any` slot. There is no
        "shape then any" merge — that would change behavior versus the
        legacy wrapper. Preserving the legacy semantics keeps NS4.1
        bit-for-bit parity with the legacy path on this origin.

        Returns `(entries, shape_key)` where `shape_key` is the slot
        actually used (e.g. "iff" or "any" or None when no tb skeletons
        are configured at all).
        """
        from evolve.strategy_wrapper import _render_template

        tb_skels = [
            s for s in self.all_skeletons()
            if s.origin == "term_builder" and s.enabled
        ]
        if not tb_skels:
            return [], None

        shape_key: str | None = None
        if any(s.shape == goal_shape for s in tb_skels):
            shape_key = goal_shape
        elif any(s.shape == SHAPE_ANY for s in tb_skels):
            shape_key = SHAPE_ANY
        if shape_key is None:
            return [], None

        seen = already_seen if already_seen is not None else set()
        out: list[EmittedTactic] = []
        emitted = 0
        matching = [s for s in tb_skels if s.shape == shape_key]
        for skel in matching:
            for rendered in _render_template(
                skel.template, nat_vars, hypotheses
            ):
                if not rendered or rendered in seen:
                    continue
                seen.add(rendered)
                out.append(EmittedTactic(
                    tactic=rendered,
                    origin=skel.origin,
                    skeleton_name=skel.name,
                    skeleton_stable_id=skel.stable_id,
                    shape=skel.shape,
                    family=skel.family,
                    specificity=skel.specificity,
                    priority=skel.priority,
                    template_source=skel.template,
                    family_source=shape_key,
                ))
                emitted += 1
                if budget and emitted >= budget:
                    return out, shape_key
            if budget and emitted >= budget:
                return out, shape_key
        return out, shape_key

    # ------------------------------------------------------------------
    # Legacy adapter.
    # ------------------------------------------------------------------

    @classmethod
    def from_legacy_strategy_config(
        cls, cfg: dict[str, Any]
    ) -> "SkeletonBag":
        """Convert a strategy_config dict (the JSON the eval subprocess
        reads) into a SkeletonBag.

        The conversion is total: every entry in every legacy field is
        represented as a Skeleton. In the NS4 prototype, only the
        priority_template skeletons are emitted through the new path —
        the others are present for introspection / future migration.

        Specificity is computed via the legacy classifier so the new
        bag's ordering matches the legacy wrapper exactly.
        """
        from evolve.strategy_wrapper import classify_template_specificity

        bag = cls()
        idx = 0  # global counter for unique skeleton names

        # priority_templates
        for shape, templates in (cfg.get("priority_templates") or {}).items():
            for raw in templates or []:
                if not raw or not str(raw).strip():
                    continue
                rank, _label = classify_template_specificity(raw)
                bag.add(Skeleton(
                    name=f"pt_{shape}_{idx}",
                    shape=str(shape),
                    template=raw,
                    origin="priority_template",
                    family=None,
                    priority=PRIORITY_PRIORITY_TEMPLATE,
                    specificity=rank,
                    tags=["legacy_adapter"],
                ))
                idx += 1

        # theorem_family_tactics: family-gated, shape=any.
        for fam, tactics in (cfg.get("theorem_family_tactics") or {}).items():
            for raw in tactics or []:
                if not raw or not str(raw).strip():
                    continue
                rank, _ = classify_template_specificity(raw)
                bag.add(Skeleton(
                    name=f"fam_{fam}_{idx}",
                    shape=SHAPE_ANY,
                    template=raw,
                    origin="family_tactic",
                    family=fam,
                    priority=PRIORITY_FAMILY,
                    specificity=rank,
                    tags=["legacy_adapter", "family"],
                ))
                idx += 1

        # term_builder_templates
        for shape, templates in (cfg.get("term_builder_templates") or {}).items():
            for raw in templates or []:
                if not raw or not str(raw).strip():
                    continue
                rank, _ = classify_template_specificity(raw)
                bag.add(Skeleton(
                    name=f"tb_{shape}_{idx}",
                    shape=str(shape),
                    template=raw,
                    origin="term_builder",
                    family=None,
                    priority=PRIORITY_TERM_BUILDER,
                    specificity=rank,
                    tags=["legacy_adapter", "term_builder"],
                ))
                idx += 1

        # fallback_tactics: flat list, shape=any, generic by default.
        for raw in (cfg.get("fallback_tactics") or []):
            if not raw or not str(raw).strip():
                continue
            rank, _ = classify_template_specificity(raw)
            bag.add(Skeleton(
                name=f"fb_{idx}",
                shape=SHAPE_ANY,
                template=raw,
                origin="fallback_tactic",
                family=None,
                priority=PRIORITY_FALLBACK,
                specificity=rank,
                tags=["legacy_adapter", "fallback"],
            ))
            idx += 1

        # tactic_templates: flat list, shape=any.
        for raw in (cfg.get("tactic_templates") or []):
            if not raw or not str(raw).strip():
                continue
            rank, _ = classify_template_specificity(raw)
            bag.add(Skeleton(
                name=f"tt_{idx}",
                shape=SHAPE_ANY,
                template=raw,
                origin="tactic_template",
                family=None,
                priority=PRIORITY_TACTIC_TEMPLATE,
                specificity=rank,
                tags=["legacy_adapter", "tactic_template"],
            ))
            idx += 1

        return bag

    # ------------------------------------------------------------------
    # Introspection helpers (used by reports / debug).
    # ------------------------------------------------------------------

    def all_skeletons(self) -> list[Skeleton]:
        out: list[Skeleton] = []
        for shape, skels in self.skeletons.items():
            out.extend(skels)
        return out

    def count_by_origin(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.all_skeletons():
            counts[s.origin] = counts.get(s.origin, 0) + 1
        return counts

    def __len__(self) -> int:
        return sum(len(v) for v in self.skeletons.values())
