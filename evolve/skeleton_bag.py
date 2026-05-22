"""NS4 skeleton-bag prototype.

A minimal skeleton-based representation that coexists with the existing
flat-field wrapper. Implements:

  - `Skeleton`: one emit unit, gated by shape and (optionally) family.
  - `SkeletonBag`: ordered collection keyed by shape; produces EmittedTactic
    lists in the same order today's wrapper produces them.
  - `EmittedTactic`: a rendered tactic with origin / attribution metadata.

Scope (NS4 4-hour prototype):
  - Only `priority_templates` is actually emitted through this path. The
    other origins (family / fallback / term_builder / tactic_template) are
    *parsed* into Skeletons by the adapter for introspection but still
    emitted via the legacy code path in `strategy_wrapper.py`.
  - No slot-vocabulary mutation. Templates rendered through
    `strategy_wrapper._render_template`.
  - No changes to genome JSON schema beyond an optional `use_skeleton_bag`
    flag.

See `project/evolve/reports/ns4_skeleton_bag_design_note.md` for context.
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


@dataclass
class EmittedTactic:
    """A rendered tactic together with its attribution metadata.

    The `tactic` field is the post-render Lean tactic string. Everything
    else is for trace / scoring attribution.
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


class SkeletonBag:
    """Ordered collection of Skeletons, keyed by shape.

    Internal storage: `self.skeletons` is `dict[shape, list[Skeleton]]`,
    insertion-ordered per shape. The `for_state` and `emit_*` methods
    pick up the shape-matching list plus the `any` list and order them
    by `priority`, then `specificity`, then insertion order.
    """

    def __init__(self) -> None:
        self.skeletons: dict[str, list[Skeleton]] = {}

    def add(self, skeleton: Skeleton) -> None:
        if skeleton.shape not in self.skeletons:
            self.skeletons[skeleton.shape] = []
        self.skeletons[skeleton.shape].append(skeleton)

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
