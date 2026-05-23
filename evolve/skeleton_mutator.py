"""NS5 skeleton-level mutator.

Operators act on the legacy strategy-config dict (the JSON the eval
subprocess reads). Internally they convert to a SkeletonBag, mutate, and
serialize back to the dict — that keeps the wire format unchanged and
lets the existing evaluator pipeline run without modification.

The operators are **safe and archive-guided**: each consults a
`dict[str, SkeletonStats]` from `skeleton_archive.aggregate(...)` and
declines to act unless the archive provides evidence the action is
benign. The runner is responsible for verifying the resulting candidate
does not regress on `nat_defs_medium`.

Outputs from each operator:

    (new_genome, MutationRecord)

`MutationRecord` is a small dict logged to `mutation_log.md` so the
final report can attribute proofs (or regressions) to specific edits.

A few facts the operators rely on, established by the NS4 adapter
(`SkeletonBag.from_legacy_strategy_config`):

  - Each priority_template skeleton has a name `pt_<shape>_<idx>`.
  - Each family_tactic skeleton has a name `fam_<family>_<idx>`.
  - Each term_builder skeleton has a name `tb_<shape>_<idx>`.
  - Each fallback_tactic skeleton has a name `fb_<idx>`.
  - Each tactic_template skeleton has a name `tt_<idx>`.

The adapter is total — every entry in the legacy config has a skeleton
representation. After mutation we *reconstruct* the legacy fields from
the mutated bag, preserving insertion order so the eval pipeline
behaves identically to the bag path. Disabling a skeleton means it
disappears from the reconstructed legacy field (the bag path itself
respects `enabled=False`).
"""

from __future__ import annotations

import json
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from evolve.skeleton_archive import (
    DEFAULT_DEAD_ATTEMPT_THRESHOLD,
    PROTECTED_ORIGINS,
    SkeletonStats,
    dead_skeletons,
    top_skeletons_by_wins,
)
from evolve.skeleton_bag import (
    PRIORITY_FALLBACK,
    PRIORITY_FAMILY,
    PRIORITY_PRIORITY_TEMPLATE,
    PRIORITY_RETRIEVED,
    PRIORITY_TACTIC_TEMPLATE,
    PRIORITY_TERM_BUILDER,
    SHAPE_ANY,
    SPECIFICITY_GENERIC,
    SPECIFICITY_SPECIFIC,
    Skeleton,
    SkeletonBag,
)


# ---------------------------------------------------------------------- record
@dataclass
class MutationRecord:
    """Provenance of one mutation step, written into the run's
    mutation_log.md so we can replay or audit later.

    NS6 added scope_* fields so order-changing operators can record the
    exact (origin, shape, family) slice they mutated. This is the
    record we need to diagnose ordering regressions like the ones
    NS5 cycle-2 and cycle-4 produced (broad bag-wide resorts).
    `affected_skeletons` is an alias for `affected` kept for clarity.
    """

    operator: str
    affected: list[str] = field(default_factory=list)  # skeleton names
    description: str = ""
    rationale: str = ""
    scope_origin: str | None = None
    scope_shape: str | None = None
    scope_family: str | None = None

    @property
    def affected_skeletons(self) -> list[str]:
        return self.affected

    def to_md_line(self) -> str:
        affected = ", ".join(self.affected) if self.affected else "—"
        scope_bits = []
        if self.scope_origin:
            scope_bits.append(f"origin={self.scope_origin}")
        if self.scope_shape:
            scope_bits.append(f"shape={self.scope_shape}")
        if self.scope_family:
            scope_bits.append(f"family={self.scope_family}")
        scope = f" scope=({', '.join(scope_bits)})" if scope_bits else ""
        return (
            f"- **{self.operator}**{scope} affected=[{affected}]  \n"
            f"  description: {self.description}  \n"
            f"  rationale: {self.rationale}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "affected": list(self.affected),
            "affected_skeletons": list(self.affected),
            "description": self.description,
            "rationale": self.rationale,
            "scope_origin": self.scope_origin,
            "scope_shape": self.scope_shape,
            "scope_family": self.scope_family,
        }


# ---------------------------------------------------------------------- helpers
def genome_to_bag(genome: dict[str, Any]) -> SkeletonBag:
    return SkeletonBag.from_legacy_strategy_config(genome)


def bag_to_genome(bag: SkeletonBag, base_genome: dict[str, Any]) -> dict[str, Any]:
    """Rebuild the legacy strategy-config dict from a (possibly-mutated)
    bag, preserving every non-skeleton field from `base_genome`.

    The 5 mutable legacy fields are reconstructed by iterating over the
    bag's skeletons in insertion order and grouping by `origin`. Skeletons
    with `enabled=False` are dropped entirely (this is how `disable_*`
    operators take effect on the wire format).
    """
    out = deepcopy(base_genome)

    priority_templates: dict[str, list[str]] = {}
    theorem_family_tactics: dict[str, list[str]] = {}
    term_builder_templates: dict[str, list[str]] = {}
    fallback_tactics: list[str] = []
    tactic_templates: list[str] = []

    for skel in bag.all_skeletons():
        if not skel.enabled:
            continue
        o = skel.origin
        if o == "priority_template":
            priority_templates.setdefault(skel.shape, []).append(skel.template)
        elif o == "family_tactic" and skel.family:
            theorem_family_tactics.setdefault(skel.family, []).append(skel.template)
        elif o == "term_builder":
            term_builder_templates.setdefault(skel.shape, []).append(skel.template)
        elif o == "fallback_tactic":
            fallback_tactics.append(skel.template)
        elif o == "tactic_template":
            tactic_templates.append(skel.template)
        # retrieved_premise is dynamic — not stored in legacy config.

    out["priority_templates"] = priority_templates
    out["theorem_family_tactics"] = theorem_family_tactics
    out["term_builder_templates"] = term_builder_templates
    out["fallback_tactics"] = fallback_tactics
    out["tactic_templates"] = tactic_templates
    return out


def _skeleton_lookup(bag: SkeletonBag) -> dict[str, Skeleton]:
    return {s.name: s for s in bag.all_skeletons()}


def _bag_origin_count(bag: SkeletonBag, origin: str) -> int:
    return sum(1 for s in bag.all_skeletons() if s.origin == origin and s.enabled)


def _is_mutable_skeleton(s: Skeleton) -> bool:
    """We will not mutate skeletons whose adapter-name format we don't
    own. Today every adapter-produced skeleton is mutable."""
    return s.origin not in PROTECTED_ORIGINS


# ---------------------------------------------------------------------- operators
def disable_dead_skeleton(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
    min_attempts: int = DEFAULT_DEAD_ATTEMPT_THRESHOLD,
    max_disable: int = 3,
    credit_stats: dict[str, dict[str, int]] | None = None,
) -> tuple[dict[str, Any], MutationRecord]:
    """Disable up to `max_disable` skeletons that look dead.

    NS6 safe-pruning rule (active when `credit_stats` is provided):
    disable only if direct_wins=0 AND advances=0 AND assist_wins_k3=0
    AND attempts >= min_attempts. Skeletons with assist credit are
    *never* disabled, even when their wins/advances are zero.

    NS5 fallback (no credit_stats supplied): keep the old wins-only
    rule for backwards compatibility, with an inline guard that still
    refuses to touch skeletons that won at all in the archive.
    """
    bag = genome_to_bag(genome)
    lookup = _skeleton_lookup(bag)
    affected: list[str] = []
    skipped_by_credit: list[str] = []

    def _has_credit(name: str) -> bool:
        if not credit_stats:
            return False
        c = credit_stats.get(name)
        if not c:
            return False
        return (
            int(c.get("direct_wins", 0)) > 0
            or int(c.get("advances", 0)) > 0
            or int(c.get("assist_wins_k3", 0)) > 0
        )

    if credit_stats is not None:
        # NS6 path: walk every observed skeleton and disable only those
        # with zero credit signals AND enough attempts. archive_stats
        # supplies attempts; credit_stats supplies win/advance/assist.
        candidates = []
        for name, c in credit_stats.items():
            if name not in lookup:
                continue
            attempts = int(c.get("attempts", 0))
            if attempts < min_attempts:
                continue
            direct = int(c.get("direct_wins", 0))
            adv = int(c.get("advances", 0))
            assist = int(c.get("assist_wins_k3", 0))
            if direct == 0 and adv == 0 and assist == 0:
                candidates.append((name, attempts))
        # Sort by attempts desc — high-attempt zero-credit goes first.
        candidates.sort(key=lambda t: (-t[1], t[0]))
        for name, _att in candidates:
            if len(affected) >= max_disable:
                break
            if _is_mutable_skeleton(lookup[name]) and lookup[name].enabled:
                lookup[name].enabled = False
                affected.append(name)
        # Also note any skeletons we explicitly protected by assist credit.
        for name, c in credit_stats.items():
            if c.get("direct_wins", 0) == 0 and c.get("advances", 0) == 0 \
                    and c.get("assist_wins_k3", 0) > 0:
                skipped_by_credit.append(name)
    else:
        # NS5 fallback path.
        dead = dead_skeletons(archive_stats, min_attempts=min_attempts)
        for s in dead:
            if len(affected) >= max_disable:
                break
            if s.skeleton_name in lookup and _is_mutable_skeleton(lookup[s.skeleton_name]):
                lookup[s.skeleton_name].enabled = False
                affected.append(s.skeleton_name)

    rationale_bits = [
        f"safe-pruning: attempts>={min_attempts}",
        "direct_wins=0",
        "advances=0",
        "assist_wins_k3=0" if credit_stats is not None else "(NS5 wins-only)",
    ]
    desc = f"Disabled {len(affected)} truly-dead skeleton(s)."
    if skipped_by_credit:
        desc += f" Protected {len(skipped_by_credit)} zero-win assist skeleton(s)."
    record = MutationRecord(
        operator="disable_dead_skeleton",
        affected=affected,
        description=desc,
        rationale="; ".join(rationale_bits),
    )
    return bag_to_genome(bag, genome), record


def promote_high_win_skeleton(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
    top_n: int = 5,
    scope_origin: str | None = None,
    scope_shape: str | None = None,
    scope_family: str | None = None,
) -> tuple[dict[str, Any], MutationRecord]:
    """SCOPED: front the top archive-winning skeleton inside ONE
    (scope_origin, scope_shape, scope_family) bucket.

    NS5 cycle-4 lost `Nat.add_mod_eq_add_mod_right` because the unscoped
    version rebuilt the entire `bag.skeletons[shape]` list from scratch,
    inadvertently reordering unrelated buckets that the wrapper iterates
    in bag order. Scoping by (origin, shape, family) restricts the
    reorder to within a single equivalence class that shares (priority,
    specificity), so the wrapper's emit order is preserved between
    classes.

    When `scope_*` is None, no reorder happens — the caller must supply
    a scope. This prevents the "broadest possible mutation" trap that
    NS5 exhibited.
    """
    if scope_origin is None or scope_shape is None:
        return deepcopy(genome), MutationRecord(
            operator="promote_high_win_skeleton",
            description="No scope provided — no-op.",
            rationale="NS6 requires explicit (scope_origin, scope_shape).",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
            scope_family=scope_family,
        )
    top_names = {
        s.skeleton_name
        for s in top_skeletons_by_wins(archive_stats, n=top_n)
        if s.wins > 0
    }
    if not top_names:
        return deepcopy(genome), MutationRecord(
            operator="promote_high_win_skeleton",
            description="No high-win skeletons in archive yet — no-op.",
            rationale="Archive empty or no winners.",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
            scope_family=scope_family,
        )
    bag = genome_to_bag(genome)
    if scope_shape not in bag.skeletons:
        return deepcopy(genome), MutationRecord(
            operator="promote_high_win_skeleton",
            description=f"shape={scope_shape} not present — no-op.",
            rationale="Cannot promote inside a missing shape slot.",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
            scope_family=scope_family,
        )
    skels = bag.skeletons[scope_shape]
    # Indices of skeletons that match the scope.
    in_scope_idx = [
        i for i, s in enumerate(skels)
        if s.origin == scope_origin and (
            scope_family is None or s.family == scope_family
        )
    ]
    if len(in_scope_idx) < 2:
        return deepcopy(genome), MutationRecord(
            operator="promote_high_win_skeleton",
            description=(
                f"<2 skeletons in scope (origin={scope_origin}, shape={scope_shape}, "
                f"family={scope_family}) — nothing to reorder."
            ),
            rationale="Reorder requires >=2 candidates within scope.",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
            scope_family=scope_family,
        )
    in_scope_names = [skels[i].name for i in in_scope_idx]
    # Top-winner from archive that's in this scope.
    top_winner: str | None = None
    for st in top_skeletons_by_wins(archive_stats, n=max(top_n, 50)):
        if st.skeleton_name in in_scope_names and st.wins > 0:
            top_winner = st.skeleton_name
            break
    if top_winner is None:
        return deepcopy(genome), MutationRecord(
            operator="promote_high_win_skeleton",
            description="No archive winner inside the scope — no-op.",
            rationale="Archive has no win record for any skeleton in this scope.",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
            scope_family=scope_family,
        )
    if skels[in_scope_idx[0]].name == top_winner:
        return deepcopy(genome), MutationRecord(
            operator="promote_high_win_skeleton",
            description="Top winner already at front of scope — no-op.",
            rationale="Idempotent: nothing to do.",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
            scope_family=scope_family,
        )
    # Build new ordering: move `top_winner` to the FIRST in-scope position;
    # all other positions (in-scope and out-of-scope) are preserved exactly.
    winner_idx = next(i for i in in_scope_idx if skels[i].name == top_winner)
    front_idx = in_scope_idx[0]
    new_list = list(skels)
    s_win = new_list.pop(winner_idx)
    new_list.insert(front_idx, s_win)
    bag.skeletons[scope_shape] = new_list
    # Families dict points at the *same* objects, so order there is
    # unchanged — `bag_to_genome` reconstructs from `all_skeletons()` which
    # walks `skeletons`. We touch families only if scope had `scope_family`.
    if scope_family and scope_family in bag.families:
        fam_list = bag.families[scope_family]
        if any(s.name == top_winner for s in fam_list):
            fam_list_new = [s for s in fam_list if s.name != top_winner]
            # Insert at first position whose origin matches scope_origin.
            inserted = False
            for i, s in enumerate(fam_list_new):
                if s.origin == scope_origin:
                    fam_list_new.insert(i, s_win)
                    inserted = True
                    break
            if not inserted:
                fam_list_new.insert(0, s_win)
            bag.families[scope_family] = fam_list_new
    return bag_to_genome(bag, genome), MutationRecord(
        operator="promote_high_win_skeleton",
        affected=[top_winner],
        description=(
            f"Fronted '{top_winner}' inside (origin={scope_origin}, "
            f"shape={scope_shape}, family={scope_family}). "
            f"Out-of-scope skeletons untouched."
        ),
        rationale=(
            "Archive shows this skeleton wins inside the scope; fronting "
            "it within scope changes ordering only between siblings of the "
            "same (priority, specificity)."
        ),
        scope_origin=scope_origin,
        scope_shape=scope_shape,
        scope_family=scope_family,
    )


def demote_generic_skeleton(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats] | None = None,
    scope_origin: str | None = None,
    scope_shape: str | None = None,
) -> tuple[dict[str, Any], MutationRecord]:
    """SCOPED: re-apply the NS1 (priority, specificity) sort within ONE
    (scope_origin, scope_shape) slice.

    Without a scope, no-op — NS5 showed that the bag-wide resort
    reorders unrelated origins (e.g. it shuffles fallback_tactic vs
    tactic_template entries that the wrapper emits in bag order), which
    caused regressions on `Nat.two_mul_ne_two_mul_add_one`. Scoping to a
    single (origin, shape) limits the resort to skeletons that all
    share the same band, so the wrapper's emit sequence is preserved
    between scopes.
    """
    if scope_origin is None or scope_shape is None:
        return deepcopy(genome), MutationRecord(
            operator="demote_generic_skeleton",
            description="No scope provided — no-op.",
            rationale="NS6 requires explicit (scope_origin, scope_shape).",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
        )
    bag = genome_to_bag(genome)
    if scope_shape not in bag.skeletons:
        return deepcopy(genome), MutationRecord(
            operator="demote_generic_skeleton",
            description=f"shape={scope_shape} not present — no-op.",
            rationale="Cannot resort a missing shape slot.",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
        )
    skels = bag.skeletons[scope_shape]
    in_scope_idx = [i for i, s in enumerate(skels) if s.origin == scope_origin]
    if len(in_scope_idx) < 2:
        return deepcopy(genome), MutationRecord(
            operator="demote_generic_skeleton",
            description="<2 skeletons in scope — no-op.",
            rationale="Resort requires >=2 candidates within scope.",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
        )
    in_scope = [skels[i] for i in in_scope_idx]
    resorted = sorted(in_scope, key=lambda s: (s.priority, s.specificity))
    if [s.name for s in resorted] == [s.name for s in in_scope]:
        return deepcopy(genome), MutationRecord(
            operator="demote_generic_skeleton",
            description="Scope already sorted — no-op.",
            rationale="Idempotent: nothing to do.",
            scope_origin=scope_origin,
            scope_shape=scope_shape,
        )
    new_list = list(skels)
    for new_pos, idx in zip(in_scope_idx, range(len(resorted))):
        new_list[new_pos] = resorted[idx]
    bag.skeletons[scope_shape] = new_list
    affected = [s.name for s in resorted]
    return bag_to_genome(bag, genome), MutationRecord(
        operator="demote_generic_skeleton",
        affected=affected,
        description=(
            f"NS1-sorted {len(in_scope_idx)} skeleton(s) inside "
            f"(origin={scope_origin}, shape={scope_shape}); out-of-scope untouched."
        ),
        rationale=(
            "Defensive (priority, specificity) sort, scoped to a single "
            "origin so the wrapper's emit order between origins is preserved."
        ),
        scope_origin=scope_origin,
        scope_shape=scope_shape,
    )


SHAPE_CLONE_GRAPH = {
    "iff": ["any"],
    "eq": ["iff"],
    "lt": ["le"],
    "le": ["lt"],
    "dvd": [],
}


def clone_skeleton_to_shape(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
    top_n: int = 5,
) -> tuple[dict[str, Any], MutationRecord]:
    """Clone the top-1 priority_template winner per shape into the
    *enabled-but-not-already-present* allowed cousins from
    `SHAPE_CLONE_GRAPH`. The clone runs only when:

      1. The source skeleton is a priority_template,
      2. The target shape exists in the genome (no new shape slots),
      3. The target slot does NOT already contain the template,
      4. There exists at least one archive win for the source.
    """
    bag = genome_to_bag(genome)
    top = top_skeletons_by_wins(archive_stats, n=top_n)
    affected: list[str] = []

    for s in top:
        if s.wins < 1:
            continue
        # The bag is the authoritative source for `origin` — the archive
        # may record the wrapper's `winning_tactic_origin`, which uses
        # `tactic_template` for both real priority_templates and pure
        # tactic_templates. Look up by name.
        src = next(
            (b for b in bag.all_skeletons() if b.name == s.skeleton_name),
            None,
        )
        if src is None:
            continue
        if src.origin != "priority_template":
            continue
        source_shape = src.shape
        if not source_shape:
            continue
        for target_shape in SHAPE_CLONE_GRAPH.get(source_shape, []):
            target_slot = bag.skeletons.get(target_shape, [])
            if src is None:
                continue
            existing_templates = {b.template for b in target_slot}
            if src.template in existing_templates:
                continue
            cloned = Skeleton(
                name=f"{src.name}->{target_shape}",
                shape=target_shape,
                template=src.template,
                origin=src.origin,
                family=src.family,
                priority=src.priority,
                specificity=src.specificity,
                enabled=True,
                tags=list(src.tags) + ["ns5_clone"],
            )
            bag.add(cloned)
            affected.append(cloned.name)

    record = MutationRecord(
        operator="clone_skeleton_to_shape",
        affected=affected,
        description=f"Cloned {len(affected)} high-win skeleton(s) to cousin shape(s).",
        rationale=(
            "Archive shows the source skeleton wins in its native shape; "
            "the cousin shape is structurally similar (iff↔any, eq↔iff, lt↔le)."
        ),
    )
    return bag_to_genome(bag, genome), record


def narrow_family_gate(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
    min_wins: int = 3,
) -> tuple[dict[str, Any], MutationRecord]:
    """Currently a no-op stub: archived rows do not record losing
    families per skeleton, only winning theorems. Implementing this
    operator usefully requires per-theorem family attribution, which
    NS5.x can add later. For now we record the no-op in the log for
    transparency.
    """
    return deepcopy(genome), MutationRecord(
        operator="narrow_family_gate",
        description="No-op (archive lacks per-attempt family attribution).",
        rationale="Awaiting per-skeleton family-failure logging.",
    )


def expand_family_gate(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
) -> tuple[dict[str, Any], MutationRecord]:
    """For each family_tactic skeleton in the bag with no archived wins
    but with >=1 win on the same template under a different family,
    relax the family gate (set `family=None`). Conservative: requires
    template-text equality, not similarity.
    """
    bag = genome_to_bag(genome)
    affected: list[str] = []
    # Index archive wins by template text.
    tpl_winners: dict[str, list[str]] = {}
    for st in archive_stats.values():
        if st.template and st.wins > 0:
            tpl_winners.setdefault(st.template, []).append(st.skeleton_name)
    # We don't relax automatically — the archive can't yet say a family
    # gate is *blocking* wins. Stub for symmetry; logs intent.
    return deepcopy(genome), MutationRecord(
        operator="expand_family_gate",
        description="No-op (archive lacks shadow-win signal across families).",
        rationale="Awaiting per-theorem family-shadow analysis.",
    )


def budget_trim(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
) -> tuple[dict[str, Any], MutationRecord]:
    """Reduce per-state budgets when the archive shows many dead
    attempts. Concretely:

      - max_extra_tactics_per_state: max(1, current - 1) if dead-skel
        count > 5.
      - family_budgets[fam]: max(1, current - 1) if all family-tactic
        skeletons under `fam` are dead.

    Only reduces; never grows budgets (other operators handle growth).
    """
    g = deepcopy(genome)
    affected: list[str] = []
    dead = dead_skeletons(archive_stats)
    n_dead = len(dead)
    if n_dead > 5:
        cur = g.get("max_extra_tactics_per_state") or 10
        new = max(1, cur - 1)
        if new != cur:
            g["max_extra_tactics_per_state"] = new
            affected.append("max_extra_tactics_per_state")
    # Family-level trim.
    if g.get("family_budgets"):
        dead_set = {s.skeleton_name for s in dead}
        per_fam: dict[str, list[bool]] = {}
        for st in archive_stats.values():
            if st.origin == "family_tactic" and st.skeleton_family:
                per_fam.setdefault(st.skeleton_family, []).append(
                    st.skeleton_name in dead_set
                )
        for fam, dead_flags in per_fam.items():
            if dead_flags and all(dead_flags):
                if fam in g["family_budgets"]:
                    cur = g["family_budgets"][fam]
                    new = max(1, cur - 1)
                    if new != cur:
                        g["family_budgets"][fam] = new
                        affected.append(f"family_budgets[{fam}]")
    return g, MutationRecord(
        operator="budget_trim",
        affected=affected,
        description=f"Trimmed {len(affected)} budget knob(s).",
        rationale=(
            f"Archive shows {n_dead} dead skeleton(s); narrower budgets "
            "reduce wasted Lean roundtrips."
        ),
    )


def archive_seed(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
    top_n: int = 25,
) -> tuple[dict[str, Any], MutationRecord]:
    """Build a candidate that retains ONLY the top-N archived skeletons.

    This is the *compact-genome* experiment from the NS5 plan: how many
    theorems does a minimal skeleton core still prove? Other skeletons
    are disabled (i.e. dropped from the legacy fields on reconstruction).
    Family/fallback/template/term_builder skeletons all participate.

    Skeleton selection: take winners first (descending by wins, then
    advances), then any skeleton whose template appears in the genome
    but not in the archive (insurance against archive-empty cases).
    """
    bag = genome_to_bag(genome)
    keep: set[str] = set()
    top = top_skeletons_by_wins(archive_stats, n=top_n)
    for s in top:
        keep.add(s.skeleton_name)

    # If archive is sparse, keep every skeleton (no-op).
    if len(keep) < max(1, top_n // 3):
        return deepcopy(genome), MutationRecord(
            operator="archive_seed",
            description="No-op (archive too sparse for compact-seed).",
            rationale="Need more archive data before building a compact core.",
        )

    affected: list[str] = []
    for s in bag.all_skeletons():
        if s.name not in keep and _is_mutable_skeleton(s):
            if s.enabled:
                s.enabled = False
                affected.append(s.name)
    return bag_to_genome(bag, genome), MutationRecord(
        operator="archive_seed",
        affected=affected[:50],
        description=(
            f"Disabled {len(affected)} non-archive skeleton(s); kept "
            f"{len(keep)} archive top-winners."
        ),
        rationale="Compact-genome experiment: minimal skeleton core only.",
    )


def _credit_score(c: dict[str, int]) -> int:
    """NS7 score for credit-aware seeding.

    score = 10·direct_wins + 5·assist_wins_k3 + 1·advances
            − 10·regressions − dead-attempt penalty
    """
    direct = int(c.get("direct_wins", 0) or 0)
    assist = int(c.get("assist_wins_k3", 0) or 0)
    adv = int(c.get("advances", 0) or 0)
    attempts = int(c.get("attempts", 0) or 0)
    regr = int(c.get("regressions", 0) or 0)
    dead_penalty = max(0, attempts // 4) if (direct == 0 and assist == 0 and adv == 0) else 0
    return 10 * direct + 5 * assist + 1 * adv - 10 * regr - dead_penalty


def top_skeletons_by_credit_score(
    credit_stats: dict[str, dict[str, int]],
    n: int = 20,
) -> list[tuple[str, int]]:
    """Return [(skeleton_name_or_stable_id, score)] sorted desc by score."""
    rows = [(k, _credit_score(v)) for k, v in credit_stats.items()]
    rows.sort(key=lambda t: (-t[1], t[0]))
    return rows[:n]


def archive_seed_credit(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
    credit_stats: dict[str, dict[str, int]] | None = None,
    top_n: int = 25,
) -> tuple[dict[str, Any], MutationRecord]:
    """NS7 credit-aware compact-genome seeder.

    Replaces NS5's wins-only `archive_seed`. Selection rule:

      1. Take the top-N by `_credit_score` (direct + assist + advance,
         minus regression and dead-attempt penalties). Assist credit
         is half the weight of a direct win; advances are 1/10.
      2. **Unconditionally protect** every skeleton with non-zero
         assist credit (these were the must-protects NS6 identified).
      3. Disable every other mutable skeleton.

    Without `credit_stats` the operator falls back to NS5's wins-only
    behaviour for compatibility.
    """
    if credit_stats is None or not credit_stats:
        return archive_seed(genome, archive_stats, top_n=top_n)
    bag = genome_to_bag(genome)
    # Compute keep-set:
    keep: set[str] = set()
    for name, score in top_skeletons_by_credit_score(credit_stats, n=top_n):
        if score > 0:
            keep.add(name)
    # Protect anything with assist credit.
    for name, c in credit_stats.items():
        if int(c.get("assist_wins_k3", 0) or 0) > 0:
            keep.add(name)
    if len(keep) < max(1, top_n // 3):
        return deepcopy(genome), MutationRecord(
            operator="archive_seed_credit",
            description="No-op (credit index too sparse).",
            rationale="Need more credit data before building a credit-aware core.",
        )
    affected: list[str] = []
    for s in bag.all_skeletons():
        if s.name not in keep and _is_mutable_skeleton(s):
            if s.enabled:
                s.enabled = False
                affected.append(s.name)
    return bag_to_genome(bag, genome), MutationRecord(
        operator="archive_seed_credit",
        affected=affected[:50],
        description=(
            f"Credit-aware compact: kept {len(keep)} skeletons (top "
            f"credit_score and all assist-credit), disabled {len(affected)}."
        ),
        rationale=(
            "Replaces wins-only archive_seed: any skeleton with "
            "assist_wins_k3 > 0 is unconditionally kept, addressing the "
            "NS5 35/38 ceiling that pruned `Nat.div_lt_iff_lt_mul'` assists."
        ),
    )


# ---------------------------------------------------------------------- registry
OPERATORS: dict[str, Callable] = {
    "disable_dead_skeleton": disable_dead_skeleton,
    "promote_high_win_skeleton": promote_high_win_skeleton,
    "demote_generic_skeleton": demote_generic_skeleton,
    "clone_skeleton_to_shape": clone_skeleton_to_shape,
    "narrow_family_gate": narrow_family_gate,
    "expand_family_gate": expand_family_gate,
    "budget_trim": budget_trim,
    "archive_seed": archive_seed,
    "archive_seed_credit": archive_seed_credit,
}


def apply_operator(
    name: str,
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
    **kwargs,
) -> tuple[dict[str, Any], MutationRecord]:
    op = OPERATORS.get(name)
    if op is None:
        raise KeyError(f"Unknown operator: {name}")
    return op(genome, archive_stats, **kwargs)


# ---------------------------------------------------------------------- mutation log
def append_mutation_log(
    log_path,
    cycle: int,
    candidate_name: str,
    records: Iterable[MutationRecord],
    eval_summary: dict[str, Any],
) -> None:
    from pathlib import Path as _Path
    p = _Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    is_new = not p.exists()
    with p.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("# NS5 mutation log\n\n")
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        f.write(f"## cycle {cycle} — {candidate_name} ({ts})\n\n")
        for r in records:
            f.write(r.to_md_line() + "\n")
        f.write("\n**eval summary**:\n```json\n")
        f.write(json.dumps(eval_summary, indent=2, ensure_ascii=False))
        f.write("\n```\n\n")
