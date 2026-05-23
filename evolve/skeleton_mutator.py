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
    mutation_log.md so we can replay or audit later."""

    operator: str
    affected: list[str] = field(default_factory=list)  # skeleton names
    description: str = ""
    rationale: str = ""

    def to_md_line(self) -> str:
        affected = ", ".join(self.affected) if self.affected else "—"
        return (
            f"- **{self.operator}** affected=[{affected}]  \n"
            f"  description: {self.description}  \n"
            f"  rationale: {self.rationale}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "affected": list(self.affected),
            "description": self.description,
            "rationale": self.rationale,
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
) -> tuple[dict[str, Any], MutationRecord]:
    """Disable up to `max_disable` skeletons that the archive flags as
    dead. Skeletons with any archived win are kept regardless of attempt
    count."""
    bag = genome_to_bag(genome)
    lookup = _skeleton_lookup(bag)
    dead = dead_skeletons(archive_stats, min_attempts=min_attempts)
    affected: list[str] = []
    for s in dead:
        if len(affected) >= max_disable:
            break
        if s.skeleton_name in lookup and _is_mutable_skeleton(lookup[s.skeleton_name]):
            lookup[s.skeleton_name].enabled = False
            affected.append(s.skeleton_name)
    record = MutationRecord(
        operator="disable_dead_skeleton",
        affected=affected,
        description=f"Disabled {len(affected)} dead skeleton(s).",
        rationale=(
            f"Archive flagged each as 0-win after >={min_attempts} attempts; "
            "removing them shortens the per-state emit list."
        ),
    )
    return bag_to_genome(bag, genome), record


def promote_high_win_skeleton(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats],
    top_n: int = 5,
) -> tuple[dict[str, Any], MutationRecord]:
    """Move the top-winning enabled skeleton in each (origin, shape, family)
    bucket to the front of its slot. Already-first skeletons are skipped.

    Reordering inside the (origin, shape, family) bucket does NOT change
    the band ordering — it only fronts the strongest template within a
    band, which can help when the per-state budget cuts off early.
    """
    top = {s.skeleton_name: s for s in top_skeletons_by_wins(archive_stats, n=top_n) if s.wins > 0}
    if not top:
        return deepcopy(genome), MutationRecord(
            operator="promote_high_win_skeleton",
            description="No high-win skeletons in archive yet — no-op.",
            rationale="Archive empty or no winners.",
        )

    bag = genome_to_bag(genome)
    affected: list[str] = []

    # Group bag skeletons by (origin, shape, family) and pin top-winners first.
    groups: dict[tuple[str, str, str | None], list[Skeleton]] = {}
    for s in bag.all_skeletons():
        groups.setdefault((s.origin, s.shape, s.family), []).append(s)

    new_skeletons: dict[str, list[Skeleton]] = {}
    new_families: dict[str, list[Skeleton]] = {}
    # Walk shapes in original insertion order.
    for shape in list(bag.skeletons.keys()):
        new_skeletons[shape] = []
    for fam in list(bag.families.keys()):
        new_families[fam] = []

    # Rebuild per-shape, prioritizing top-winners inside their bucket.
    visited: set[int] = set()
    for shape in list(bag.skeletons.keys()):
        for s in bag.skeletons[shape]:
            if id(s) in visited:
                continue
            key = (s.origin, s.shape, s.family)
            bucket = groups[key]
            # Pull winners first.
            winners = [b for b in bucket if b.name in top]
            losers = [b for b in bucket if b.name not in top]
            for b in winners + losers:
                if id(b) in visited:
                    continue
                visited.add(id(b))
                new_skeletons[b.shape].append(b)
                if b.family is not None:
                    new_families.setdefault(b.family, []).append(b)
                if b.name in top and b is winners[0] if winners else False:
                    affected.append(b.name)

    bag.skeletons = new_skeletons
    bag.families = new_families
    record = MutationRecord(
        operator="promote_high_win_skeleton",
        affected=sorted(set(affected)),
        description=(
            f"Promoted {len(set(affected))} high-win skeleton(s) "
            "to the front of their (origin, shape, family) bucket."
        ),
        rationale=(
            f"Archive shows each won >=1 theorem; fronting reduces the "
            "chance of being cut off by per-state budget."
        ),
    )
    return bag_to_genome(bag, genome), record


def demote_generic_skeleton(
    genome: dict[str, Any],
    archive_stats: dict[str, SkeletonStats] | None = None,
) -> tuple[dict[str, Any], MutationRecord]:
    """Within each shape slot, push generics (`specificity=1`) after all
    specifics (`specificity=0`). This is the NS1 invariant — applied here
    as a defensive re-sort after other mutations may have shuffled order.
    """
    bag = genome_to_bag(genome)
    moved: list[str] = []
    for shape, skels in bag.skeletons.items():
        ordered = sorted(
            skels, key=lambda s: (s.priority, s.specificity)
        )
        if [s.name for s in ordered] != [s.name for s in skels]:
            moved.append(shape)
            bag.skeletons[shape] = ordered
    record = MutationRecord(
        operator="demote_generic_skeleton",
        affected=moved,
        description=f"Re-applied NS1 specificity sort in {len(moved)} shape slot(s).",
        rationale="Defensive: maintain (priority, specificity) order after other ops.",
    )
    return bag_to_genome(bag, genome), record


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
