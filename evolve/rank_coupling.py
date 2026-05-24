"""NS7 — pre-flight rank-coupling detector.

The credit-aware safe-pruning rule in NS6 still produced regressions
on `Nat.div_lt_iff_lt_mul'`: disabling an apparently-uncredited
skeleton can shift the wrapper's top-K window so that a separate,
credit-bearing skeleton drops out of the ranked list and the proof
fails at a later step.

This module provides a pure-Python check that compares the
*skeleton-emission rank order* of a baseline genome vs. a mutated
genome at every protected (theorem, state_hash) pair, and flags
mutations that would push a protected skeleton past its observed
required_rank_max.

The check operates entirely on the bag's deterministic emit order —
no policy model is loaded, no Lean is run. This is what makes it
"pre-flight": it can reject a mutation before paying for a Lean eval.

Approximation: we do not simulate generative-model output. The
ranked list the wrapper produces interleaves model outputs with
skeleton emissions; we only check the skeleton-emission *index* of
each protected skeleton. If that index is preserved (or improved)
across the mutation, the wrapper's merged top-K should not push the
skeleton out — model outputs are unchanged because the model only
sees the state, not the genome.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable

from evolve.skeleton_bag import SkeletonBag, Skeleton, SHAPE_ANY


@dataclass
class RankViolation:
    skeleton_stable_id: str
    skeleton_name: str | None
    theorem: str | None
    state_hash: str | None
    reason: str  # protection reason (direct_win / assist_win / critical_advance)
    baseline_rank: int | None  # observed required_rank_max
    mutated_rank: int | None   # None if skeleton dropped from bag entirely
    kind: str                   # "dropped" | "pushed_back"
    notes: str = ""


def _enabled_skeletons_by_shape(
    bag: SkeletonBag, shape: str
) -> list[Skeleton]:
    """Return the bag's enabled skeletons that would emit for `shape`,
    ordered the same way the wrapper iterates them:

      1. priority_template skeletons in (priority, specificity) order
         within the shape slot and the any slot
      2. family_tactic skeletons (subset of any slot by origin) in
         insertion order
      3. term_builder skeletons in (priority, specificity) order
      4. fallback_tactic skeletons (any slot, insertion order)
      5. tactic_template skeletons (any slot, insertion order)

    Retrieved_premise emissions are dynamic per-state; they are
    materialized at runtime and not part of the bag's static list. We
    handle them separately by checking enabled state of the retrieval
    pipeline (which is genome-level, not per-skeleton).
    """
    out: list[Skeleton] = []
    # Build priority_template emit by joining shape slot + any slot,
    # sorted by (priority, specificity).
    shape_slot = list(bag.skeletons.get(shape, []))
    any_slot = list(bag.skeletons.get(SHAPE_ANY, []))
    pt_pool = [
        s for s in (shape_slot + any_slot)
        if s.enabled and s.origin == "priority_template"
    ]
    pt_pool.sort(key=lambda s: (s.priority, s.specificity))
    out.extend(pt_pool)
    # family_tactic — iterates `any` slot in insertion order (only
    # those that match an active family; we approximate by counting
    # all enabled family_tactic skeletons that exist).
    out.extend(
        s for s in any_slot
        if s.enabled and s.origin == "family_tactic"
    )
    # term_builder
    out.extend(sorted(
        (s for s in (shape_slot + any_slot)
         if s.enabled and s.origin == "term_builder"),
        key=lambda s: (s.priority, s.specificity),
    ))
    # fallback_tactic (any slot, insertion order)
    out.extend(
        s for s in any_slot
        if s.enabled and s.origin == "fallback_tactic"
    )
    # tactic_template (any slot, insertion order)
    out.extend(
        s for s in any_slot
        if s.enabled and s.origin == "tactic_template"
    )
    return out


def _rank_by_stable_id(skels: list[Skeleton]) -> dict[str, int]:
    """Build {stable_id: rank} for a deterministic emission list."""
    out: dict[str, int] = {}
    for i, s in enumerate(skels):
        if s.stable_id not in out:
            out[s.stable_id] = i
    return out


def check_rank_coupling(
    baseline_genome: dict[str, Any],
    mutated_genome: dict[str, Any],
    protected_entries: Iterable[dict[str, Any]],
    *,
    rank_slack: int = 0,
) -> list[RankViolation]:
    """Return a list of violations (empty list = mutation is safe).

    A violation is reported when a protected skeleton's index in the
    mutated bag's emit order is either:
      - missing entirely (skeleton disabled / removed), OR
      - pushed back past `baseline_rank + rank_slack`.

    `rank_slack` lets the caller tolerate small reorderings. Default 0
    is the strictest — any backward movement is a violation.
    """
    baseline_bag = SkeletonBag.from_legacy_strategy_config(baseline_genome)
    mutated_bag = SkeletonBag.from_legacy_strategy_config(mutated_genome)

    # Group protected entries by their goal-shape (which we approximate
    # using the skeleton's own shape field — protected entries record
    # `shape`).
    by_shape: dict[str, list[dict[str, Any]]] = {}
    for e in protected_entries:
        shape = e.get("shape") or "any"
        by_shape.setdefault(shape, []).append(e)

    violations: list[RankViolation] = []
    for shape, entries in by_shape.items():
        baseline_list = _enabled_skeletons_by_shape(baseline_bag, shape)
        mutated_list = _enabled_skeletons_by_shape(mutated_bag, shape)
        baseline_rank = _rank_by_stable_id(baseline_list)
        mutated_rank = _rank_by_stable_id(mutated_list)
        for e in entries:
            sid = e["skeleton_stable_id"]
            req = e.get("required_rank_max")
            base_idx = baseline_rank.get(sid)
            mut_idx = mutated_rank.get(sid)
            if mut_idx is None:
                # Skeleton dropped entirely. Only flag if it was
                # present in the baseline list (i.e. it was reachable
                # for this shape in the baseline genome).
                if base_idx is not None:
                    violations.append(RankViolation(
                        skeleton_stable_id=sid,
                        skeleton_name=e.get("skeleton_name"),
                        theorem=e.get("theorem"),
                        state_hash=e.get("state_hash"),
                        reason=e.get("reason", "?"),
                        baseline_rank=base_idx,
                        mutated_rank=None,
                        kind="dropped",
                        notes="skeleton not present in mutated bag",
                    ))
                continue
            # The trace's `required_rank_max` is in wrapper-merged-list
            # scale (skeleton emissions interleaved with model output);
            # the bag-side `base_idx` is the skeleton-only emit-list
            # index. Comparing across scales is incorrect — we use the
            # bag-side index throughout. The required_rank_max is kept
            # in the protected file for downstream tools but not used
            # as the pre-flight threshold.
            if base_idx is None:
                continue
            if mut_idx > base_idx + rank_slack:
                violations.append(RankViolation(
                    skeleton_stable_id=sid,
                    skeleton_name=e.get("skeleton_name"),
                    theorem=e.get("theorem"),
                    state_hash=e.get("state_hash"),
                    reason=e.get("reason", "?"),
                    baseline_rank=base_idx,
                    mutated_rank=mut_idx,
                    kind="pushed_back",
                    notes=f"skeleton-emit-index moved from {base_idx} to {mut_idx}",
                ))
    return violations


@dataclass
class StateViolation:
    """NS8 — violation found by the full ranked-list simulator.

    Distinct from the NS7 bag-only `RankViolation`: this is computed
    from the wrapper's actual merged ranked list (skeleton emissions
    interleaved with cached model outputs), so it catches the
    second-order rank-coupling effects NS7 could not.
    """
    theorem: str
    state_hash: str | None
    critical_tactic: str
    critical_skeleton_stable_id: str | None
    reason: str
    baseline_rank: int | None
    mutated_rank: int | None
    kind: str   # "dropped" | "pushed_back"
    notes: str = ""


def check_state_coupling(
    baseline_genome: dict[str, Any],
    mutated_genome: dict[str, Any],
    protected_states: list[dict[str, Any]],
    simulator,
    *,
    rank_slack: int = 0,
    only_critical_tactic_drop: bool = True,
    k: int = 8,
) -> list[StateViolation]:
    """For each protected state, simulate both genomes' ranked lists
    and flag the mutation when the critical_tactic disappears (or is
    pushed back past `baseline_rank + rank_slack`).

    `only_critical_tactic_drop=True` (default) reports only the
    "tactic vanished" case. Setting it False also reports
    backward-rank-movement.

    `simulator` must be an instance of `evolve.rank_simulator.RankSimulator`.
    """
    violations: list[StateViolation] = []
    for st in protected_states:
        state_pp = st.get("state_pp")
        full_name = st.get("full_name")
        if not state_pp or not full_name:
            continue
        if not simulator.has_cache(state_pp, full_name):
            # Skip states without cached model outputs — we can't
            # simulate them faithfully.
            continue
        critical_tactic = st.get("critical_tactic")
        if not critical_tactic:
            continue
        base_res = simulator.simulate(
            baseline_genome, state_pp, full_name, k=k,
            state_hash=st.get("state_hash"),
        )
        mut_res = simulator.simulate(
            mutated_genome, state_pp, full_name, k=k,
            state_hash=st.get("state_hash"),
        )
        base_rank = base_res.find(critical_tactic)
        mut_rank = mut_res.find(critical_tactic)
        if base_rank is None:
            # Critical tactic isn't even in the baseline — protected
            # set has a stale entry. Skip silently.
            continue
        if mut_rank is None:
            violations.append(StateViolation(
                theorem=st.get("theorem") or full_name,
                state_hash=st.get("state_hash"),
                critical_tactic=critical_tactic,
                critical_skeleton_stable_id=st.get("critical_skeleton_stable_id"),
                reason=st.get("reason", "?"),
                baseline_rank=base_rank,
                mutated_rank=None,
                kind="dropped",
                notes="critical tactic absent from mutated ranked list",
            ))
            continue
        if not only_critical_tactic_drop and mut_rank > base_rank + rank_slack:
            violations.append(StateViolation(
                theorem=st.get("theorem") or full_name,
                state_hash=st.get("state_hash"),
                critical_tactic=critical_tactic,
                critical_skeleton_stable_id=st.get("critical_skeleton_stable_id"),
                reason=st.get("reason", "?"),
                baseline_rank=base_rank,
                mutated_rank=mut_rank,
                kind="pushed_back",
                notes=f"rank moved from {base_rank} to {mut_rank}",
            ))
    return violations


def summarize_state_violations(
    violations: list[StateViolation],
) -> dict[str, Any]:
    by_kind: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    affected_theorems: set[str] = set()
    for v in violations:
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
        by_reason[v.reason] = by_reason.get(v.reason, 0) + 1
        affected_theorems.add(v.theorem)
    return {
        "total": len(violations),
        "by_kind": by_kind,
        "by_reason": by_reason,
        "affected_theorems": sorted(affected_theorems),
    }


def summarize_violations(violations: list[RankViolation]) -> dict[str, Any]:
    """Compact summary suitable for logging."""
    by_kind: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    affected_theorems: set[str] = set()
    for v in violations:
        by_kind[v.kind] = by_kind.get(v.kind, 0) + 1
        by_reason[v.reason] = by_reason.get(v.reason, 0) + 1
        if v.theorem:
            affected_theorems.add(v.theorem)
    return {
        "total": len(violations),
        "by_kind": by_kind,
        "by_reason": by_reason,
        "affected_theorems": sorted(affected_theorems),
    }
