"""EvalMetrics + scalar fitness function.

The score deliberately leads with proved_count: at this stage we want the loop
to chase verified proofs, not to over-optimize secondary signals.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class EvalMetrics:
    """Outcome of evaluating one candidate over a theorem set.

    attempted_count      — theorems where a proof search actually ran
    proved_count         — theorems closed and verified by Lean
    progress_count       — theorems that did NOT error and did NOT finish.
                           Sourced from eval_rollout_all.py's `exhausted`
                           bucket. This is "non-erroring but unfinished":
                           the search applied at least one valid-looking
                           tactic but ran out of max_steps without closing
                           the goal. It is NOT a guarantee of real
                           mathematical progress — a candidate that emits
                           valid no-ops like `skip` would also land here.
                           Weighted very lightly in score_metrics() so no
                           realistic amount of "exhausted" can outscore a
                           single PROVED theorem.
    total_steps          — sum of rollout steps across all theorems
    timeout_count        — theorems abandoned to a timeout (or, when the
                           whole subprocess timed out, the full set size)
    invalid_tactic_count — theorems where the search dead-ended on tactic errors
    """

    attempted_count: int = 0
    proved_count: int = 0
    progress_count: int = 0
    total_steps: int = 0
    timeout_count: int = 0
    invalid_tactic_count: int = 0
    # NS4.1 skeleton-level counters. Surfaced for future scoring /
    # archive consumers; not read by score_metrics in the default path
    # so adding them does NOT change the scalar fitness today.
    #   skeleton_attempt_count   — skeleton-sourced candidates run on Lean
    #   skeleton_advanced_count  — skeleton-sourced advances (close or step)
    #   skeleton_proved_count    — proofs whose winning tactic was a skeleton
    skeleton_attempt_count: int = 0
    skeleton_advanced_count: int = 0
    skeleton_proved_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EvalMetrics":
        known = {f.name for f in fields(cls)}
        return cls(**{k: int(v) for k, v in d.items() if k in known})


def score_metrics(metrics: EvalMetrics) -> float:
    """Scalar fitness. Higher is better.

    proved_count dominates (100x). progress_count is weighted at 0.5x —
    deliberately small so that even a worst-case "exhausted on every theorem"
    candidate (e.g. progress_count = attempted_count = ~200) scores below
    a single PROVED theorem. Steps, timeouts and dead-ends are mild
    penalties so the loop prefers cheap, robust strategies among those
    that prove the same count.

    Weight history: progress_count was 5.0x in v1. Lowered to 0.5x in the
    v2 hardening pass because, semantically, progress_count is just
    eval_rollout_all.py's `exhausted` (non-erroring-but-unfinished) and
    does not reliably indicate real mathematical advancement.
    """
    return (
        100.0 * metrics.proved_count
        + 0.5 * metrics.progress_count
        - 0.1 * metrics.total_steps
        - 10.0 * metrics.timeout_count
        - 1.0 * metrics.invalid_tactic_count
    )
