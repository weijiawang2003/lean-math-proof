"""evolve/ — an AlphaEvolve-style outer loop for Lean proof-search strategies.

This package does NOT train neural networks. It evolves *search strategies*:
configurations (a `SearchCandidate`) that wrap the existing rollout machinery.
Lean is the evaluator. The analogy to AlphaEvolve:

    AlphaEvolve searches for programs that generate mathematical objects.
    evolve/ searches for configurations that generate proof-search behaviour.

Public API:
    SearchCandidate   — the genome (one proof-search strategy)
    EvalMetrics       — metrics produced by evaluating a candidate
    score_metrics     — scalar fitness from EvalMetrics
    CandidateRecord   — one (generation, candidate, metrics, score) row
    evaluate_candidate— run a candidate (dry-run fake metrics, or real Lean eval)
    mutate_candidate  — produce a child candidate by local mutation
"""

from __future__ import annotations

from evolve.candidate import SearchCandidate
from evolve.scoring import EvalMetrics, score_metrics
from evolve.population import (
    CandidateRecord,
    DEFAULT_POPULATION_PATH,
    append_record,
    load_records,
    select_top,
)
from evolve.evaluator import evaluate_candidate
from evolve.mutator import mutate_candidate

__all__ = [
    "SearchCandidate",
    "EvalMetrics",
    "score_metrics",
    "CandidateRecord",
    "DEFAULT_POPULATION_PATH",
    "append_record",
    "load_records",
    "select_top",
    "evaluate_candidate",
    "mutate_candidate",
]
