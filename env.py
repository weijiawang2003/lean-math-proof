from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lean_dojo import LeanError, LeanGitRepo, ProofFinished, TacticState, Theorem

from core_types import TheoremConfig, TransitionRecord

REPO_URL = "https://github.com/leanprover-community/mathlib4"
COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"


@dataclass(frozen=True)
class TransitionOutcome:
    record: TransitionRecord
    next_state: TacticState | None
    is_error: bool
    is_finished: bool


def make_repo() -> LeanGitRepo:
    return LeanGitRepo(url=REPO_URL, commit=COMMIT)


def make_theorem(repo: LeanGitRepo, cfg: TheoremConfig) -> Theorem:
    return Theorem(repo=repo, file_path=cfg.file_path, full_name=cfg.full_name)


def run_transition(
    dojo: Any,
    theorem: Theorem,
    state: TacticState,
    tactic: str,
    *,
    step: int | None = None,
    domain: str | None = None,
    run_id: str | None = None,
    episode_id: str | None = None,
    method: str | None = None,
) -> TransitionOutcome:
    """Run a tactic and normalize output into a serializable transition."""
    result = dojo.run_tac(state, tactic)
    is_finished = isinstance(result, ProofFinished)
    is_error = isinstance(result, LeanError)
    goals_after = None
    if not is_error:
        goals_after = 0 if is_finished else result.num_goals

    record = TransitionRecord(
        file_path=str(theorem.file_path),
        full_name=theorem.full_name,
        state_pp=state.pp,
        tactic=tactic,
        result_kind=type(result).__name__,
        proof_finished=is_finished,
        num_goals_before=state.num_goals,
        num_goals_after=goals_after,
        step=step,
        domain=domain,
        error_message=(
            getattr(result, "message", None)
            or getattr(result, "error", None)
            or (str(result) if is_error else None)
        ),
        run_id=run_id,
        episode_id=episode_id,
        method=method,
    )
    next_state = None if (is_finished or is_error) else result
    return TransitionOutcome(record=record, next_state=next_state, is_error=is_error, is_finished=is_finished)
