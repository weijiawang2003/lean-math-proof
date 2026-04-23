from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lean_dojo import LeanError, LeanGitRepo, ProofFinished, TacticState, Theorem
from lean_dojo.interaction.dojo import DojoCrashError

try:
    from lean_dojo.interaction.dojo import DojoTacticTimeoutError
except ImportError:
    # Older LeanDojo versions may not have this
    DojoTacticTimeoutError = None

from core_types import TheoremConfig, TransitionRecord

REPO_URL = "https://github.com/leanprover-community/mathlib4"
COMMIT = "29dcec074de168ac2bf835a77ef68bbe069194c5"


@dataclass(frozen=True)
class TransitionOutcome:
    record: TransitionRecord
    next_state: TacticState | None
    is_error: bool
    is_finished: bool
    session_dead: bool = False  # True when the Lean REPL crashed (Dojo unusable)


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
    """Run a tactic and normalize output into a serializable transition.

    If the Lean REPL crashes (DojoCrashError), the outcome has
    ``session_dead=True``.  Callers MUST stop using the Dojo after that.
    """
    # Build tuple of fatal exceptions that kill the Dojo session
    _fatal_exceptions = (DojoCrashError,)
    if DojoTacticTimeoutError is not None:
        _fatal_exceptions = (DojoCrashError, DojoTacticTimeoutError)

    try:
        result = dojo.run_tac(state, tactic)
    except _fatal_exceptions as exc:
        # The Lean REPL process died or timed out — session is unrecoverable.
        kind = type(exc).__name__
        record = TransitionRecord(
            file_path=str(theorem.file_path),
            full_name=theorem.full_name,
            state_pp=state.pp,
            tactic=tactic,
            result_kind=kind,
            proof_finished=False,
            num_goals_before=state.num_goals,
            num_goals_after=None,
            step=step,
            domain=domain,
            error_message=f"REPL fatal: {exc}",
            run_id=run_id,
            episode_id=episode_id,
            method=method,
        )
        return TransitionOutcome(
            record=record, next_state=None,
            is_error=True, is_finished=False, session_dead=True,
        )

    is_finished = isinstance(result, ProofFinished)
    is_tactic_state = isinstance(result, TacticState)
    is_error = isinstance(result, LeanError)

    # Treat anything that is not ProofFinished or TacticState as an error.
    # This covers LeanError, ProofGivenUp, and any future result types.
    if not is_finished and not is_tactic_state:
        is_error = True

    goals_after = None
    if is_finished:
        goals_after = 0
    elif is_tactic_state:
        goals_after = result.num_goals

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
    next_state = result if is_tactic_state else None
    return TransitionOutcome(record=record, next_state=next_state, is_error=is_error, is_finished=is_finished)
