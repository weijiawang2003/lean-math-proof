from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TheoremConfig:
    """Lightweight theorem descriptor used by rollout/search scripts."""

    file_path: str
    full_name: str


@dataclass(frozen=True)
class TransitionRecord:
    """Serializable state-action transition produced by Lean interaction."""

    file_path: str
    full_name: str
    state_pp: str
    tactic: str
    result_kind: str
    proof_finished: bool
    num_goals_before: int | None = None
    num_goals_after: int | None = None
    step: int | None = None
    domain: str | None = None
    error_message: str | None = None
    run_id: str | None = None
    episode_id: str | None = None
    method: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_prompt(state_pp: str, full_name: str = "") -> str:
    """Shared prompt format for policy training/inference."""
    return f"Theorem: {full_name}\n\nProof state:\n{state_pp}\n"
