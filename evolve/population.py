"""Population manager — stores candidate evaluation results as JSONL.

Each line of the population file is one CandidateRecord: a (generation,
candidate, metrics, score) row. The file accumulates across evolve runs; the
run-id stored in candidate.metadata distinguishes them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from evolve.candidate import SearchCandidate
from evolve.scoring import EvalMetrics, score_metrics

# Default location, relative to the repo root (where run_evolve is launched).
DEFAULT_POPULATION_PATH = Path("project/evolve/population.jsonl")


@dataclass
class CandidateRecord:
    generation: int
    candidate: SearchCandidate
    metrics: EvalMetrics
    score: float
    run_dir: Optional[str] = None

    @classmethod
    def build(
        cls,
        generation: int,
        candidate: SearchCandidate,
        metrics: EvalMetrics,
        run_dir: Optional[str] = None,
    ) -> "CandidateRecord":
        """Construct a record, computing the score from the metrics."""
        return cls(
            generation=generation,
            candidate=candidate,
            metrics=metrics,
            score=score_metrics(metrics),
            run_dir=run_dir,
        )

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "candidate": self.candidate.to_dict(),
            "metrics": self.metrics.to_dict(),
            "score": self.score,
            "run_dir": self.run_dir,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "CandidateRecord":
        return cls(
            generation=int(d["generation"]),
            candidate=SearchCandidate.from_dict(d["candidate"]),
            metrics=EvalMetrics.from_dict(d["metrics"]),
            score=float(d["score"]),
            run_dir=d.get("run_dir"),
        )


def append_record(path: str | Path, record: CandidateRecord) -> None:
    """Append one record as a JSON line, creating parent dirs if needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def load_records(path: str | Path) -> list[CandidateRecord]:
    """Load all records from a JSONL file. Returns [] if the file is absent."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[CandidateRecord] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(CandidateRecord.from_dict(json.loads(line)))
    return records


def select_top(records: list[CandidateRecord], k: int) -> list[CandidateRecord]:
    """Return the k highest-scoring records (descending). Ties broken by
    fewer total_steps, so cheaper strategies win ties."""
    ranked = sorted(
        records,
        key=lambda r: (r.score, -r.metrics.total_steps),
        reverse=True,
    )
    return ranked[: max(0, k)]
