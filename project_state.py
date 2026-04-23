"""Persistent project state for incremental theorem proving.

Tracks everything across runs so no work is ever repeated:
  - Which theorems have been discovered, searched, and proved
  - Accumulated trace corpus (append-only)
  - Model versions and their training data hashes
  - Per-step progress within a curriculum stage

State is stored in a single JSON file (project_state.json) that is
updated after every atomic operation. The trace corpus is a separate
append-only JSONL file.

Usage:
    state = ProjectState("my_project")       # loads or creates
    state.mark_searched("Set.union_comm", traces_added=42)
    state.mark_proved("Set.union_comm", tactic="simp", steps=1)
    state.save()                             # explicit save (also auto-saves)

Design principles:
    - Append-only: traces are never deleted, only accumulated
    - Idempotent: re-running a step on an already-searched theorem is a no-op
    - Versioned: each model retrain gets a version number
    - Resumable: any interruption loses at most the current theorem
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ProjectState:
    """Persistent, JSON-backed project state."""

    def __init__(self, project_dir: str = "project"):
        self._dir = Path(project_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._state_path = self._dir / "project_state.json"
        self._traces_path = self._dir / "all_traces.jsonl"
        self._models_dir = self._dir / "models"
        self._models_dir.mkdir(exist_ok=True)

        if self._state_path.exists():
            self._data = json.loads(self._state_path.read_text(encoding="utf-8"))
            print(f"Loaded project state from {self._state_path}")
        else:
            self._data = self._empty_state()
            self.save()
            print(f"Created new project at {self._dir}")

    # ------------------------------------------------------------------
    # State schema
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_state() -> dict:
        return {
            "created": _utc_now(),
            "last_updated": _utc_now(),
            "version": 1,

            # Theorem registry: full_name → status dict
            "theorems": {},

            # Trace stats
            "total_traces": 0,
            "traces_by_method": {},

            # Model versions: list of {version, type, ckpt_dir, data_hash, ...}
            "models": [],

            # Curriculum progress
            "curriculum_stage": 0,
            "curriculum_log": [],

            # Search config used (for reproducibility)
            "search_config": {},
        }

    # ------------------------------------------------------------------
    # Save / backup
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Write state to disk. Creates a backup of the previous state."""
        self._data["last_updated"] = _utc_now()

        # Backup previous state
        if self._state_path.exists():
            backup = self._dir / "project_state.backup.json"
            shutil.copy2(self._state_path, backup)

        self._state_path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ------------------------------------------------------------------
    # Theorem management
    # ------------------------------------------------------------------

    @property
    def theorems(self) -> dict[str, dict]:
        return self._data["theorems"]

    def register_theorem(
        self,
        full_name: str,
        file_path: str,
        difficulty: str = "unknown",
        available: bool | None = None,
    ) -> None:
        """Register a theorem. Idempotent — doesn't overwrite existing data."""
        if full_name in self.theorems:
            # Update availability if newly known
            if available is not None and self.theorems[full_name].get("available") is None:
                self.theorems[full_name]["available"] = available
            return

        self.theorems[full_name] = {
            "file_path": file_path,
            "difficulty": difficulty,
            "available": available,
            "searched": False,
            "search_traces": 0,
            "proved": False,
            "proof_tactics": None,
            "proof_steps": None,
            "search_timestamp": None,
            "proof_timestamp": None,
            "error_message": None,
        }

    def register_theorems_from_configs(self, configs: list) -> None:
        """Register TheoremConfig objects."""
        for cfg in configs:
            self.register_theorem(cfg.full_name, cfg.file_path)

    def mark_searched(
        self,
        full_name: str,
        traces_added: int = 0,
        method: str = "beam_search",
    ) -> None:
        """Mark a theorem as searched (traces collected)."""
        if full_name not in self.theorems:
            return
        t = self.theorems[full_name]
        t["searched"] = True
        t["search_traces"] += traces_added
        t["search_timestamp"] = _utc_now()

        # Update global trace stats
        self._data["total_traces"] += traces_added
        counts = self._data["traces_by_method"]
        counts[method] = counts.get(method, 0) + traces_added

    def mark_proved(
        self,
        full_name: str,
        tactic: str | None = None,
        steps: int | None = None,
        method: str = "policy_rollout",
    ) -> None:
        """Mark a theorem as proved."""
        if full_name not in self.theorems:
            return
        t = self.theorems[full_name]
        t["proved"] = True
        t["proof_tactics"] = tactic
        t["proof_steps"] = steps
        t["proof_timestamp"] = _utc_now()

    def mark_unavailable(self, full_name: str, reason: str = "") -> None:
        if full_name not in self.theorems:
            return
        self.theorems[full_name]["available"] = False
        self.theorems[full_name]["error_message"] = reason

    def get_unsearched(self, difficulty: str | None = None) -> list[str]:
        """Get theorem names that haven't been searched yet."""
        results = []
        for name, t in self.theorems.items():
            if t["searched"]:
                continue
            if t.get("available") is False:
                continue
            if difficulty and t.get("difficulty") != difficulty:
                continue
            results.append(name)
        return results

    def get_unproved(self) -> list[str]:
        """Get theorem names that have been searched but not proved."""
        return [
            name for name, t in self.theorems.items()
            if t["searched"] and not t["proved"] and t.get("available") is not False
        ]

    def get_proved(self) -> list[str]:
        return [name for name, t in self.theorems.items() if t["proved"]]

    # ------------------------------------------------------------------
    # Trace corpus
    # ------------------------------------------------------------------

    @property
    def traces_path(self) -> str:
        return str(self._traces_path)

    def append_traces(self, source_path: str) -> int:
        """Append traces from a file to the cumulative corpus. Returns count."""
        if not Path(source_path).exists():
            return 0
        count = 0
        with open(self._traces_path, "a", encoding="utf-8") as fout:
            with open(source_path, "r", encoding="utf-8") as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line)
                        count += 1
        return count

    @property
    def total_traces(self) -> int:
        return self._data["total_traces"]

    # ------------------------------------------------------------------
    # Model versioning
    # ------------------------------------------------------------------

    @property
    def models(self) -> list[dict]:
        return self._data["models"]

    @property
    def latest_model(self) -> dict | None:
        return self.models[-1] if self.models else None

    def model_dir(self, version: int, model_type: str = "gen") -> str:
        return str(self._models_dir / f"{model_type}_v{version}")

    def needs_retrain(self, model_type: str = "gen") -> bool:
        """Check if new traces exist since last training."""
        latest = None
        for m in reversed(self.models):
            if m.get("type") == model_type:
                latest = m
                break
        if latest is None:
            return True  # No model exists yet
        return self._data["total_traces"] > latest.get("trained_on_traces", 0)

    def register_model(
        self,
        version: int,
        model_type: str,
        ckpt_dir: str,
        base_model: str = "",
        trained_on_traces: int = 0,
        data_hash: str = "",
        extra: dict | None = None,
    ) -> None:
        """Register a trained model version."""
        entry = {
            "version": version,
            "type": model_type,
            "ckpt_dir": ckpt_dir,
            "base_model": base_model,
            "trained_on_traces": trained_on_traces,
            "data_hash": data_hash,
            "timestamp": _utc_now(),
        }
        if extra:
            entry.update(extra)
        self.models.append(entry)

    def next_model_version(self, model_type: str = "gen") -> int:
        versions = [m["version"] for m in self.models if m.get("type") == model_type]
        return max(versions, default=0) + 1

    # ------------------------------------------------------------------
    # Curriculum progress
    # ------------------------------------------------------------------

    @property
    def curriculum_stage(self) -> int:
        return self._data["curriculum_stage"]

    def advance_curriculum(self, stage: int, note: str = "") -> None:
        self._data["curriculum_stage"] = stage
        self._data["curriculum_log"].append({
            "stage": stage,
            "timestamp": _utc_now(),
            "note": note,
        })

    # ------------------------------------------------------------------
    # Summary / reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Print a human-readable summary of project state."""
        total = len(self.theorems)
        available = sum(1 for t in self.theorems.values() if t.get("available") is not False)
        searched = sum(1 for t in self.theorems.values() if t["searched"])
        proved = sum(1 for t in self.theorems.values() if t["proved"])
        unsearched = len(self.get_unsearched())

        diff_counts = Counter(t.get("difficulty", "?") for t in self.theorems.values())

        lines = [
            f"Project: {self._dir}",
            f"  Theorems: {total} registered, {available} available, "
            f"{searched} searched, {proved} proved, {unsearched} unsearched",
            f"  Difficulty: " + ", ".join(f"{k}={v}" for k, v in sorted(diff_counts.items())),
            f"  Traces: {self.total_traces} total",
            f"  Models: {len(self.models)} trained",
            f"  Curriculum stage: {self.curriculum_stage}",
        ]

        if self.models:
            latest = self.models[-1]
            lines.append(
                f"  Latest model: v{latest['version']} ({latest['type']}) "
                f"trained on {latest.get('trained_on_traces', '?')} traces"
            )

        return "\n".join(lines)

    @property
    def project_dir(self) -> str:
        return str(self._dir)
