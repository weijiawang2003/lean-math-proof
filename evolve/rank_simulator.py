"""NS8 — full ranked-list simulator.

Instantiates the real `StrategyWrapperPolicy` with a *fake* base
policy that returns cached model outputs. This lets us simulate the
wrapper's merged ranked list exactly — same dedup, same cap, same
priority/base/extra ordering — without loading the actual model.

The wrapper is unchanged: it still calls `base.rank_tactics(state_pp,
full_name, k=k)`. Our `CachedBasePolicy` just looks up the cached
outputs by `(state_pp, full_name)` key.

This gives us a ground-truth simulation of what the eval rollout
would see — modulo Lean's actual response to the proposed tactic.

API:

    sim = RankSimulator(model_cache_path)
    rl = sim.simulate(genome, state_pp, full_name, k=8)
    # rl is a list of dicts: {tactic, origin, skeleton_name,
    #                          skeleton_stable_id, rank}
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class CachedBasePolicy:
    """Stand-in for `GenerativePolicy` that returns cached top-K for
    each (state_pp, full_name)."""

    def __init__(
        self,
        cache: dict[tuple[str, str], list[str]],
        ckpt_dir: str = "(cached)",
    ):
        self._cache = cache
        self.ckpt_dir = ckpt_dir

    def rank_tactics(
        self,
        state_pp: str,
        full_name: str = "",
        k: int = 8,
    ) -> list[str]:
        key = (state_pp or "", full_name or "")
        return list(self._cache.get(key, []))[:k]


@dataclass
class SimEntry:
    rank: int
    tactic: str
    origin: str | None
    skeleton_name: str | None
    skeleton_stable_id: str | None
    skeleton_shape: str | None
    skeleton_family: str | None


@dataclass
class SimResult:
    state_hash: str | None
    state_pp: str
    full_name: str
    entries: list[SimEntry] = field(default_factory=list)

    def find(self, tactic: str) -> int | None:
        for e in self.entries:
            if e.tactic == tactic:
                return e.rank
        return None


class RankSimulator:
    """Simulate the wrapper's merged ranked list deterministically
    using cached model outputs."""

    def __init__(
        self,
        model_cache_path: Path | str,
    ):
        cache_rows: list[dict[str, Any]] = []
        p = Path(model_cache_path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cache_rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        self._cache: dict[tuple[str, str], list[str]] = {}
        for r in cache_rows:
            key = (r.get("state_pp") or "", r.get("full_name") or "")
            self._cache[key] = list(r.get("model_outputs") or [])
        self._wrapper_cache: dict[int, Any] = {}

    def _build_wrapper(self, genome: dict[str, Any]):
        """Build a StrategyWrapperPolicy from a genome using the cached
        base. We re-build per genome because the wrapper holds
        skeleton-bag state."""
        from evolve.strategy_wrapper import StrategyWrapperPolicy
        base = CachedBasePolicy(self._cache)
        wrapper = StrategyWrapperPolicy(
            base_policy=base,
            fallback_tactics=list(genome.get("fallback_tactics") or []),
            tactic_templates=list(genome.get("tactic_templates") or []),
            max_extra_tactics_per_state=genome.get("max_extra_tactics_per_state"),
            theorem_family_tactics=dict(genome.get("theorem_family_tactics") or {}),
            family_budgets=dict(genome.get("family_budgets") or {}),
            theorem_tactic_denylist=dict(genome.get("theorem_tactic_denylist") or {}),
            retrieval_enabled=bool(genome.get("retrieval_enabled", False)),
            retrieval_top_k=int(genome.get("retrieval_top_k") or 0),
            retrieval_tactic_forms=list(genome.get("retrieval_tactic_forms") or []),
            retrieval_filter_self=bool(genome.get("retrieval_filter_self", True)),
            retrieval_filter_unavailable=bool(genome.get("retrieval_filter_unavailable", True)),
            retrieval_shape_filter=bool(
                genome.get("retrieval_shape_filter", True)
            ),
            retrieval_requires_family=bool(
                genome.get("retrieval_requires_family", True)
            ),
            retrieval_family_gates=list(
                genome.get("retrieval_family_gates") or []
            ),
            term_builder_templates=dict(genome.get("term_builder_templates") or {}),
            term_builder_budget=int(genome.get("term_builder_budget") or 0),
            priority_templates=dict(genome.get("priority_templates") or {}),
            priority_template_budget=int(genome.get("priority_template_budget") or 0),
            use_skeleton_bag=bool(genome.get("use_skeleton_bag", False)),
        )
        return wrapper

    def simulate(
        self,
        genome: dict[str, Any],
        state_pp: str,
        full_name: str,
        k: int = 8,
        state_hash: str | None = None,
    ) -> SimResult:
        """Return a `SimResult` listing every tactic that would appear
        in the wrapper's `last_ranked_tactics` for this state."""
        wrapper = self._build_wrapper(genome)
        # Call rank_tactics — populates wrapper.last_* arrays.
        ranked = wrapper.rank_tactics(state_pp, full_name=full_name, k=k)
        origins = list(wrapper.last_origins)
        skel_names = list(wrapper.last_skeleton_names)
        skel_sids = list(getattr(wrapper, "last_skeleton_stable_ids", []) or [None] * len(ranked))
        skel_shapes = list(wrapper.last_skeleton_shapes)
        skel_families = list(wrapper.last_skeleton_families)
        out_entries: list[SimEntry] = []
        for i, tac in enumerate(ranked):
            out_entries.append(SimEntry(
                rank=i,
                tactic=tac,
                origin=origins[i] if i < len(origins) else None,
                skeleton_name=skel_names[i] if i < len(skel_names) else None,
                skeleton_stable_id=skel_sids[i] if i < len(skel_sids) else None,
                skeleton_shape=skel_shapes[i] if i < len(skel_shapes) else None,
                skeleton_family=skel_families[i] if i < len(skel_families) else None,
            ))
        return SimResult(
            state_hash=state_hash,
            state_pp=state_pp,
            full_name=full_name,
            entries=out_entries,
        )

    def has_cache(self, state_pp: str, full_name: str) -> bool:
        return (state_pp or "", full_name or "") in self._cache
