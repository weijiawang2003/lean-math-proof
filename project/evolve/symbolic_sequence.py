"""SX1 Stage 3 — experimental depth-2 symbolic *sequence* search.

A `SymbolicActionSequence` is a symbolic first action followed by a single
follow-up step (depth 2 only). This module turns the `symbolic_sequence_search`
config block into concrete candidate plans for the live proof state, behind an
explicit `enabled` flag. It is purely additive:

* When the flag is disabled (default), `plan_sequences` returns `[]`, so the
  wrapper's normal raw / single-action symbolic path is byte-for-byte unchanged.
* When enabled, it emits, for each gated symbolic first action that instantiates
  on the state, a depth-2 plan whose follow-ups are the base-model top-k (passed
  in by the caller, if available) and/or a fixed small battery. The plans are
  *added* to the search frontier — they never replace or reorder the NS9 ranked
  list. Stop condition, depth, and caps are all bounded.

There is no live Lean here; the SX1 offline evaluator (scripts/
sx1_sequence_probe_eval.py) reconstructs which plans would have fired and
whether they close, by replaying already-mined trace states. This module is the
single source of truth for *what* a sequence plan is; the evaluator consumes it.
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    from project.evolve.symbolic_actions import (
        SymbolicActionSequence, battery_for_namespace,
        instantiate_symbolic_action, load_sequences,
    )
except ImportError:  # allow import when repo root isn't on sys.path yet
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from project.evolve.symbolic_actions import (
        SymbolicActionSequence, battery_for_namespace,
        instantiate_symbolic_action, load_sequences,
    )


@dataclass(frozen=True)
class SequenceSearchConfig:
    """Parsed `symbolic_sequence_search` config block."""
    enabled: bool = False
    max_depth: int = 2
    max_symbolic_first_actions: int = 2
    max_followup_tactics: int = 6
    namespace_gates: tuple[str, ...] = ()
    followup_modes: tuple[str, ...] = ("base_topk", "fixed_battery")
    timeout_per_theorem: int = 60
    sequences: list[SymbolicActionSequence] = field(default_factory=list)

    @classmethod
    def from_config(cls, cfg: dict | None) -> "SequenceSearchConfig":
        cfg = cfg or {}
        block = cfg.get("symbolic_sequence_search") or {}
        seqs = load_sequences(block.get("sequences") or [])
        fm = block.get("followup_mode") or block.get("followup_modes") \
            or ["base_topk", "fixed_battery"]
        if isinstance(fm, str):
            fm = [fm]
        return cls(
            enabled=bool(block.get("enabled", False)),
            max_depth=int(block.get("max_depth", 2) or 2),
            max_symbolic_first_actions=int(
                block.get("max_symbolic_first_actions", 2) or 2),
            max_followup_tactics=int(
                block.get("max_followup_tactics", 6) or 6),
            namespace_gates=tuple(block.get("namespace_gates") or []),
            followup_modes=tuple(fm),
            timeout_per_theorem=int(block.get("timeout_per_theorem", 60) or 60),
            sequences=seqs,
        )

    def gate_allows(self, full_name: str) -> bool:
        """True if any configured namespace gate matches `full_name`.

        Empty gate list => no namespace allowed (sequence mode is strictly
        opt-in per namespace; it never fires globally).
        """
        if not self.namespace_gates:
            return False
        if not full_name:
            return False
        return any(full_name.startswith(g + ".") for g in self.namespace_gates)


@dataclass(frozen=True)
class SequencePlan:
    """A concrete depth-2 plan instantiated for one proof state."""
    sequence_id: str
    first_action_id: str
    first_tactic: str
    followup_mode: str          # "base_topk" | "fixed_battery" | "simp_all"
    followup_tactics: tuple[str, ...]
    family_source: str
    namespace: str
    full_name: str
    max_depth: int = 2

    def to_dict(self) -> dict:
        return {
            "sequence_id": self.sequence_id,
            "first_action_id": self.first_action_id,
            "first_tactic": self.first_tactic,
            "followup_mode": self.followup_mode,
            "followup_tactics": list(self.followup_tactics),
            "family_source": self.family_source,
            "namespace": self.namespace,
            "full_name": self.full_name,
            "max_depth": self.max_depth,
        }


def _namespace_of(full_name: str) -> str:
    return full_name.split(".", 1)[0] if full_name else ""


def _followups(seq: SymbolicActionSequence, cfg: SequenceSearchConfig,
               full_name: str, base_topk: list[str] | None) -> list[str]:
    """Build the ordered follow-up tactic list for one sequence on a state.

    `base_topk` are the base policy's suggested tactics from the *post-first-
    action* state, if the caller can supply them (offline this comes from the
    mined traces); when absent we degrade to the fixed battery so the plan is
    still well-defined. The result is deduped and capped.
    """
    cap = min(seq.max_followup_tactics, cfg.max_followup_tactics)
    out: list[str] = []
    seen: set[str] = set()

    def add(tacs):
        for t in tacs:
            t = (t or "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)

    mode = seq.followup_mode
    if mode == "simp_all":
        add(["simp_all"])
    elif mode == "base_topk":
        if base_topk:
            add(base_topk)
        else:  # degrade gracefully — never emit an empty follow-up set
            add(battery_for_namespace(full_name))
    else:  # "fixed_battery"
        if "base_topk" in cfg.followup_modes and base_topk:
            add(base_topk)
        add(battery_for_namespace(full_name))
    return out[:cap]


def plan_sequences(
    state_pp: str,
    full_name: str,
    cfg: SequenceSearchConfig,
    base_topk: list[str] | None = None,
) -> list[SequencePlan]:
    """Instantiate all gated depth-2 sequence plans for the current state.

    Returns `[]` when sequence mode is disabled or the theorem's namespace is
    not gated — i.e. the normal path is untouched. Otherwise each configured
    sequence whose first symbolic action instantiates on the state yields one
    plan (capped to `max_symbolic_first_actions` distinct first tactics).
    """
    if not cfg.enabled:
        return []
    if not cfg.gate_allows(full_name):
        return []

    ns = _namespace_of(full_name)
    plans: list[SequencePlan] = []
    first_tactics_emitted: set[str] = set()

    for seq in cfg.sequences:
        if cfg.max_depth != 2 or seq.max_depth != 2:
            continue  # SX1 is depth-2 only; skip anything else defensively
        if not seq.gate_allows(full_name):
            continue
        rendered = instantiate_symbolic_action(
            seq.first_action, state_pp, full_name)
        for first_tac, fam, action_id in rendered:
            if first_tac in first_tactics_emitted:
                continue
            if len(first_tactics_emitted) >= cfg.max_symbolic_first_actions:
                break
            first_tactics_emitted.add(first_tac)
            followups = _followups(seq, cfg, full_name, base_topk)
            if not followups:
                continue
            plans.append(SequencePlan(
                sequence_id=seq.sequence_id or seq.auto_sequence_id,
                first_action_id=action_id,
                first_tactic=first_tac,
                followup_mode=seq.followup_mode,
                followup_tactics=tuple(followups),
                family_source=seq.family_source or seq.default_family_source(),
                namespace=ns,
                full_name=full_name,
            ))
    return plans
