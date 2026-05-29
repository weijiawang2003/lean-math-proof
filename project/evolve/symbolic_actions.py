"""AX1 Stage 1+3 — symbolic action schema and instantiation.

Motivation (WX1/WX2): `cases <var> <;> simp_all` adds +29 wins beyond NS9
but is not SFT-ready as a raw tactic string because `<var>` is
state-dependent. A *symbolic* action — CASES_SIMP(var_type=List,
simp_mode=simp_all) — is a stable, learnable label; the wrapper
instantiates it from the live proof state.

This module defines the typed `SymbolicAction`, (de)serialization, a
stable id, validation, and `instantiate_symbolic_action`, which renders
an action into concrete Lean tactics using the variables found in the
state. No neural training here.
"""
from __future__ import annotations

from dataclasses import dataclass, field

try:
    from project.evolve.state_vars import vars_of_type
except ImportError:  # allow import when repo root isn't on sys.path yet
    import pathlib
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
    from project.evolve.state_vars import vars_of_type

ACTION_TYPES = ("CASES_SIMP", "INDUCTION_SIMP")
VAR_TYPES = ("Option", "List", "Bool")
SIMP_MODES = ("simp", "simp_all", "decide")

# action_type -> Lean head tactic that consumes a variable
_HEAD = {"CASES_SIMP": "cases", "INDUCTION_SIMP": "induction"}


@dataclass(frozen=True)
class SymbolicAction:
    action_type: str
    var_type: str | None = None
    simp_mode: str | None = None
    namespace_gate: str | None = None
    max_vars: int = 2
    priority: int = 50
    family_source: str = ""

    # ---- identity & (de)serialization --------------------------------
    @property
    def action_id(self) -> str:
        """Stable, human-readable id, e.g. `CASES_SIMP[List,simp_all]`."""
        return f"{self.action_type}[{self.var_type},{self.simp_mode}]"

    def default_family_source(self) -> str:
        vt = (self.var_type or "any").lower()
        head = "cases" if self.action_type == "CASES_SIMP" else "induction"
        return f"symbolic_{vt}_{head}_{self.simp_mode}"

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "var_type": self.var_type,
            "simp_mode": self.simp_mode,
            "namespace_gate": self.namespace_gate,
            "max_vars": self.max_vars,
            "priority": self.priority,
            "family_source": self.family_source or self.default_family_source(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolicAction":
        a = cls(
            action_type=str(d["action_type"]),
            var_type=d.get("var_type"),
            simp_mode=d.get("simp_mode"),
            namespace_gate=d.get("namespace_gate"),
            max_vars=int(d.get("max_vars", 2) or 2),
            priority=int(d.get("priority", 50) or 50),
            family_source=str(d.get("family_source", "") or ""),
        )
        return a

    # ---- validation --------------------------------------------------
    def validate(self) -> list[str]:
        """Return a list of problems (empty == valid)."""
        errs: list[str] = []
        if self.action_type not in ACTION_TYPES:
            errs.append(f"action_type {self.action_type!r} not in {ACTION_TYPES}")
        if self.var_type not in VAR_TYPES:
            errs.append(f"var_type {self.var_type!r} not in {VAR_TYPES}")
        if self.simp_mode not in SIMP_MODES:
            errs.append(f"simp_mode {self.simp_mode!r} not in {SIMP_MODES}")
        if self.action_type == "INDUCTION_SIMP" and self.var_type not in ("List",):
            errs.append("INDUCTION_SIMP is only meaningful for inductive "
                        "recursive types (List); got var_type="
                        f"{self.var_type!r}")
        if self.max_vars < 1:
            errs.append("max_vars must be >= 1")
        return errs

    def is_valid(self) -> bool:
        return not self.validate()

    # ---- namespace gating --------------------------------------------
    def gate_allows(self, full_name: str) -> bool:
        """True if this action may fire for `full_name`. A None gate is
        permissive; a gate `G` requires full_name to start with `G.`."""
        if not self.namespace_gate:
            return True
        if not full_name:
            return False
        return full_name.startswith(self.namespace_gate + ".")


def instantiate_symbolic_action(
    action: SymbolicAction, state_pp: str, full_name: str = "",
) -> list[tuple[str, str, str]]:
    """Render `action` into concrete tactics for the current state.

    Returns a list of `(tactic, family_source, action_id)`. Empty when the
    namespace gate blocks the action or no matching variable is present.
    Variables are read from the state (no hardcoded names), capped at
    `action.max_vars`, deduplicated, in goal-preference order.
    """
    if full_name and not action.gate_allows(full_name):
        return []
    if action.var_type is None or action.simp_mode is None:
        return []
    head = _HEAD.get(action.action_type)
    if head is None:
        return []
    names = vars_of_type(state_pp, action.var_type, max_vars=action.max_vars)
    fam = action.family_source or action.default_family_source()
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for v in names:
        tac = f"{head} {v} <;> {action.simp_mode}"
        if tac in seen:
            continue
        seen.add(tac)
        out.append((tac, fam, action.action_id))
    return out


def load_actions(specs: list[dict]) -> list[SymbolicAction]:
    """Build and validate a list of actions from dict specs (config)."""
    actions: list[SymbolicAction] = []
    for spec in specs or []:
        a = SymbolicAction.from_dict(spec)
        problems = a.validate()
        if problems:
            raise ValueError(f"invalid symbolic action {spec}: {problems}")
        actions.append(a)
    return actions
