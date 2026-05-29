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

# WX3 adds two Multiset-oriented action types:
#   MULTISET_INDUCTION_SIMP — `induction {var} using Multiset.induction_on
#                              <;> {simp_mode}` (quotient-aware induction).
#   EXT_SIMP                 — `ext x <;> {simp_mode}` (extensionality; the
#                              tactic is variable-independent, but emission is
#                              gated on a matching variable being present so
#                              it stays state-aware).
ACTION_TYPES = ("CASES_SIMP", "INDUCTION_SIMP",
                "MULTISET_INDUCTION_SIMP", "EXT_SIMP")
VAR_TYPES = ("Option", "List", "Bool", "Multiset")
SIMP_MODES = ("simp", "simp_all", "decide")

# action_type -> Lean head tactic that consumes a variable (var-consuming
# families only; MULTISET_INDUCTION_SIMP and EXT_SIMP render specially).
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
        if self.action_type == "EXT_SIMP":
            return f"symbolic_{vt}_ext_{self.simp_mode}"
        if self.action_type == "MULTISET_INDUCTION_SIMP":
            return f"symbolic_{vt}_induction_on_{self.simp_mode}"
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
        if self.action_type == "MULTISET_INDUCTION_SIMP" and self.var_type != "Multiset":
            errs.append("MULTISET_INDUCTION_SIMP requires var_type=Multiset; "
                        f"got {self.var_type!r}")
        if self.action_type == "EXT_SIMP" and self.var_type not in ("Multiset",):
            errs.append("EXT_SIMP is gated to var_type=Multiset for WX3; "
                        f"got {self.var_type!r}")
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
    fam = action.family_source or action.default_family_source()
    out: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    at = action.action_type

    # EXT_SIMP: variable-independent tactic, but gated on a matching variable
    # being present so it only fires on states that actually carry a value of
    # the target type. Emitted once.
    if at == "EXT_SIMP":
        names = vars_of_type(state_pp, action.var_type,
                             max_vars=action.max_vars)
        if not names:
            return []
        tac = f"ext x <;> {action.simp_mode}"
        return [(tac, fam, action.action_id)]

    # MULTISET_INDUCTION_SIMP: quotient-aware induction principle.
    if at == "MULTISET_INDUCTION_SIMP":
        names = vars_of_type(state_pp, action.var_type,
                             max_vars=action.max_vars)
        for v in names:
            tac = (f"induction {v} using Multiset.induction_on "
                   f"<;> {action.simp_mode}")
            if tac in seen:
                continue
            seen.add(tac)
            out.append((tac, fam, action.action_id))
        return out

    # CASES_SIMP / INDUCTION_SIMP (AX1 behavior — unchanged).
    head = _HEAD.get(at)
    if head is None:
        return []
    names = vars_of_type(state_pp, action.var_type, max_vars=action.max_vars)
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


# =====================================================================
# SX1 Stage 2 — symbolic action *sequences* (depth-2 only).
#
# Motivation (SX1): AX1–AX4 emit a single symbolic action. A symbolic
# action often *advances* the state (`TacticState`) without closing it —
# e.g. `induction s using Multiset.induction_on <;> simp_all` leaves an
# inductive-step goal that a follow-up (`aesop`, `simp_all`, base-model
# top-k) closes. A `SymbolicActionSequence` makes that two-step shape a
# first-class, namespace-gated, depth-bounded object so the wrapper can
# emit it deliberately instead of relying on the open best-first search to
# stumble onto the follow-up.
#
# Design: the FIRST step is always a typed `SymbolicAction` (the stable,
# learnable label). The SECOND step is a *follow-up mode*, not another
# symbolic action, because the closers we observe are plain tactics
# (`SIMP_ALL`) or the base policy's top-k. `actions` therefore returns the
# leading symbolic action(s); the follow-up is described separately. This
# keeps the schema additive and faithful to the spec wording
# ("actions: list[SymbolicAction]") while modelling the real two-step shape.
# SX1 supports max_depth == 2 ONLY.
# =====================================================================

# Follow-up modes for the second step of a depth-2 sequence.
FOLLOWUP_MODES = ("base_topk", "fixed_battery", "simp_all")
SEQUENCE_STOP_CONDITIONS = ("proof_finished", "max_depth", "no_progress")

# Fixed small battery, applied in order. `omega`/`decide` are only added for
# arithmetic-flavoured namespaces by `battery_for_namespace` below.
FIXED_BATTERY = ("simp", "simp_all", "aesop", "rfl")
FIXED_BATTERY_ARITH = ("omega", "decide")
_ARITH_NAMESPACES = ("Nat", "Int")


def battery_for_namespace(full_name: str,
                          extra_arith: bool = True) -> list[str]:
    """The fixed follow-up battery for a theorem, in attempt order.

    `omega`/`decide` are appended only for arithmetic namespaces so we never
    emit `omega` on a Multiset/Option/List goal where it cannot apply.
    """
    bat = list(FIXED_BATTERY)
    if extra_arith and full_name:
        head = full_name.split(".", 1)[0]
        if head in _ARITH_NAMESPACES:
            bat += list(FIXED_BATTERY_ARITH)
    return bat


@dataclass(frozen=True)
class SymbolicActionSequence:
    """A depth-bounded symbolic-action sequence (SX1; depth 2 only).

    `first_action` is the symbolic step; `followup_mode` selects how the
    second step is produced (base-model top-k, a fixed small battery, or a
    single `simp_all`). `namespace_gate` (falling back to the first action's
    gate) restricts where the whole sequence may fire.
    """
    first_action: SymbolicAction
    followup_mode: str = "fixed_battery"
    max_depth: int = 2
    namespace_gate: str | None = None
    max_followup_tactics: int = 6
    priority: int = 50
    family_source: str = ""
    sequence_id: str = ""
    stop_condition: str = "proof_finished"

    # ---- identity & (de)serialization --------------------------------
    @property
    def actions(self) -> list[SymbolicAction]:
        """The leading symbolic action(s). SX1 sequences carry exactly one."""
        return [self.first_action]

    @property
    def gate(self) -> str | None:
        return self.namespace_gate or self.first_action.namespace_gate

    @property
    def auto_sequence_id(self) -> str:
        return f"SEQ[{self.first_action.action_id}=>{self.followup_mode}]"

    def default_family_source(self) -> str:
        base = self.first_action.family_source or \
            self.first_action.default_family_source()
        return f"seq_{base}__then__{self.followup_mode}"

    def to_dict(self) -> dict:
        return {
            "first_action": self.first_action.to_dict(),
            "followup_mode": self.followup_mode,
            "max_depth": self.max_depth,
            "namespace_gate": self.gate,
            "max_followup_tactics": self.max_followup_tactics,
            "priority": self.priority,
            "family_source": self.family_source or self.default_family_source(),
            "sequence_id": self.sequence_id or self.auto_sequence_id,
            "stop_condition": self.stop_condition,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SymbolicActionSequence":
        fa = d.get("first_action")
        first = SymbolicAction.from_dict(fa) if isinstance(fa, dict) \
            else fa
        return cls(
            first_action=first,
            followup_mode=str(d.get("followup_mode", "fixed_battery")),
            max_depth=int(d.get("max_depth", 2) or 2),
            namespace_gate=d.get("namespace_gate"),
            max_followup_tactics=int(d.get("max_followup_tactics", 6) or 6),
            priority=int(d.get("priority", 50) or 50),
            family_source=str(d.get("family_source", "") or ""),
            sequence_id=str(d.get("sequence_id", "") or ""),
            stop_condition=str(d.get("stop_condition", "proof_finished")),
        )

    # ---- validation --------------------------------------------------
    def validate(self) -> list[str]:
        errs = list(self.first_action.validate())
        if self.followup_mode not in FOLLOWUP_MODES:
            errs.append(f"followup_mode {self.followup_mode!r} not in "
                        f"{FOLLOWUP_MODES}")
        if self.max_depth != 2:
            errs.append(f"SX1 supports max_depth == 2 only; got {self.max_depth}")
        if self.stop_condition not in SEQUENCE_STOP_CONDITIONS:
            errs.append(f"stop_condition {self.stop_condition!r} not in "
                        f"{SEQUENCE_STOP_CONDITIONS}")
        if self.max_followup_tactics < 1:
            errs.append("max_followup_tactics must be >= 1")
        return errs

    def is_valid(self) -> bool:
        return not self.validate()

    def gate_allows(self, full_name: str) -> bool:
        g = self.gate
        if not g:
            return True
        if not full_name:
            return False
        return full_name.startswith(g + ".")


def load_sequences(specs: list[dict]) -> list[SymbolicActionSequence]:
    """Build and validate a list of sequences from dict specs (config)."""
    seqs: list[SymbolicActionSequence] = []
    for spec in specs or []:
        s = SymbolicActionSequence.from_dict(spec)
        problems = s.validate()
        if problems:
            raise ValueError(f"invalid symbolic sequence {spec}: {problems}")
        seqs.append(s)
    return seqs
