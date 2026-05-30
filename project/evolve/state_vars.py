"""AX1 Stage 2 — robust proof-state variable extraction.

Parses a Lean pretty-printed proof state into a list of `StateVar`,
classifying each local-context binder by a coarse type. Used by the
symbolic action layer (project/evolve/symbolic_actions.py) to instantiate
parameterized actions like CASES_SIMP(var_type=List) from the live state.

Conservative by design: it only reads the local context (lines before the
`⊢` goal), skips anything it cannot parse, and returns an empty list when
uncertain rather than guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A local-context binder line: "names... : type". Names may carry the
# inaccessible dagger (✝); the caller decides whether to keep them.
_BINDER_LINE = re.compile(
    r"^\s*([A-Za-z_][\w'✝]*(?:\s+[A-Za-z_][\w'✝]*)*)\s*:\s*(.+?)\s*$"
)

# Coarse type vocabulary (AX1 spec; WX3 adds Multiset; MX1 adds Finset/Set).
COARSE_TYPES = ("Option", "List", "Bool", "Nat", "Int", "Multiset",
                "Finset", "Set", "unknown")

# Symbols that mark a binder type as a proposition (hypothesis).
_PROP_MARKERS = ("=", "≤", "<", "≥", ">", "∈", "∉", "∧", "∨", "↔", "¬",
                 "→", "≠", "∣", "∀", "∃")


@dataclass(frozen=True)
class StateVar:
    name: str
    type_pp: str
    coarse_type: str          # one of COARSE_TYPES
    is_hypothesis: bool       # best-effort: Prop-typed single binder

    def to_dict(self) -> dict:
        return {
            "name": self.name, "type_pp": self.type_pp,
            "coarse_type": self.coarse_type,
            "is_hypothesis": self.is_hypothesis,
        }


def _coarse_type(type_pp: str) -> str:
    """Classify a binder type by its head, conservatively.

    A *value* of an inductive type is matched by the type starting with the
    type name (so `g : α → Option β` is NOT an Option value — it's a
    function). Nat/Int accept both ASCII and unicode notation.
    """
    t = type_pp.strip()
    if t.startswith("Option"):
        return "Option"
    if t.startswith("Multiset"):
        return "Multiset"
    if t.startswith("Finset"):
        return "Finset"
    if t.startswith("Set"):
        return "Set"
    if t.startswith("List"):
        return "List"
    if t == "Bool" or t.startswith("Bool"):
        return "Bool"
    if t in ("ℕ", "Nat") or t.startswith("ℕ") or t.startswith("Nat"):
        return "Nat"
    if t in ("ℤ", "Int") or t.startswith("ℤ") or t.startswith("Int"):
        return "Int"
    return "unknown"


def _looks_like_prop(type_pp: str) -> bool:
    return any(m in type_pp for m in _PROP_MARKERS)


def extract_state_variables(state_pp: str) -> list[StateVar]:
    """Return the local-context variables of a Lean proof state.

    Only context lines before the `⊢` goal are parsed. Multi-name binders
    (`a b c : ℕ`) expand to one StateVar per name. Returns [] on empty /
    unparseable input.
    """
    if not state_pp:
        return []
    out: list[StateVar] = []
    seen: set[str] = set()
    for line in state_pp.splitlines():
        s = line.lstrip()
        if s.startswith("⊢"):
            break  # goal begins; stop reading context
        m = _BINDER_LINE.match(line)
        if not m:
            continue
        names_str, type_pp = m.group(1), m.group(2).strip()
        names = names_str.split()
        coarse = _coarse_type(type_pp)
        # A single-name Prop binder is a hypothesis; multi-name binders
        # (`a b : ℕ`) are data, never hypotheses.
        is_hyp = len(names) == 1 and _looks_like_prop(type_pp)
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            out.append(StateVar(
                name=name, type_pp=type_pp,
                coarse_type=coarse, is_hypothesis=is_hyp,
            ))
    return out


def vars_of_type(
    state_pp: str, coarse_type: str, max_vars: int = 0,
    exclude_inaccessible: bool = True, prefer_goal: bool = True,
) -> list[str]:
    """Names of accessible, non-hypothesis variables of a coarse type,
    optionally preferring those that occur in the goal. `max_vars=0` means
    no cap. Inaccessible (daggered) names are dropped by default since
    `cases`/`induction` cannot reference them."""
    allv = extract_state_variables(state_pp)
    cands = [v for v in allv
             if v.coarse_type == coarse_type and not v.is_hypothesis]
    names = [v.name for v in cands
             if not (exclude_inaccessible and "✝" in v.name)]

    if prefer_goal:
        goal = ""
        grab = False
        for line in state_pp.splitlines():
            s = line.lstrip()
            if s.startswith("⊢"):
                grab = True
            if grab:
                goal += "\n" + (s[1:] if s.startswith("⊢") else line)
        if goal and names:
            used = [n for n in names
                    if re.search(rf"(?<![\w']){re.escape(n)}(?![\w'])", goal)]
            if used:
                names = used + [n for n in names if n not in used]

    return names[:max_vars] if max_vars else names
