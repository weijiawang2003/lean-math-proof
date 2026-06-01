#!/usr/bin/env python3
"""SX2 Part 3 — off-by-default SET2 experimental wrapper (candidate policy).

This is an EXTERNAL candidate policy, NOT a router integration. It evaluates the
SET2 gate policy against a (theorem_name, goal_pp) pair and returns the ordered
list of gated tactic emissions. It NEVER touches RC1 / NS24 / NS9.

Design contract:
  * Off-by-default: `global_enabled` in the policy is false. eval_gates() returns
    NO emissions unless `force_enable=True` is passed (used only by the SX2
    experiment runners, which record `production_default_emits: false`).
  * Narrow gates: each gate ANDs its conditions over name + goal pretty-print.
  * Emission logging: every evaluation appends a structured record (gate, tactic,
    conditions met, off_gate flag, theorem, goal shape) to an in-memory log that
    callers persist.
  * SET_EXT_BYCASES is hard-disabled (`emit: false` / needs_binder_inference): it
    is never emitted because its proposition / local-hypothesis cannot be inferred.

Used as a library by sx2_run_set2_eval.py and sx2_gate_sanity_check.py.
"""
from __future__ import annotations

import json

# Set-relation tokens used by the goal_has_set_relation predicate.
_SET_REL_TOKENS = ["=", "∈", "⊆", "⊂", "∩", "∪", "\\", "∅"]
_SETOP_TOKENS = ["⊆", "\\", "∪", "∩"]
_ITE_TOKENS = [".ite", " ite ", "ite ", "if "]


def load_policy(path):
    return json.load(open(path))


# ----------------------------- predicates ---------------------------------
def _name_or_ns_has_set(name, goal):
    return "Set" in (name or "")


def _goal_has_ite(name, goal):
    g = goal or ""
    if ".ite" in g:
        return True
    # bare `ite`/`if` token (avoid matching substrings like 'finite')
    import re
    return bool(re.search(r"(?<![A-Za-z])ite(?![A-Za-z])", g) or
                re.search(r"(?<![A-Za-z])if(?![A-Za-z])", g))


def _goal_has_set_relation(name, goal):
    g = goal or ""
    return any(tok in g for tok in _SET_REL_TOKENS)


def _no_multiset_target(name, goal):
    return "Multiset" not in (name or "") and "Multiset" not in (goal or "")


def _goal_is_eq(name, goal):
    g = goal or ""
    # top-level Set equality, not an iff
    return "=" in g and "↔" not in g.split("⊢")[-1]


def _goal_is_iff(name, goal):
    return "↔" in (goal or "")


def _goal_has_subset_or_setop(name, goal):
    g = goal or ""
    return any(tok in g for tok in _SETOP_TOKENS)


def _binder_proposition_inferable(name, goal):
    # We cannot synthesize the by_cases proposition / local hypothesis generically.
    return False


_PREDS = {
    "name_or_ns_has_set": _name_or_ns_has_set,
    "goal_has_ite": _goal_has_ite,
    "goal_has_set_relation": _goal_has_set_relation,
    "no_multiset_target": _no_multiset_target,
    "goal_is_eq": _goal_is_eq,
    "goal_is_iff": _goal_is_iff,
    "goal_has_subset_or_setop": _goal_has_subset_or_setop,
    "binder_proposition_inferable": _binder_proposition_inferable,
    # not_arithmetic_only is subsumed by name_or_ns_has_set; treat as always-true
    "not_arithmetic_only": lambda n, g: True,
}


def _gate_emits(gate):
    """Whether a gate is permitted to emit at all (hard disable check)."""
    if gate.get("emit") is False:
        return False
    if gate.get("needs_binder_inference"):
        return False
    if gate.get("status", "").endswith("disabled"):
        return False
    return True


def eval_gates(policy, full_name, goal_pp, force_enable=False, log=None):
    """Return ordered list of emissions for one (theorem, goal).

    Each emission: {gate_id, tactic, conditions_met, off_gate, theorem, ...}.
    Off-by-default: returns [] unless policy.global_enabled or force_enable.
    """
    enabled = bool(policy.get("global_enabled")) or bool(force_enable)
    emissions = []
    name = full_name or ""
    goal = goal_pp or ""
    is_set_surface = ("Set" in name) and ("Multiset" not in name)
    for gate in policy.get("gates", []):
        cond_results = {c: bool(_PREDS.get(c, lambda n, g: False)(name, goal))
                        for c in gate.get("conditions", [])}
        all_met = all(cond_results.values()) and len(cond_results) > 0
        permitted = _gate_emits(gate)
        will_emit = enabled and permitted and all_met
        if not (all_met and permitted):
            # record near-misses only when conditions matched but gate disabled
            if all_met and not permitted and log is not None:
                log.append({"theorem": name, "gate_id": gate["gate_id"],
                            "emitted": False, "reason": "gate_hard_disabled",
                            "conditions_met": cond_results})
            continue
        # off_gate := fired on a non-Set / Multiset surface (should never happen)
        off_gate = not is_set_surface
        rec = {"theorem": name, "gate_id": gate["gate_id"],
               "tactic": gate["tactic"], "template_family": gate["template_family"],
               "conditions_met": cond_results, "emitted": bool(will_emit),
               "off_gate": bool(off_gate),
               "gate_strength": gate.get("gate_strength"),
               "production_default_emits": bool(policy.get("global_enabled"))}
        if will_emit:
            emissions.append(rec)
        if log is not None:
            log.append(rec)
    return emissions


def describe(policy):
    """Short human description of which gates can emit."""
    out = []
    for g in policy.get("gates", []):
        out.append({"gate_id": g["gate_id"], "tactic": g["tactic"],
                    "can_emit": _gate_emits(g),
                    "gate_strength": g.get("gate_strength"),
                    "mined_theorems": g.get("mined_support", {}).get("num_theorems", 0)})
    return out


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Describe SET2 gate policy (no Lean).")
    p.add_argument("--gate-policy",
                   default="project/evolve/experiments/sx2/set2_gate_policy.json")
    a = p.parse_args()
    pol = load_policy(a.gate_policy)
    print(f"policy={pol['policy_id']} global_enabled={pol['global_enabled']} "
          f"promotion_allowed={pol['promotion_allowed']}")
    for d in describe(pol):
        print(f"  {d['gate_id']:20s} can_emit={d['can_emit']} "
              f"strength={d['gate_strength']} mined={d['mined_theorems']} "
              f":: {d['tactic']}")
