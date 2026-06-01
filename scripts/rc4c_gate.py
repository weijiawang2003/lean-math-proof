#!/usr/bin/env python3
"""RC4C shared gate (imported by the RC4C scripts).

The gate is the narrow d2_simp_aesop allowlist. Each policy action carries its own gate:
a required namespace plus a set of name/goal tokens (any-match, case-insensitive) and an
optional forbid list. When an action's gate matches a theorem it emits the single depth-2
tactic `simp [L] <;> aesop`. Several actions may match one theorem (e.g. Multiset disjoint
goals match both Multiset.disjoint_left and Multiset.disjoint_right) — the candidate is
additive, so the goal counts as solved if ANY emitted tactic closes it.

`mode` selects the action subset:
  * "all"        -> every allowlisted action (overlap_policy.RC4C_all)
  * "nonoverlap" -> drop actions tagged overlap_family == "RC4B" (overlap_policy.RC4C_nonoverlap)

The live probe is reused verbatim from rc4b_gate (which itself reuses rc4a_gate) so the
LeanDojo harness is identical across RC4A / RC4B / RC4C.
"""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4b_gate as B  # noqa: E402  (reuse run_tactics_live / namespace_of / statement_from_source)

# re-exported helpers
run_tactics_live = B.run_tactics_live
namespace_of = B.namespace_of
statement_from_source = B.statement_from_source


def load_policy(path):
    return json.load(open(path if os.path.isabs(path) else os.path.join(_REPO, path)))


def actions_for_mode(policy, mode="all"):
    names = policy["overlap_policy"]["RC4C_nonoverlap" if mode == "nonoverlap" else "RC4C_all"]
    nameset = set(names)
    return [a for a in policy["actions"] if a["name"] in nameset]


def _blob(name, goal_text):
    return ((name or "") + " " + (goal_text or "")).lower()


def _gate_match(gate, blob, ns):
    if gate.get("requires_namespace") and gate["requires_namespace"] != ns:
        return False
    toks = gate.get("requires_name_or_goal_contains") or []
    if toks and not any(t.lower() in blob for t in toks):
        return False
    forb = gate.get("forbids_name_or_goal_contains") or []
    if any(t.lower() in blob for t in forb):
        return False
    return True


def gate_fires(policy, namespace, goal_text, full_name, mode="all"):
    """Return (fires, matched_actions, tactics, action_names, lemmas).

    matched_actions is a list of the full action dicts that fired (each gives name,
    tactic, lemma, overlap_family). tactics/action_names/lemmas are the parallel lists.
    """
    ns = namespace_of(namespace, full_name)
    blob = _blob(full_name, goal_text)
    matched = []
    for a in actions_for_mode(policy, mode):
        if _gate_match(a["gate"], blob, ns):
            matched.append(a)
    if not matched:
        return False, [], [], [], []
    tactics = [a["tactic"] for a in matched]
    names = [a["name"] for a in matched]
    lemmas = [a["lemma"] for a in matched]
    return True, matched, tactics, names, lemmas


def overlap_family_of(matched_actions):
    """RC4B if every matched action is overlap, none if no overlap, mixed otherwise."""
    fams = {a.get("overlap_family", "none") for a in matched_actions}
    if fams == {"RC4B"}:
        return "RC4B"
    if "RC4B" not in fams:
        return "none"
    return "mixed"
