#!/usr/bin/env python3
"""RC4D composition gate (imported by the RC4D scripts).

RC4D = RC2 ⊕ RC4A ⊕ RC4B ⊕ RC4C_residue. The gate is the ORDERED UNION of the three
validated component gates:

  1. RC4A  — def_unfold_simp: emit `simp [<allowlisted defs in goal>]` (depth-1).
  2. RC4B  — disjoint_left bridge: for NS ∈ {Set,Multiset} disjoint goals emit
             `simp [<NS>.disjoint_left]` and `<;> aesop`.
  3. RC4C_residue — the NON-overlap d2 lemmas only (Multiset.disjoint_right,
             Set.subset_pair_iff_eq, List.forall_iff_forall_mem), each emitted RC4B-style
             as BOTH `simp [L]` and `simp [L] <;> aesop`.

`gate_fires` returns, per theorem, the ordered list of (component, action, tactic) the
candidate would try. ORDERING MATTERS: the additive evaluator credits the FIRST component
whose tactic closes the goal, so a Multiset-disjoint theorem solved by both RC4B
(`disjoint_left`) and RC4C_residue (`disjoint_right`) is credited to RC4B — this is the
composition de-duplication (`drop_rc4c_overlap_rc4b` at the theorem-coverage level).

The live probe is the same `run_tactics_live` used by RC4A/RC4B/RC4C (imported via
rc4b_gate). The component gates themselves are imported verbatim from rc4a_gate / rc4b_gate
so the composition reuses exactly the validated matching logic — no re-implementation.
"""
from __future__ import annotations

import json
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4a_gate as A  # noqa: E402  def-unfold gate
import rc4b_gate as B  # noqa: E402  disjoint_left bridge gate + run_tactics_live

# re-exported helpers (identical LeanDojo harness across RC4A/B/C/D)
run_tactics_live = B.run_tactics_live
namespace_of = B.namespace_of
statement_from_source = B.statement_from_source


def load_policy(path):
    return json.load(open(path if os.path.isabs(path) else os.path.join(_REPO, path)))


def _blob(name, goal_text):
    return ((name or "") + " " + (goal_text or "")).lower()


def _residue_action_fires(action, ns, blob):
    g = action["gate"]
    if g.get("requires_namespace") and g["requires_namespace"] != ns:
        return False
    toks = g.get("requires_name_or_goal_contains") or []
    if toks and not any(t.lower() in blob for t in toks):
        return False
    forb = g.get("forbids_name_or_goal_contains") or []
    if any(t.lower() in blob for t in forb):
        return False
    return True


def gate_fires(policy, namespace, goal_text, full_name):
    """Return (fires, emissions) where emissions is the ORDERED list of dicts:
        {component, action, tactic, lemma_or_defs}
    Ordering follows policy['ordering'] = [RC4A, RC4B, RC4C_residue]; within a component
    the depth-1 simp precedes the depth-2 `<;> aesop`. The additive evaluator credits the
    first emission that closes the goal, so component order == credit precedence.
    """
    ns = namespace_of(namespace, full_name)
    blob = _blob(full_name, goal_text)
    comp = policy["components"]
    emissions = []

    for component in policy["ordering"]:
        if component == "RC4A":
            cfg = comp["RC4A"]
            defs = A.matched_defs(cfg["allowlist"], goal_text, full_name)
            if defs:
                emissions.append({"component": "RC4A",
                                  "action": "def_unfold_simp_allowlist",
                                  "tactic": "simp [" + ", ".join(defs) + "]",
                                  "lemma_or_defs": defs})
        elif component == "RC4B":
            for a in comp["RC4B"]["actions"]:
                g = a["gate"]
                if g["requires_namespace"] != ns:
                    continue
                if not any(t.lower() in blob for t in g["requires_name_or_goal_contains"]):
                    continue
                emissions.append({"component": "RC4B", "action": a["name"],
                                  "tactic": a["tactic"], "lemma_or_defs": a["lemma"]})
        elif component == "RC4C_residue":
            for a in comp["RC4C_residue"]["actions"]:
                if _residue_action_fires(a, ns, blob):
                    for tac in a["tactics"]:
                        emissions.append({"component": "RC4C_residue", "action": a["name"],
                                          "tactic": tac, "lemma_or_defs": a["lemma"]})
    return (len(emissions) > 0), emissions


def components_firing(emissions):
    out = []
    for e in emissions:
        if e["component"] not in out:
            out.append(e["component"])
    return out


def tactics_of(emissions):
    seen, out = set(), []
    for e in emissions:
        if e["tactic"] not in seen:
            seen.add(e["tactic"])
            out.append(e["tactic"])
    return out


def component_of_tactic(emissions, tactic):
    """First component (in emission order) that emitted `tactic`."""
    for e in emissions:
        if e["tactic"] == tactic:
            return e["component"], e["action"], e["lemma_or_defs"]
    return None, None, None


# ----------------------- reuse map from component caches -----------------------
_REUSE_CACHES = [
    "project/evolve/experiments/rc4_candidates/disjoint_left_bridge/out/candidate_runs/probe_checkpoint.json",
    "project/evolve/experiments/rc4_candidates/d2_simp_aesop/out/candidate_runs/probe_checkpoint.json",
]
_REUSE_RESULTS = [
    # RC4A has no probe_checkpoint; its per-theorem candidate solve outcomes live here.
    "project/evolve/experiments/rc4_candidates/def_unfold_simp/out/candidate_results.json",
    "project/evolve/experiments/rc4_candidates/disjoint_left_bridge/out/candidate_results.json",
    "project/evolve/experiments/rc4_candidates/d2_simp_aesop/out/candidate_results.json",
]


def build_reuse_map():
    """{(full_name, tactic): solved_bool} harvested from the three components' live probe
    caches + candidate results. Lets RC4D reuse already-executed (theorem,tactic) outcomes
    instead of re-probing; the LeanDojo harness/timeouts were identical."""
    m = {}
    for rel in _REUSE_CACHES:
        p = os.path.join(_REPO, rel)
        if not os.path.exists(p):
            continue
        for fn, rec in json.load(open(p)).items():
            for ran in rec.get("ran", []):
                key = (fn, ran["tactic"])
                if ran.get("solved"):
                    m[key] = True
                else:
                    m.setdefault(key, False)
    for rel in _REUSE_RESULTS:
        p = os.path.join(_REPO, rel)
        if not os.path.exists(p):
            continue
        data = json.load(open(p))
        for r in data.get("results", []):
            fn = r["full_name"]
            outs = r.get("candidate_probe_outcomes") or {}
            for tac, outcome in outs.items():
                key = (fn, tac)
                if outcome == "success":
                    m[key] = True
                else:
                    m.setdefault(key, False)
            wt = r.get("winning_tactic")
            if wt:
                m[(fn, wt)] = True
    return m
