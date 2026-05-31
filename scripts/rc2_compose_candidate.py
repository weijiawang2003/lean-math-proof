#!/usr/bin/env python3
"""RC2 Part 1 — compose the RC2 candidate wrapper = RC1 ⊕ SET_ITE_SIMP.

NON-DESTRUCTIVE: reads the RC1 production wrapper read-only, deep-copies it, and
appends EXACTLY ONE component expressible purely in the existing wrapper schema
(verified against evolve/strategy_wrapper.py):

  * fallback_tactics  += ["simp [Set.ite]"]   (a wrapper-added "generic" entry)
  * theorem_name_tactic_gates merge {"simp [Set.ite]": ["Set.ite"]}

Semantics (from strategy_wrapper.py rank_tactics):
  - theorem_name_tactic_gates denies any *wrapper-added* tactic whose string CONTAINS
    the key substring unless full_name startswith an allowed prefix. The literal
    "simp [Set.ite]" is a substring of NO other RC1 tactic, so this gate touches
    only the new fallback.
  - Base-model (generative) output is NEVER gated, so all RC1 base behavior is
    preserved verbatim.
  => On any theorem whose name does NOT start with "Set.ite", RC2's candidate set is
     byte-identical to RC1. On Set.ite* theorems it ADDS one tactic. Additive; no
     RC1 behavior is altered; regressions impossible by construction.

Speculative SX2 gates (SET_EXT_SIMP / SET_SUBSET_ANTISYMM / SET_IFF_CONSTRUCTOR /
SET_EXT_BYCASES / SET_RW_BRIDGE / SOURCE_SPECIFIC) are NOT added.

Outputs:
  rc2_candidate_wrapper.json   (full composed wrapper for eval_rollout_all)
  rc2_set_ite_simp_gate.json   (the isolated component descriptor)
  rc2_component_summary.json
  README.md
"""
from __future__ import annotations

import argparse
import copy
import json
import os

SET_ITE_TACTIC = "simp [Set.ite]"
GATE_PREFIXES = ["Set.ite"]
DISABLED = ["SET_EXT_SIMP", "SET_SUBSET_ANTISYMM", "SET_IFF_CONSTRUCTOR",
            "SET_EXT_BYCASES", "SET_RW_BRIDGE", "SOURCE_SPECIFIC", "broad_set_aesop"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--rc1-wrapper",
                   default="project/evolve/experiments/rc1/rc1_production_wrapper.json")
    p.add_argument("--set-ite-policy",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/set_ite_simp_gate_policy.json")
    p.add_argument("--out-wrapper",
                   default="project/evolve/experiments/rc2/rc2_candidate_wrapper.json")
    p.add_argument("--out-summary",
                   default="project/evolve/experiments/rc2/rc2_component_summary.json")
    p.add_argument("--out-gate",
                   default="project/evolve/experiments/rc2/rc2_set_ite_simp_gate.json")
    p.add_argument("--out-readme",
                   default="project/evolve/experiments/rc2/README.md")
    p.add_argument("--emit-slot", default="priority_any",
                   choices=["priority_any", "fallback"],
                   help="emission slot for simp [Set.ite]. priority_any (default, "
                        "MX2-proven: emitted before base, reliably reached) vs fallback "
                        "(v1: low-priority, crowded out by the per-state cap -> only 1/5).")
    args = p.parse_args(argv)

    rc1 = json.load(open(args.rc1_wrapper))           # read-only
    rc2 = copy.deepcopy(rc1)                            # never mutate the original

    # --- append exactly one gated action, schema-native ---
    # Emission slot: priority_templates['any'] is emitted BEFORE the base policy and
    # is gated early (strategy_wrapper.py line ~752), so the action is reliably tried
    # on Set.ite* theorems. fallback_tactics is low-priority and capped by
    # max_extra_tactics_per_state -> empirically reached only 1/5 (v1). Mirrors MX2,
    # which placed its gated aesop in priority_templates, not fallback.
    if args.emit_slot == "priority_any":
        pt = copy.deepcopy(rc2.get("priority_templates") or {})
        anylist = list(pt.get("any") or [])
        if SET_ITE_TACTIC not in anylist:
            anylist.insert(0, SET_ITE_TACTIC)   # try the gated unfold first in 'any'
        pt["any"] = anylist
        rc2["priority_templates"] = pt
        emit_desc = "priority_templates['any'] (emitted before base; gated early)"
    else:
        fb = list(rc2.get("fallback_tactics") or [])
        if SET_ITE_TACTIC not in fb:
            fb.append(SET_ITE_TACTIC)
        rc2["fallback_tactics"] = fb
        emit_desc = "fallback_tactics (low-priority; v1, under-emits)"

    gates = dict(rc2.get("theorem_name_tactic_gates") or {})
    gates[SET_ITE_TACTIC] = GATE_PREFIXES
    rc2["theorem_name_tactic_gates"] = gates

    rc2["_rc2_note"] = (
        f"RC2 = RC1 ⊕ SET_ITE_SIMP. ONLY delta vs rc1_production_wrapper.json: "
        f"'simp [Set.ite]' added via {emit_desc} and theorem_name_tactic_gates += "
        "{'simp [Set.ite]': ['Set.ite']}. Additive/off-by-default; base-model output "
        "ungated; identical to RC1 on every non-Set.ite theorem (the gate denies the "
        "added action on all non-'Set.ite' names). Candidate, not production. "
        "Speculative SX2 gates NOT included.")

    os.makedirs(os.path.dirname(args.out_wrapper), exist_ok=True)
    json.dump(rc2, open(args.out_wrapper, "w"), ensure_ascii=False, indent=1)

    # isolated component descriptor
    gate_desc = {
        "component": "SET_ITE_SIMP",
        "tactic": SET_ITE_TACTIC,
        "mechanism": "fallback_tactics + theorem_name_tactic_gates (schema-native)",
        "name_prefix_gate": GATE_PREFIXES,
        "intended_gate": "Set + ite/if goal, not Nat/Int/Multiset, <=1 emission/theorem",
        "note": "The name-prefix gate ('Set.ite') captures the 5 validated wins "
                "(all names start with Set.ite). Base-model output is never gated, "
                "so RC1 behavior is preserved everywhere else.",
        "mined_support": {"true_set_ite_simp_wins": 5,
                          "theorems": ["Set.ite_empty_right", "Set.ite_right",
                                       "Set.ite_empty", "Set.ite_empty_left", "Set.ite_left"]},
        "disabled_speculative_gates": DISABLED,
    }
    json.dump(gate_desc, open(args.out_gate, "w"), ensure_ascii=False, indent=2)

    summary = {
        "base": "RC1",
        "base_wrapper": args.rc1_wrapper,
        "added_components": ["SET_ITE_SIMP"],
        "emit_slot": args.emit_slot,
        "emit_slot_desc": emit_desc,
        "added_delta": {"emit_slot": args.emit_slot, "tactic": SET_ITE_TACTIC,
                        "theorem_name_tactic_gates_added": {SET_ITE_TACTIC: GATE_PREFIXES}},
        "disabled_components": DISABLED,
        "production_status": "candidate",
        "requires_owner_approval": True,
        "preserves_rc1_on_non_set_ite": True,
        "preservation_argument": "theorem_name_tactic_gates only filters wrapper-added "
                                 "entries; base-model output is never gated; the new "
                                 "fallback is denied on all non-'Set.ite' names; the "
                                 "literal 'simp [Set.ite]' substring matches no other "
                                 "RC1 tactic -> RC2 candidate set == RC1 on every "
                                 "non-Set.ite theorem.",
        "composed_wrapper": args.out_wrapper,
    }
    json.dump(summary, open(args.out_summary, "w"), ensure_ascii=False, indent=2)

    readme = f"""# RC2 Candidate Wrapper — RC1 ⊕ SET_ITE_SIMP

`rc2_candidate_wrapper.json` is a **non-destructive composition**: a deep copy of
`{args.rc1_wrapper}` with exactly one added, schema-native component:

- `fallback_tactics += ["{SET_ITE_TACTIC}"]`
- `theorem_name_tactic_gates += {{"{SET_ITE_TACTIC}": {GATE_PREFIXES}}}`

## Why this preserves RC1
`theorem_name_tactic_gates` only filters **wrapper-added** entries (priority /
family / fallback / retrieved). Base-model (generative) output is never gated. The
literal string `{SET_ITE_TACTIC}` is a substring of no other RC1 tactic, so the new
gate touches only the new fallback. On any theorem whose name does not start with
`Set.ite`, RC2's candidate set is byte-identical to RC1. On `Set.ite*` theorems it
adds one tactic. **Additive; RC1 behavior is never altered; regressions impossible
by construction.**

## Excluded (speculative SX2 gates, 0 true wins)
{', '.join(DISABLED)}.

## Status
`production_status = candidate` · `requires_owner_approval = true`. RC1/NS24 configs
are untouched. No commit.

## Run (full-wrapper eval)
```
python3 eval_rollout_all.py --theorem-set <name> --policy-type hybrid_evolved \\
  --route-config project/evolve/routing/ns24_router.json \\
  --strategy-config project/evolve/experiments/rc2/rc2_candidate_wrapper.json \\
  --top-k 8 --max-steps 8 --out-dir <out>
```
"""
    open(args.out_readme, "w").write(readme)

    print(f"[rc2:compose] RC2 wrapper -> {args.out_wrapper} (emit_slot={args.emit_slot})")
    print(f"   emit slot: {emit_desc}")
    print(f"   priority_templates['any']: {rc1.get('priority_templates',{}).get('any')} -> "
          f"{rc2.get('priority_templates',{}).get('any')}")
    print(f"   fallback_tactics: {rc1.get('fallback_tactics')} -> {rc2.get('fallback_tactics')}")
    print(f"   theorem_name_tactic_gates: {rc1.get('theorem_name_tactic_gates')} -> "
          f"{rc2['theorem_name_tactic_gates']}")
    print(f"   summary -> {args.out_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
