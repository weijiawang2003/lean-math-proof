#!/usr/bin/env python3
"""RC5S Part 2 — build the strict dynamic policy.

Derives a strict, timeout-safe policy from the RC5H policy + safety audit + program plan:
removes the stall-prone tactic families (simp_all combos, depth-3 try chains, bare tactics),
keeps the low-risk grammar (8 patterns, aesop-tail gated to historically-safe namespaces),
enforces strict grammar matching, sets B5-default/B10-safe-only/no-B20 budgets, namespace gates,
and a process-kill timeout policy. Emits the policy + an explicit diff vs RC5H with rationale.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc5s_grammar as G

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC5H_POLICY = "project/evolve/experiments/rc5_hybrid/rc5h_policy.json"
RC5H_AUDIT = "project/evolve/experiments/rc5_hybrid/out/rc5h_dynamic_safety_audit.json"
RC5H_PLAN = "project/evolve/experiments/rc5_hybrid/out/rc5h_dynamic_program_plan.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-policy", required=True)
    ap.add_argument("--out-diff-json", required=True)
    ap.add_argument("--out-diff-md", required=True)
    ap.add_argument("--per-theorem-cap", type=int, default=60)
    ap.add_argument("--per-tactic", type=int, default=8)
    args = ap.parse_args()

    rc5h = json.load(open(_p(RC5H_POLICY)))
    audit = json.load(open(_p(RC5H_AUDIT))) if os.path.exists(_p(RC5H_AUDIT)) else {}

    # classify the RC5H plan's emitted tactics to quantify what the strict policy removes
    removed_hist = Counter()
    if os.path.exists(_p(RC5H_PLAN)):
        for t in json.load(open(_p(RC5H_PLAN)))["theorems"]:
            for pgm in t.get("programs_ranked", []):
                k, _ = G.classify_program(pgm.get("tactic"), t.get("namespace"))
                removed_hist[k] += 1

    policy = {
        "family": "rc5s_strict_timeout_safe",
        "base": "RC5H dynamic stage (hardened)",
        "status": "experimental_hardening_prototype",
        "promotion_allowed": False,
        "allowed_grammar": [
            {"pattern_id": pid, "regex": rx.pattern,
             "example": ex} for (pid, rx), ex in zip(G.ALLOWED_PATTERNS, [
                "exact L", "simpa using L", "simpa [L]", "simp [L]", "rw [L]",
                "simp [L] <;> aesop", "rw [L] <;> aesop", "ext x <;> simp [L]",
                "constructor <;> intro h <;> aesop"])],
        "removed_by_default": {
            "simp_all": "any tactic containing simp_all (simp [L] <;> simp_all, rw [L] <;> simp_all, "
                        "bare simp_all) — top Dojo-stall cause in RC5H (230+ programs, ignores SIGALRM)",
            "depth3_try_chains": "simp [L] <;> try aesop <;> try simp_all and similar — 112 programs, "
                                 "depth-3 try chains stall and are off-grammar",
            "depth3_other": "any program with >=2 `<;>` except the single `constructor <;> intro h "
                            "<;> aesop` pattern (e.g. ext x <;> simp [L] <;> aesop)",
            "bare_tactics": "aesop / omega / nlinarith / tauto / decide with no lemma — off-policy "
                            "TR6 grammar leakage (74 off-policy programs in RC5H)",
        },
        "quarantine_only": {
            "note": "simp_all and depth-3 chains may be re-enabled ONLY in an explicit offline "
                    "quarantine mode (never B5/B10), out of scope for RC5S.",
        },
        "aesop_namespaces": list(G.DEFAULT_AESOP_NAMESPACES),
        "aesop_namespace_rationale": "`<;> aesop` (and constructor<;>aesop) only on Set/Finset/List/"
                                     "Multiset where RC5H/TR6 had aesop wins without stalls; NOT Nat.",
        "allowed_namespaces": list(G.DEFAULT_ALLOWED_NAMESPACES),
        "disabled_namespaces": ["Order", "root/Other (unless explicitly whitelisted)"],
        "budgets": {
            "default": "B5",
            "B5": {"max_programs": 5, "controls": True, "all_families": True},
            "B10": {"max_programs": 10, "controls": False,
                    "safe_families_only": ["exact_L", "simpa_using_L", "simpa_L", "simp_L", "rw_L"],
                    "note": "B10 only continues with NON-aesop low-risk families to avoid residual stalls"},
            "B20": "DISABLED in safe mode",
        },
        "timeout_policy": {
            "per_theorem_wall_cap_seconds": args.per_theorem_cap,
            "per_tactic_seconds": args.per_tactic,
            "process_group_kill_fallback": True,
            "kill_method": "run_with_timeout.py: SIGTERM process group, then SIGKILL after 5s; "
                           "exit 124 on timeout (works even when LeanDojo/aesop ignore SIGALRM)",
            "checkpoint_each_theorem": True,
            "deterministic_resume": True,
            "rationale": "the outer wall cap is the hard guarantee; per-tactic SIGALRM is best-effort.",
        },
        "strict_grammar_enforcement": {
            "reject_off_policy_before_scoring": True,
            "every_program_matches_allowed_grammar": True,
            "target_off_policy_in_final_plan": 0,
        },
        "rc5h_winners_preserved": [
            "Finset.biUnion_subset_iff_forall_subset (simp [Finset.biUnion_subset] <;> aesop)",
            "Multiset.add_bind (simp [Multiset.bind])",
            "Finset.image_subset_iff (simp [Finset.subset_iff])",
        ],
    }
    os.makedirs(os.path.dirname(_p(args.out_policy)), exist_ok=True)
    json.dump(policy, open(_p(args.out_policy), "w"), ensure_ascii=False, indent=2)

    rc5h_grammar = rc5h.get("dynamic_stage", {}).get("program_grammar", [])
    strict_grammar = [g["example"] for g in policy["allowed_grammar"]]
    removed_from_grammar = [g for g in rc5h_grammar if "simp_all" in g]
    diff = {
        "generated_by": "scripts/rc5s_build_strict_policy.py",
        "rc5h_grammar": rc5h_grammar,
        "strict_grammar": strict_grammar,
        "removed_from_rc5h_grammar": removed_from_grammar,
        "removed_tactic_families": list(policy["removed_by_default"].keys()),
        "rc5h_plan_classification": dict(removed_hist),
        "would_remove_from_rc5h_plan": {k: v for k, v in removed_hist.items() if k != "POLICY_ALLOWED"},
        "budgets": {"rc5h": rc5h.get("dynamic_stage", {}).get("max_programs_per_theorem"),
                    "rc5s": {"default": "B5", "B10": "safe-families-only", "B20": "DISABLED"}},
        "timeout": {"rc5h": rc5h.get("dynamic_stage", {}).get("gates", {}),
                    "rc5s": policy["timeout_policy"]},
        "aesop_namespace_gate_added": list(G.DEFAULT_AESOP_NAMESPACES),
    }
    json.dump(diff, open(_p(args.out_diff_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5S strict policy diff vs RC5H", "",
          f"- RC5H plan classification under strict grammar: {dict(removed_hist)}",
          f"- would remove {sum(v for k,v in removed_hist.items() if k!='POLICY_ALLOWED')} of "
          f"{sum(removed_hist.values())} RC5H programs", "",
          "## Removed tactic families (rationale)"]
    for k, v in policy["removed_by_default"].items():
        md.append(f"- **{k}**: {v}")
    md += ["", "## Allowed grammar (strict, 8 patterns + simpa variants)"]
    for g in policy["allowed_grammar"]:
        md.append(f"- `{g['example']}` ({g['pattern_id']})")
    md += ["", "## Budgets", f"- default B5; B10 = safe non-aesop families only; **B20 disabled**",
           "", "## Timeout policy",
           f"- per-theorem wall cap **{args.per_theorem_cap}s** (process-group kill); per-tactic {args.per_tactic}s",
           f"- aesop-tail namespace gate: {list(G.DEFAULT_AESOP_NAMESPACES)} (not Nat)",
           "", "## RC5H winners preserved", ""] + [f"- {w}" for w in policy["rc5h_winners_preserved"]]
    open(_p(args.out_diff_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5s-policy] strict grammar {len(policy['allowed_grammar'])} patterns; "
          f"plan classification {dict(removed_hist)}")
    print(f"[rc5s-policy] per_theorem_cap={args.per_theorem_cap}s aesop_ns={list(G.DEFAULT_AESOP_NAMESPACES)}")


if __name__ == "__main__":
    main()
