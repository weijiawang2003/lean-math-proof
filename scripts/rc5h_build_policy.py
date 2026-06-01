#!/usr/bin/env python3
"""RC5H Part 2 — build the hybrid policy.

Static stage = the frozen RC4R wrapper (unchanged). Dynamic stage = TR4 HGB ranker over a small
program grammar, enabled only after static failure, gated to a namespace allowlist + a retrieval-
confidence gate, with B5/B10/B20 budgets. RC4A gate tightening from TR7 is recorded as a
recommendation only (NOT implemented in the static core). No live Lean.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RC4R_WRAPPER = "project/evolve/experiments/rc4_release_candidate/rc4_release_candidate_wrapper.json"
TR7_REC = "project/evolve/experiments/tr7/out/tr7_rc5_recommendations.json"
TR7_GATE = "project/evolve/experiments/tr7/out/tr7_gate_refinement_analysis.json"
TR4_MODEL = "project/evolve/experiments/tr4/models/hgb_program_ranker.joblib"
TR4_VEC = "project/evolve/experiments/tr4/data/tr4_vectorizers.joblib"
TR4_META = "project/evolve/experiments/tr4/data/tr4_feature_metadata.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-policy", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    tr7 = json.load(open(_p(TR7_REC))) if os.path.exists(_p(TR7_REC)) else {}
    gate = json.load(open(_p(TR7_GATE))) if os.path.exists(_p(TR7_GATE)) else {}

    policy = {
        "family": "rc5_hybrid_static_plus_ranker",
        "base_static_core": "RC4R",
        "status": "experimental_prototype",
        "promotion_allowed": False,
        "source_recommendation": tr7.get("primary_recommendation", "RC5_HYBRID_STATIC_PLUS_RANKER"),
        "static_stage": {
            "wrapper": RC4R_WRAPPER,
            "route_config": "project/evolve/routing/ns24_router.json",
            "policy_type": "hybrid_evolved", "top_k": 8, "max_steps": 8,
            "note": "frozen RC4R, byte-identical; floors + known wins preserved by construction.",
        },
        "dynamic_stage": {
            "enabled_only_after_static_failure": True,
            "ranker": "TR4_HGB",
            "ranker_model": TR4_MODEL, "ranker_vectorizers": TR4_VEC, "ranker_metadata": TR4_META,
            "retrieval_index": ["project/evolve/experiments/tr3/out/tr3_retrieval_index.jsonl",
                                "project/evolve/experiments/sf5/out/sf5_lemma_index.jsonl"],
            "max_programs_per_theorem": {"B5": 5, "B10": 10, "B20": 20},
            "default_budget": 10,
            "retrieval_top_k": 20,
            "program_grammar": [
                "exact L", "simpa using L", "simp [L]", "rw [L]",
                "simp [L] <;> aesop", "simp [L] <;> simp_all", "rw [L] <;> aesop",
                "ext x <;> simp [L]", "constructor <;> intro h <;> aesop",
            ],
            "controls": ["simp", "simp_all", "aesop", "classical <;> aesop",
                         "exact L", "simpa using L", "simp [L]"],
            "gates": {
                "allowed_namespaces": ["Set", "Finset", "List", "Multiset", "Nat"],
                "order_family": "disabled_or_analysis_only",
                "max_unknown_name_rate": 0.10,
                "disable_if_no_retrieval_confidence": True,
                "min_retrieval_best_score": 0.0,
            },
            "stop_after_first_success": True,
            "deterministic_ordering": True,
        },
        "rc4a_gate_tightening": {
            "status": "RECOMMENDATION_ONLY_NOT_IMPLEMENTED",
            "from": "TR7 RC4A_TIGHTEN_MONO_GATE",
            "precision": (gate.get("rc4a_broad_gate") or {}).get("precision"),
            "note": "RC4A def-unfold gate is broad (precision ~0.09); TR7 recommends tightening to "
                    "the iff-unfold shape. NOT applied to the RC5H static core — the static core "
                    "remains exactly RC4R so the prototype measures the dynamic stage in isolation.",
        },
        "evaluation": {
            "true_hybrid_delta_rule": "RC2 failed AND RC4 static failed AND a dynamic program "
                                      "solved AND bare controls did not solve AND not source-specific",
            "additive": "dynamic stage runs only on static failures -> regressions over RC4 "
                        "structurally impossible.",
        },
    }
    os.makedirs(os.path.dirname(_p(args.out_policy)), exist_ok=True)
    json.dump(policy, open(_p(args.out_policy), "w"), ensure_ascii=False, indent=2)

    md = ["# RC5H policy", "",
          f"- family: {policy['family']} | base static core: {policy['base_static_core']} | "
          "promotion_allowed: False", "",
          "## Static stage", f"- wrapper: `{RC4R_WRAPPER}` (frozen RC4R, unchanged)",
          "- config: hybrid_evolved, top-k 8, max-steps 8", "",
          "## Dynamic stage (only after static failure)",
          f"- ranker: TR4 HGB | retrieval top-k: {policy['dynamic_stage']['retrieval_top_k']}",
          f"- budgets: {policy['dynamic_stage']['max_programs_per_theorem']}",
          f"- grammar ({len(policy['dynamic_stage']['program_grammar'])}): "
          f"{policy['dynamic_stage']['program_grammar']}",
          f"- gates: namespaces {policy['dynamic_stage']['gates']['allowed_namespaces']}, "
          f"max_unknown_name_rate {policy['dynamic_stage']['gates']['max_unknown_name_rate']}, "
          "disable_if_no_retrieval_confidence True; Order family disabled", "",
          "## RC4A gate tightening",
          f"- **{policy['rc4a_gate_tightening']['status']}** (precision "
          f"{policy['rc4a_gate_tightening']['precision']}) — recommendation only, not implemented.", "",
          "## Evaluation",
          f"- TRUE_HYBRID_DELTA: {policy['evaluation']['true_hybrid_delta_rule']}",
          f"- additive: {policy['evaluation']['additive']}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-policy] static_core={policy['base_static_core']} ranker=TR4_HGB "
          f"budgets={policy['dynamic_stage']['max_programs_per_theorem']} "
          f"namespaces={policy['dynamic_stage']['gates']['allowed_namespaces']}")


if __name__ == "__main__":
    main()
