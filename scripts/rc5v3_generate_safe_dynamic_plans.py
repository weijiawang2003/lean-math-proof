#!/usr/bin/env python3
"""RC5V3 Part 8 — generate the strict safe dynamic plan with B1/B3/B5 budget slices.

Shells out to the validated TR6 generator (TR3/TR5 grammar + TR4 HGB ranker), then STRICT-FILTERS
every program through the RC5S grammar (rejecting off-policy / stall-risk / namespace-disabled),
keeps the top-5 safe programs/theorem, and marks B1 (rank 1) / B3 (ranks 1-3) / B5 (ranks 1-5)
slices. Hard requirement: final off-policy count = 0. No B10/B20, no simp_all, no depth-3 chains.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rc5s_grammar as G

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--eligible", default="project/evolve/experiments/rc5_v3/cases/rc5v3_dynamic_eligible.json")
    ap.add_argument("--tr4-model-dir", required=True)
    ap.add_argument("--tr4-vectorizers", required=True)
    ap.add_argument("--tr4-metadata", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-budget-slices", required=True)
    args = ap.parse_args()

    policy = json.load(open(_p(args.policy)))
    pol = {"allowed_namespaces": policy["allowed_namespaces"], "aesop_namespaces": policy["aesop_namespaces"]}

    raw = args.out_json.replace(".json", "_raw.json")
    cmd = [sys.executable, _p("scripts/tr6_generate_ranked_programs.py"),
           "--confirmation", _p(args.eligible), "--retrieval", _p(args.retrieval),
           "--tr4-model-dir", _p(args.tr4_model_dir), "--tr4-vectorizers", _p(args.tr4_vectorizers),
           "--tr4-metadata", _p(args.tr4_metadata), "--out-json", _p(raw),
           "--out-md", _p(raw.replace(".json", ".md")), "--model", "hgb", "--keep-top", "20"]
    print("[rc5v3-genplan] generating ranked programs via TR6 generator ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(_p(raw)):
        print(r.stdout[-1500:]); print(r.stderr[-1500:]); raise SystemExit("generation failed")

    plan = json.load(open(_p(raw)))
    theorems, total_gen, total_rej, total_b5, off_final = [], 0, 0, 0, 0
    fam_hist, ns_hist, score_buckets = Counter(), Counter(), Counter()
    b1_count = b3_count = b5_count = 0
    for t in plan["theorems"]:
        ns = t.get("namespace")
        safe = []
        for pgm in t.get("programs_ranked", []):
            total_gen += 1
            tac = pgm.get("tactic")
            klass, ok = G.classify_program(tac, ns, pol)
            if not ok:
                total_rej += 1
                continue
            safe.append({**pgm, "rc5s_pattern": G.pattern_of(tac)})
        safe.sort(key=lambda p: p.get("rank", 99))
        b5 = safe[:5]
        for i, p in enumerate(b5, 1):
            p["rank"] = i
            p["budget_stage"] = "B1" if i == 1 else ("B3" if i <= 3 else "B5")
            p.setdefault("used_lemmas", p.get("lemmas", []))
            fam_hist[p["rc5s_pattern"]] += 1
            sc = p.get("ranker_score")
            if sc is not None:
                score_buckets[round(min(0.99, max(0.0, sc)) * 10) / 10] += 1
            if G.classify_program(p["tactic"], ns, pol)[1] is not True:
                off_final += 1
        if b5:
            total_b5 += len(b5)
            ns_hist[ns] += 1
            b1_count += sum(1 for p in b5 if p["rank"] == 1)
            b3_count += sum(1 for p in b5 if p["rank"] <= 3)
            b5_count += len(b5)
            theorems.append({"full_name": t["full_name"], "namespace": ns, "file_path": t.get("file_path"),
                             "rc2_status": t.get("rc2_status"), "num_programs": len(b5),
                             "programs_ranked": b5})

    out = {"generated_by": "scripts/rc5v3_generate_safe_dynamic_plans.py",
           "num_eligible_theorems": len(plan["theorems"]),
           "num_theorems_with_programs": len(theorems),
           "programs_generated": total_gen, "programs_rejected_offpolicy": total_rej,
           "total_b5_programs": total_b5, "off_policy_in_final_plan": off_final,
           "budget_program_counts": {"B1": b1_count, "B3": b3_count, "B5": b5_count},
           "pattern_histogram": dict(fam_hist), "namespace_histogram": dict(ns_hist),
           "predicted_score_distribution": {str(k): v for k, v in sorted(score_buckets.items())},
           "theorems": theorems}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    slices = {"generated_by": "scripts/rc5v3_generate_safe_dynamic_plans.py",
              "budgets": {"B1": {"rank_lo": 1, "rank_hi": 1}, "B3": {"rank_lo": 2, "rank_hi": 3},
                          "B5": {"rank_lo": 4, "rank_hi": 5}},
              "cumulative": {"B1": {"rank_lo": 1, "rank_hi": 1}, "B3": {"rank_lo": 1, "rank_hi": 3},
                             "B5": {"rank_lo": 1, "rank_hi": 5}},
              "program_counts": {"B1": b1_count, "B3": b3_count, "B5": b5_count},
              "num_theorems": len(theorems)}
    json.dump(slices, open(_p(args.out_budget_slices), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5V3 safe dynamic plan (B1/B3/B5, strict)", "",
          f"- eligible: {len(plan['theorems'])} | with safe programs: {len(theorems)}",
          f"- generated: {total_gen} | rejected off-policy: {total_rej} | B5 programs: {total_b5}",
          f"- **off-policy in final plan: {off_final}** (must be 0)",
          f"- budget program counts (cumulative): B1={b1_count} B3={b3_count} B5={b5_count}",
          f"- pattern histogram: {dict(fam_hist)}",
          f"- namespace histogram: {dict(ns_hist)}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v3-genplan] theorems={len(theorems)} generated={total_gen} rejected={total_rej} "
          f"B5={total_b5} off_policy_final={off_final}")
    print(f"[rc5v3-genplan] patterns={dict(fam_hist)} budget_counts B1={b1_count} B3={b3_count} B5={b5_count}")


if __name__ == "__main__":
    main()
