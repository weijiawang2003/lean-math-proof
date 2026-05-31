#!/usr/bin/env python3
"""RC2 Part 5 — NS23 minimal-sufficient relabel for SET_ITE_SIMP new wins.

Promotion filter against LITERAL RC1. For every candidate new win over literal RC1,
classify using the baseline battery already run live by the candidate runner
(simp / simp_all / aesop / classical<;>aesop) plus the candidate tactic itself:

  TRUE_SET_ITE_SIMP_WIN  literal RC1 failed, ALL baselines failed, gate valid,
                         `simp [Set.ite]` (non-baseline) closed it.
  BASELINE_DUPLICATE     a simpler baseline also closed it (or the candidate tactic
                         is itself a baseline — cannot happen for simp [Set.ite]).
  RC1_ALREADY_SOLVED     literal RC1 solved it — not a win.
  PARSER_ARTIFACT        the candidate solve came after a prior parser/runner failure
                         shape (flagged if parse_error observed on the candidate).
  SOURCE_SPECIFIC        included for completeness; should not occur for simp [Set.ite].
  NEEDS_DEEPER_SEQUENCE  gate fired but candidate did not close it.

Outputs:
  minimal_relabel_results.json / .md
"""
from __future__ import annotations

import argparse
import json
import os

BASELINES = {"simp", "simp_all", "aesop", "classical <;> aesop"}


def classify(r):
    if r.get("literal_rc1_finished"):
        return "RC1_ALREADY_SOLVED", "literal RC1 solved it"
    if not r.get("set_ite_gate_fired"):
        return "NEEDS_DEEPER_SEQUENCE", "gate did not fire"
    if not r.get("set_ite_finished"):
        return "NEEDS_DEEPER_SEQUENCE", "gate fired but simp [Set.ite] did not close it"
    # gate fired and solved, RC1 failed
    base_solved = [b["probe"] for b in r.get("baseline_outcomes", []) if b.get("solved")]
    if base_solved:
        return "BASELINE_DUPLICATE", f"baseline(s) also closed it: {base_solved}"
    tac = (r.get("set_ite_tactic") or "").strip()
    if tac in BASELINES:
        return "BASELINE_DUPLICATE", f"candidate tactic `{tac}` is itself a baseline"
    if r.get("parse_error"):
        return "PARSER_ARTIFACT", "candidate solve associated with a parse-error shape"
    return "TRUE_SET_ITE_SIMP_WIN", ("literal RC1 failed, all baselines failed, "
                                     "non-baseline `simp [Set.ite]` closed it")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--candidate-results",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/candidate_results.json")
    p.add_argument("--literal-rc1-results",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/literal_rc1_results.json")
    p.add_argument("--out-json",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/minimal_relabel_results.json")
    p.add_argument("--out-md",
                   default="project/evolve/experiments/rc2_candidates/set_ite_simp/out/minimal_relabel_results.md")
    args = p.parse_args(argv)

    cand = json.load(open(args.candidate_results))["results"]
    rows, hist = [], {}
    for r in cand:
        cls, reason = classify(r)
        hist[cls] = hist.get(cls, 0) + 1
        rows.append({"full_name": r["full_name"], "in_sets": r.get("in_sets"),
                     "literal_rc1_finished": r.get("literal_rc1_finished"),
                     "set_ite_gate_fired": r.get("set_ite_gate_fired"),
                     "set_ite_finished": r.get("set_ite_finished"),
                     "baseline_solved_by": r.get("baseline_solved_by"),
                     "attribution": cls, "reason": reason})
    true_wins = [r for r in rows if r["attribution"] == "TRUE_SET_ITE_SIMP_WIN"]
    # dedupe true wins by theorem (a theorem appears across multiple sets)
    true_unique = sorted({r["full_name"] for r in true_wins})

    out = {
        "attribution_histogram": hist,
        "true_set_ite_simp_win_count": len(true_unique),
        "true_set_ite_simp_wins": true_unique,
        "policy": "A candidate solve is TRUE_SET_ITE_SIMP_WIN only if literal RC1 AND all "
                  "four baselines failed and non-baseline `simp [Set.ite]` closed it. "
                  "RC1_ALREADY_SOLVED / BASELINE_DUPLICATE are not wins.",
        "rows": rows,
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = ["# RC2 — SET_ITE_SIMP Minimal-Sufficient Relabel (vs literal RC1)", ""]
    L.append(f"- attribution histogram: `{hist}`")
    L.append(f"- **TRUE_SET_ITE_SIMP_WIN (unique theorems) = {len(true_unique)}**: {true_unique}")
    L.append(f"- {out['policy']}")
    L.append("")
    L.append("| theorem | sets | rc1 | gate | set_ite | baseline_solved_by | attribution |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| `{r['full_name']}` | {','.join(r.get('in_sets') or [])} | "
                 f"{r['literal_rc1_finished']} | {r['set_ite_gate_fired']} | "
                 f"{r['set_ite_finished']} | {r['baseline_solved_by']} | "
                 f"**{r['attribution']}** |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[rc2:relabel] rows={len(rows)} hist={hist} "
          f"TRUE_SET_ITE_SIMP_WIN(unique)={len(true_unique)} -> {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
