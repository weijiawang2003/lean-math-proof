#!/usr/bin/env python3
"""FLI2 Part 9 — researcher-facing retrieval-gap deployment atlas (md + json)."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    return [json.loads(l) for l in open(_p(path)) if l.strip()] if os.path.exists(_p(path)) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--actions", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--rules", required=True)
    ap.add_argument("--rc4bc-comparison", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    pool = _rows(args.pool)
    actions = _rows(args.actions)
    results = _rows(args.results)
    attr = _rows(args.attribution)
    rules = json.load(open(_p(args.rules))).get("rules", []) if os.path.exists(_p(args.rules)) else []
    rc4bc = json.load(open(_p(args.rc4bc_comparison))) if os.path.exists(_p(args.rc4bc_comparison)) else {}

    true_r = [r for r in attr if r["classification"] == "TRUE_RETRIEVAL_GAP_RESCUE"]
    partials = [r for r in attr if r["classification"] == "PARTIAL_PROGRESS"]
    cls_hist = Counter(r["classification"] for r in attr)

    atlas = {
        "generated_by": "scripts/fli2_write_deployment_atlas.py",
        "pool_size": len(pool), "actions_generated": len(actions),
        "actions_run": len(results), "theorems_evaluated": len({r["theorem"] for r in results}),
        "action_classification": dict(cls_hist.most_common()),
        "true_rescues": [{"theorem": r["theorem"], "lemma": r["lemma"], "tactic": r["tactic"],
                          "namespace": r["namespace"], "controls_failed": r["control_solved"] == [],
                          "why": r.get("why")} for r in true_r],
        "partial_progress": [{"theorem": r["theorem"], "lemma": r["lemma"], "tactic": r["tactic"],
                              "residual_before": (r.get("residual_before") or "")[:200],
                              "residual_after": (r.get("residual_after") or "")[:200]}
                             for r in partials][:20],
        "deployment_rules": rules,
        "rc4bc_comparison": {k: rc4bc.get(k) for k in
                             ("fli2_true_rescues", "overlap_with_rc4bc_families",
                              "new_family_signatures", "is_scalable_rc_candidate_generator")},
    }
    with open(_p(args.out_json), "w") as f:
        json.dump(atlas, f, ensure_ascii=False, indent=2)

    md = ["# FLI2 Retrieval-Gap Deployment Atlas", "",
          "_Can failure analysis turn retrieved-but-undeployed lemmas into reusable deployment "
          "rules that rescue failed theorems? Hedged language; at-position tests only._", "",
          "## 1. Overview",
          f"Pool {len(pool)} retrieval-gap/high-signal failures → {len(actions)} gated deployment "
          f"actions → {len(results)} run live over {atlas['theorems_evaluated']} theorems.", "",
          "## 2. Why FLI2 follows FLI1",
          "FLI1 found 1 robust rescue and 15 retrieval gaps — the relevant lemma often already "
          "exists but is not deployed. FLI2 scales the test: do retrieval gaps form reusable "
          "deployment families?", "",
          "## 3. Retrieval-gap pool",
          f"{len(pool)} cases (FLI1 retrieval-gap + FLI0 high-signal), namespaces "
          f"{dict(Counter(p['namespace'] for p in pool).most_common())}.", "",
          "## 4. Deployment action families",
          f"{dict(Counter(a['template'] for a in actions).most_common())}", "",
          "## 5. Live evaluation results",
          f"action classifications: {atlas['action_classification']}", "",
          "## 6. True retrieval-gap rescues", ""]
    if true_r:
        md += ["| theorem | lemma | tactic |", "|---|---|---|"]
        for r in atlas["true_rescues"]:
            md.append(f"| `{r['theorem']}` | `{r['lemma']}` | `{r['tactic']}` |")
    else:
        md.append("_None beyond known cases in this batch._")
    md += ["", "## 7. Partial progress cases", ""]
    if partials:
        for r in atlas["partial_progress"][:8]:
            md += [f"### `{r['theorem']}` via `{r['tactic']}`",
                   f"- before: `{r['residual_before'][:120]}`",
                   f"- after:  `{r['residual_after'][:120]}`", ""]
    else:
        md.append("_None._")
    md += ["## 8. Deployment rules mined", ""]
    if rules:
        md += ["| rule | family | actions | rescues | partials | FP | status |",
               "|---|---|---|---|---|---|---|"]
        for r in rules:
            md.append(f"| {r['rule_id']} | {r['lemma_family']} | "
                      f"{','.join(r['recommended_actions'])} | {r['num_rescues']} | "
                      f"{r['num_partials']} | {r['num_false_positives']} | {r['promotion_status']} |")
    else:
        md.append("_No rule met the support threshold._")
    md += ["", "## 9. Comparison to RC4B/RC4C", "",
           rc4bc.get("assessment", "_n/a_"), "",
           "## 10. Failure modes", "",
           "- Many candidate solves are CONTROL_DUPLICATE: a bare control already closes the goal "
           "at position, i.e. RC5's failure does not reproduce in plain LeanDojo (its grammar was "
           "stricter / context differed).",
           "- UNKNOWN_NAME: a retrieved lemma is not in scope at the theorem's file position "
           "(defined later) — an honest availability gap, not a deployable bridge.",
           "- The deployable signal concentrates where a control genuinely fails but a specific "
           "retrieved lemma closes it.", "",
           "## 11. Recommended next step (FLI3)", "",
           "1. Promote confirmed deployment-rule candidates into the RC4-style literal-RC2 additive "
           "validation harness (off-gate/floors/determinism) — turn discovery into validated "
           "candidates.",
           "2. For CONTROL_DUPLICATE cases, re-mine the FLI0 corpus distinguishing 'RC5-grammar gap' "
           "(control works at position) from 'genuine bridge needed'.",
           "3. For PARTIAL_PROGRESS, feed the new residual to FLI1-style lemma invention (multi-step)."]
    with open(_p(args.out_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli2-atlas] pool={len(pool)} actions_run={len(results)} true_rescues={len(true_r)} "
          f"partials={len(partials)} rules={len(rules)}")


if __name__ == "__main__":
    main()
