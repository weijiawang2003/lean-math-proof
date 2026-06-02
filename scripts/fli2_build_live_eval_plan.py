#!/usr/bin/env python3
"""FLI2 Part 4 — theorem-centric live evaluation plan (vacuity-safe, at theorem position).

Groups deployment actions by theorem (one LeanDojo Dojo per theorem at its real file position),
attaches controls, and selects theorems in priority order until the action budget is reached.
Controls deliberately exclude `exact <target>` and never import the target theorem (avoids the
FLI1 self-import vacuity issue).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, OrderedDict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROLS = ["simp", "aesop", "classical <;> aesop", "constructor <;> simp", "ext x <;> simp"]


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--max-actions", type=int, default=200)
    ap.add_argument("--timeout-per-tactic", type=int, default=20)
    ap.add_argument("--timeout-per-theorem", type=int, default=240)
    args = ap.parse_args()

    actions = [json.loads(l) for l in open(_p(args.actions)) if l.strip()]
    # group by theorem, preserving the (already priority-sorted) action order
    by_thm = OrderedDict()
    for a in actions:
        by_thm.setdefault(a["theorem"], []).append(a)

    plan, used = [], 0
    for thm, acts in by_thm.items():
        if used >= args.max_actions:
            break
        a0 = acts[0]
        if not a0.get("file_path"):
            continue
        plan.append({
            "theorem": thm, "namespace": a0["namespace"], "file_path": a0["file_path"],
            "priority": a0["priority"], "source": a0["source"],
            "expected_pattern": a0["expected_pattern"],
            "controls": CONTROLS,
            "actions": [{"action_id": a["action_id"], "case_id": a["case_id"], "lemma": a["lemma"],
                         "template": a["template"], "tactic": a["tactic"], "risk": a["risk"]}
                        for a in acts],
            "timeout_per_tactic": args.timeout_per_tactic,
            "timeout_per_theorem": args.timeout_per_theorem,
        })
        used += len(acts)

    out = {"generated_by": "scripts/fli2_build_live_eval_plan.py",
           "num_theorems": len(plan), "num_actions": used,
           "max_actions_budget": args.max_actions,
           "controls": CONTROLS, "theorems": plan}
    with open(_p(args.out_json), "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    summary = {"generated_by": "scripts/fli2_build_live_eval_plan.py",
               "num_theorems": len(plan), "num_actions": used,
               "by_priority": dict(Counter(t["priority"] for t in plan)),
               "by_namespace": dict(Counter(t["namespace"] for t in plan).most_common()),
               "by_source": dict(Counter(t["source"] for t in plan)),
               "controls_per_theorem": len(CONTROLS)}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI2 live eval plan summary", "",
          f"- theorems: {summary['num_theorems']} | actions: {summary['num_actions']} "
          f"(budget {args.max_actions}) | controls/thm: {len(CONTROLS)}",
          f"- by priority: {summary['by_priority']}",
          f"- by namespace: {summary['by_namespace']}",
          f"- by source: {summary['by_source']}", "",
          "_At-position LeanDojo; controls exclude `exact <target>`; no self-import._"]
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli2-plan] theorems={len(plan)} actions={used} by_priority={summary['by_priority']}")


if __name__ == "__main__":
    main()
