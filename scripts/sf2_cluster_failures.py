#!/usr/bin/env python3
"""SF2 Part 6 — failure clustering + pattern ranking.

Consumes corrected eval results (RC1/NS9), the classified frontier, and any
trace/goal context, then groups GENUINE failures (solved == false) into clusters
keyed by namespace / top candidate family / name tokens / failure type / goal
shape, and ranks each cluster high|medium|low with a next_action.

Only `finished`/`solved == false` rows count as failures. Rows that are actually
environment problems (theorem path not resolvable, junk frontier rows) are tagged
`unresolved_or_junk` and ranked low, never mixed with genuine proof failures.

Inputs (any subset; missing ones skipped):
  --eval-results  one or more eval_matrix_results.json / rc1_eval_results.json
  --classified-frontier classified_frontier.jsonl
Outputs:
  failure_clusters.json , failure_clusters.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

JUNK_NAMES = {"traces_from_search.jsonl"}
JUNK_NAMESPACES = {"traces_from_search", "GENERAL_FRONTIER"}
# names that are almost certainly not real Mathlib decls (mining artifacts)
SUSPECT_NAMES = {"Prop.compl_singleton", "Eq.subset", "coe_notMemRangeEquiv_symm"}


def read_jsonl(path):
    rows = []
    if path and os.path.isfile(path):
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def load_eval_failures(paths):
    """Return list of failure dicts {full_name, namespace, policy, theorem_set,
    winning_tactic, error_message, final_goal} for solved==false rows."""
    failures = []
    for p in paths:
        if not os.path.isfile(p):
            continue
        try:
            data = json.load(open(p))
        except Exception:
            continue
        for res in data.get("results", []):
            if res.get("status") != "ran":
                continue
            policy = res.get("policy")
            tset = res.get("theorem_set")
            run_dir = res.get("run_dir")
            for t in (res.get("per_theorem") or []):
                if t.get("solved"):
                    continue
                fn = t.get("full_name")
                goal = _trace_goal(run_dir, fn)
                failures.append({
                    "full_name": fn,
                    "namespace": fn.split(".")[0] if fn and "." in fn else (fn or None),
                    "policy": policy, "theorem_set": tset,
                    "winning_tactic": t.get("winning_tactic"),
                    "error_message": t.get("error_message"),
                    "num_steps": t.get("num_steps"),
                    "final_goal": goal,
                    "source_results": os.path.basename(p),
                })
    return failures


def _trace_goal(run_dir, full_name):
    if not run_dir:
        return None
    last = None
    for tp in glob.glob(os.path.join(run_dir, "**", "*.jsonl"), recursive=True):
        for rec in read_jsonl(tp):
            if isinstance(rec, dict) and (rec.get("full_name") or rec.get("theorem")) == full_name:
                if rec.get("state_pp"):
                    last = rec["state_pp"]
    return last


def goal_shape(name, goal, tags):
    g = (goal or "")
    nm = (name or "")
    shapes = []
    if "↔" in g or nm.endswith("_iff") or "_iff_" in nm or "has_iff" in tags:
        shapes.append("iff")
    if "toFinset" in nm or "toFinset" in g or "has_toFinset" in tags:
        shapes.append("toFinset")
    if "{" in g and "}" in g or "singleton" in nm or "has_singleton" in tags:
        shapes.append("singleton")
    if "Disjoint" in g or "disjoint" in nm:
        shapes.append("disjoint")
    if "⊆" in g or "subset" in nm or "has_subset" in tags:
        shapes.append("subset")
    if "•" in g or "nsmul" in nm or "nsmul" in g:
        shapes.append("nsmul")
    if (" = " in g and "↔" not in g) or nm.endswith("_eq") or "has_eq" in tags:
        shapes.append("equality")
    if "∈" in g or "mem" in nm:
        shapes.append("membership")
    return sorted(set(shapes)) or ["unknown"]


def failure_type(err, goal):
    e = (err or "").lower()
    if "expected end of input" in e or "unexpected token" in e:
        return "parse_error"
    if "maximum recursion" in e:
        return "max_recursion"
    if "applyexttheorem only applies" in e:
        return "ext_not_applicable"
    if "all top-" in e and "errored" in e:
        return "all_tactics_errored"
    if "induction" in e and "premise" in e:
        return "induction_misapplied"
    if (goal and ("::ₘ" in goal or "• {" in goal) and "↔" in goal):
        return "wrong_induction_residual"
    if "timeout" in e or "deterministic timeout" in e:
        return "timeout"
    if not e:
        return "unknown_no_error"
    return "tactics_failed"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--eval-results", nargs="+", default=[
        "project/evolve/experiments/sf1/out/real/eval_matrix_results.json",
        "project/evolve/experiments/sf2/out/frontier_expansion/rc1_eval_results.json",
    ])
    p.add_argument("--classified-frontier",
                   default="project/evolve/experiments/sf1/out/real/classified_frontier.jsonl")
    p.add_argument("--out-json",
                   default="project/evolve/experiments/sf2/out/frontier_expansion/failure_clusters.json")
    p.add_argument("--out-md",
                   default="project/evolve/experiments/sf2/out/frontier_expansion/failure_clusters.md")
    args = p.parse_args(argv)

    classified = {r.get("decl_name"): r for r in read_jsonl(args.classified_frontier)}
    failures = load_eval_failures(args.eval_results)
    # dedup by (full_name, theorem_set): the same set may appear in multiple result
    # files (e.g. a copied rc1_eval_results.json); never double-count a theorem.
    seen, uniq = set(), []
    for f in failures:
        k = (f.get("full_name"), f.get("theorem_set"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(f)
    failures = uniq

    # enrich + bucket
    enriched = []
    for f in failures:
        cl = classified.get(f["full_name"], {})
        tags = cl.get("tags", [])
        top = cl.get("top_candidate_family")
        is_junk = (f["full_name"] in JUNK_NAMES or f["namespace"] in JUNK_NAMESPACES
                   or f["full_name"] in SUSPECT_NAMES)
        is_unresolved = (f.get("final_goal") is None and not f.get("error_message"))
        ft = ("unresolved_or_junk" if (is_junk) else failure_type(f.get("error_message"),
                                                                  f.get("final_goal")))
        shapes = goal_shape(f["full_name"], f.get("final_goal"), tags)
        feat_tags = [t for t in ["has_iff", "has_eq", "has_membership", "has_subset",
                                 "has_toFinset", "has_finite", "likely_extensionality",
                                 "likely_aesop", "likely_multiset_induction"] if t in tags]
        enriched.append({**f, "top_candidate_family": top, "tags": feat_tags,
                         "failure_type": ft, "goal_shapes": shapes,
                         "is_junk_or_unresolved": bool(is_junk or is_unresolved),
                         "has_trace": bool(f.get("final_goal"))})

    # cluster key: (namespace, top_family, primary goal shape, failure_type)
    groups = defaultdict(list)
    for e in enriched:
        key = (e["namespace"], e["top_candidate_family"], e["goal_shapes"][0],
               e["failure_type"])
        groups[key].append(e)

    clusters = []
    for (ns, fam, shape, ft), items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        names = [i["full_name"] for i in items]
        any_trace = any(i["has_trace"] for i in items)
        junk = all(i["is_junk_or_unresolved"] for i in items)
        # priority
        if junk or ft == "unresolved_or_junk":
            prio, action = "low", "ignore"
        elif len(items) >= 2 and not junk:
            prio, action = "high", "probe"
        elif any_trace and ft in ("wrong_induction_residual", "all_tactics_errored",
                                  "max_recursion", "ext_not_applicable"):
            prio, action = "high", "probe"
        elif any_trace:
            prio, action = "medium", "probe"
        else:
            prio, action = "medium", "needs_trace"
        # capability hypothesis
        if "toFinset" in (shape, ) or "toFinset" in (fam or ""):
            cap = "toFinset membership/iff orchestration (split-iff -> ext/membership)"
        elif shape == "iff":
            cap = "iff decomposition before simp/ext"
        elif fam == "wx3_multiset_induction":
            cap = "Multiset induction routing (avoid on membership/iff goals)"
        else:
            cap = "needs source-proof inspection"
        clusters.append({
            "cluster_id": f"{ns}|{fam}|{shape}|{ft}",
            "size": len(items),
            "namespace": ns, "top_candidate_family": fam,
            "primary_goal_shape": shape, "failure_type": ft,
            "representative_theorems": names[:6],
            "shared_features": sorted(set(t for i in items for t in i["tags"])),
            "has_full_trace": any_trace,
            "likely_missing_capability": cap,
            "candidate_probe_family": ("split_iff_then_ext_membership"
                                       if shape in ("iff", "toFinset", "singleton")
                                       else "simp/aesop baseline"),
            "candidate_lemma_template": None,
            "priority": prio, "expected_utility": "multi" if len(items) >= 2 else "single",
            "next_action": action,
        })

    out = {"num_failures": len(enriched),
           "num_genuine_failures": sum(1 for e in enriched if not e["is_junk_or_unresolved"]),
           "num_junk_or_unresolved": sum(1 for e in enriched if e["is_junk_or_unresolved"]),
           "num_clusters": len(clusters),
           "eval_results_used": [p for p in args.eval_results if os.path.isfile(p)],
           "clusters": clusters, "failures": enriched}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), ensure_ascii=False, indent=2)

    L = []
    a = L.append
    a("# SF2 Failure Clusters")
    a("")
    a(f"- failures: {out['num_failures']} (genuine {out['num_genuine_failures']}, "
      f"junk/unresolved {out['num_junk_or_unresolved']}) | clusters: {out['num_clusters']}")
    a(f"- eval results used: {out['eval_results_used']}")
    a("")
    a("| priority | cluster_id | size | trace | capability | next |")
    a("|---|---|---|---|---|---|")
    for c in sorted(clusters, key=lambda c: ({"high": 0, "medium": 1, "low": 2}[c["priority"]], -c["size"])):
        a(f"| {c['priority']} | `{c['cluster_id']}` | {c['size']} | {c['has_full_trace']} | "
          f"{c['likely_missing_capability']} | {c['next_action']} |")
    a("")
    a("## Representative theorems per cluster")
    for c in sorted(clusters, key=lambda c: ({"high": 0, "medium": 1, "low": 2}[c["priority"]], -c["size"])):
        a(f"- **{c['priority']}** `{c['cluster_id']}` (n={c['size']}): "
          f"{', '.join(c['representative_theorems'])}")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[sf2:cluster] failures={out['num_failures']} genuine={out['num_genuine_failures']} "
          f"clusters={len(clusters)} -> {args.out_json}")
    for c in sorted(clusters, key=lambda c: -c["size"])[:8]:
        print(f"  [{c['priority']}] {c['cluster_id']} n={c['size']} {c['next_action']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
