#!/usr/bin/env python3
"""FLI2 Part 2 — build the retrieval-gap candidate pool.

Sources (deduped by theorem; FLI1 records take precedence as they carry residual/rescue detail):
  A. FLI1 confirmed RETRIEVAL_GAP cases.
  B. FLI1 EXISTS_CLOSE cases not rescued.
  C. FLI0 clean high-signal failures: nonempty retrieved lemmas, namespace in {Finset,List,
     Multiset,Set,Nat}, high-value bridge pattern, not unknown-name-only / infra (clean ⇒ already
     excludes those, and FLI0 failures are by construction not RC2/RC4/RC5-solved).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOOD_NS = {"Finset", "List", "Multiset", "Set", "Nat"}
HIGH_VALUE = {"MEMBERSHIP_BRIDGE", "SUBSET_BRIDGE", "MAP_FILTER_BIND_BRIDGE",
              "SINGLETON_CHARACTERIZATION", "DISJOINT_BRIDGE", "IFF_SPLIT",
              "EXTENSIONALITY_NEEDED", "INDUCTION_GENERALIZATION"}
HIGH_PRIO = {"MEMBERSHIP_BRIDGE", "SUBSET_BRIDGE", "MAP_FILTER_BIND_BRIDGE",
             "SINGLETON_CHARACTERIZATION", "DISJOINT_BRIDGE"}
FLI1_RESID = "project/evolve/experiments/fli1/cases/fli1_residual_goals.jsonl"


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    return [json.loads(l) for l in open(_p(path)) if l.strip()] if os.path.exists(_p(path)) else []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fli1-checked", required=True)
    ap.add_argument("--fli1-rescues", required=True)
    ap.add_argument("--fli0-enriched", required=True)
    ap.add_argument("--fli0-patterns", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    checked = _rows(args.fli1_checked)
    rescues = {r["theorem"]: r for r in _rows(args.fli1_rescues)}
    enriched = {e["theorem"]: e for e in _rows(args.fli0_enriched)}
    patterns = {p["theorem"]: p for p in _rows(args.fli0_patterns)}
    resid = {r["theorem"]: r for r in _rows(FLI1_RESID)}

    pool = {}
    n = 0

    def add(theorem, source, prio, why, extra):
        nonlocal n
        if theorem in pool:
            return
        e = enriched.get(theorem, {})
        pat = patterns.get(theorem, {})
        rl = [t.get("lemma") for t in (e.get("top_retrieved_lemmas_detailed") or []) if t.get("lemma")][:8]
        rg = resid.get(theorem, {})
        residual = (rg.get("residual_goals") or [None])[0] if rg.get("status") == "captured" else None
        n += 1
        pool[theorem] = {
            "case_id": f"FLI2-{n:03d}",
            "source": source,
            "theorem": theorem,
            "namespace": e.get("namespace") or pat.get("namespace"),
            "statement": e.get("statement") or pat.get("statement"),
            "failure_patterns": pat.get("pattern_labels") or [pat.get("primary_pattern")],
            "primary_pattern": pat.get("primary_pattern"),
            "retrieved_lemmas": rl,
            "candidate_existing_lemmas": extra.get("candidate_existing_lemmas") or rl[:4],
            "residual_goal": residual,
            "file_path": e.get("file_path"),
            "source_stage": extra.get("source_stage") or e.get("source_stage") or "FLI0",
            "priority": prio,
            "why_in_pool": why,
        }

    # A. FLI1 confirmed retrieval gaps
    for c in checked:
        if c.get("retrieval_gap"):
            thm = c["downstream_targets"][0]
            add(thm, "FLI1_RETRIEVAL_GAP", "high",
                f"FLI1 retrieval-gap: close existing `{c.get('closest_existing_lemma')}` retrieved but undeployed",
                {"candidate_existing_lemmas": [c.get("closest_existing_lemma")] +
                 [t.get("lemma") for t in (enriched.get(thm, {}).get("top_retrieved_lemmas_detailed") or [])[:3]],
                 "source_stage": "FLI1"})
    # B. FLI1 EXISTS_CLOSE not rescued (not already added)
    for c in checked:
        if c.get("existing_check") == "EXISTS_CLOSE":
            thm = c["downstream_targets"][0]
            rc = rescues.get(thm, {}).get("classification")
            if thm not in pool and rc != "DOWNSTREAM_RESCUE":
                add(thm, "FLI1_EXISTS_CLOSE", "high",
                    f"FLI1 exists-close not rescued (rescue={rc}); retry deployment",
                    {"candidate_existing_lemmas": [c.get("closest_existing_lemma")],
                     "source_stage": "FLI1"})
    # C. FLI0 high-signal
    for thm, p in patterns.items():
        if thm in pool:
            continue
        e = enriched.get(thm, {})
        if not (p.get("clean_failure") and e.get("top_retrieved_lemmas_detailed")):
            continue
        ns = (p.get("namespace") or "").split(".")[0]
        if ns not in GOOD_NS or p.get("primary_pattern") not in HIGH_VALUE:
            continue
        prio = "high" if p["primary_pattern"] in HIGH_PRIO else "medium"
        add(thm, "FLI0_HIGH_SIGNAL", prio,
            f"FLI0 clean {p['primary_pattern']} failure with {len(e['top_retrieved_lemmas_detailed'])} retrieved lemmas",
            {"source_stage": e.get("source_stage")})

    items = sorted(pool.values(), key=lambda r: ({"high": 0, "medium": 1, "low": 2}[r["priority"]],
                                                 r["namespace"] or "", r["case_id"]))
    with open(_p(args.out_jsonl), "w") as f:
        for r in items:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {"generated_by": "scripts/fli2_build_retrieval_gap_pool.py",
               "pool_size": len(items),
               "by_source": dict(Counter(r["source"] for r in items)),
               "by_priority": dict(Counter(r["priority"] for r in items)),
               "by_namespace": dict(Counter(r["namespace"] for r in items).most_common()),
               "by_pattern": dict(Counter(r["primary_pattern"] for r in items).most_common()),
               "with_residual_goal": sum(1 for r in items if r["residual_goal"]),
               "with_file_path": sum(1 for r in items if r["file_path"])}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI2 retrieval-gap pool summary", "",
          f"- **pool size: {summary['pool_size']}**",
          f"- by source: {summary['by_source']}",
          f"- by priority: {summary['by_priority']}",
          f"- by namespace: {summary['by_namespace']}",
          f"- by pattern: {summary['by_pattern']}",
          f"- with residual goal (FLI1): {summary['with_residual_goal']} | with file_path: "
          f"{summary['with_file_path']}", ""]
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli2-pool] size={len(items)} by_source={summary['by_source']} "
          f"by_priority={summary['by_priority']}")


if __name__ == "__main__":
    main()
