#!/usr/bin/env python3
"""TR7 Part 10 — export the fresh-delta-gap diagnostic dataset.

Joins the comparison corpus + static coverage audit + replay results + missing-allowlist
analysis + dynamic/static classification into one per-TR6-win diagnostic dataset (jsonl) plus a
summary. New TR7 artifact — does not overwrite any prior training dataset.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _index(path, key="full_name", sub="records"):
    d = json.load(open(_p(path)))
    return {r[key]: r for r in d.get(sub, d if isinstance(d, list) else [])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--coverage", required=True)
    ap.add_argument("--replay", required=True)
    ap.add_argument("--allowlist", required=True)
    ap.add_argument("--dynamic-static", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    corpus = {r["full_name"]: r for r in (json.loads(l) for l in open(_p(args.corpus)))}
    cov = _index(args.coverage)
    replay = _index(args.replay)
    allow = _index(args.allowlist)
    dynstat = _index(args.dynamic_static)

    NEXT = {
        "STATIC_WRAPPER_COMPATIBLE_NOW": "none (already in RC4)",
        "STATIC_WRAPPER_COMPATIBLE_WITH_SCHEMA_FIX": "RC5H: deploy as bare simp[L] enabling action (RC4B-style)",
        "STATIC_WRAPPER_COMPATIBLE_WITH_GATE_REFINEMENT": "RC5H: widen the gate prefix",
        "STATIC_WRAPPER_COMPATIBLE_WITH_ALLOWLIST_EXPANSION": "TR8: gather recurrence evidence, then add lemma",
        "DYNAMIC_RETRIEVAL_PREFERRED": "RC5H: ranker-guided dynamic retrieval stage",
        "SEARCH_ONLY_FAMILY": "keep search-only (not production)",
    }

    rows = []
    for fn, c in cov.items():
        cp = corpus.get(fn, {})
        rp = replay.get(fn, {})
        al = allow.get(fn, {})
        ds = dynstat.get(fn, {})
        dscls = ds.get("dynamic_vs_static_class")
        rows.append({
            "full_name": fn, "namespace": c["namespace"],
            "tr6_status": "FRESH_TRUE_DELTA",
            "rc4r_status": rp.get("rc4_wrapper_status") or cp.get("rc4_static_result"),
            "winning_program": c.get("tr6_winning_program") or cp.get("tr6_winning_program"),
            "winning_lemma": c.get("tr6_winning_lemma"),
            "static_coverage_class": c["classification"],
            "replay_class": rp.get("classification"),
            "tr6_program_reproduces": rp.get("tr6_program_reproduces"),
            "rc4_action_single_shot_solves": rp.get("rc4_action_single_shot_solves"),
            "allowlist_recommendation": al.get("recommendation"),
            "dynamic_vs_static_class": dscls,
            "missing_reason": (None if c["classification"] == "STATIC_COVERED_AND_SHOULD_SOLVE"
                               else c["classification"]),
            "recommended_next_action": NEXT.get(dscls, "review"),
        })

    summary = {
        "generated_by": "scripts/tr7_export_diagnostic_dataset.py",
        "num_examples": len(rows),
        "static_coverage_histogram": dict(Counter(r["static_coverage_class"] for r in rows)),
        "replay_histogram": dict(Counter(r["replay_class"] for r in rows)),
        "dynamic_vs_static_histogram": dict(Counter(r["dynamic_vs_static_class"] for r in rows)),
        "recommended_next_action_histogram": dict(Counter(r["recommended_next_action"] for r in rows)),
        "tr6_program_reproduces_all": all(r["tr6_program_reproduces"] for r in rows
                                          if r["tr6_program_reproduces"] is not None),
        "usage": "Diagnostic set for RC5 design: each row says whether a TR6 fresh win is "
                 "static-reachable and the concrete next action. Use to scope RC5H (hybrid) and "
                 "to seed TR8's recurrence search; NOT a prover training set.",
    }
    with open(_p(args.out_jsonl), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR7 diagnostic dataset summary", "",
          f"- examples: {summary['num_examples']}",
          f"- static coverage: {summary['static_coverage_histogram']}",
          f"- replay: {summary['replay_histogram']}",
          f"- dynamic/static: {summary['dynamic_vs_static_histogram']}",
          f"- recommended next action: {summary['recommended_next_action_histogram']}",
          f"- TR6 program reproduces on all wins: {summary['tr6_program_reproduces_all']}", "",
          "| theorem | static_coverage | replay | dynamic/static | next action |",
          "|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda x: x["dynamic_vs_static_class"]):
        md.append(f"| `{r['full_name']}` | {r['static_coverage_class']} | {r['replay_class']} | "
                  f"{r['dynamic_vs_static_class']} | {r['recommended_next_action']} |")
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[tr7-export] examples={len(rows)} dynstat={summary['dynamic_vs_static_histogram']}")
    print(f"[tr7-export] replay={summary['replay_histogram']}")


if __name__ == "__main__":
    main()
