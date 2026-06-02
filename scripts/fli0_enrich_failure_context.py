#!/usr/bin/env python3
"""FLI0 Part 4 — enrich failed cases with statement / source / retrieval context.

Adds, where available (no live Lean, no OCR): theorem statement, file path, difficulty +
num_tactics (from discovered_theorems catalog), name tokens, involved constants/definitions
(from retrieval goal_defs + statement parse), top retrieved lemmas with statement text, a similar
SOLVED theorem in the same namespace (from the dynamic success sets), the last error message, and
the failed-tactic trace. Residual goal states are absent from all artifacts → residual_goal_status
is set to MISSING (handed to FLI1 for optional live capture).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V2 = "project/evolve/experiments/rc5_v2"
V3 = "project/evolve/experiments/rc5_v3"
# constants worth flagging if they appear in a statement (membership/structure operators)
_CONST_TOKENS = ["Disjoint", "insert", "image", "preimage", "biUnion", "iUnion", "sUnion",
                 "biInter", "iInter", "filter", "map", "bind", "powerset", "toFinset", "toList",
                 "singleton", "Nonempty", "Subset", "Finset", "Multiset", "List", "Set"]
_NAME_TOK = re.compile(r"[A-Za-z][A-Za-z0-9]*")


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    p = _p(path)
    return json.load(open(p)) if os.path.exists(p) else None


def _retr_index(stage_root, fname):
    d = _load(f"{stage_root}/out/{fname}")
    return {r["target"]: r for r in (d or {}).get("results", [])}


def _success_by_ns(paths):
    """Map namespace -> [solved theorem names] from dynamic success_targets across stages."""
    by_ns = {}
    elig = {}
    for root, elig_rel in paths:
        e = _load(f"{root}/{elig_rel}")
        for r in (e or {}).get("results", []):
            elig[r["full_name"]] = r.get("namespace")
    return elig


def _name_tokens(full_name):
    short = full_name.split(".")[-1]
    return [t for t in _NAME_TOK.findall(short) if len(t) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--failed-cases", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(_p(args.failed_cases)) if l.strip()]
    cat = _load(args.catalog) or {}
    cat_map = {t["full_name"]: t for t in cat.get("theorems", [])}

    retr = {"RC5V2": _retr_index(V2, "rc5v2_retrieval_results.json"),
            "RC5V3": _retr_index(V3, "rc5v3_retrieval_results.json")}

    # solved theorems per namespace (for "similar successful theorem")
    solved_by_ns = {}
    for root, b in [(V2, "out/rc5v2_b5_dynamic_results.json"),
                    (V3, "out/rc5v3_b1_dynamic_results.json"),
                    (V3, "out/rc5v3_b5_dynamic_results.json")]:
        d = _load(f"{root}/{b}")
        for fn in (d or {}).get("success_targets", []):
            ns = fn.split(".")[0]
            solved_by_ns.setdefault(ns, set()).add(fn)
    # also from static new_over_rc2 (RC4 wins) as solved exemplars
    for root, s in [(V2, "out/rc5v2_static_stage_results.json"),
                    (V3, "out/rc5v3_static_stage_results.json")]:
        d = _load(f"{root}/{s}")
        for fn in (d or {}).get("new_over_rc2", []):
            solved_by_ns.setdefault(fn.split(".")[0], set()).add(fn)

    enriched = []
    have_stmt = have_retr = have_cat = have_similar = 0
    for c in cases:
        fn = c["theorem"]
        ns = c.get("namespace")
        rrec = retr.get(c["source_stage"], {}).get(fn, {})
        stmt = c.get("statement_text") or rrec.get("goal_text")
        top_lemmas = rrec.get("top_lemmas", []) or []
        goal_defs = rrec.get("goal_defs", []) or []
        catrec = cat_map.get(fn, {})
        involved = sorted({tok for tok in _CONST_TOKENS if stmt and tok in stmt})
        # last error / failed-tactic trace
        last_error = None
        for e in c.get("errors", []):
            if e.get("error"):
                last_error = e["error"]
                break
            if e.get("setup_error"):
                last_error = "setup_error: " + e["setup_error"]
        similar = None
        pool = solved_by_ns.get(ns) or solved_by_ns.get((ns or "").split(".")[0]) or set()
        # sort first so max() tie-breaks deterministically (set iteration order varies with
        # PYTHONHASHSEED across processes).
        pool = sorted(s for s in pool if s != fn)
        if pool:
            toks = set(_name_tokens(fn))
            similar = max(pool, key=lambda s: len(toks & set(_name_tokens(s))))
            if not (toks & set(_name_tokens(similar))):
                similar = pool[0]
        e = dict(c)
        e.update({
            "statement": stmt,
            "statement_status": "PRESENT" if stmt else "MISSING",
            "file_path": c.get("file_path") or catrec.get("file_path"),
            "difficulty": catrec.get("difficulty"),
            "difficulty_score": catrec.get("difficulty_score"),
            "catalog_num_tactics": catrec.get("num_tactics"),
            "name_tokens": _name_tokens(fn),
            "involved_constants": involved,
            "involved_definitions": goal_defs,
            "top_retrieved_lemmas_detailed": [
                {"lemma": tl.get("lemma"), "score": tl.get("score"),
                 "statement_text": tl.get("statement_text"), "decl_kind": tl.get("decl_kind")}
                for tl in top_lemmas[:5]],
            "best_retrieval_score": rrec.get("best_score"),
            "similar_solved_theorem": similar,
            "residual_goal_status": "MISSING",
            "last_error": last_error,
            "failed_tactic_trace": c.get("attempted_tactics", []),
        })
        enriched.append(e)
        have_stmt += bool(stmt)
        have_retr += bool(top_lemmas)
        have_cat += bool(catrec)
        have_similar += bool(similar)

    with open(_p(args.out_jsonl), "w") as f:
        for r in enriched:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n = len(enriched) or 1
    summary = {"generated_by": "scripts/fli0_enrich_failure_context.py",
               "num_cases": len(enriched),
               "with_statement": have_stmt, "statement_coverage": round(have_stmt / n, 3),
               "with_retrieved_lemmas": have_retr, "retrieval_coverage": round(have_retr / n, 3),
               "with_catalog_difficulty": have_cat,
               "with_similar_solved_theorem": have_similar,
               "residual_goal_status": "MISSING for all (no per-tactic goal states in artifacts)",
               "involved_constant_histogram":
                   dict(Counter(t for r in enriched for t in r["involved_constants"]).most_common(15))}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI0 context enrichment summary", "",
          f"- cases: {summary['num_cases']}",
          f"- statement coverage: {summary['with_statement']}/{summary['num_cases']} "
          f"({summary['statement_coverage']})",
          f"- retrieved-lemma coverage: {summary['with_retrieved_lemmas']}/{summary['num_cases']} "
          f"({summary['retrieval_coverage']})",
          f"- catalog difficulty present: {summary['with_catalog_difficulty']}",
          f"- similar solved theorem found: {summary['with_similar_solved_theorem']}",
          f"- **residual goal: {summary['residual_goal_status']}**", "",
          "## Most-involved constants", "", "| constant | count |", "|---|---|"]
    for k, v in summary["involved_constant_histogram"].items():
        md.append(f"| {k} | {v} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli0-enrich] cases={len(enriched)} stmt={have_stmt} retr={have_retr} "
          f"cat={have_cat} similar={have_similar}")


if __name__ == "__main__":
    main()
