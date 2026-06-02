#!/usr/bin/env python3
"""FLI1 Part 6 — check whether each candidate lemma already exists.

Compares the candidate residual-goal against (a) the FLI0 retrieved lemma statements for the same
seed, (b) the discovered_theorems catalog names, and (c) name/token heuristics. Classifies
EXISTS_EXACT / EXISTS_CLOSE / PROBABLY_NEW / TOO_VAGUE_TO_CHECK / ILL_TYPED_STATEMENT /
NEEDS_REVIEW. If a close lemma was ALREADY in the seed's retrieved list but the search still failed
→ flags RETRIEVAL_GAP (a valuable discovery: the bridge exists, the searcher didn't use it).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = "project/evolve/experiments/fli0/cases/fli0_seed_cases.json"
_TOK = re.compile(r"[A-Za-z][A-Za-z0-9']+|[∈⊆↔→∃∀=≤]")
_STOP = {"Type", "by", "fun", "the", "and"}


def _p(*a):
    return os.path.join(_REPO, *a)


def _toks(s):
    return {t for t in _TOK.findall(s or "") if t not in _STOP and len(t) > 1 or t in "∈⊆↔→∃∀=≤"}


def _jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--catalog", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    cands = [json.loads(l) for l in open(_p(args.candidates)) if l.strip()]
    seeds = {s["seed_id"]: s for s in json.load(open(_p(SEEDS))).get("seeds", [])}
    # FLI0 seeds store top_retrieved_lemmas as names only; pull detailed statements from enriched
    enr = {}
    ep = _p("project/evolve/experiments/fli0/cases/fli0_failed_cases_enriched.jsonl")
    if os.path.exists(ep):
        for l in open(ep):
            r = json.loads(l)
            enr[r["theorem"]] = r.get("top_retrieved_lemmas_detailed", [])
    catalog = json.load(open(_p(args.catalog))) if os.path.exists(_p(args.catalog)) else {}
    cat_names = {t["full_name"] for t in catalog.get("theorems", [])}

    out = []
    for c in cands:
        goal = c.get("lemma_goal", "")
        # compare on the FULL statement (binders carry the content; the goal alone can look thin
        # because single-letter locals like s/t are not multi-char tokens).
        full = (c.get("lemma_binders", "") + " " + goal).strip()
        gtok = _toks(full)
        has_rel = any(r in goal for r in ("∈", "⊆", "↔", "→", "=", "≤", "<", "∃", "∀", "Disjoint"))
        sid = c["source_seed_ids"][0]
        thm = (c.get("downstream_targets") or [None])[0]
        retrieved = enr.get(thm, [])
        best = {"lemma": None, "score": 0.0, "statement": None}
        for tl in retrieved:
            sc = _jaccard(gtok, _toks(tl.get("statement_text", "")))
            if sc > best["score"]:
                best = {"lemma": tl.get("lemma"), "score": round(sc, 3),
                        "statement": tl.get("statement_text")}
        # classify
        if not goal or "?m" in goal or "sorry" in goal:
            cls = "ILL_TYPED_STATEMENT"
        elif not has_rel and len(gtok) < 3:
            cls = "TOO_VAGUE_TO_CHECK"
        elif best["score"] >= 0.85:
            cls = "EXISTS_EXACT"
        elif best["score"] >= 0.45:
            cls = "EXISTS_CLOSE"
        else:
            cls = "PROBABLY_NEW"
        # retrieval gap: a close/exact lemma was in the seed's retrieved set yet search failed
        retrieval_gap = cls in ("EXISTS_EXACT", "EXISTS_CLOSE") and best["lemma"] is not None
        rec = dict(c)
        rec.update({
            "existing_check": cls,
            "closest_existing_lemma": best["lemma"],
            "closest_existing_statement": best["statement"],
            "closest_overlap_score": best["score"],
            "retrieval_gap": retrieval_gap,
            "why_close": (f"residual goal shares {best['score']} token-jaccard with retrieved "
                          f"`{best['lemma']}`" if best["lemma"] else "no close retrieved lemma"),
            "might_rescue_downstream": retrieval_gap,
        })
        out.append(rec)

    with open(_p(args.out_jsonl), "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    hist = Counter(r["existing_check"] for r in out)
    summary = {"generated_by": "scripts/fli1_check_existing_lemmas.py",
               "num_candidates": len(out), "classification_histogram": dict(hist),
               "retrieval_gaps": sum(1 for r in out if r["retrieval_gap"]),
               "probably_new": hist.get("PROBABLY_NEW", 0),
               "retrieval_gap_targets": sorted({r["downstream_targets"][0] for r in out
                                                if r["retrieval_gap"]})}
    with open(_p(args.out_summary_json), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    md = ["# FLI1 existing-lemma check summary", "",
          f"- candidates: {summary['num_candidates']} | classes: {summary['classification_histogram']}",
          f"- **retrieval gaps (bridge exists, search didn't use): {summary['retrieval_gaps']}**",
          f"- probably new: {summary['probably_new']}", "",
          "| id | seed | class | closest existing | score | retr-gap |",
          "|---|---|---|---|---|---|"]
    for r in out:
        md.append(f"| {r['candidate_id']} | {r['source_seed_ids'][0]} | {r['existing_check']} | "
                  f"`{r['closest_existing_lemma']}` | {r['closest_overlap_score']} | "
                  f"{r['retrieval_gap']} |")
    with open(_p(args.out_summary_md), "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"[fli1-existing] candidates={len(out)} classes={dict(hist)} "
          f"retrieval_gaps={summary['retrieval_gaps']}")


if __name__ == "__main__":
    main()
