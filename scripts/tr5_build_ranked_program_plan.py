#!/usr/bin/env python3
"""TR5 Part 4 — build the ranker-ordered live program plan.

For each confirmed-RC2-failure target, take the TR3-grammar candidate programs (reused
from tr3_depth_program_plan; same families incl. def_unfold_simp / d1_simp_lemma /
d2_simp_aesop / RC4B Set.disjoint_left bridge / RC4C simp[L]<;>aesop), attach the
retrieval rank/score for each program's lemmas, score every program with the TR4 HGB
ranker (full-data model, featurized identically to tr4_featurize_programs.py) plus the
heuristic, and emit a ranked list per theorem with budget cutoffs B1/B3/B5/B10/B20.
No live Lean here.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

sys_path = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, sys_path)
import tr5_score as S

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    fp = _p(path)
    return json.load(open(fp)) if os.path.exists(fp) else None


def _retrieval_map(retr):
    """target -> {lemma: (rank, score, source)}."""
    m = {}
    if not retr:
        return m
    for r in retr.get("results", []):
        d = {}
        for i, t in enumerate(r.get("top_lemmas", r.get("retrieved", []))):
            d[t["lemma"]] = (i, t.get("score"), t.get("source"))
        m[r.get("target")] = d
    return m


def _best_retrieval(lemmas, rmap):
    rank, score, src = None, None, None
    for L in (lemmas or []):
        if L in rmap:
            rk, sc, sr = rmap[L]
            if rank is None or rk < rank:
                rank, score, src = rk, sc, sr
    return rank, score, src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-pool", required=True)
    ap.add_argument("--confirmation", required=True)
    ap.add_argument("--tr4-model-dir", required=True)
    ap.add_argument("--tr4-vectorizers", required=True)
    ap.add_argument("--tr4-metadata", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--model", default="hgb")
    args = ap.parse_args()

    pool = {json.loads(l)["full_name"]: json.loads(l)
            for l in open(_p(args.target_pool)) if l.strip()}
    conf = {r["full_name"]: r for r in _load(args.confirmation)["results"]}
    plan = {t["full_name"]: t for t in
            _load("project/evolve/experiments/tr3/out/tr3_depth_program_plan.json")["theorems"]}
    rmap_all = _retrieval_map(_load("project/evolve/experiments/tr3/out/tr3_retrieval_results.json"))

    scorer = S.RankerScorer(
        os.path.join(args.tr4_vectorizers),
        os.path.join(args.tr4_model_dir, f"{args.model}_program_ranker.joblib"),
        os.path.join(args.tr4_model_dir, "heuristic_ranker.json"))

    BUDGETS = [1, 3, 5, 10, 20]
    theorems = []
    fam_hist = Counter()
    score_count = 0
    for fn, tgt in pool.items():
        c = conf.get(fn, {})
        rc2_status = c.get("classification", tgt.get("known_rc2_status"))
        t = plan.get(fn)
        goal = (t.get("goal_text") if t else None) or tgt.get("goal_text")
        ns = tgt.get("namespace")
        rmap = rmap_all.get(fn, {})
        progs = (t.get("programs", []) if t else [])
        # build scoring rows
        rows, raw = [], []
        seen_tac = set()
        for pgm in progs:
            tac = pgm.get("tactic")
            if not tac or tac in seen_tac:
                continue
            seen_tac.add(tac)
            lemmas = pgm.get("lemmas", [])
            rk, sc, src = _best_retrieval(lemmas, rmap)
            row = S.build_row(fn, goal, ns, tac, lemmas, pgm.get("family"),
                              pgm.get("depth", 1), retrieval_rank=rk, retrieval_score=sc,
                              lemma_source=src, source="tr5")
            rows.append(row)
            raw.append(pgm)
        if not rows:
            theorems.append({"full_name": fn, "namespace": ns, "rc2_status": rc2_status,
                             "target_category": tgt.get("target_category"),
                             "candidate_family_tags": tgt.get("candidate_family_tags"),
                             "num_programs": 0, "programs_ranked": []})
            continue
        hgb = scorer.score(rows)
        score_count += len(rows)
        ranked = []
        for i, (row, pgm) in enumerate(zip(rows, raw)):
            tags = []
            if any("disjoint_left" in (L or "") for L in pgm.get("lemmas", [])):
                tags.append("rc4b_set_disjoint_left")
            if pgm.get("family") == "d2_simp_aesop":
                tags.append("rc4c_d2_simp_aesop")
            if pgm.get("family") == "def_unfold_simp":
                tags.append("rc4a_def_unfold")
            ranked.append({
                "program_id": pgm.get("program_id"),
                "family": pgm.get("family"), "depth": pgm.get("depth", 1),
                "tactic": pgm.get("tactic"), "used_lemmas": pgm.get("lemmas", []),
                "ranker_score": round(float(hgb[i]), 6),
                "heuristic_score": scorer.heuristic_score(row),
                "retrieval_rank": row["retrieval_rank"],
                "candidate_family_tags": tags,
            })
        ranked.sort(key=lambda r: (-r["ranker_score"], r["program_id"] or ""))
        for rk, r in enumerate(ranked, 1):
            r["rank"] = rk
            fam_hist[r["family"]] += 1
        theorems.append({
            "full_name": fn, "namespace": ns, "rc2_status": rc2_status,
            "target_category": tgt.get("target_category"),
            "candidate_family_tags": tgt.get("candidate_family_tags"),
            "num_programs": len(ranked),
            "programs_ranked": ranked,
        })

    out = {
        "generated_by": "scripts/tr5_build_ranked_program_plan.py",
        "ranker_model": args.model, "budgets": BUDGETS,
        "num_theorems": len(theorems),
        "num_confirmed_failures": sum(1 for t in theorems if t["rc2_status"] == "CONFIRMED_RC2_FAILURE"),
        "total_programs_scored": score_count,
        "family_histogram": dict(fam_hist),
        "theorems": theorems,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    # budget program counts
    bcount = {b: sum(min(b, t["num_programs"]) for t in theorems) for b in BUDGETS}
    md = ["# TR5 ranked program plan", "",
          f"- ranker model: **{args.model}** | theorems: {len(theorems)} | "
          f"confirmed failures: {out['num_confirmed_failures']}",
          f"- total programs scored: {score_count}",
          f"- family histogram: {dict(fam_hist)}",
          f"- programs to run per budget: {bcount}", "",
          "## Top program per theorem (rank 1)", "",
          "| theorem | ns | rank1 family | rank1 score | rank1 tactic |",
          "|---|---|---|---|---|"]
    for t in sorted(theorems, key=lambda x: -(x["programs_ranked"][0]["ranker_score"] if x["programs_ranked"] else -1))[:25]:
        if not t["programs_ranked"]:
            continue
        r1 = t["programs_ranked"][0]
        md.append(f"| `{t['full_name']}` | {t['namespace']} | {r1['family']} | "
                  f"{r1['ranker_score']} | `{r1['tactic'][:55]}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr5-plan] {len(theorems)} theorems, {score_count} programs scored; "
          f"budget program counts={bcount}")
    print(f"  family_histogram={dict(fam_hist)}")


if __name__ == "__main__":
    main()
