#!/usr/bin/env python3
"""TR6 Part 7 — generate ranker-scored candidate programs for fresh RC2 failures.

Reuses the TR3/TR5 program grammar (depth-1/2/3 retrieval-aware + def-unfold + depth-only
controls; helpers imported from tr3_generate_depth_programs) and adds RC4B
(`simp [Set.disjoint_left]` / `... <;> aesop`) probes on disjoint-shaped Set/Finset/List
goals. Up to 100 raw programs/theorem are scored with the TR4 HGB ranker (featurized via
tr5_score, identical to tr4_featurize_programs), the top 20 kept, ranks + B5/B10/B20 budget
tags assigned. No live Lean here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr5_score as S
from tr3_generate_depth_programs import _shape, _malformed, _rw_safe

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _gen_programs(fn, goal, namespace, features, ret, max_lemmas=10, max_programs=100):
    shape = _shape(goal, namespace, features or {})
    programs, seen = [], set()

    def add(tactic, family, depth, lemmas, tags=()):
        if tactic in seen or len(programs) >= max_programs:
            return
        seen.add(tactic)
        programs.append({"program_id": f"{fn}::{len(programs):02d}", "target": fn,
                         "family": family, "depth": depth, "lemmas": list(lemmas),
                         "tactic": tactic, "candidate_family_tags": list(tags)})

    lemmas = [t for t in ret.get("top_lemmas", [])
              if not _malformed(t["lemma"]) and t.get("decl_kind") != "def"
              and t["lemma"] != fn][:max_lemmas]
    goal_defs = [d for d in ret.get("goal_defs", []) if not _malformed(d) and d != fn][:4]

    # (1) def-unfold (RC4A family)
    if goal_defs:
        add("simp [" + ", ".join(goal_defs) + "]", "def_unfold_simp", 1, goal_defs,
            ("rc4a_def_unfold",))
    # (2) depth-1 retrieval
    for t in lemmas[:6]:
        L = t["lemma"]
        add(f"exact {L}", "d1_exact", 1, [L])
        add(f"simpa using {L}", "d1_simpa_using", 1, [L])
        add(f"simp [{L}]", "d1_simp_lemma", 1, [L])
        add(f"simpa [{L}]", "d1_simpa_lemma", 1, [L])
        if _rw_safe(t.get("statement_text")):
            add(f"rw [{L}]", "d1_rw_lemma", 1, [L])
    # (3) RC4B candidate probes (disjoint-shaped)
    if (features or {}).get("has_disjoint") or "disjoint" in (goal or "").lower():
        add("simp [Set.disjoint_left]", "d1_simp_lemma", 1, ["Set.disjoint_left"],
            ("rc4b_set_disjoint_left",))
        add("simp [Set.disjoint_left] <;> aesop", "d2_simp_aesop", 2, ["Set.disjoint_left"],
            ("rc4b_set_disjoint_left", "rc4c_d2_simp_aesop"))
    # (4) depth-only controls (gated)
    if shape["set_eq"]:
        add("ext x <;> aesop", "d2_ext_aesop", 2, [])
        add("apply Set.Subset.antisymm <;> intro x <;> aesop", "d3_antisymm_aesop", 3, [])
    if shape["iff"]:
        add("constructor <;> intro h <;> aesop", "d3_constructor_aesop", 3, [])
        add("constructor <;> intro h <;> simp_all", "d3_constructor_simp_all", 3, [])
    if shape["nat"]:
        add("omega", "d1_omega", 1, [])
        add("nlinarith", "d1_nlinarith", 1, [])
    if shape["multiset_tofinset"]:
        add("simp [Multiset.toFinset, Multiset.mem_toFinset]", "d1_tofinset_simp", 1, [])
    add("aesop", "d1_aesop", 1, [])
    add("simp_all", "d1_simp_all", 1, [])
    add("tauto", "d1_tauto", 1, [])
    # (5) depth-2 retrieval-aware (RC4C family = d2_simp_aesop)
    for t in lemmas[:3]:
        L = t["lemma"]
        add(f"simp [{L}] <;> aesop", "d2_simp_aesop", 2, [L], ("rc4c_d2_simp_aesop",))
        add(f"simp [{L}] <;> simp_all", "d2_simp_simpall", 2, [L])
        if _rw_safe(t.get("statement_text")):
            add(f"rw [{L}] <;> aesop", "d2_rw_aesop", 2, [L])
            add(f"rw [{L}] <;> simp_all", "d2_rw_simpall", 2, [L])
        if shape["iff"]:
            add(f"constructor <;> intro h <;> simpa using {L}", "d2_constructor_simpa", 2, [L])
        if shape["set_eq"]:
            add(f"ext x <;> simp [{L}]", "d2_ext_simp", 2, [L])
    # (6) depth-3 conservative
    for t in lemmas[:2]:
        L = t["lemma"]
        if shape["set_eq"]:
            add(f"ext x <;> simp [{L}] <;> aesop", "d3_ext_simp_aesop", 3, [L])
        add(f"simp [{L}] <;> try aesop <;> try simp_all", "d3_simp_try", 3, [L])
    return programs, shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation", required=True)
    ap.add_argument("--retrieval", required=True)
    ap.add_argument("--tr4-model-dir", required=True)
    ap.add_argument("--tr4-vectorizers", required=True)
    ap.add_argument("--tr4-metadata", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--model", default="hgb")
    ap.add_argument("--keep-top", type=int, default=20)
    args = ap.parse_args()

    conf = json.load(open(_p(args.confirmation)))
    failures = [r for r in conf["results"] if r["classification"] == "CONFIRMED_RC2_FAILURE"]
    ret = {r["target"]: r for r in json.load(open(_p(args.retrieval)))["results"]}
    scorer = S.RankerScorer(args.tr4_vectorizers,
                            os.path.join(args.tr4_model_dir, f"{args.model}_program_ranker.joblib"),
                            os.path.join(args.tr4_model_dir, "heuristic_ranker.json"))
    BUDGETS = [1, 3, 5, 10, 20]

    theorems = []
    fam_hist = Counter()
    scored_total = 0
    for fr in failures:
        fn = fr["full_name"]
        goal = fr.get("statement_text") or ""
        ns = fr.get("namespace")
        r = ret.get(fn, {})
        progs, shape = _gen_programs(fn, goal, ns, fr.get("features"), r)
        # retrieval rank/score per lemma
        rmap = {t["lemma"]: (i, t.get("score")) for i, t in enumerate(r.get("top_lemmas", []))}
        rows = []
        for p in progs:
            rk, sc = None, None
            for L in p["lemmas"]:
                if L in rmap:
                    if rk is None or rmap[L][0] < rk:
                        rk, sc = rmap[L]
            rows.append(S.build_row(fn, goal, ns, p["tactic"], p["lemmas"], p["family"],
                                    p["depth"], retrieval_rank=rk, retrieval_score=sc, source="tr6"))
        hgb = scorer.score(rows) if rows else []
        scored_total += len(rows)
        ranked = []
        for i, (p, row) in enumerate(zip(progs, rows)):
            ranked.append({**p, "ranker_score": round(float(hgb[i]), 6),
                           "heuristic_score": scorer.heuristic_score(row),
                           "retrieval_rank": row["retrieval_rank"]})
        ranked.sort(key=lambda x: (-x["ranker_score"], x["program_id"]))
        ranked = ranked[: args.keep_top]
        for rk, p in enumerate(ranked, 1):
            p["rank"] = rk
            fam_hist[p["family"]] += 1
        theorems.append({"full_name": fn, "namespace": ns, "file_path": fr.get("file_path"),
                         "rc2_status": "CONFIRMED_RC2_FAILURE",
                         "candidate_family_tags": fr.get("candidate_family_tags", []),
                         "shape": [k for k, v in shape.items() if v],
                         "num_programs": len(ranked), "programs_ranked": ranked})

    out = {"generated_by": "scripts/tr6_generate_ranked_programs.py",
           "ranker_model": args.model, "budgets": BUDGETS,
           "num_theorems": len(theorems), "total_programs_scored": scored_total,
           "family_histogram": dict(fam_hist), "theorems": theorems}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    bcount = {b: sum(min(b, t["num_programs"]) for t in theorems) for b in BUDGETS}
    md = ["# TR6 ranked program plan", "",
          f"- ranker: {args.model} | theorems: {len(theorems)} | scored {scored_total} programs",
          f"- family histogram: {dict(fam_hist)}",
          f"- programs per budget: {bcount}", "",
          "## rank-1 program per theorem (first 25)", "",
          "| theorem | ns | rank1 family | score | tactic |", "|---|---|---|---|---|"]
    for t in theorems[:25]:
        if t["programs_ranked"]:
            r1 = t["programs_ranked"][0]
            md.append(f"| `{t['full_name']}` | {t['namespace']} | {r1['family']} | "
                      f"{r1['ranker_score']} | `{r1['tactic'][:45]}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-plan] {len(theorems)} theorems, {scored_total} scored; budget counts={bcount}")
    print(f"  families={dict(fam_hist)}")


if __name__ == "__main__":
    main()
