#!/usr/bin/env python3
"""TR6 Part 6 — retrieve candidate lemmas for fresh confirmed RC2 failures.

Same deterministic combined scorer as TR3 (lexical TF-IDF cosine + namespace/path
proximity + feature overlap + name similarity), reusing the SF5 helpers so scoring is
identical. Uses the theorem's `statement_text` as the query goal (TR6 records carry the
signature, not a live goal). Emits top-20 lemmas + a goal-driven def-unfold channel.
Index = TR3 retrieval index (10,790 Mathlib decls) ∪ SF5 lemma index; only lemma NAMES
are used downstream (as `simp [L]` args), so the index's munged file_paths are irrelevant.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import sf5_retrieve_lemmas as SF5R  # noqa: E402
from sf5_build_targets import _features as feat_fn  # noqa: E402


def _p(*a):
    return os.path.join(_REPO, *a)


def _name_tokens(fn):
    return set(re.split(r"[._]", fn.lower())) | {fn.split(".")[-1].lower()}


def _load_index(paths):
    seen, index = set(), []
    for path in paths:
        fp = _p(path)
        if not os.path.exists(fp):
            continue
        for l in open(fp):
            if not l.strip():
                continue
            try:
                rec = json.loads(l)
            except json.JSONDecodeError:
                continue
            fn = rec.get("full_name")
            if not fn or fn in seen:
                continue
            seen.add(fn)
            index.append(rec)
    return index


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--indexes", nargs="*", default=[
        "project/evolve/experiments/tr3/out/tr3_retrieval_index.jsonl",
        "project/evolve/experiments/sf5/out/sf5_lemma_index.jsonl"])
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    conf = json.load(open(_p(args.confirmation)))
    failures = [r for r in conf["results"] if r["classification"] == "CONFIRMED_RC2_FAILURE"]
    index = _load_index(args.indexes)

    def_by_short = {}
    for rec in index:
        if rec.get("decl_kind") in ("def", "abbrev"):
            def_by_short.setdefault(rec["full_name"].split(".")[-1], []).append(rec["full_name"])

    def goal_defs(goal, ns):
        if not goal:
            return []
        idents = set(re.findall(r"[A-Za-z_][A-Za-z0-9_']+", goal))
        idents |= {m[1:] for m in re.findall(r"\.[A-Za-z_][A-Za-z0-9_']+", goal)}
        out, seen = [], set()
        for ident in idents:
            for full in def_by_short.get(ident, []):
                if full in seen:
                    continue
                seen.add(full)
                fns = full.rsplit(".", 1)[0] if "." in full else ""
                out.append((0 if (fns == "" or fns == ns) else 1, full))
        out.sort()
        return [f for _, f in out]

    df = Counter()
    lemma_tok, lemma_feat, name_tok = [], [], []
    for rec in index:
        toks = set(SF5R._doc_tokens(rec))
        lemma_tok.append(toks)
        lemma_feat.append(SF5R._feat_set(rec.get("features", {})))
        name_tok.append(_name_tokens(rec["full_name"]))
        for t in toks:
            df[t] += 1
    N = len(index)
    idf = {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def tfidf(tokens):
        tf = Counter(tokens)
        v = {t: (1 + math.log(c)) * idf.get(t, math.log(N + 1) + 1.0) for t, c in tf.items()}
        nrm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / nrm for t, x in v.items()}

    lemma_vecs = [tfidf(list(t)) for t in lemma_tok]
    W_TFIDF, W_NS, W_PATH, W_FEAT, W_NAME = 1.0, 0.35, 0.25, 0.40, 0.30

    results = []
    coverage_hits = 0
    for fr in failures:
        fn = fr["full_name"]
        goal = fr.get("statement_text") or fr.get("goal_text") or ""
        qv = tfidf([t for t in (SF5R._word_tokens(goal) + re.split(r"[._]", fn.lower()))
                    if len(t) >= 2])
        qfeat = SF5R._feat_set(feat_fn(goal or fn))
        qname = _name_tokens(fn)
        scored = []
        for i, rec in enumerate(index):
            if rec["full_name"] == fn:
                continue
            lv = lemma_vecs[i]
            cos = (sum(qv[t] * lv.get(t, 0.0) for t in qv) if len(qv) <= len(lv)
                   else sum(lv[t] * qv.get(t, 0.0) for t in lv))
            nsp = SF5R._ns_prox(fr.get("namespace"), rec.get("namespace"))
            pp = SF5R._path_prox(fr.get("file_path"), rec.get("file_path"))
            lf = lemma_feat[i]
            fj = (len(qfeat & lf) / len(qfeat | lf)) if (qfeat | lf) else 0.0
            nm = (len(qname & name_tok[i]) / len(qname | name_tok[i])) if (qname | name_tok[i]) else 0.0
            score = W_TFIDF * cos + W_NS * nsp + W_PATH * pp + W_FEAT * fj + W_NAME * nm
            if score <= 0:
                continue
            reasons = []
            if cos > 0.05:
                reasons.append(f"lex={cos:.2f}")
            if nsp:
                reasons.append(f"ns={nsp:.2f}")
            if pp:
                reasons.append(f"path={pp:.2f}")
            if fj:
                reasons.append(f"feat={fj:.2f}")
            if nm > 0.1:
                reasons.append(f"name={nm:.2f}")
            scored.append((score, rec, "; ".join(reasons)))
        scored.sort(key=lambda x: (-x[0], x[1]["full_name"]))
        top = scored[: args.top_k]
        top_lemmas = [{"lemma": rec["full_name"], "score": round(sc, 4), "reason": rsn,
                       "source": rec.get("source"), "decl_kind": rec.get("decl_kind"),
                       "statement_text": rec.get("statement_text")}
                      for sc, rec, rsn in top]
        gd = goal_defs(goal, fr.get("namespace"))
        if top_lemmas:
            coverage_hits += 1
        results.append({
            "target": fn, "namespace": fr.get("namespace"), "cluster_id": fr.get("cluster_id"),
            "goal_text": goal, "num_retrieved": len(top_lemmas),
            "top_lemmas": top_lemmas, "goal_defs": gd,
            "best_score": top_lemmas[0]["score"] if top_lemmas else 0.0,
        })

    avg_top = round(sum(r["best_score"] for r in results) / max(1, len(results)), 3)
    out = {"generated_by": "scripts/tr6_retrieve_lemmas.py",
           "index_size": N, "top_k": args.top_k, "num_targets": len(results),
           "coverage_targets_with_lemmas": coverage_hits,
           "avg_best_score": avg_top,
           "weights": {"tfidf": W_TFIDF, "ns": W_NS, "path": W_PATH, "feat": W_FEAT, "name": W_NAME},
           "results": results}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 retrieval results", "",
          f"- targets: {len(results)} | index {N} | top-{args.top_k} | coverage "
          f"{coverage_hits}/{len(results)} | avg best score {avg_top}", "",
          "## Per-target top-3 (first 25 targets)"]
    for r in results[:25]:
        md.append(f"### {r['target']}")
        for t in r["top_lemmas"][:3]:
            md.append(f"- `{t['lemma']}` ({t['score']}) — {t['reason']}")
        if r["goal_defs"]:
            md.append(f"- goal_defs: {r['goal_defs'][:4]}")
        md.append("")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-retrieve] {len(results)} targets, coverage {coverage_hits}, avg_best {avg_top}")


if __name__ == "__main__":
    main()
