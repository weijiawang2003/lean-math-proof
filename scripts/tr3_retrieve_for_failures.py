#!/usr/bin/env python3
"""TR3 Part 5 — retrieve candidate lemmas for confirmed RC2 failures.

For each CONFIRMED_RC2_FAILURE, rank top-20 lemmas by a deterministic combined score:
lexical TF-IDF cosine + namespace/path proximity + feature overlap + name-pattern
similarity. Injects SF5 winning lemmas for the relevant targets, and emits a
goal-driven definition channel (defs named in the goal) for the def-unfold programs.
Reuses SF5 retrieval helpers so scoring matches SF5.
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


def _p(*a):
    return os.path.join(_REPO, *a)


def _name_tokens(fn):
    return set(re.split(r"[._]", fn.lower())) | {fn.split(".")[-1].lower()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirmation", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--sf5-attribution",
                    default="project/evolve/experiments/sf5/out/sf5_retrieval_attribution.json")
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    conf = json.load(open(_p(args.confirmation)))
    failures = [r for r in conf["results"] if r["classification"] == "CONFIRMED_RC2_FAILURE"]
    index = [json.loads(l) for l in open(_p(args.index)) if l.strip()]

    # SF5 winning lemmas to inject
    sf5_win = {}
    if os.path.exists(_p(args.sf5_attribution)):
        for r in json.load(open(_p(args.sf5_attribution))).get("records", []):
            if r.get("winning_lemma"):
                sf5_win[r["full_name"]] = r["winning_lemma"]

    # def short-name map (for goal-driven unfolds)
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

    # IDF
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
    target_retrieved = {}
    for fr in failures:
        fn = fr["full_name"]
        goal = fr.get("goal_text") or ""
        qv = tfidf([t for t in (SF5R._word_tokens(goal) + re.split(r"[._]", fn.lower()))
                    if len(t) >= 2])
        # query feature flags from goal text (reuse SF5 target feature extractor)
        from sf5_build_targets import _features as feat_fn
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
                       "statement_text": rec.get("statement_text"),
                       "file_path": rec.get("file_path")}
                      for sc, rec, rsn in top]
        # inject SF5 winning lemma at the front if present and not already there
        win = sf5_win.get(fn)
        if win and "+" not in win and not any(t["lemma"] == win for t in top_lemmas):
            top_lemmas.insert(0, {"lemma": win, "score": 99.0,
                                  "reason": "sf5_winning_lemma", "source": "sf5",
                                  "decl_kind": "theorem", "statement_text": None,
                                  "file_path": None})
            top_lemmas = top_lemmas[: args.top_k]
        results.append({
            "target": fn, "namespace": fr.get("namespace"), "cluster_id": fr.get("cluster_id"),
            "goal_text": goal, "num_retrieved": len(top_lemmas),
            "top_lemmas": top_lemmas,
            "goal_defs": goal_defs(goal, fr.get("namespace")),
            "sf5_winning_lemma": win,
        })
        target_retrieved[fn] = {t["lemma"] for t in top_lemmas}

    # cluster-shared
    by_cluster = defaultdict(list)
    for fr in failures:
        by_cluster[fr.get("cluster_id")].append(fr["full_name"])
    cluster_shared = {}
    for cid, members in by_cluster.items():
        cnt = Counter()
        for m in members:
            for l in target_retrieved.get(m, ()):
                cnt[l] += 1
        cluster_shared[cid] = {"size": len(members),
                               "shared_retrieved_lemmas": [{"lemma": l, "appears_in_targets": c}
                                                           for l, c in cnt.most_common(15) if c >= 2]}

    out = {"generated_by": "scripts/tr3_retrieve_for_failures.py",
           "index_size": N, "top_k": args.top_k, "num_targets": len(results),
           "weights": {"tfidf": W_TFIDF, "ns": W_NS, "path": W_PATH, "feat": W_FEAT, "name": W_NAME},
           "results": results, "cluster_shared_lemmas": cluster_shared}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR3 retrieval results", "",
          f"- confirmed-failure targets: {len(results)} | index {N} | top-{args.top_k}", "",
          "## Per-target top-5"]
    for r in results:
        md.append(f"### {r['target']}")
        for t in r["top_lemmas"][:5]:
            md.append(f"- `{t['lemma']}` ({t['score']}) — {t['reason']}")
        md.append("")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr3-retrieve] {len(results)} targets, top-{args.top_k} each")


if __name__ == "__main__":
    main()
