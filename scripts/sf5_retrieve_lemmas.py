#!/usr/bin/env python3
"""SF5 Part 4 — retrieve candidate lemmas per target.

Three combined signals (deterministic, pure-python, no sklearn dependency):
  1. lexical TF-IDF cosine over {name tokens + statement word tokens}
  2. namespace / file-path proximity
  3. feature overlap (Jaccard over Set/iff/monotone/strictmono/subset/compl/
     singleton/insert/ssubset/ite/pair/pairwiseDisjoint/union/empty flags)

Top-20 candidates per target (the target's own declaration is excluded). Targets
are then grouped by shared retrieved lemmas.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_SYMBOL_WORDS = {
    "↔": "iff", "⊆": "subset", "⊂": "ssubset", "∪": "union", "∩": "inter",
    "∅": "empty", "ᶜ": "compl", "∈": "mem", "∉": "notmem", "\\": "sdiff",
    "¬": "not", "∧": "and", "∨": "or", "→": "imp",
}
FEATURE_KEYS = ("has_iff", "has_subset", "has_ssubset", "has_monotone",
                "has_strictmono", "has_set", "has_singleton", "has_insert",
                "has_compl", "has_pair", "has_ite", "has_union", "has_empty")


def _p(*a):
    return os.path.join(_REPO, *a)


def _word_tokens(text):
    if not text:
        return []
    t = text
    for sym, w in _SYMBOL_WORDS.items():
        t = t.replace(sym, f" {w} ")
    toks = re.findall(r"[A-Za-z][A-Za-z0-9_']+", t.lower())
    # also split camelCase within tokens
    out = []
    for tok in toks:
        out.append(tok)
        for piece in re.findall(r"[A-Z]?[a-z0-9]+", tok):
            if len(piece) >= 2 and piece != tok:
                out.append(piece)
    return [w for w in out if len(w) >= 2]


def _doc_tokens(rec):
    toks = list(rec.get("name_tokens", []))
    toks += _word_tokens(rec.get("statement_text") or "")
    return toks


def _feat_set(features):
    return {k for k in FEATURE_KEYS if features.get(k)}


def _path_prox(a, b):
    if not a or not b:
        return 0.0
    pa, pb = a.split("/"), b.split("/")
    common = 0
    for x, y in zip(pa, pb):
        if x == y:
            common += 1
        else:
            break
    return common / max(len(pa), len(pb))


def _ns_prox(a, b):
    if not a or not b:
        return 0.0
    pa, pb = a.split("."), b.split(".")
    common = 0
    for x, y in zip(pa, pb):
        if x == y:
            common += 1
        else:
            break
    return common / max(len(pa), len(pb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    targets = json.load(open(_p(args.targets)))
    index = [json.loads(l) for l in open(_p(args.index)) if l.strip()]

    # map def/abbrev short-names -> full-names (for goal-driven definitional unfolds)
    def_by_short = {}
    for rec in index:
        if rec.get("decl_kind") in ("def", "abbrev"):
            short = rec["full_name"].split(".")[-1]
            def_by_short.setdefault(short, []).append(rec["full_name"])

    def goal_defs(goal, ns):
        """Indexed defs whose name literally appears in the goal (CamelCase ident or
        `.proj` notation). These are exactly the definitions to unfold."""
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
                # prefer root-namespace or same-namespace defs (e.g. Monotone, Set.ite)
                fns = full.rsplit(".", 1)[0] if "." in full else ""
                pref = 0 if (fns == "" or fns == ns) else 1
                out.append((pref, full))
        out.sort()
        return [f for _, f in out]

    # build IDF over the lemma index
    df = Counter()
    lemma_tok = []
    for rec in index:
        toks = set(_doc_tokens(rec))
        lemma_tok.append(toks)
        for t in toks:
            df[t] += 1
    N = len(index)
    idf = {t: math.log((N + 1) / (c + 1)) + 1.0 for t, c in df.items()}

    def tfidf_vec(tokens):
        tf = Counter(tokens)
        v = {t: (1 + math.log(c)) * idf.get(t, math.log(N + 1) + 1.0)
             for t, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {t: x / norm for t, x in v.items()}

    lemma_vecs = [tfidf_vec(list(toks)) for toks in lemma_tok]
    lemma_feats = [_feat_set(rec.get("features", {})) for rec in index]

    W_TFIDF, W_NS, W_PATH, W_FEAT = 1.0, 0.35, 0.25, 0.40

    results = []
    target_retrieved = {}
    for tg in targets:
        q_text = (tg.get("goal_text") or "") + " " + tg["full_name"]
        q_tokens = _word_tokens(tg.get("goal_text") or "") + \
            re.split(r"[._]", tg["full_name"].lower())
        qv = tfidf_vec([t for t in q_tokens if len(t) >= 2])
        qfeat = _feat_set(tg.get("features_extended") or tg.get("features", {}))
        scored = []
        for i, rec in enumerate(index):
            if rec["full_name"] == tg["full_name"]:
                continue  # never retrieve the target itself
            lv = lemma_vecs[i]
            # cosine
            if len(qv) <= len(lv):
                cos = sum(qv[t] * lv.get(t, 0.0) for t in qv)
            else:
                cos = sum(lv[t] * qv.get(t, 0.0) for t in lv)
            nsp = _ns_prox(tg.get("namespace"), rec.get("namespace"))
            pp = _path_prox(tg.get("file_path"), rec.get("file_path"))
            lf = lemma_feats[i]
            fj = (len(qfeat & lf) / len(qfeat | lf)) if (qfeat | lf) else 0.0
            score = W_TFIDF * cos + W_NS * nsp + W_PATH * pp + W_FEAT * fj
            if score <= 0:
                continue
            reasons = []
            if cos > 0.05:
                reasons.append(f"lexical={cos:.2f}")
            if nsp > 0:
                reasons.append(f"ns={nsp:.2f}")
            if pp > 0:
                reasons.append(f"path={pp:.2f}")
            if fj > 0:
                reasons.append(f"feat={fj:.2f}")
            scored.append((score, rec, "; ".join(reasons)))
        scored.sort(key=lambda x: (-x[0], x[1]["full_name"]))
        top = scored[: args.top_k]
        # separate channel: top definition/abbrev candidates (for def-unfold probes).
        # Defs lose the namespace/path-proximity bonus that Set.* lemmas enjoy, so they
        # rarely make the lexical top-k even when their name strongly matches the goal.
        top_defs = [{
            "lemma": rec["full_name"], "score": round(sc, 4),
            "decl_kind": rec.get("decl_kind"), "reason": reason,
        } for sc, rec, reason in scored
            if rec.get("decl_kind") in ("def", "abbrev")][:5]
        retrieved = [{
            "lemma": rec["full_name"],
            "score": round(sc, 4),
            "reason": reason,
            "source": rec["source"],
            "decl_kind": rec.get("decl_kind", "theorem"),
            "statement_text": rec.get("statement_text"),
            "file_path": rec.get("file_path"),
        } for sc, rec, reason in top]
        results.append({
            "target": tg["full_name"],
            "cluster_id": tg.get("cluster_id"),
            "namespace": tg.get("namespace"),
            "goal_text": tg.get("goal_text"),
            "num_retrieved": len(retrieved),
            "retrieved": retrieved,
            "goal_defs": goal_defs(tg.get("goal_text"), tg.get("namespace")),
            "retrieved_defs": top_defs,
        })
        target_retrieved[tg["full_name"]] = {r["lemma"] for r in retrieved}

    # cluster grouping by shared retrieved lemmas
    by_cluster = defaultdict(list)
    for tg in targets:
        by_cluster[tg.get("cluster_id")].append(tg["full_name"])
    cluster_shared = {}
    for cid, members in by_cluster.items():
        counter = Counter()
        for m in members:
            for lemma in target_retrieved.get(m, ()):
                counter[lemma] += 1
        shared = [{"lemma": l, "appears_in_targets": c}
                  for l, c in counter.most_common(15) if c >= 2]
        cluster_shared[cid] = {
            "size": len(members),
            "members": sorted(members),
            "shared_retrieved_lemmas": shared,
        }

    out = {
        "generated_by": "scripts/sf5_retrieve_lemmas.py",
        "index_size": N,
        "top_k": args.top_k,
        "weights": {"tfidf": W_TFIDF, "namespace": W_NS, "path": W_PATH, "feature": W_FEAT},
        "num_targets": len(results),
        "results": results,
        "cluster_shared_lemmas": cluster_shared,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# SF5 retrieval results", "",
          f"- targets: {len(results)} | index: {N} lemmas | top-k: {args.top_k}", ""]
    for cid, info in sorted(cluster_shared.items(), key=lambda kv: -kv[1]["size"]):
        md.append(f"## Cluster `{cid}` (size {info['size']})")
        if info["shared_retrieved_lemmas"]:
            md.append("Shared retrieved lemmas (≥2 targets):")
            for s in info["shared_retrieved_lemmas"]:
                md.append(f"- `{s['lemma']}` — in {s['appears_in_targets']} targets")
        else:
            md.append("_no lemma retrieved for ≥2 targets_")
        md.append("")
    md.append("## Per-target top-5")
    md.append("")
    for r in results:
        md.append(f"### {r['target']}")
        for c in r["retrieved"][:5]:
            md.append(f"- `{c['lemma']}` ({c['score']}) — {c['reason']}")
        md.append("")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")

    print(f"[sf5-retrieve] {len(results)} targets, top-{args.top_k} each")
    for cid, info in sorted(cluster_shared.items(), key=lambda kv: -kv[1]["size"]):
        ns = len(info["shared_retrieved_lemmas"])
        print(f"  {cid}: size {info['size']}, {ns} shared lemmas")


if __name__ == "__main__":
    main()
