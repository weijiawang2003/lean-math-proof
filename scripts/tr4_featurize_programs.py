#!/usr/bin/env python3
"""TR4 Part 3 — featurize (theorem, program) rows into sparse matrices.

Groups: theorem name (char+token n-grams, namespace, path tokens); goal TF-IDF
(word+char); lemma (name n-grams, namespace, token overlap, retrieval rank/score);
program (family/depth one-hot, tactic tokens, tactic-op flags); symbolic flags;
interactions (ns-match, rank-bucket × family, overlap × family). Saves
tr4_features.npz + feature_metadata.json + vectorizers.joblib.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    return [json.loads(l) for l in open(_p(path)) if l.strip()]


def _name_text(fn):
    import re
    return " ".join(re.split(r"[._]", fn))


def _lemma_text(lemmas):
    import re
    return " ".join(" ".join(re.split(r"[._]", L)) for L in (lemmas or []))


def _rank_bucket(rank):
    if rank is None:
        return "none"
    if rank == 0:
        return "r0"
    if rank <= 2:
        return "r1-2"
    if rank <= 5:
        return "r3-5"
    if rank <= 10:
        return "r6-10"
    return "r11+"


def _dict_feats(r):
    f = dict(r.get("features", {}))
    d = {f"flag={k}": (1.0 if v else 0.0) for k, v in f.items()}
    d[f"ns={r.get('namespace')}"] = 1.0
    d[f"fam={r.get('program_family')}"] = 1.0
    d[f"depth={r.get('program_depth')}"] = 1.0
    d[f"lemma_src={r.get('lemma_source')}"] = 1.0
    d[f"source={r.get('source')}"] = 1.0
    rb = _rank_bucket(r.get("retrieval_rank"))
    d[f"rankbucket={rb}"] = 1.0
    sc = r.get("retrieval_score")
    d["retrieval_score"] = float(sc) if isinstance(sc, (int, float)) else 0.0
    d["retrieval_rank"] = float(r["retrieval_rank"]) if r.get("retrieval_rank") is not None else 99.0
    d["has_retrieval"] = 1.0 if r.get("retrieval_rank") is not None else 0.0
    # interactions
    fam = r.get("program_family")
    d[f"int_rankbucket×fam={rb}|{fam}"] = 1.0
    nsmatch = r.get("features", {}).get("lemma_namespace_matches")
    d[f"int_nsmatch×fam={int(bool(nsmatch))}|{fam}"] = 1.0
    d[f"int_usesretr×fam={int(bool(r.get('features',{}).get('program_uses_retrieved_lemma')))}|{fam}"] = 1.0
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--out-features", required=True)
    ap.add_argument("--out-metadata", required=True)
    ap.add_argument("--out-vectorizers", required=True)
    args = ap.parse_args()

    rows = _rows(args.examples)

    name_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    name_tok = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
    goal_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3)
    tactic_tok = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                                 token_pattern=r"[^\s]+")
    lemma_tok = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
    dv = DictVectorizer(sparse=True)

    Xnc = name_char.fit_transform([_name_text(r["full_name"]) for r in rows])
    Xnt = name_tok.fit_transform([_name_text(r["full_name"]) for r in rows])
    Xgw = goal_word.fit_transform([(r.get("goal_text") or "") for r in rows])
    Xtt = tactic_tok.fit_transform([(r.get("tactic") or "") for r in rows])
    Xlt = lemma_tok.fit_transform([_lemma_text(r.get("used_lemmas")) for r in rows])
    Xd = dv.fit_transform([_dict_feats(r) for r in rows])

    blocks = {"name_char": Xnc, "name_tok": Xnt, "goal_word": Xgw,
              "tactic_tok": Xtt, "lemma_tok": Xlt, "dict": Xd}
    X = sp.hstack([blocks[k] for k in blocks]).tocsr()

    y_success = np.array([r["label_success"] for r in rows], dtype=np.int8)
    y_credit = np.array([r["label_credit"] for r in rows], dtype=np.int8)

    sp.save_npz(_p(args.out_features), X)
    np.savez(_p(args.out_features).replace(".npz", "_labels.npz"),
             y_success=y_success, y_credit=y_credit)
    joblib.dump({"name_char": name_char, "name_tok": name_tok, "goal_word": goal_word,
                 "tactic_tok": tactic_tok, "lemma_tok": lemma_tok, "dict": dv},
                _p(args.out_vectorizers))

    # block column offsets for ablation
    offsets, c = {}, 0
    for k in blocks:
        w = blocks[k].shape[1]
        offsets[k] = [c, c + w]
        c += w
    meta = {
        "generated_by": "scripts/tr4_featurize_programs.py",
        "num_rows": X.shape[0], "num_features": X.shape[1],
        "block_dims": {k: blocks[k].shape[1] for k in blocks},
        "block_col_ranges": offsets,
        "labels_path": os.path.relpath(_p(args.out_features).replace(".npz", "_labels.npz"), _REPO),
        "num_positive_success": int(y_success.sum()),
        "num_positive_credit": int(y_credit.sum()),
    }
    json.dump(meta, open(_p(args.out_metadata), "w"), ensure_ascii=False, indent=2)
    print(f"[tr4-featurize] X={X.shape} blocks={meta['block_dims']} "
          f"pos_success={int(y_success.sum())} pos_credit={int(y_credit.sum())}")


if __name__ == "__main__":
    main()
