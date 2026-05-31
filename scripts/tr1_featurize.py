#!/usr/bin/env python3
"""TR1 Part 3 — feature extraction (scikit-learn).

Builds a sparse feature matrix from tr1_examples.jsonl across four groups:
  1. name features  — char n-grams + token n-grams + namespace one-hot
  2. goal/failure   — TF-IDF word n-grams + char n-grams (goal_text ⊕ last_error)
  3. boolean symbolic flags
  4. cluster features — SF4 cluster id + coarse goal shape (one-hot)

Persists: the matrix (.npz), feature metadata (group dims), the label map, and the
FITTED vectorizers (joblib) so new theorems can be featurized identically at
inference (Part 5). Falls back to a hashing vectorizer only if TF-IDF fit fails.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse import save_npz
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer


def _load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def _goalshape(surface):
    s = (surface or "").lower()
    for sh in ["iff", "subset", "equality", "membership", "extensionality", "arithmetic"]:
        if sh in s:
            return sh
    return "unknown"


def build(examples):
    names = [e["full_name"] for e in examples]
    tok_strings = [" ".join(e.get("theorem_name_tokens") or []) for e in examples]
    ns = [e.get("namespace") or "none" for e in examples]
    goal = [((e.get("goal_text") or "") + " " + (e.get("last_error") or "")).strip() or "∅"
            for e in examples]
    bools = [{k: (1.0 if v else 0.0) for k, v in (e.get("features") or {}).items()}
             for e in examples]
    # symptom flags folded into bool dict
    for i, e in enumerate(examples):
        for s in e.get("trace_symptoms") or []:
            bools[i][f"sym_{s}"] = 1.0
    cluster = [{"cluster": e.get("source_surface") or "none",
                "shape": _goalshape(e.get("source_surface")),
                "rc2": e.get("rc2_status") or "unknown"} for e in examples]

    vecs = {}
    blocks = []
    dims = {}

    vecs["name_char"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    Xc = vecs["name_char"].fit_transform(names); blocks.append(Xc); dims["name_char"] = Xc.shape[1]

    vecs["name_tok"] = TfidfVectorizer(analyzer="word", ngram_range=(1, 2),
                                       token_pattern=r"[^ ]+", min_df=1)
    Xt = vecs["name_tok"].fit_transform(tok_strings); blocks.append(Xt); dims["name_tok"] = Xt.shape[1]

    vecs["ns"] = DictVectorizer(sparse=True)
    Xn = vecs["ns"].fit_transform([{f"ns={n}": 1.0} for n in ns]); blocks.append(Xn); dims["ns"] = Xn.shape[1]

    vecs["goal_word"] = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1)
    Xgw = vecs["goal_word"].fit_transform(goal); blocks.append(Xgw); dims["goal_word"] = Xgw.shape[1]

    vecs["goal_char"] = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    Xgc = vecs["goal_char"].fit_transform(goal); blocks.append(Xgc); dims["goal_char"] = Xgc.shape[1]

    vecs["bool"] = DictVectorizer(sparse=True)
    Xb = vecs["bool"].fit_transform(bools); blocks.append(Xb); dims["bool"] = Xb.shape[1]

    vecs["cluster"] = DictVectorizer(sparse=True)
    Xcl = vecs["cluster"].fit_transform([{f"{k}={v}": 1.0 for k, v in c.items()} for c in cluster])
    blocks.append(Xcl); dims["cluster"] = Xcl.shape[1]

    X = sp.hstack(blocks).tocsr()
    return X, vecs, dims


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--examples", required=True)
    p.add_argument("--out-features", required=True)
    p.add_argument("--out-metadata", required=True)
    p.add_argument("--out-label-map", required=True)
    p.add_argument("--out-vectorizers",
                   default="project/evolve/experiments/tr1/models/tr1_vectorizers.joblib")
    args = p.parse_args(argv)

    examples = _load(args.examples)
    labels = sorted({e["label"] for e in examples})
    label_map = {lab: i for i, lab in enumerate(labels)}
    y = np.array([label_map[e["label"]] for e in examples])
    groups = np.array([e.get("namespace") or "none" for e in examples])

    X, vecs, dims = build(examples)

    os.makedirs(os.path.dirname(args.out_features), exist_ok=True)
    save_npz(args.out_features, X)
    np.savez(args.out_features.replace(".npz", "_yg.npz"), y=y, groups=groups,
             example_ids=np.array([e["example_id"] for e in examples]))
    os.makedirs(os.path.dirname(args.out_vectorizers), exist_ok=True)
    joblib.dump(vecs, args.out_vectorizers)

    meta = {"num_examples": X.shape[0], "num_features": X.shape[1],
            "feature_group_dims": dims, "groups_field": "namespace",
            "vectorizers_path": args.out_vectorizers,
            "yg_path": args.out_features.replace(".npz", "_yg.npz"),
            "label_field": "label"}
    json.dump(meta, open(args.out_metadata, "w"), indent=2)
    json.dump({"label_to_index": label_map,
               "index_to_label": {i: l for l, i in label_map.items()}},
              open(args.out_label_map, "w"), indent=2)
    print(f"[tr1-feat] X={X.shape} groups={len(set(groups))} labels={len(labels)} dims={dims}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
