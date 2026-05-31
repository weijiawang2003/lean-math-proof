#!/usr/bin/env python3
"""TR1 Part 7 (optional) — active-learning case selection.

Scores SF1 frontier theorems (not already RC2-solved, with file_path) by router
uncertainty (entropy of the predicted distribution), boosting underrepresented
label families and Set iff / missing-bridge shapes. Produces a ranked list of
theorems where live probing would be most informative (feeds SF5 / larger eval).
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz
import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier


def _factory(name):
    if name == "logistic":
        return lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    if name == "random_forest":
        return lambda: RandomForestClassifier(n_estimators=200, random_state=0,
                                              class_weight="balanced_subsample")
    return lambda: SGDClassifier(loss="log_loss", max_iter=3000, class_weight="balanced",
                                 alpha=1e-4, random_state=0)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--frontier", required=True)
    p.add_argument("--confirmed-failures", required=True)
    p.add_argument("--examples", default="project/evolve/experiments/tr1/data/tr1_examples.jsonl")
    p.add_argument("--features", default="project/evolve/experiments/tr1/data/tr1_features.npz")
    p.add_argument("--metadata", default="project/evolve/experiments/tr1/data/tr1_feature_metadata.json")
    p.add_argument("--label-map", default="project/evolve/experiments/tr1/data/tr1_label_map.json")
    p.add_argument("--training-results", default="project/evolve/experiments/tr1/out/tr1_training_results.json")
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--top", type=int, default=25)
    args = p.parse_args(argv)

    X = load_npz(args.features)
    meta = json.load(open(args.metadata))
    yg = np.load(meta["yg_path"], allow_pickle=True); y = yg["y"]
    lm = json.load(open(args.label_map))
    idx_to_label = {int(k): v for k, v in lm["index_to_label"].items()}
    support = {idx_to_label[i]: int((y == i).sum()) for i in range(len(idx_to_label))}
    best = json.load(open(args.training_results)).get("best_model", "sgd")
    fac = _factory(best if best != "rule_baseline" else "sgd")
    model = fac(); model.fit(X, y)
    vecs = joblib.load(meta["vectorizers_path"])

    examples = [json.loads(l) for l in open(args.examples) if l.strip()]
    in_data = {e["full_name"] for e in examples}
    solved = set()
    conf = json.load(open(args.confirmed_failures))
    for r in conf.get("results", []):
        if r.get("classification") == "NOW_SOLVED_BY_RC2":
            solved.add(r["full_name"])

    rows = [json.loads(l) for l in open(args.frontier) if l.strip()]
    rare_labels = {l for l, n in support.items() if n < 5}

    def featurize(fn):
        toks = " ".join(re.split(r"[._]", fn)).lower()
        blocks = [vecs["name_char"].transform([fn]), vecs["name_tok"].transform([toks]),
                  vecs["ns"].transform([{f"ns={fn.split('.')[0]}": 1.0}]),
                  vecs["goal_word"].transform(["∅"]), vecs["goal_char"].transform(["∅"]),
                  vecs["bool"].transform([{}]), vecs["cluster"].transform([{}])]
        return sp.hstack(blocks).tocsr()

    cands = []
    for r in rows:
        fn = r.get("name") or r.get("full_name")
        fp = r.get("file_path")
        if not fn or not fp or fn in solved:
            continue
        Xr = featurize(fn)
        if hasattr(model, "predict_proba"):
            pr = model.predict_proba(Xr)[0]
            full = np.zeros(len(idx_to_label))
            for j, cls in enumerate(model.classes_):
                full[cls] = pr[j]
        else:
            full = np.zeros(len(idx_to_label)); full[model.predict(Xr)[0]] = 1.0
        entropy = float(-np.sum([q * np.log(q + 1e-12) for q in full if q > 0]))
        pred = idx_to_label[int(np.argmax(full))]
        low = (fn.lower().find("iff") >= 0) or (fn.lower().find("subset") >= 0)
        boost = (1.0 if pred in rare_labels else 0.0) + (0.5 if low else 0.0) \
            + (0.5 if fn not in in_data else 0.0)
        score = round(entropy + boost, 4)
        cands.append({"full_name": fn, "file_path": fp, "namespace": fn.split(".")[0],
                      "predicted_label": pred, "entropy": round(entropy, 4),
                      "uncertainty_boost": round(boost, 3), "selection_score": score,
                      "already_in_dataset": fn in in_data})
    cands.sort(key=lambda c: -c["selection_score"])
    top = cands[:args.top]

    out = {"num_frontier_considered": len(cands), "num_selected": len(top),
           "selection_criteria": "high entropy + underrepresented predicted family + Set iff/subset shape + not already in dataset; excludes RC2-solved",
           "label_support": support, "selected": top}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)
    L = ["# TR1 active-learning case selection", "",
         f"- frontier considered: {len(cands)}; selected top **{len(top)}**",
         f"- criteria: {out['selection_criteria']}", "",
         "| rank | theorem | predicted | entropy | score |", "|---|---|---|---|---|"]
    for i, c in enumerate(top):
        L.append(f"| {i+1} | `{c['full_name']}` | {c['predicted_label']} | {c['entropy']} | {c['selection_score']} |")
    open(args.out_md, "w").write("\n".join(L))
    print(f"[tr1-active] considered={len(cands)} selected={len(top)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
