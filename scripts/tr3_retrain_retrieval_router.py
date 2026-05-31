#!/usr/bin/env python3
"""TR3 Part 11 (OPTIONAL, EXPLORATORY) — retrain the small TR1-class router with
retrieval/depth labels.

Trains a single small model (LogisticRegression, class_weight=balanced — the TR1
family) over the UNION label space of three additive dataset combos:
  TR1 only / TR1+SF5 / TR1+SF5+TR3
Features: TF-IDF over {name tokens + goal text} ⊕ boolean feature flags. Reports
label coverage, macro-F1 (leave-one-out), grouped leave-one-namespace-out
generalization, and top-3 accuracy. EXPLORATORY ONLY — not production routing, not a
promoted model. Degrades gracefully if a dataset is missing.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _load(path):
    fp = _p(path)
    if not os.path.exists(fp):
        return []
    return [json.loads(l) for l in open(fp) if l.strip()]


def _text(ex):
    toks = ex.get("theorem_name_tokens") or []
    return " ".join(toks) + " " + (ex.get("goal_text") or "")


def _featdict(ex):
    f = dict(ex.get("features") or {})
    return {k: (1.0 if v else 0.0) for k, v in f.items()}


def _evaluate(examples, label):
    import numpy as np
    from scipy.sparse import hstack, csr_matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
    from sklearn.metrics import f1_score

    ys = [e["label"] for e in examples]
    labels = sorted(set(ys))
    if len(examples) < 6 or len(labels) < 2:
        return {"n": len(examples), "labels": labels, "note": "too small to train"}
    tfidf = TfidfVectorizer(min_df=1)
    Xt = tfidf.fit_transform([_text(e) for e in examples])
    dv = DictVectorizer(sparse=True)
    Xf = dv.fit_transform([_featdict(e) for e in examples])
    X = hstack([Xt, Xf]).tocsr()
    y = np.array(ys)
    groups = np.array([e.get("namespace") or "?" for e in examples])

    def _clf():
        return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)

    # leave-one-out macro-F1 + top-3 acc
    loo = LeaveOneOut()
    preds, top3 = [], 0
    for tr, te in loo.split(X):
        if len(set(y[tr])) < 2:
            preds.append(y[te][0]); top3 += 1; continue
        m = _clf().fit(X[tr], y[tr])
        p = m.predict(X[te])[0]
        preds.append(p)
        proba = m.predict_proba(X[te])[0]
        cls = list(m.classes_)
        order = sorted(range(len(cls)), key=lambda i: -proba[i])[:3]
        if y[te][0] in [cls[i] for i in order]:
            top3 += 1
    loo_f1 = f1_score(y, preds, average="macro", zero_division=0)
    loo_acc = float((np.array(preds) == y).mean())

    # grouped leave-one-namespace-out
    grp_f1 = None
    uniq_groups = set(groups)
    if len(uniq_groups) >= 2:
        logo = LeaveOneGroupOut()
        gp = []
        for tr, te in logo.split(X, y, groups):
            if len(set(y[tr])) < 2:
                gp.extend(y[te]); continue
            m = _clf().fit(X[tr], y[tr])
            gp.extend(m.predict(X[te]))
        grp_f1 = f1_score(y, gp, average="macro", zero_division=0)

    return {
        "n": len(examples), "num_labels": len(labels), "labels": labels,
        "loo_macro_f1": round(loo_f1, 3), "loo_acc": round(loo_acc, 3),
        "loo_top3_acc": round(top3 / len(examples), 3),
        "grouped_lono_macro_f1": (round(grp_f1, 3) if grp_f1 is not None else None),
        "num_namespaces": len(uniq_groups),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr1", default="project/evolve/experiments/tr1/data/tr1_examples.jsonl")
    ap.add_argument("--sf5", default="project/evolve/experiments/sf5/out/sf5_training_examples.jsonl")
    ap.add_argument("--tr3", default="project/evolve/experiments/tr3/data/tr3_training_examples.jsonl")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    tr1, sf5, tr3 = _load(args.tr1), _load(args.sf5), _load(args.tr3)
    combos = {
        "TR1": tr1,
        "TR1+SF5": tr1 + sf5,
        "TR1+SF5+TR3": tr1 + sf5 + tr3,
    }
    try:
        results = {name: _evaluate(ex, name) for name, ex in combos.items()}
        err = None
    except Exception as e:
        import traceback
        results = {}
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()[-400:]}"

    out = {
        "generated_by": "scripts/tr3_retrain_retrieval_router.py",
        "exploratory": True, "not_production": True,
        "component_sizes": {"tr1": len(tr1), "sf5": len(sf5), "tr3": len(tr3)},
        "results": results, "error": err,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR3 retrained router (EXPLORATORY)", "",
          f"- component sizes: TR1={len(tr1)}, SF5={len(sf5)}, TR3={len(tr3)}",
          "- **exploratory only — not production routing**", ""]
    if err:
        md.append(f"Error during training: `{err.splitlines()[0]}`")
    else:
        md.append("| combo | n | labels | LOO macroF1 | LOO acc | top3 | grouped LONO F1 |")
        md.append("|---|---|---|---|---|---|---|")
        for name, r in results.items():
            md.append(f"| {name} | {r.get('n')} | {r.get('num_labels','-')} | "
                      f"{r.get('loo_macro_f1','-')} | {r.get('loo_acc','-')} | "
                      f"{r.get('loo_top3_acc','-')} | {r.get('grouped_lono_macro_f1','-')} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr3-retrain] {[(k, results.get(k, {}).get('loo_macro_f1')) for k in combos]}")


if __name__ == "__main__":
    main()
