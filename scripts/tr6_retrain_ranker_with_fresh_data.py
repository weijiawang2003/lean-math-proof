#!/usr/bin/env python3
"""TR6 Part 13 (optional, EXPLORATORY) — mini-retrain with fresh TR6 data.

Compares TR4-only / TR4+TR5 / TR4+TR6 / TR4+TR5+TR6 on leakage-free GroupKFold OOF
(by theorem, namespace, cluster) PR-AUC + top-5 recovery, refitting the same featurizer +
HGB on each union. The question: does fresh-frontier data (esp. non-Set positives) improve
by-namespace generalization? Marks exploratory; does NOT replace the TR4 model.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr5_score as S

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    fp = _p(path) if not os.path.isabs(path) else path
    return [json.loads(l) for l in open(fp) if l.strip()] if os.path.exists(fp) else []


def _featurize(rows):
    nt = lambda fn: " ".join(re.split(r"[._]", fn or ""))
    lt = lambda ls: " ".join(" ".join(re.split(r"[._]", L)) for L in (ls or []))
    vs = {"name_char": TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2),
          "name_tok": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2),
          "goal_word": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3),
          "tactic_tok": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, token_pattern=r"[^\s]+"),
          "lemma_tok": TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2),
          "dict": DictVectorizer(sparse=True)}
    X = [vs["name_char"].fit_transform([nt(r["full_name"]) for r in rows]),
         vs["name_tok"].fit_transform([nt(r["full_name"]) for r in rows]),
         vs["goal_word"].fit_transform([(r.get("goal_text") or "") for r in rows]),
         vs["tactic_tok"].fit_transform([(r.get("tactic") or "") for r in rows]),
         vs["lemma_tok"].fit_transform([lt(r.get("used_lemmas")) for r in rows]),
         vs["dict"].fit_transform([S._dict_feats(r) for r in rows])]
    return sp.hstack(X).tocsr()


def _oof_pr(X, y, groups):
    if len(set(y)) < 2:
        return None
    ng = len(set(groups))
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=min(5, ng)).split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = float(y[tr].mean()); continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=4)
        m.fit(X[tr].toarray(), y[tr])
        oof[te] = m.predict_proba(X[te].toarray())[:, 1]
    return (round(float(average_precision_score(y, oof)), 4), oof)


def _top5(rows, y, oof):
    by = {}
    for i, r in enumerate(rows):
        by.setdefault(r["full_name"], []).append(i)
    have = rec = 0
    for fn, idxs in by.items():
        if not any(y[i] for i in idxs):
            continue
        have += 1
        order = sorted(idxs, key=lambda i: -(oof[i] if not np.isnan(oof[i]) else -1))
        if any(y[i] for i in order[:5]):
            rec += 1
    return round(rec / max(1, have), 4), have


def _evalset(rows, tag):
    if not rows:
        return {"tag": tag, "n": 0}
    for r in rows:
        r.setdefault("label_success", 0)
        r.setdefault("cluster_id", None)
    X = _featurize(rows)
    y = np.array([r.get("label_success", 0) for r in rows], dtype=int)
    g_thm = np.array([r["full_name"] for r in rows])
    g_ns = np.array([r.get("namespace") for r in rows])
    g_cl = np.array([r.get("cluster_id") or r.get("namespace") for r in rows])
    pr_t = _oof_pr(X, y, g_thm)
    pr_n = _oof_pr(X, y, g_ns)
    pr_c = _oof_pr(X, y, g_cl)
    top5, have = _top5(rows, y, pr_t[1]) if pr_t else (None, 0)
    pos_ns = {}
    from collections import Counter
    pos_ns = dict(Counter(r["namespace"] for r in rows if r.get("label_success")))
    return {"tag": tag, "n": len(rows), "positives": int(y.sum()),
            "positive_namespaces": pos_ns,
            "oof_pr_auc_by_theorem": pr_t[0] if pr_t else None,
            "oof_pr_auc_by_namespace": pr_n[0] if pr_n else None,
            "oof_pr_auc_by_cluster": pr_c[0] if pr_c else None,
            "top5_recovery": top5, "theorems_with_success": have}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr4-examples", required=True)
    ap.add_argument("--tr5-examples", required=True)
    ap.add_argument("--tr6-examples", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    tr4 = _rows(args.tr4_examples)
    tr5 = _rows(args.tr5_examples)
    tr6 = _rows(args.tr6_examples)
    sets = {"tr4_only": tr4, "tr4_plus_tr5": tr4 + tr5,
            "tr4_plus_tr6": tr4 + tr6, "tr4_plus_tr5_tr6": tr4 + tr5 + tr6}
    res = {tag: _evalset(rows, tag) for tag, rows in sets.items()}
    base = res["tr4_only"].get("oof_pr_auc_by_theorem") or 0
    base_ns = res["tr4_only"].get("oof_pr_auc_by_namespace") or 0
    out = {"generated_by": "scripts/tr6_retrain_ranker_with_fresh_data.py",
           "exploratory": True, "note": "TR4 model NOT replaced.",
           "results": res,
           "delta_by_theorem_tr6": round((res["tr4_plus_tr6"].get("oof_pr_auc_by_theorem") or 0) - base, 4),
           "delta_by_namespace_tr6": round((res["tr4_plus_tr6"].get("oof_pr_auc_by_namespace") or 0) - base_ns, 4)}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 optional retrain (EXPLORATORY — TR4 model not replaced)", "",
          "| set | n | pos | pos-ns | PR(thm) | PR(ns) | PR(cl) | top5 |",
          "|---|---|---|---|---|---|---|---|"]
    for tag in ("tr4_only", "tr4_plus_tr5", "tr4_plus_tr6", "tr4_plus_tr5_tr6"):
        r = res[tag]
        md.append(f"| {tag} | {r.get('n')} | {r.get('positives')} | "
                  f"{len(r.get('positive_namespaces') or {})} | {r.get('oof_pr_auc_by_theorem')} | "
                  f"{r.get('oof_pr_auc_by_namespace')} | {r.get('oof_pr_auc_by_cluster')} | "
                  f"{r.get('top5_recovery')} |")
    md += ["", f"- Δ PR-AUC by-theorem (TR4→TR4+TR6): **{out['delta_by_theorem_tr6']}**",
           f"- Δ PR-AUC **by-namespace** (TR4→TR4+TR6): **{out['delta_by_namespace_tr6']}** "
           "(the key generalization metric)"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr6-retrain] PR(thm) tr4={base} +tr6={res['tr4_plus_tr6'].get('oof_pr_auc_by_theorem')} "
          f"| PR(ns) tr4={base_ns} +tr6={res['tr4_plus_tr6'].get('oof_pr_auc_by_namespace')}")


if __name__ == "__main__":
    main()
