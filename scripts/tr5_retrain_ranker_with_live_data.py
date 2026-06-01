#!/usr/bin/env python3
"""TR5 Part 11 (optional, EXPLORATORY) — mini-retrain with TR5 live data.

Compares TR4-only vs TR4+TR5 on leakage-free GroupKFold OOF PR-AUC (by theorem,
namespace, cluster) and top-k recovery, refitting the same featurizer + HGB on the union.
Marks exploratory; does NOT replace the TR4 model globally.
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
from sklearn.metrics import average_precision_score, roc_auc_score

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
    name_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    name_tok = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
    goal_word = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=3)
    tactic_tok = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, token_pattern=r"[^\s]+")
    lemma_tok = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2)
    dv = DictVectorizer(sparse=True)
    nt = lambda fn: " ".join(re.split(r"[._]", fn or ""))
    lt = lambda ls: " ".join(" ".join(re.split(r"[._]", L)) for L in (ls or []))
    Xnc = name_char.fit_transform([nt(r["full_name"]) for r in rows])
    Xnt = name_tok.fit_transform([nt(r["full_name"]) for r in rows])
    Xgw = goal_word.fit_transform([(r.get("goal_text") or "") for r in rows])
    Xtt = tactic_tok.fit_transform([(r.get("tactic") or "") for r in rows])
    Xlt = lemma_tok.fit_transform([lt(r.get("used_lemmas")) for r in rows])
    Xd = dv.fit_transform([S._dict_feats(r) for r in rows])
    return sp.hstack([Xnc, Xnt, Xgw, Xtt, Xlt, Xd]).tocsr()


def _oof(X, y, groups):
    if len(set(y)) < 2:
        return None, None
    ng = len(set(groups))
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(5, ng))
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = float(y[tr].mean()); continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=4)
        m.fit(X[tr].toarray(), y[tr])
        oof[te] = m.predict_proba(X[te].toarray())[:, 1]
    pr = round(float(average_precision_score(y, oof)), 4)
    try:
        roc = round(float(roc_auc_score(y, oof)), 4)
    except Exception:
        roc = None
    return pr, roc


def _topk_recovery(rows, X, y, groups, k=5):
    """per-theorem: fraction of theorems-with-a-success where a success is in top-k by OOF."""
    pr, _ = None, None
    oof = np.full(len(y), np.nan)
    ng = len(set(groups))
    gkf = GroupKFold(n_splits=min(5, ng))
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = float(y[tr].mean()); continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=4)
        m.fit(X[tr].toarray(), y[tr])
        oof[te] = m.predict_proba(X[te].toarray())[:, 1]
    by_thm = {}
    for i, r in enumerate(rows):
        by_thm.setdefault(r["full_name"], []).append(i)
    have = rec = 0
    for fn, idxs in by_thm.items():
        if not any(y[i] for i in idxs):
            continue
        have += 1
        order = sorted(idxs, key=lambda i: -(oof[i] if not np.isnan(oof[i]) else -1))
        if any(y[i] for i in order[:k]):
            rec += 1
    return {"theorems_with_success": have, "topk_recovered": rec,
            "recovery_frac": round(rec / max(1, have), 4), "k": k}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tr4-examples", required=True)
    ap.add_argument("--tr5-examples", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    tr4 = _rows(args.tr4_examples)
    tr5 = _rows(args.tr5_examples)
    # normalize label fields for TR5 rows (already have label_success)
    for r in tr5:
        r.setdefault("label_success", 0)
        r.setdefault("cluster_id", None)

    def evalset(rows, tag):
        if not rows:
            return {"tag": tag, "n": 0}
        X = _featurize(rows)
        y = np.array([r.get("label_success", 0) for r in rows], dtype=int)
        g_thm = np.array([r["full_name"] for r in rows])
        g_ns = np.array([r.get("namespace") for r in rows])
        g_cl = np.array([r.get("cluster_id") or r.get("namespace") for r in rows])
        pr_t, roc_t = _oof(X, y, g_thm)
        pr_n, _ = _oof(X, y, g_ns)
        pr_c, _ = _oof(X, y, g_cl)
        topk = _topk_recovery(rows, X, y, g_thm, k=5)
        return {"tag": tag, "n": len(rows), "positives": int(y.sum()),
                "oof_pr_auc_by_theorem": pr_t, "oof_roc_by_theorem": roc_t,
                "oof_pr_auc_by_namespace": pr_n, "oof_pr_auc_by_cluster": pr_c,
                "top5_recovery": topk}

    res_tr4 = evalset(tr4, "tr4_only")
    res_union = evalset(tr4 + tr5, "tr4_plus_tr5")
    out = {"generated_by": "scripts/tr5_retrain_ranker_with_live_data.py",
           "exploratory": True, "note": "TR4 model NOT replaced globally.",
           "tr4_only": res_tr4, "tr4_plus_tr5": res_union,
           "delta_pr_auc_by_theorem": (round((res_union.get("oof_pr_auc_by_theorem") or 0)
                                              - (res_tr4.get("oof_pr_auc_by_theorem") or 0), 4))}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR5 optional retrain (EXPLORATORY — TR4 model not replaced)", "",
          "| set | n | pos | PR-AUC(thm) | ROC(thm) | PR-AUC(ns) | PR-AUC(cluster) | top5 rec |",
          "|---|---|---|---|---|---|---|---|"]
    for r in (res_tr4, res_union):
        md.append(f"| {r['tag']} | {r['n']} | {r.get('positives')} | "
                  f"{r.get('oof_pr_auc_by_theorem')} | {r.get('oof_roc_by_theorem')} | "
                  f"{r.get('oof_pr_auc_by_namespace')} | {r.get('oof_pr_auc_by_cluster')} | "
                  f"{(r.get('top5_recovery') or {}).get('recovery_frac')} |")
    md += ["", f"- Δ PR-AUC (by theorem), TR4→TR4+TR5: **{out['delta_pr_auc_by_theorem']}**"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr5-retrain] tr4 PR-AUC(thm)={res_tr4.get('oof_pr_auc_by_theorem')} "
          f"-> tr4+tr5={res_union.get('oof_pr_auc_by_theorem')} "
          f"(Δ {out['delta_pr_auc_by_theorem']})")


if __name__ == "__main__":
    main()
