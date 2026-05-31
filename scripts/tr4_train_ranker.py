#!/usr/bin/env python3
"""TR4 Part 4 — train program rankers + grouped evaluation.

Models: heuristic baseline (rule score, no training), LogisticRegression,
SGDClassifier(log_loss), HistGradientBoosting (RF-class). Because positives are rare
and theorem-correlated, the headline scores are LEAKAGE-FREE out-of-fold (OOF)
predictions from GroupKFold grouped by theorem; these OOF scores are saved for the
ranking/budget evaluation (Parts 5-6). Also reports grouped leave-one-namespace-out /
leave-one-cluster-out PR-AUC and a source split. Full-data models are saved for the
active-probe queue (Part 8).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import average_precision_score, roc_auc_score

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    return [json.loads(l) for l in open(_p(path)) if l.strip()]


WIN_FAMILIES = {"def_unfold_simp": 1.0, "d1_simp_lemma": 0.8, "d2_simp_aesop": 0.7}


def heuristic_score(r):
    """Rule-based: prefer known winning families, namespace match, low retrieval rank,
    using a retrieved lemma; penalize depth and missing-retrieval."""
    f = r.get("features", {})
    s = 0.0
    s += WIN_FAMILIES.get(r.get("program_family"), 0.0)
    if f.get("lemma_namespace_matches"):
        s += 0.4
    if f.get("program_uses_retrieved_lemma"):
        s += 0.5
    rank = r.get("retrieval_rank")
    if rank is not None:
        s += max(0.0, 0.6 - 0.05 * rank)
    else:
        s -= 0.3
    s -= 0.1 * (int(r.get("program_depth") or 1) - 1)
    if f.get("is_def_unfold"):
        s += 0.3
    return s


def _models():
    return {
        "logistic": lambda: LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0),
        "sgd": lambda: SGDClassifier(loss="log_loss", class_weight="balanced",
                                     max_iter=3000, tol=1e-4),
        "hgb": lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                                      max_depth=4),
    }


def _oof_scores(X, y, groups, make):
    """Leakage-free OOF probabilities via GroupKFold by theorem."""
    oof = np.full(len(y), np.nan)
    ng = len(set(groups))
    nsplits = min(5, ng)
    if nsplits < 2:
        return oof
    gkf = GroupKFold(n_splits=nsplits)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = float(y[tr].mean())
            continue
        m = make()
        Xtr = X[tr]
        if isinstance(m, HistGradientBoostingClassifier):
            Xtr = Xtr.toarray()
            m.fit(Xtr, y[tr])
            oof[te] = m.predict_proba(X[te].toarray())[:, 1]
        else:
            m.fit(Xtr, y[tr])
            oof[te] = m.predict_proba(X[te])[:, 1]
    return oof


def _grouped_aucs(X, y, groups, make):
    """Mean PR-AUC over leave-one-group-out style (GroupKFold) for generalization."""
    ng = len(set(groups))
    if ng < 2:
        return None
    nsplits = min(5, ng)
    gkf = GroupKFold(n_splits=nsplits)
    oof = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = float(y[tr].mean()); continue
        m = make()
        if isinstance(m, HistGradientBoostingClassifier):
            m.fit(X[tr].toarray(), y[tr]); oof[te] = m.predict_proba(X[te].toarray())[:, 1]
        else:
            m.fit(X[tr], y[tr]); oof[te] = m.predict_proba(X[te])[:, 1]
    if len(set(y)) < 2:
        return None
    return {"pr_auc": round(float(average_precision_score(y, oof)), 4),
            "roc_auc": round(float(roc_auc_score(y, oof)), 4)}


def _topk_global(y, scores, ks=(10, 20, 50)):
    order = np.argsort(-scores)
    out = {}
    npos = int(y.sum())
    for k in ks:
        topk = order[:k]
        tp = int(y[topk].sum())
        out[f"precision@{k}"] = round(tp / k, 4)
        out[f"recall@{k}"] = round(tp / max(1, npos), 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-model-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = _rows(args.examples)
    X = sp.load_npz(_p(args.features)).tocsr()
    lab = np.load(_p(args.features).replace(".npz", "_labels.npz"))
    y = lab["y_success"].astype(int)
    yc = lab["y_credit"].astype(int)
    groups_thm = np.array([r["full_name"] for r in rows])
    groups_ns = np.array([r["namespace"] or "?" for r in rows])
    groups_cl = np.array([str(r.get("cluster_id")) for r in rows])
    os.makedirs(_p(args.out_model_dir), exist_ok=True)

    # heuristic OOF (no training → score is the same regardless of split)
    heur = np.array([heuristic_score(r) for r in rows], dtype=float)

    oof = {"heuristic": heur}
    full_models = {}
    grouped = {"heuristic": {}}
    for name, make in _models().items():
        oof[name] = _oof_scores(X, y, groups_thm, make)
        # full-data model for the queue
        m = make()
        if isinstance(m, HistGradientBoostingClassifier):
            m.fit(X.toarray(), y)
        else:
            m.fit(X, y)
        full_models[name] = m
        joblib.dump(m, os.path.join(_p(args.out_model_dir), f"{name}_program_ranker.joblib"))
        grouped[name] = {
            "by_theorem": _grouped_aucs(X, y, groups_thm, make),
            "by_namespace": _grouped_aucs(X, y, groups_ns, make),
            "by_cluster": _grouped_aucs(X, y, groups_cl, make),
        }

    # save heuristic + oof
    json.dump({"family_weights": WIN_FAMILIES,
               "rules": ["+win_family", "+0.4 ns_match", "+0.5 uses_retrieved",
                         "+max(0,0.6-0.05*rank)", "-0.3 if no retrieval",
                         "-0.1*(depth-1)", "+0.3 def_unfold"]},
              open(os.path.join(_p(args.out_model_dir), "heuristic_ranker.json"), "w"), indent=2)
    np.savez(os.path.join(_p(args.out_model_dir), "oof_scores.npz"),
             **{k: v for k, v in oof.items()})

    # OOF classification metrics (success + credit)
    def _metrics(scores):
        mask = ~np.isnan(scores)
        ys, sc = y[mask], scores[mask]
        out = {}
        if len(set(ys)) > 1:
            out["pr_auc"] = round(float(average_precision_score(ys, sc)), 4)
            out["roc_auc"] = round(float(roc_auc_score(ys, sc)), 4)
        out.update(_topk_global(ys, sc))
        # credit recovery
        ycm = yc[mask]
        if len(set(ycm)) > 1:
            out["credit_pr_auc"] = round(float(average_precision_score(ycm, sc)), 4)
        return out

    results = {
        "generated_by": "scripts/tr4_train_ranker.py",
        "num_examples": len(rows), "num_pos_success": int(y.sum()),
        "num_pos_credit": int(yc.sum()),
        "oof_classification": {k: _metrics(v) for k, v in oof.items()},
        "grouped_generalization": grouped,
        "models_saved": list(full_models) + ["heuristic"],
        "oof_scores_path": "oof_scores.npz",
        "note": "OOF = GroupKFold-by-theorem (leakage-free). Grouped = PR/ROC-AUC under "
                "GroupKFold by theorem/namespace/cluster.",
    }
    json.dump(results, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR4 training results", "",
          f"- examples {len(rows)} | success pos {int(y.sum())} | credit pos {int(yc.sum())}",
          "", "## OOF classification (leakage-free, group=theorem)", "",
          "| model | PR-AUC | ROC-AUC | prec@10 | rec@20 | rec@50 | credit PR-AUC |",
          "|---|---|---|---|---|---|---|"]
    for k, m in results["oof_classification"].items():
        md.append(f"| {k} | {m.get('pr_auc','-')} | {m.get('roc_auc','-')} | "
                  f"{m.get('precision@10','-')} | {m.get('recall@20','-')} | "
                  f"{m.get('recall@50','-')} | {m.get('credit_pr_auc','-')} |")
    md += ["", "## Grouped generalization PR-AUC", "",
           "| model | by_theorem | by_namespace | by_cluster |", "|---|---|---|---|"]
    for k in _models():
        g = grouped[k]
        def _pr(x):
            return x.get("pr_auc") if x else "-"
        md.append(f"| {k} | {_pr(g['by_theorem'])} | {_pr(g['by_namespace'])} | "
                  f"{_pr(g['by_cluster'])} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")

    # model card
    card = ["# TR4 model card", "",
            "Program-level success rankers. EXPLORATORY — not production routing, not promoted.",
            f"- training rows: {len(rows)} ((theorem,program) pairs)",
            f"- success positives: {int(y.sum())} ({round(100*y.sum()/len(rows),2)}%)",
            "- models: heuristic / logistic / sgd / hgb",
            "- headline metric: leakage-free OOF (GroupKFold by theorem)",
            "- intended use: rank candidate programs to cut probe budget in future TR3-style search",
            "- NOT for: proof generation, production routing, automatic promotion"]
    open(os.path.join(_p(args.out_model_dir), "model_card.md"), "w").write("\n".join(card) + "\n")

    print(f"[tr4-train] OOF PR-AUC: " +
          ", ".join(f"{k}={results['oof_classification'][k].get('pr_auc','-')}" for k in oof))


if __name__ == "__main__":
    main()
