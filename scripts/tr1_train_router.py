#!/usr/bin/env python3
"""TR1 Part 4 — train baseline failure-to-action routers.

Models:
  A. rule baseline  — handcrafted symbolic rules (deterministic, no training)
  B. logistic       — multinomial LogisticRegression (class_weight balanced)
  C. sgd            — SGDClassifier(loss=log_loss) (linear, probabilistic)
  D. random_forest  — if it helps (small data; reported, not relied on)

Because the corpus is tiny and several classes are singletons, honest out-of-fold
predictions use LeaveOneOut (every example predicted by a model that never saw it).
A grouped leave-one-namespace-out pass is reported as a leakage check.

Metrics: accuracy, macro F1, per-label precision/recall, top-1, top-3, confusion,
and abstention accuracy (accuracy among examples whose max prob >= threshold).
Models are then refit on ALL data and saved for inference.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import scipy.sparse as sp
from scipy.sparse import load_npz
import joblib
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut, LeaveOneGroupOut
from sklearn.metrics import f1_score, precision_recall_fscore_support, confusion_matrix


# ----------------------------- rule baseline ------------------------------
def rule_predict(ex):
    f = ex.get("features", {})
    fn = ex["full_name"]
    rc2 = ex.get("rc2_status")
    if f.get("has_set") and f.get("has_ite"):
        return "SET_ITE_SIMP"
    if f.get("has_multiset") and f.get("has_induction_signal"):
        return "WX3_MULTISET_INDUCTION"
    if f.get("has_tofinset"):
        return "MX2_TOFINSET_AESOP"
    if rc2 == "failed" and (f.get("has_iff") or f.get("has_subset")):
        return "MISSING_BRIDGE_LEMMA_CANDIDATE"
    if rc2 == "failed":
        return "NO_CHEAP_ACTION"
    if rc2 == "solved":
        return "BASELINE_DUPLICATE"
    return "NO_CHEAP_ACTION"


RULES_DOC = [
    "Set ∧ ite -> SET_ITE_SIMP",
    "Multiset ∧ induction-signal -> WX3_MULTISET_INDUCTION",
    "toFinset/Finite -> MX2_TOFINSET_AESOP",
    "rc2_failed ∧ (iff ∨ subset) -> MISSING_BRIDGE_LEMMA_CANDIDATE",
    "rc2_failed (else) -> NO_CHEAP_ACTION",
    "rc2_solved (else) -> BASELINE_DUPLICATE",
]


def _metrics(y_true, y_pred, labels, proba=None, idx_to_label=None, k=3, abstain_thr=0.35):
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(f1_score(y_true, y_pred, labels=range(len(labels)),
                              average="macro", zero_division=0))
    P, R, F, S = precision_recall_fscore_support(y_true, y_pred, labels=range(len(labels)),
                                                 zero_division=0)
    per_label = {labels[i]: {"precision": round(float(P[i]), 3), "recall": round(float(R[i]), 3),
                             "f1": round(float(F[i]), 3), "support": int(S[i])}
                 for i in range(len(labels))}
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels))).tolist()
    out = {"accuracy": round(acc, 3), "top1_accuracy": round(acc, 3),
           "macro_f1": round(macro_f1, 3), "per_label": per_label, "confusion_matrix": cm}
    if proba is not None:
        topk = np.argsort(-proba, axis=1)[:, :k]
        out[f"top{k}_accuracy"] = round(float(np.mean([y_true[i] in topk[i]
                                                       for i in range(len(y_true))])), 3)
        maxp = proba.max(axis=1)
        keep = maxp >= abstain_thr
        out["abstention"] = {
            "threshold": abstain_thr, "kept": int(keep.sum()), "abstained": int((~keep).sum()),
            "accuracy_on_kept": round(float((y_true[keep] == y_pred[keep]).mean()), 3) if keep.any() else None}
    return out


def _oof(model_factory, X, y, groups, proba_classes):
    """LeaveOneOut OOF preds + per-sample proba aligned to global label order."""
    n, C = X.shape[0], proba_classes
    preds = np.zeros(n, dtype=int)
    proba = np.zeros((n, C))
    loo = LeaveOneOut()
    for tr, te in loo.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
        if hasattr(m, "predict_proba"):
            pr = m.predict_proba(X[te])
            for j, cls in enumerate(m.classes_):
                proba[te, cls] = pr[:, j]
    return preds, proba


def _grouped_acc(model_factory, X, y, groups):
    logo = LeaveOneGroupOut()
    if len(set(groups)) < 2:
        return None
    preds = np.zeros(len(y), dtype=int)
    for tr, te in logo.split(X, y, groups):
        m = model_factory()
        m.fit(X[tr], y[tr])
        preds[te] = m.predict(X[te])
    return round(float((preds == y).mean()), 3)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--examples", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--metadata", required=True)
    p.add_argument("--label-map", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    args = p.parse_args(argv)

    examples = [json.loads(l) for l in open(args.examples) if l.strip()]
    X = load_npz(args.features)
    meta = json.load(open(args.metadata))
    yg = np.load(meta["yg_path"], allow_pickle=True)
    y, groups = yg["y"], yg["groups"]
    lm = json.load(open(args.label_map))
    idx_to_label = {int(k): v for k, v in lm["index_to_label"].items()}
    labels = [idx_to_label[i] for i in range(len(idx_to_label))]
    C = len(labels)

    # ---- rule baseline (full-set; deterministic) ----
    rule_pred = np.array([lm["label_to_index"].get(rule_predict(e), -1) for e in examples])
    rule_metrics = _metrics(y, np.where(rule_pred < 0, y * 0, rule_pred), labels)

    results = {"num_examples": int(X.shape[0]), "num_features": int(X.shape[1]),
               "labels": labels, "label_support": {labels[i]: int((y == i).sum()) for i in range(C)},
               "cv": "LeaveOneOut OOF (tiny corpus; honest per-sample held-out)",
               "models": {}}
    results["models"]["rule_baseline"] = {"type": "handcrafted_rules", "rules": RULES_DOC,
                                          "metrics": rule_metrics}

    factories = {
        "logistic": lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0),
        "sgd": lambda: SGDClassifier(loss="log_loss", max_iter=3000, class_weight="balanced",
                                     alpha=1e-4, random_state=0),
        "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0,
                                                        class_weight="balanced_subsample"),
    }
    Xd = X  # sparse OK for these
    best_name, best_f1 = "rule_baseline", rule_metrics["macro_f1"]
    saved = {}
    for name, fac in factories.items():
        try:
            preds, proba = _oof(fac, Xd, y, groups, C)
            m = _metrics(y, preds, labels, proba=proba, idx_to_label=idx_to_label)
            m["grouped_leave_one_namespace_out_accuracy"] = _grouped_acc(fac, Xd, y, groups)
            # refit on all + save
            full = fac(); full.fit(Xd, y)
            path = os.path.join(args.out_dir, f"{name}_router.joblib")
            os.makedirs(args.out_dir, exist_ok=True)
            joblib.dump({"model": full, "labels": labels}, path)
            saved[name] = path
            results["models"][name] = {"type": type(full).__name__, "metrics": m, "model_path": path}
            if m["macro_f1"] > best_f1:
                best_f1, best_name = m["macro_f1"], name
        except Exception as e:
            results["models"][name] = {"error": f"{type(e).__name__}: {e}"}

    results["best_model"] = best_name
    results["best_macro_f1"] = best_f1
    results["beats_rule_baseline"] = best_name != "rule_baseline" and best_f1 > rule_metrics["macro_f1"]

    # rule baseline + model card
    json.dump({"rules": RULES_DOC, "label_priority": labels}, open(os.path.join(args.out_dir, "rule_baseline.json"), "w"), indent=2)
    json.dump(results, open(args.out_json, "w"), indent=2)

    L = ["# TR1 training results", "",
         f"- examples: **{results['num_examples']}**, features: {results['num_features']}",
         f"- CV: {results['cv']}",
         f"- **best model: `{best_name}` (macro-F1 {best_f1})**",
         f"- beats rule baseline: **{results['beats_rule_baseline']}**", "",
         "## Model comparison", "",
         "| model | accuracy | macro_F1 | top3 | grouped(LONO) acc |", "|---|---|---|---|---|"]
    for name, mr in results["models"].items():
        if "metrics" not in mr:
            L.append(f"| {name} | ERROR | | | |"); continue
        mm = mr["metrics"]
        L.append(f"| {name} | {mm['accuracy']} | {mm['macro_f1']} | {mm.get('top3_accuracy','—')} | "
                 f"{mm.get('grouped_leave_one_namespace_out_accuracy','—')} |")
    L += ["", "## Label support", ""]
    for lab, n in results["label_support"].items():
        L.append(f"- `{lab}`: {n}")
    L += ["", f"## Best model per-label ({best_name})", "",
          "| label | precision | recall | f1 | support |", "|---|---|---|---|---|"]
    bm = results["models"][best_name]["metrics"]["per_label"]
    for lab, s in bm.items():
        L.append(f"| `{lab}` | {s['precision']} | {s['recall']} | {s['f1']} | {s['support']} |")
    open(args.out_md, "w").write("\n".join(L))

    # model card
    card = ["# TR1 model card", "",
            "Pilot failure-to-action router. NOT production routing. Trained on verified",
            "proof-search artifacts (see tr1_methodology.md).", "",
            f"- examples: {results['num_examples']}, labels: {C}",
            f"- best model: {best_name} (macro-F1 {best_f1})",
            f"- beats rule baseline: {results['beats_rule_baseline']}",
            "- intended use: prioritize a next-work queue (SF5 retrieval / deeper search).",
            "- limitations: tiny corpus, several singleton classes, name features may dominate.",
            "- do NOT use for production routing or to credit any candidate."]
    open(os.path.join(args.out_dir, "model_card.md"), "w").write("\n".join(card))
    print(f"[tr1-train] best={best_name} macro_f1={best_f1} beats_rule={results['beats_rule_baseline']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
