"""AX3 Stage 7 — train the first symbolic-action predictor.

A deliberately small, deterministic baseline (per the AX3 spec: sklearn
TF-IDF + logistic regression is acceptable as the first symbolic-action
learner — a 60M DistilBERT fine-tune is overkill and unstable at ~20-40
labels). Input features: the proof-state prompt (theorem name + state text).
Classes: NULL plus each clean Multiset symbolic action id observed in the
training split.

Trains on split==train_candidate, evaluates on split==heldout_eval. Reports
top-1 / top-2 accuracy and the false-positive rate on the non-Multiset
control rows (where the correct action is NULL). The model is persisted to a
git-ignored dir for the Stage-8 wrapper integration.

Outputs:
  project/models/ax3_multiset_symbolic_clf/model.joblib   (gitignored)
  project/data/ax3_classifier_metrics.json                (committed)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=1, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                  C=4.0)),
    ])

ROOT = Path(__file__).resolve().parent.parent
JSONL = ROOT / "project/data/ax3_multiset_symbolic_dataset.jsonl"
MODEL_DIR = ROOT / "project/models/ax3_multiset_symbolic_clf"
OUT = ROOT / "project/data/ax3_classifier_metrics.json"
NULL = "NULL"


def load_rows():
    return [json.loads(l) for l in open(JSONL) if l.strip()]


def topk_acc(clf, X, y, classes, k):
    proba = clf.predict_proba(X)
    idx = {c: i for i, c in enumerate(classes)}
    correct = 0
    for row, gold in zip(proba, y):
        topk = sorted(range(len(row)), key=lambda i: -row[i])[:k]
        if idx.get(gold) in topk:
            correct += 1
    return correct / len(y) if y else 0.0


def main() -> None:
    rows = load_rows()
    train = [r for r in rows if r["split"] == "train_candidate"]
    held = [r for r in rows if r["split"] == "heldout_eval"]
    if not train:
        print("no training rows; abort")
        sys.exit(1)

    Xtr = [r["prompt"] for r in train]
    ytr = [r["label"] for r in train]
    classes = sorted(set(ytr))

    clf = make_pipeline()
    clf.fit(Xtr, ytr)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf, "classes": list(clf.classes_)},
                MODEL_DIR / "model.joblib")

    cls_list = list(clf.classes_)

    def evalset(rs):
        if not rs:
            return None
        X = [r["prompt"] for r in rs]
        y = [r["label"] for r in rs]
        preds = clf.predict(X)
        acc1 = sum(p == g for p, g in zip(preds, y)) / len(y)
        acc2 = topk_acc(clf, X, y, cls_list, 2)
        # per-class
        cm = Counter()
        for p, g in zip(preds, y):
            cm[(g, p)] += 1
        return {"n": len(y), "top1": round(acc1, 4),
                "top2": round(acc2, 4),
                "gold_dist": dict(Counter(y)),
                "pred_dist": dict(Counter(preds.tolist())),
                "confusions": {f"{g}=>{p}": c for (g, p), c in
                               sorted(cm.items(), key=lambda x: -x[1])[:12]}}

    train_metrics = evalset(train)
    held_metrics = evalset(held)

    # false-positive analysis on non-Multiset control rows (gold == NULL,
    # any non-NULL prediction is a false positive / leakage risk).
    control = [r for r in rows if r["arc"] == "control"]
    fp_control = 0
    if control:
        preds = clf.predict([r["prompt"] for r in control])
        fp_control = sum(1 for p in preds if p != NULL)
    # FP on Multiset NULL (predicting induction where it isn't the clean closer)
    ms_null = [r for r in rows if r["label"] == NULL and r["is_multiset"]
               and r["split"] == "heldout_eval"]
    fp_ms_null = 0
    if ms_null:
        preds = clf.predict([r["prompt"] for r in ms_null])
        fp_ms_null = sum(1 for p in preds if p != NULL)

    # ---- stratified 5-fold CV over ALL rows (the natural held-out split has
    # only 1 positive, too thin for recall; CV gives a robust estimate). ----
    Xall = [r["prompt"] for r in rows]
    yall = [r["label"] for r in rows]
    cv_metrics = None
    min_class = min(Counter(yall).values())
    if min_class >= 2 and len(set(yall)) >= 2:
        k = min(5, min_class)
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
        cvpred = cross_val_predict(make_pipeline(), Xall, yall, cv=skf)
        acc = sum(p == g for p, g in zip(cvpred, yall)) / len(yall)
        # per-class recall
        per_class = {}
        gold_c = Counter(yall)
        hit_c = Counter(g for g, p in zip(yall, cvpred) if g == p)
        for c in sorted(gold_c):
            per_class[c] = {"n": gold_c[c],
                            "recall": round(hit_c[c] / gold_c[c], 4)}
        # positive recall: any clean symbolic label predicted as some
        # non-NULL action (action-family recall, mode-agnostic)
        pos = [(g, p) for g, p in zip(yall, cvpred) if g != NULL]
        fam_recall = (sum(1 for g, p in pos if p != NULL) / len(pos)
                      if pos else None)
        null_rows = [(g, p) for g, p in zip(yall, cvpred) if g == NULL]
        null_fp = (sum(1 for g, p in null_rows if p != NULL) / len(null_rows)
                   if null_rows else None)
        cv_metrics = {
            "folds": k, "overall_top1": round(acc, 4),
            "per_class": per_class,
            "positive_family_recall": round(fam_recall, 4)
            if fam_recall is not None else None,
            "null_false_positive_rate": round(null_fp, 4)
            if null_fp is not None else None,
        }

    out = {
        "model": "TfidfVectorizer(char_wb 3-5) + LogisticRegression "
                 "(balanced, C=4)",
        "readiness": "YELLOW (smoke training)",
        "cross_validation_all_rows": cv_metrics,
        "model_path": str((MODEL_DIR / "model.joblib").relative_to(ROOT)),
        "classes": cls_list,
        "n_train": len(train), "n_heldout": len(held),
        "train_metrics": train_metrics,
        "heldout_metrics": held_metrics,
        "false_positive_control": {
            "n_control_null": len(control),
            "false_positives": fp_control,
            "fp_rate": round(fp_control / len(control), 4) if control else None,
        },
        "false_positive_multiset_null_heldout": {
            "n": len(ms_null), "false_positives": fp_ms_null,
            "fp_rate": round(fp_ms_null / len(ms_null), 4) if ms_null else None,
        },
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"classes: {cls_list}")
    print(f"train n={len(train)} heldout n={len(held)}")
    if held_metrics:
        print(f"HELDOUT top1={held_metrics['top1']} top2={held_metrics['top2']}"
              f" gold={held_metrics['gold_dist']}")
    print(f"control NULL FP: {fp_control}/{len(control)}")
    print(f"heldout Multiset-NULL FP: {fp_ms_null}/{len(ms_null)}")


if __name__ == "__main__":
    main()
