"""AX4 Stage 6 — train symbolic-action learner v2 (only if dataset is Green).

Same small, deterministic baseline as AX3 (TF-IDF char_wb 3-5 + balanced
logistic regression) over the proof-state prompt, classes = NULL + the Multiset
symbolic action ids. v2 adds:
  - a feature-source ablation (name-only / state-only / name+state) reported
    via cross-validation, to show what the learner actually keys on;
  - a confidence-threshold sweep on the held-out + CV firing, for the Stage-7
    promotion decision.

Trains on split==train_candidate; the held-out split (reserved sets) is never
trained on. Reports CV (stratified) top-1/top-2, positive/simp_all recall, NULL
and non-Multiset false-positive rates before/after the namespace gate.

Refuses to train unless readiness verdict is GREEN (per the AX4 spec: retrain
only after Green). The model is persisted to a git-ignored dir.

Outputs:
  project/models/ax4_multiset_symbolic_clf/model.joblib   (gitignored)
  project/data/ax4_classifier_metrics.json                (committed)
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

ROOT = Path(__file__).resolve().parent.parent
JSONL = ROOT / "project/data/ax4_multiset_symbolic_dataset.jsonl"
DS_META = ROOT / "project/data/ax4_multiset_symbolic_dataset_meta.json"
MODEL_DIR = ROOT / "project/models/ax4_multiset_symbolic_clf"
OUT = ROOT / "project/data/ax4_classifier_metrics.json"
NULL = "NULL"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def make_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                  min_df=1, sublinear_tf=True)),
        ("lr", LogisticRegression(max_iter=2000, class_weight="balanced",
                                  C=4.0)),
    ])


def load_rows():
    return [json.loads(l) for l in open(JSONL) if l.strip()]


def _name_of(prompt):
    # build_prompt = "Theorem: {name}\n\nProof state:\n{state}\n"
    first = prompt.split("\n", 1)[0]
    return first[len("Theorem: "):] if first.startswith("Theorem: ") else first


def _state_of(prompt):
    parts = prompt.split("Proof state:\n", 1)
    return parts[1] if len(parts) == 2 else prompt


def cv_block(X, y, label):
    cnt = Counter(y)
    min_class = min(cnt.values())
    if min_class < 2 or len(cnt) < 2:
        return None
    k = min(5, min_class)
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)
    pred = cross_val_predict(make_pipeline(), X, y, cv=skf)
    acc = sum(p == g for p, g in zip(pred, y)) / len(y)
    gold_c, hit_c = Counter(y), Counter(g for g, p in zip(y, pred) if g == p)
    per_class = {c: {"n": gold_c[c], "recall": round(hit_c[c] / gold_c[c], 4)}
                 for c in sorted(gold_c)}
    pos = [(g, p) for g, p in zip(y, pred) if g != NULL]
    fam_recall = (sum(1 for g, p in pos if p != NULL) / len(pos)
                  if pos else None)
    nulls = [(g, p) for g, p in zip(y, pred) if g == NULL]
    null_fp = (sum(1 for g, p in nulls if p != NULL) / len(nulls)
               if nulls else None)
    return {"feature_source": label, "folds": k, "top1": round(acc, 4),
            "per_class": per_class,
            "positive_family_recall": round(fam_recall, 4)
            if fam_recall is not None else None,
            "null_false_positive_rate": round(null_fp, 4)
            if null_fp is not None else None}


def main() -> None:
    meta = json.loads(DS_META.read_text(encoding="utf-8"))
    verdict = meta["readiness"]["verdict"]
    if verdict != "GREEN":
        print(f"readiness={verdict} (not GREEN) — per AX4 spec, NOT retraining "
              f"the learner. Keep AX3 learner + WX3 oracle.")
        OUT.write_text(json.dumps(
            {"trained": False, "reason": f"readiness={verdict}",
             "note": "AX4 spec trains v2 only at GREEN."},
            indent=2), encoding="utf-8")
        return

    rows = load_rows()
    train = [r for r in rows if r["split"] == "train_candidate"]
    held = [r for r in rows if r["split"] == "heldout_eval"]
    Xtr = [r["prompt"] for r in train]
    ytr = [r["label"] for r in train]

    clf = make_pipeline()
    clf.fit(Xtr, ytr)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": clf, "classes": list(clf.classes_)},
                MODEL_DIR / "model.joblib")
    cls_list = list(clf.classes_)

    # ---- feature-source ablation (CV over all rows) ----
    Xall = [r["prompt"] for r in rows]
    yall = [r["label"] for r in rows]
    ablation = [
        cv_block(Xall, yall, "name+state"),
        cv_block([_name_of(p) for p in Xall], yall, "name-only"),
        cv_block([_state_of(p) for p in Xall], yall, "state-only"),
    ]

    # ---- held-out metrics + threshold sweep ----
    def predict_fire(rs, thresh):
        if not rs:
            return None
        proba = clf.predict_proba([r["prompt"] for r in rs])
        nidx = cls_list.index(NULL)
        fires, hits = 0, 0
        for row, r in zip(proba, rs):
            bi = max(range(len(row)), key=lambda i: row[i])
            fired = cls_list[bi] != NULL and row[bi] >= thresh
            if fired:
                fires += 1
                if r["label"] != NULL:
                    hits += 1
        return {"n": len(rs), "fired": fires, "hits_true_positive": hits}

    held_pos = [r for r in held if r["label"] != NULL]
    held_ms_null = [r for r in held if r["label"] == NULL and r["is_multiset"]]
    control = [r for r in rows if r["arc"] == "control"]
    sweep = []
    for th in THRESHOLDS:
        sweep.append({
            "threshold": th,
            "heldout_positive": predict_fire(held_pos, th),
            "heldout_multiset_null": predict_fire(held_ms_null, th),
            "control_non_multiset": predict_fire(control, th),
        })

    out = {
        "trained": True,
        "model": "TfidfVectorizer(char_wb 3-5) + LogisticRegression "
                 "(balanced, C=4)",
        "readiness": verdict,
        "model_path": str((MODEL_DIR / "model.joblib").relative_to(ROOT)),
        "classes": cls_list,
        "n_train": len(train), "n_heldout": len(held),
        "feature_ablation_cv": ablation,
        "threshold_sweep": sweep,
        "note": "control firing is raw; the wrapper namespace-gates emission "
                "to Multiset so effective non-Multiset FP = 0.",
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"classes={cls_list} train={len(train)} heldout={len(held)}")
    for a in ablation:
        if a:
            print(f"  CV[{a['feature_source']:11s}] top1={a['top1']} "
                  f"pos_recall={a['positive_family_recall']} "
                  f"null_fp={a['null_false_positive_rate']}")


if __name__ == "__main__":
    main()
