"""AX2 Stage 7 — tiny symbolic-action classifier smoke test (optional).

Gated on Stage 6 readiness (runs only for YELLOW/GREEN). Trains a small
TF-IDF + logistic-regression classifier that maps a proof-state goal
snippet to a symbolic action label. This is a SMOKE TEST of learnability,
not a production model: the dataset is tiny, so metrics are illustrative.

No model artifact is written or committed — only a metrics JSON. With very
few examples per label a single split is high-variance, so we report
leave-one-out (LOO) top-1/top-2 accuracy plus a per-label confusion table.

Output: project/data/ax2_smoke_classifier_metrics.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DS = ROOT / "project/data/ax2_symbolic_label_dataset_meta.json"
READY = ROOT / "project/data/ax2_readiness_meta.json"
OUT = ROOT / "project/data/ax2_smoke_classifier_metrics.json"


def main() -> None:
    if not READY.exists():
        sys.exit("missing readiness meta; run Stage 6 first")
    readiness = json.load(open(READY)).get("readiness")
    if readiness not in ("YELLOW", "GREEN"):
        print(f"readiness={readiness}: smoke classifier skipped (RED).")
        OUT.write_text(json.dumps(
            {"skipped": True, "reason": f"readiness={readiness}"}, indent=2),
            encoding="utf-8")
        return

    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    ds = json.load(open(DS))

    def label_of(e):
        return e.get("final_training_label") or e.get("symbolic_label")

    def eligible(e):
        if e.get("symbolic_action_needed") is True:
            return True
        if not ds.get("minimal_relabel_applied") and e.get("symbolic_label"):
            return True
        return False

    rows = []
    for e in ds["examples"]:
        lab = label_of(e)
        if not eligible(e) or not lab or str(lab).startswith("NON_SYMBOLIC"):
            continue
        # feature text: goal snippet + coarse local var types (state cues)
        vtypes = " ".join(f"VAR_{v.get('coarse_type')}"
                          for v in e.get("local_variables", []))
        text = (e.get("goal_snippet") or "") + " " + vtypes
        rows.append((text.strip(), lab))

    texts = [t for t, _ in rows]
    labels = [l for _, l in rows]
    label_counts = Counter(labels)
    # keep only labels with >=2 examples (need at least one train + one test)
    keep = {l for l, c in label_counts.items() if c >= 2}
    idx = [i for i, l in enumerate(labels) if l in keep]
    X = [texts[i] for i in idx]
    y = [labels[i] for i in idx]

    n = len(X)
    classes = sorted(set(y))
    if n < 6 or len(classes) < 2:
        out = {"skipped": True,
               "reason": f"too few usable examples (n={n}, classes={len(classes)})",
               "label_counts": dict(label_counts)}
        OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        print(out["reason"])
        return

    def make_clf():
        return make_pipeline(
            TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4),
                            min_df=1),
            LogisticRegression(max_iter=2000, class_weight="balanced"))

    # leave-one-out top-1 / top-2
    top1 = top2 = 0
    confusion = Counter()
    for i in range(n):
        Xtr = [X[j] for j in range(n) if j != i]
        ytr = [y[j] for j in range(n) if j != i]
        if len(set(ytr)) < 2:
            continue
        clf = make_clf()
        clf.fit(Xtr, ytr)
        cls = list(clf.classes_)
        proba = clf.predict_proba([X[i]])[0]
        order = np.argsort(proba)[::-1]
        pred1 = cls[order[0]]
        pred2 = {cls[order[0]]} | ({cls[order[1]]} if len(order) > 1 else set())
        top1 += int(pred1 == y[i])
        top2 += int(y[i] in pred2)
        confusion[(y[i], pred1)] += 1

    out = {
        "readiness": readiness,
        "note": ("SMOKE TEST: tiny dataset, LOO cross-validation, char-ngram "
                 "TF-IDF + logistic regression. Illustrative learnability "
                 "signal only — NOT a production model; no artifact saved."),
        "usable_examples": n,
        "classes": classes,
        "class_counts": {c: y.count(c) for c in classes},
        "loo_top1_accuracy": round(top1 / n, 4),
        "loo_top2_accuracy": round(top2 / n, 4),
        "majority_baseline": round(max(Counter(y).values()) / n, 4),
        "confusion_true_pred": {f"{t}->{p}": c
                                for (t, p), c in confusion.most_common()},
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"n={n} classes={len(classes)} "
          f"LOO top1={out['loo_top1_accuracy']} top2={out['loo_top2_accuracy']} "
          f"majority_baseline={out['majority_baseline']}")


if __name__ == "__main__":
    main()
