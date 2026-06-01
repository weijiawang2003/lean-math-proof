#!/usr/bin/env python3
"""RC5H Part 12 (optional) — export RC5H program examples + ranker retrain comparison.

Exports the RC5H (theorem, program) attempts with success labels, then compares the TR4 HGB
ranker trained on TR4 / TR4+TR6 / TR4+TR6+RC5H via GroupKFold (group=theorem) PR-AUC + top-k
recovery, reusing the tr5_score featurization. Does NOT replace the global TR4 model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tr5_score as S

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR4_EX = "project/evolve/experiments/tr4/data/tr4_program_examples.jsonl"
TR6_EX = "project/evolve/experiments/tr6/data/tr6_program_examples.jsonl"


def _p(*a):
    return os.path.join(_REPO, *a)


def _row_from_example(e):
    return S.build_row(e["full_name"], e.get("goal_text"), e.get("namespace"), e.get("tactic"),
                       e.get("used_lemmas"), e.get("program_family"), e.get("program_depth", 1),
                       retrieval_rank=e.get("retrieval_rank"), retrieval_score=e.get("retrieval_score"),
                       lemma_source=e.get("lemma_source"), source=e.get("source", "x"))


def _load_examples(path):
    out = []
    if not os.path.exists(_p(path)):
        return out
    for l in open(_p(path)):
        if l.strip():
            out.append(json.loads(l))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--program-plan", required=True)
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--dynamic-b5")
    ap.add_argument("--dynamic-b10")
    ap.add_argument("--dynamic-b20")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-results-json", required=True)
    ap.add_argument("--out-results-md", required=True)
    ap.add_argument("--vectorizers", default="project/evolve/experiments/tr4/data/tr4_vectorizers.joblib")
    args = ap.parse_args()

    # ---- export RC5H examples (each attempted program with its solved label) ----
    plan = {t["full_name"]: t for t in json.load(open(_p(args.program_plan)))["theorems"]}
    rc5h_examples = []
    for path in (args.dynamic_b5, args.dynamic_b10, args.dynamic_b20):
        if not path or not os.path.exists(_p(path)):
            continue
        for r in json.load(open(_p(path)))["results"]:
            fn = r["full_name"]
            t = plan.get(fn, {})
            by_tac = {p["tactic"]: p for p in t.get("programs_ranked", [])}
            win = r.get("winning_program") or {}
            # winning program -> label 1
            if win.get("tactic"):
                p = by_tac.get(win["tactic"], win)
                rc5h_examples.append({"full_name": fn, "namespace": r.get("namespace"),
                                      "goal_text": t.get("goal_text") or t.get("statement_text"),
                                      "tactic": win["tactic"], "used_lemmas": win.get("used_lemmas") or p.get("lemmas"),
                                      "program_family": win.get("family") or p.get("family"),
                                      "program_depth": win.get("depth") or p.get("depth", 1),
                                      "retrieval_rank": p.get("retrieval_rank"),
                                      "retrieval_score": p.get("retrieval_score"),
                                      "label_success": 1, "source": "rc5h"})
            for f in r.get("failures", []):
                p = by_tac.get(f["tactic"], {})
                rc5h_examples.append({"full_name": fn, "namespace": r.get("namespace"),
                                      "goal_text": t.get("goal_text") or t.get("statement_text"),
                                      "tactic": f["tactic"], "used_lemmas": p.get("lemmas"),
                                      "program_family": f.get("family") or p.get("family"),
                                      "program_depth": p.get("depth", 1),
                                      "retrieval_rank": p.get("retrieval_rank"),
                                      "retrieval_score": p.get("retrieval_score"),
                                      "label_success": 0, "source": "rc5h"})
    # dedup
    seen, dedup = set(), []
    for e in rc5h_examples:
        k = (e["full_name"], e["tactic"])
        if k not in seen:
            seen.add(k); dedup.append(e)
    rc5h_examples = dedup
    with open(_p(args.out_jsonl), "w") as f:
        for e in rc5h_examples:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")

    # ---- retrain comparison (GroupKFold PR-AUC) ----
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import average_precision_score
    except Exception as e:
        json.dump({"error": f"sklearn unavailable: {e}", "rc5h_examples": len(rc5h_examples)},
                  open(_p(args.out_results_json), "w"), indent=2)
        open(_p(args.out_results_md), "w").write(f"# RC5H retrain\n\nsklearn unavailable; "
                                                 f"exported {len(rc5h_examples)} RC5H examples.\n")
        print(f"[rc5h-retrain] exported {len(rc5h_examples)} examples; sklearn unavailable")
        return

    scorer = S.RankerScorer(args.vectorizers, "project/evolve/experiments/tr4/models/hgb_program_ranker.joblib")
    tr4 = _load_examples(TR4_EX)
    tr6 = _load_examples(TR6_EX)

    def _xyg(examples):
        rows = [_row_from_example(e) for e in examples]
        if not rows:
            return None, None, None
        X = scorer._featurize(rows).toarray()
        y = np.array([int(e.get("label_success", 0)) for e in examples])
        g = np.array([e["full_name"] for e in examples])
        return X, y, g

    def _grouped_prauc(examples, label):
        X, y, g = _xyg(examples)
        if X is None or y.sum() < 3 or len(set(g)) < 4:
            return {"set": label, "n": len(examples), "positives": int(y.sum()) if X is not None else 0,
                    "pr_auc": None, "note": "too few positives/groups"}
        n_splits = min(4, len(set(g)))
        gkf = GroupKFold(n_splits=n_splits)
        aps = []
        for tr, te in gkf.split(X, y, g):
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_depth=3,
                                               random_state=0)
            m.fit(X[tr], y[tr])
            aps.append(average_precision_score(y[te], m.predict_proba(X[te])[:, 1]))
        return {"set": label, "n": len(examples), "positives": int(y.sum()),
                "pr_auc": round(float(np.mean(aps)), 4) if aps else None, "folds": len(aps)}

    results = [
        _grouped_prauc(tr4, "TR4_only"),
        _grouped_prauc(tr4 + tr6, "TR4+TR6"),
        _grouped_prauc(tr4 + tr6 + rc5h_examples, "TR4+TR6+RC5H"),
    ]
    out = {"generated_by": "scripts/rc5h_export_and_retrain_ranker.py",
           "rc5h_examples": len(rc5h_examples),
           "rc5h_positives": sum(1 for e in rc5h_examples if e["label_success"]),
           "comparison": results,
           "note": "grouped (group=theorem) PR-AUC; does NOT replace the global TR4 model."}
    json.dump(out, open(_p(args.out_results_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC5H ranker retrain comparison", "",
          f"- RC5H examples: {len(rc5h_examples)} (positives {out['rc5h_positives']})", "",
          "| dataset | n | positives | grouped PR-AUC |", "|---|---|---|---|"]
    for r in results:
        md.append(f"| {r['set']} | {r['n']} | {r['positives']} | {r.get('pr_auc')} |")
    open(_p(args.out_results_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-retrain] examples={len(rc5h_examples)} pos={out['rc5h_positives']} "
          f"prauc={[(r['set'], r.get('pr_auc')) for r in results]}")


if __name__ == "__main__":
    main()
