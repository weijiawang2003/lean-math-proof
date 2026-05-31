#!/usr/bin/env python3
"""TR1 Part 5 — evaluate the router on confirmed RC2 failures + build a next-work queue.

For each SF4 CONFIRMED_RC2_FAILURE, produce a HELD-OUT prediction: if the theorem is
in the training set, retrain the best model leaving it out (true held-out); otherwise
featurize fresh with the saved vectorizers and predict with the full model. Emit
top-k predictions, abstention, recommended next step, and a ranked next-work queue.
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

NEXT_STEP = {
    "SET_ITE_SIMP": "no action — RC2 already includes SET_ITE_SIMP",
    "SX3_PRODUCTION_SUBSUMED": "do not pursue — subsumed by production search",
    "NO_CHEAP_ACTION": "send to lemma retrieval / deeper search",
    "MISSING_BRIDGE_LEMMA_CANDIDATE": "send to SF5 existing-lemma retrieval",
    "PROOF_SEARCH_DEPTH_GAP": "send to deeper bounded sequence search / widen aesop routing",
    "BASELINE_DUPLICATE": "retry bare baseline (aesop/simp_all) — routing/depth gap",
    "WX3_MULTISET_INDUCTION": "try Multiset.induction_on <;> simp_all",
    "MX2_TOFINSET_AESOP": "try Set.Finite/toFinset-gated aesop",
}
# queue priority: actionable directions first
QUEUE_PRIORITY = {
    "PROOF_SEARCH_DEPTH_GAP": 0, "MISSING_BRIDGE_LEMMA_CANDIDATE": 1,
    "WX3_MULTISET_INDUCTION": 2, "MX2_TOFINSET_AESOP": 2, "SET_ITE_SIMP": 3,
    "BASELINE_DUPLICATE": 4, "NO_CHEAP_ACTION": 5, "SX3_PRODUCTION_SUBSUMED": 9,
}


def _factory(name):
    if name == "logistic":
        return lambda: LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    if name == "sgd":
        return lambda: SGDClassifier(loss="log_loss", max_iter=3000, class_weight="balanced",
                                     alpha=1e-4, random_state=0)
    return lambda: RandomForestClassifier(n_estimators=200, random_state=0,
                                          class_weight="balanced_subsample")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--examples", required=True)
    p.add_argument("--confirmed-failures", required=True)
    p.add_argument("--features",
                   default="project/evolve/experiments/tr1/data/tr1_features.npz")
    p.add_argument("--metadata",
                   default="project/evolve/experiments/tr1/data/tr1_feature_metadata.json")
    p.add_argument("--label-map",
                   default="project/evolve/experiments/tr1/data/tr1_label_map.json")
    p.add_argument("--training-results",
                   default="project/evolve/experiments/tr1/out/tr1_training_results.json")
    p.add_argument("--abstain-threshold", type=float, default=0.35)
    p.add_argument("--out-json", required=True)
    p.add_argument("--out-md", required=True)
    p.add_argument("--queue-json", required=True)
    p.add_argument("--queue-md", required=True)
    args = p.parse_args(argv)

    examples = [json.loads(l) for l in open(args.examples) if l.strip()]
    idx_of = {e["full_name"]: i for i, e in enumerate(examples)}
    X = load_npz(args.features)
    meta = json.load(open(args.metadata))
    yg = np.load(meta["yg_path"], allow_pickle=True)
    y = yg["y"]
    lm = json.load(open(args.label_map))
    idx_to_label = {int(k): v for k, v in lm["index_to_label"].items()}
    best = json.load(open(args.training_results)).get("best_model", "sgd")
    fac = _factory(best if best != "rule_baseline" else "sgd")
    vecs = joblib.load(meta["vectorizers_path"])

    full_model = fac(); full_model.fit(X, y)

    conf = json.load(open(args.confirmed_failures))
    failures = [r for r in conf.get("results", []) if r.get("classification") == "CONFIRMED_RC2_FAILURE"]

    def featurize_fresh(fn, file_path):
        # minimal fresh featurization mirroring tr1_featurize for an unseen theorem
        import re
        toks = " ".join(re.split(r"[._]", fn)).lower()
        blocks = [vecs["name_char"].transform([fn]), vecs["name_tok"].transform([toks]),
                  vecs["ns"].transform([{f"ns={fn.split('.')[0]}": 1.0}]),
                  vecs["goal_word"].transform(["∅"]), vecs["goal_char"].transform(["∅"]),
                  vecs["bool"].transform([{}]), vecs["cluster"].transform([{}])]
        return sp.hstack(blocks).tocsr()

    def proba_vec(model, Xrow):
        pr = np.zeros(len(idx_to_label))
        if hasattr(model, "predict_proba"):
            pp = model.predict_proba(Xrow)[0]
            for j, cls in enumerate(model.classes_):
                pr[cls] = pp[j]
        else:
            pred = model.predict(Xrow)[0]
            pr[pred] = 1.0
        return pr

    records = []
    for r in failures:
        fn = r["full_name"]
        if fn in idx_of:
            i = idx_of[fn]
            mask = np.ones(X.shape[0], dtype=bool); mask[i] = False
            m = fac(); m.fit(X[mask], y[mask])
            pr = proba_vec(m, X[i])
            held_out = True
            true_triage = examples[i]["label"]
        else:
            m = full_model
            pr = proba_vec(m, featurize_fresh(fn, r.get("file_path")))
            held_out = False
            true_triage = None
        order = np.argsort(-pr)
        top = [{"label": idx_to_label[int(j)], "score": round(float(pr[j]), 3)}
               for j in order[:3] if pr[j] > 0][:3]
        if not top:
            top = [{"label": idx_to_label[int(order[0])], "score": round(float(pr[order[0]]), 3)}]
        abstained = float(pr.max()) < args.abstain_threshold
        pred_label = top[0]["label"]
        records.append({
            "full_name": fn, "file_path": r.get("file_path"),
            "cluster": (examples[idx_of[fn]].get("source_surface") if fn in idx_of else None),
            "true_triage": true_triage, "held_out": held_out,
            "top_predictions": top, "abstained": abstained,
            "predicted_label": pred_label,
            "recommended_next_step": NEXT_STEP.get(pred_label, "review"),
            "correct": (true_triage == pred_label) if true_triage else None,
        })

    n_held = sum(1 for r in records if r["held_out"])
    n_correct = sum(1 for r in records if r.get("correct"))
    out = {"best_model": best, "num_failures": len(records),
           "num_held_out": n_held, "num_correct_held_out": n_correct,
           "held_out_accuracy": round(n_correct / n_held, 3) if n_held else None,
           "num_abstained": sum(1 for r in records if r["abstained"]),
           "predictions": records}
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=2)

    L = ["# TR1 router predictions on confirmed RC2 failures", "",
         f"- best model: `{best}`",
         f"- failures: **{len(records)}**, held-out accuracy: **{out['held_out_accuracy']}** "
         f"({n_correct}/{n_held}), abstained: {out['num_abstained']}", "",
         "| theorem | true triage | predicted | score | abstain | next step |", "|---|---|---|---|---|---|"]
    for r in records:
        L.append(f"| `{r['full_name']}` | {r['true_triage']} | {r['predicted_label']} | "
                 f"{r['top_predictions'][0]['score']} | {'Y' if r['abstained'] else '—'} | "
                 f"{r['recommended_next_step']} |")
    open(args.out_md, "w").write("\n".join(L))

    # ---- next-work queue ----
    queue = sorted(records, key=lambda r: (QUEUE_PRIORITY.get(r["predicted_label"], 9),
                                           -r["top_predictions"][0]["score"]))
    queue_out = [{"rank": i + 1, "full_name": r["full_name"], "file_path": r["file_path"],
                  "predicted_label": r["predicted_label"],
                  "recommended_next_step": r["recommended_next_step"],
                  "score": r["top_predictions"][0]["score"], "abstained": r["abstained"]}
                 for i, r in enumerate(queue)]
    json.dump({"num": len(queue_out), "queue": queue_out}, open(args.queue_json, "w"), indent=2)
    QL = ["# TR1 next-work queue (router-ranked)", "",
          "Ranked by actionability of the predicted triage label (depth-gap / bridge-lemma first).", "",
          "| rank | theorem | predicted | next step | score |", "|---|---|---|---|---|"]
    for q in queue_out:
        QL.append(f"| {q['rank']} | `{q['full_name']}` | {q['predicted_label']} | "
                  f"{q['recommended_next_step']} | {q['score']} |")
    open(args.queue_md, "w").write("\n".join(QL))
    print(f"[tr1-eval] failures={len(records)} held_out_acc={out['held_out_accuracy']} "
          f"abstained={out['num_abstained']} queue={len(queue_out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
