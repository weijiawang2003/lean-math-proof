#!/usr/bin/env python3
"""TR4 Part 5 — per-theorem ranking evaluation (leakage-free OOF scores).

Uses the GroupKFold-by-theorem OOF scores from training (models/oof_scores.npz) so a
theorem's programs are ranked by a model that never trained on that theorem. Per
theorem (TR3 programs = the realistic search scenario) records the rank of the first
successful / first credited program under: original TR3 order, random expectation,
heuristic, and each model. Aggregates top-k success recovery, mean/median first-success
rank, and budget saved.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    return [json.loads(l) for l in open(_p(path)) if l.strip()]


def _first_rank(order_idx, success):
    """1-based rank of first success in the given ordering (list of row-indices)."""
    for pos, i in enumerate(order_idx, 1):
        if success[i]:
            return pos
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--source", default="tr3")
    args = ap.parse_args()

    rows = _rows(args.examples)
    oof = np.load(os.path.join(_p(args.model_dir), "oof_scores.npz"))
    models = list(oof.keys())
    success = np.array([r["label_success"] for r in rows])
    credit = np.array([r["label_credit"] for r in rows])

    # group tr3 rows by theorem, preserving original (plan) order
    by_thm = {}
    for i, r in enumerate(rows):
        if r["source"] != args.source:
            continue
        by_thm.setdefault(r["full_name"], []).append(i)

    per_theorem = []
    agg = {m: {"first_success_ranks": [], "first_credit_ranks": []} for m in
           ["original_order", "random_expected"] + models}
    topk = {m: {1: 0, 3: 0, 5: 0} for m in models + ["original_order"]}
    n_with_success = 0

    for fn, idxs in by_thm.items():
        n = len(idxs)
        nsucc = int(success[idxs].sum())
        ncred = int(credit[idxs].sum())
        rec = {"full_name": fn, "num_programs": n, "num_successes": nsucc,
               "num_credited": ncred, "best_success_rank": {}, "best_credit_rank": {}}
        # original order
        orig = list(idxs)
        rec["best_success_rank"]["original_order"] = _first_rank(orig, success)
        rec["best_credit_rank"]["original_order"] = _first_rank(orig, credit)
        # random expected rank of first success
        rand_exp = round((n + 1) / (nsucc + 1), 2) if nsucc else None
        rec["best_success_rank"]["random_expected"] = rand_exp
        # model orderings
        for m in models:
            sc = oof[m][idxs]
            order = [idxs[j] for j in np.argsort(-sc)]
            rec["best_success_rank"][m] = _first_rank(order, success)
            rec["best_credit_rank"][m] = _first_rank(order, credit)
            if nsucc:
                for k in (1, 3, 5):
                    if any(success[order[:k]]):
                        topk[m][k] += 1
        rec["top_predictions"] = [
            {"tactic": rows[idxs[j]]["tactic"], "family": rows[idxs[j]]["program_family"],
             "solved": bool(success[idxs[j]]),
             "score_hgb": float(oof["hgb"][idxs[j]]) if "hgb" in oof else None}
            for j in (np.argsort(-oof["hgb"][idxs])[:5] if "hgb" in oof else range(min(5, n)))
        ]
        if nsucc:
            n_with_success += 1
            for k in (1, 3, 5):
                if any(success[orig[:k]]):
                    topk["original_order"][k] += 1
            for m in ["original_order"] + models:
                r = rec["best_success_rank"][m]
                if r:
                    agg[m]["first_success_ranks"].append(r)
            if rand_exp:
                agg["random_expected"]["first_success_ranks"].append(rand_exp)
        per_theorem.append(rec)

    def _stats(vals):
        if not vals:
            return {"n": 0}
        a = np.array(vals, dtype=float)
        return {"n": len(vals), "mean": round(float(a.mean()), 2),
                "median": round(float(np.median(a)), 2)}

    summary = {m: _stats(agg[m]["first_success_ranks"])
               for m in ["original_order", "random_expected"] + models}
    topk_recovery = {m: {f"top{k}": round(topk[m][k] / max(1, n_with_success), 3)
                         for k in (1, 3, 5)} for m in topk}

    out = {
        "generated_by": "scripts/tr4_evaluate_ranking.py",
        "source": args.source, "num_theorems": len(by_thm),
        "num_theorems_with_success": n_with_success,
        "first_success_rank_stats": summary,
        "topk_success_recovery": topk_recovery,
        "per_theorem": per_theorem,
    }
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR4 per-theorem ranking eval", "",
          f"- source: {args.source} | theorems: {len(by_thm)} | with success: {n_with_success}",
          "", "## First-success rank (lower = better)", "",
          "| ordering | n | mean | median |", "|---|---|---|---|"]
    for m in ["original_order", "random_expected"] + models:
        s = summary[m]
        md.append(f"| {m} | {s.get('n')} | {s.get('mean','-')} | {s.get('median','-')} |")
    md += ["", "## Top-k success recovery (frac of theorems-with-success)", "",
           "| ordering | top1 | top3 | top5 |", "|---|---|---|---|"]
    for m in topk:
        t = topk_recovery[m]
        md.append(f"| {m} | {t['top1']} | {t['top3']} | {t['top5']} |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr4-rankeval] theorems_with_success={n_with_success}; "
          f"topk={ {m: topk_recovery[m] for m in topk} }")


if __name__ == "__main__":
    main()
