#!/usr/bin/env python3
"""TR4 Part 9 (optional) — feature-group ablation.

Trains the HGB ranker on each feature block alone vs all blocks, with leakage-free
GroupKFold-by-theorem OOF PR-AUC, to see whether the model needs retrieval/program
INTERACTIONS (all-features ≫ any single block) or just memorizes a family/name cue
(a single block ≈ all).
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _oof_pr_auc(X, y, groups):
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=min(5, len(set(groups))))
    for tr, te in gkf.split(X, y, groups):
        if len(set(y[tr])) < 2:
            oof[te] = float(y[tr].mean()); continue
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=4)
        m.fit(X[tr].toarray(), y[tr])
        oof[te] = m.predict_proba(X[te].toarray())[:, 1]
    return round(float(average_precision_score(y, oof)), 4) if len(set(y)) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--metadata", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(_p(args.examples)) if l.strip()]
    X = sp.load_npz(_p(args.features)).tocsr()
    y = np.load(_p(args.features).replace(".npz", "_labels.npz"))["y_success"].astype(int)
    meta = json.load(open(_p(args.metadata)))
    ranges = meta["block_col_ranges"]
    groups = np.array([r["full_name"] for r in rows])

    GROUP_MAP = {
        "theorem_name_only": ["name_char", "name_tok"],
        "goal_only": ["goal_word"],
        "lemma_name_only": ["lemma_tok"],
        "program_tactic_only": ["tactic_tok"],
        "symbolic_and_interactions_only": ["dict"],
        "all_features": list(ranges.keys()),
    }
    results = {}
    for label, blocks in GROUP_MAP.items():
        cols = []
        for b in blocks:
            a, z = ranges[b]
            cols.extend(range(a, z))
        Xs = X[:, cols]
        results[label] = {"num_cols": len(cols), "oof_pr_auc": _oof_pr_auc(Xs, y, groups)}

    allv = results["all_features"]["oof_pr_auc"]
    best_single = max((v["oof_pr_auc"] or 0) for k, v in results.items() if k != "all_features")
    interaction_gain = round((allv or 0) - best_single, 4)
    interpretation = (
        "all-features materially exceeds the best single block "
        f"(+{interaction_gain} PR-AUC) → the ranker uses retrieval/program INTERACTIONS, "
        "not a single memorized cue."
        if interaction_gain > 0.05 else
        f"all-features ≈ best single block (Δ {interaction_gain}) → the signal is "
        "concentrated in one feature group (likely family/name cue), so the model is "
        "largely memorizing that cue rather than learning interactions.")

    out = {"generated_by": "scripts/tr4_ablation_study.py",
           "results": results, "all_features_pr_auc": allv,
           "best_single_block_pr_auc": round(best_single, 4),
           "interaction_gain": interaction_gain, "interpretation": interpretation}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR4 ablation study (OOF PR-AUC, group=theorem)", "",
          "| feature group | cols | OOF PR-AUC |", "|---|---|---|"]
    for k, v in results.items():
        md.append(f"| {k} | {v['num_cols']} | {v['oof_pr_auc']} |")
    md += ["", f"- all-features {allv} vs best-single {round(best_single,4)} "
           f"(interaction gain {interaction_gain})", "", interpretation]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr4-ablation] all={allv} best_single={round(best_single,4)} gain={interaction_gain}")


if __name__ == "__main__":
    main()
