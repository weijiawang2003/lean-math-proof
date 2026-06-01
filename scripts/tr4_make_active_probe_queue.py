#!/usr/bin/env python3
"""TR4 Part 8 — active probing queue for a future (TR5) live search.

Ranks the existing TR3 program rows by the leakage-free OOF HGB score and emits a
prioritized per-theorem queue: top recommended programs + a selection_reason
(high_score likely win / high_uncertainty useful-label / underrepresented_namespace /
candidate_family_validation for RC4B Set.disjoint_left & RC4C d2_simp_aesop). Does NOT
run live probes.
"""
from __future__ import annotations

import argparse
import json
import os
import numpy as np
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _rows(path):
    return [json.loads(l) for l in open(_p(path)) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--model", default="hgb")
    ap.add_argument("--topn", type=int, default=5)
    args = ap.parse_args()

    rows = _rows(args.examples)
    oof = np.load(os.path.join(_p(args.model_dir), "oof_scores.npz"))
    sc = oof[args.model]
    success = np.array([r["label_success"] for r in rows])

    # positives per namespace (to flag underrepresented)
    pos_ns = Counter(r["namespace"] for r in rows if r["label_success"])
    underrep = {ns for ns in {r["namespace"] for r in rows} if pos_ns.get(ns, 0) == 0}

    # group tr3 rows by theorem; emphasise theorems that had NO win (the open frontier)
    by_thm = {}
    for i, r in enumerate(rows):
        if r["source"] != "tr3":
            continue
        by_thm.setdefault(r["full_name"], []).append(i)

    queue = []
    for fn, idxs in by_thm.items():
        nsucc = int(success[idxs].sum())
        ns = rows[idxs[0]]["namespace"]
        s = np.where(np.isnan(sc[idxs]), 0.0, sc[idxs])
        order = [idxs[j] for j in np.argsort(-s)]
        topk = order[: args.topn]
        recs = [{"tactic": rows[i]["tactic"], "score": round(float(sc[i]), 4),
                 "family": rows[i]["program_family"],
                 "used_lemma": (rows[i]["used_lemmas"] or [None])[0],
                 "reason": f"oof_{args.model}_score",
                 "already_solved": bool(success[i])} for i in topk]
        top_score = recs[0]["score"] if recs else 0.0
        # expected value ~ top score; uncertainty ~ closeness to 0.5
        ev = round(top_score, 4)
        unc = round(1.0 - abs(top_score - 0.5) * 2, 4)
        # candidate-family validation support
        fams = {rows[i]["program_family"] for i in topk}
        lemmas = {(rows[i]["used_lemmas"] or [None])[0] for i in topk}
        if nsucc > 0:
            reason = "already_has_win"
        elif "Set.disjoint_left" in lemmas:
            reason = "candidate_family_validation"  # RC4B
        elif "d2_simp_aesop" in fams:
            reason = "candidate_family_validation"  # RC4C
        elif ns in underrep:
            reason = "underrepresented_namespace"
        elif top_score >= 0.5:
            reason = "high_score"
        else:
            reason = "high_uncertainty"
        queue.append({"full_name": fn, "namespace": ns, "num_programs": len(idxs),
                      "had_win": nsucc > 0, "recommended_programs": recs,
                      "expected_value": ev, "uncertainty": unc,
                      "selection_reason": reason})

    # order queue: open theorems first, by expected_value desc
    queue.sort(key=lambda q: (q["had_win"], -q["expected_value"]))
    cat = Counter(q["selection_reason"] for q in queue)

    out = {"generated_by": "scripts/tr4_make_active_probe_queue.py",
           "model": args.model, "num_theorems": len(queue),
           "category_histogram": dict(cat),
           "note": "Priorities from leakage-free OOF scores over EXISTING TR3 programs; "
                   "no live probes run. RC4B = Set.disjoint_left support, RC4C = "
                   "d2_simp_aesop support.",
           "queue": queue}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)

    md = ["# TR4 active probe queue", "",
          f"- theorems: {len(queue)} | categories: {dict(cat)}",
          "- (no live probes run; priorities from leakage-free OOF HGB scores)", "",
          "## Top 15 (open theorems first, by expected value)", "",
          "| theorem | ns | EV | reason | top program |", "|---|---|---|---|---|"]
    for q in queue[:15]:
        tp = q["recommended_programs"][0]["tactic"] if q["recommended_programs"] else ""
        md.append(f"| `{q['full_name']}` | {q['namespace']} | {q['expected_value']} | "
                  f"{q['selection_reason']} | `{tp}` |")
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[tr4-queue] {len(queue)} theorems; categories={dict(cat)}")


if __name__ == "__main__":
    main()
