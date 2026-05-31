#!/usr/bin/env python3
"""SF5 Part 9 — export additive training examples.

Emits SF5 retrieval-attribution labels in a schema compatible with TR1
(tr1_examples.jsonl) so a future TR3/TR4 retrieval-aware router can consume them.
ADDITIVE ONLY: writes a fresh sf5_training_examples.jsonl; never touches the TR1/TR2
datasets.

Exported labels (per spec):
  EXISTING_LEMMA_GAP        positive  — an existing lemma closes it (router should retrieve)
  RETRIEVAL_ROUTING_GAP     positive  — lemma reachable; routing should try library search
  TRUE_MISSING_BRIDGE_LEMMA negative  — genuine synthesis target (SF6)
  PROOF_DEPTH_GAP           negative  — needs deeper multi-step search
  NO_RETRIEVAL_SIGNAL       negative  — no retrieval direction
Guard classes (PRODUCTION_SUBSUMED / BASELINE_DUPLICATE) are NOT exported.
"""
from __future__ import annotations

import argparse
import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPORT_LABELS = {
    "EXISTING_LEMMA_GAP": "positive",
    "RETRIEVAL_ROUTING_GAP": "positive",
    "TRUE_MISSING_BRIDGE_LEMMA": "negative",
    "PROOF_DEPTH_GAP": "negative",
    "NO_RETRIEVAL_SIGNAL": "negative",
}


def _p(*a):
    return os.path.join(_REPO, *a)


def _tokens(fn):
    import re
    last = fn.split(".")[-1]
    parts = re.split(r"[._]", fn)
    cam = re.findall(r"[A-Z]?[a-z0-9]+", last)
    out = []
    for s in parts + cam:
        s = s.strip().lower()
        if s and s not in out:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--cluster-analysis", required=True)
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    ap.add_argument("--targets",
                    default="project/evolve/experiments/sf5/cases/"
                            "sf5_missing_bridge_targets.json")
    args = ap.parse_args()

    attr = json.load(open(_p(args.attribution)))
    targets = {t["full_name"]: t for t in json.load(open(_p(args.targets)))}
    cluster_rec = {c["cluster_id"]: c
                   for c in json.load(open(_p(args.cluster_analysis)))["clusters"]}

    examples = []
    label_hist = {}
    skipped = {}
    for i, r in enumerate(attr["records"]):
        cls = r["classification"]
        if cls not in EXPORT_LABELS:
            skipped[cls] = skipped.get(cls, 0) + 1
            continue
        fn = r["full_name"]
        tg = targets.get(fn, {})
        feat = tg.get("features_extended") or tg.get("features", {})
        crec = cluster_rec.get(r.get("cluster_id"), {})
        ex = {
            "example_id": f"sf5_{i:03d}",
            "full_name": fn,
            "file_path": tg.get("file_path"),
            "namespace": tg.get("namespace"),
            "theorem_name_tokens": _tokens(fn),
            "goal_text": tg.get("goal_text"),
            "last_error": tg.get("last_error"),
            "source_artifact": "sf5_retrieval",
            "rc2_status": r.get("rc2_status"),
            "cluster_id": r.get("cluster_id"),
            "label": cls,
            "label_type": EXPORT_LABELS[cls],
            "label_confidence": "verified" if r.get("win_over_literal_rc2") else "strong",
            "winning_lemma": r.get("winning_lemma"),
            "win_over_literal_rc2": r.get("win_over_literal_rc2", False),
            "cluster_recommendation": crec.get("recommendation"),
            "features": {
                "has_set": bool(feat.get("has_set")),
                "has_iff": bool(feat.get("has_iff")),
                "has_subset": bool(feat.get("has_subset")),
                "has_ssubset": bool(feat.get("has_ssubset")),
                "has_monotone": bool(feat.get("has_monotone")),
                "has_strictmono": bool(feat.get("has_strictmono")),
                "has_singleton": bool(feat.get("has_singleton")),
                "has_insert": bool(feat.get("has_insert")),
                "has_compl": bool(feat.get("has_compl")),
                "has_pair": bool(feat.get("has_pair")),
                "has_ite": bool(feat.get("has_ite")),
                "has_retrieval_win": bool(r.get("win_over_literal_rc2")),
                "num_named_lemma_wins": r.get("num_named_lemma_wins", 0),
            },
        }
        examples.append(ex)
        label_hist[cls] = label_hist.get(cls, 0) + 1

    os.makedirs(os.path.dirname(_p(args.out_jsonl)), exist_ok=True)
    with open(_p(args.out_jsonl), "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    summary = {
        "generated_by": "scripts/sf5_export_training_examples.py",
        "additive_only": True,
        "does_not_modify": ["project/evolve/experiments/tr1/data/tr1_examples.jsonl",
                            "project/evolve/experiments/tr2/data/tr1_plus_tr2_examples.jsonl"],
        "num_examples": len(examples),
        "label_histogram": label_hist,
        "label_types": {k: EXPORT_LABELS[k] for k in label_hist},
        "skipped_guard_classes": skipped,
        "tr3_usage": ("Merge additively with tr1_examples.jsonl as a retrieval-aware "
                      "label channel; positive labels (EXISTING_LEMMA_GAP / "
                      "RETRIEVAL_ROUTING_GAP) train a router to fire library-search / "
                      "lemma-retrieval actions; negatives keep verified-label discipline."),
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)

    md = ["# SF5 training delta summary", "",
          f"- examples exported (additive): **{len(examples)}**",
          f"- label histogram: {label_hist}",
          f"- skipped guard classes: {skipped}", "",
          "Does NOT modify TR1/TR2 datasets. ", "",
          "## How TR3/TR4 should use these", "", summary["tr3_usage"]]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")

    print(f"[sf5-export] {len(examples)} examples, labels={label_hist}, skipped={skipped}")


if __name__ == "__main__":
    main()
