#!/usr/bin/env python3
"""TR3 Part 10 — export additive training data.

Emits TR3 attribution labels in a TR1-compatible schema (additive; never overwrites
TR1/TR2/SF5 datasets) plus a dataset manifest stitching TR1 + SF5 + TR3 for a future
TR4. Labels: TRUE_RETRIEVAL_DEPTH_DELTA / TRUE_RETRIEVAL_ONLY_DELTA /
TRUE_DEPTH_ONLY_DELTA / BASELINE_DUPLICATE / PROOF_DEPTH_GAP / NO_RETRIEVAL_SIGNAL /
SOURCE_SPECIFIC / OPEN_FLAKE.
"""
from __future__ import annotations

import argparse
import json
import os
import re

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LABEL_TYPE = {
    "TRUE_RETRIEVAL_DEPTH_DELTA": "positive",
    "TRUE_RETRIEVAL_ONLY_DELTA": "positive",
    "TRUE_DEPTH_ONLY_DELTA": "positive",
    "BASELINE_DUPLICATE": "negative",
    "PROOF_DEPTH_GAP": "negative",
    "NO_RETRIEVAL_SIGNAL": "negative",
    "SOURCE_SPECIFIC": "negative",
    "OPEN_FLAKE": "exclude",
    "PRODUCTION_SUBSUMED": "exclude",
    "NEEDS_REVIEW": "exclude",
}


def _p(*a):
    return os.path.join(_REPO, *a)


def _tokens(fn):
    last = fn.split(".")[-1]
    out = []
    for s in re.split(r"[._]", fn) + re.findall(r"[A-Z]?[a-z0-9]+", last):
        s = s.strip().lower()
        if s and s not in out:
            out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attribution", required=True)
    ap.add_argument("--family-analysis", required=True)
    ap.add_argument("--lemma-analysis", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--confirmation",
                    default="project/evolve/experiments/tr3/out/tr3_rc2_confirmation.json")
    args = ap.parse_args()

    attr = json.load(open(_p(args.attribution)))
    conf = {r["full_name"]: r for r in json.load(open(_p(args.confirmation)))["results"]}
    out_dir = _p(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    examples = []
    label_hist = {}
    excluded = {}
    for i, r in enumerate(attr["records"]):
        lab = r["classification"]
        lt = LABEL_TYPE.get(lab, "exclude")
        if lt == "exclude":
            excluded[lab] = excluded.get(lab, 0) + 1
            continue
        fn = r["full_name"]
        c = conf.get(fn, {})
        feats = c.get("features") or {}
        ex = {
            "example_id": f"tr3_{i:03d}",
            "full_name": fn, "file_path": c.get("file_path"),
            "namespace": r.get("namespace"), "theorem_name_tokens": _tokens(fn),
            "goal_text": c.get("goal_text"), "cluster_id": r.get("cluster_id"),
            "source_artifact": "tr3_depth_search",
            "rc2_status": r.get("rc2_status"),
            "label": lab, "label_type": lt,
            "label_confidence": "verified" if r.get("credited") else "strong",
            "winning_program": r.get("winning_program"),
            "winning_family": r.get("winning_family"),
            "winning_depth": r.get("winning_depth"),
            "winning_lemmas": r.get("winning_lemmas", []),
            "credited_true_delta": bool(r.get("credited")),
            "features": {
                "has_set": bool(feats.get("has_set")), "has_iff": bool(feats.get("has_iff")),
                "has_subset": bool(feats.get("has_subset")), "has_eq": bool(feats.get("has_eq")),
                "has_monotone": bool(feats.get("has_monotone")),
                "has_singleton": bool(feats.get("has_singleton")),
                "has_nat": bool(feats.get("has_nat")),
                "has_tofinset": bool(feats.get("has_tofinset")),
                "has_retrieval_depth_win": r.get("classification") == "TRUE_RETRIEVAL_DEPTH_DELTA",
                "has_retrieval_only_win": r.get("classification") == "TRUE_RETRIEVAL_ONLY_DELTA",
                "has_depth_only_win": r.get("classification") == "TRUE_DEPTH_ONLY_DELTA",
            },
        }
        examples.append(ex)
        label_hist[lab] = label_hist.get(lab, 0) + 1

    jsonl_path = os.path.join(out_dir, "tr3_training_examples.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # manifest stitching TR1 + SF5 + TR3
    tr1 = "project/evolve/experiments/tr1/data/tr1_examples.jsonl"
    sf5 = "project/evolve/experiments/sf5/out/sf5_training_examples.jsonl"

    def _count(path):
        fp = _p(path)
        return sum(1 for _ in open(fp)) if os.path.exists(fp) else 0

    manifest = {
        "generated_by": "scripts/tr3_export_training_data.py",
        "additive_only": True,
        "components": {
            "tr1": {"path": tr1, "n": _count(tr1)},
            "sf5": {"path": sf5, "n": _count(sf5)},
            "tr3": {"path": os.path.relpath(jsonl_path, _REPO), "n": len(examples)},
        },
        "combined_n": _count(tr1) + _count(sf5) + len(examples),
        "does_not_modify": [tr1, sf5,
                            "project/evolve/experiments/tr2/data/tr1_plus_tr2_examples.jsonl"],
        "note": "Label spaces differ across components (TR1 action/triage families, SF5 "
                "retrieval gaps, TR3 retrieval-depth deltas); a TR4 multi-head or "
                "channel-tagged model should consume them additively, not merged blindly.",
    }
    json.dump(manifest, open(os.path.join(out_dir, "tr1_tr2_sf5_tr3_dataset_manifest.json"), "w"),
              ensure_ascii=False, indent=2)

    n_pos = sum(1 for e in examples if e["label_type"] == "positive")
    n_neg = sum(1 for e in examples if e["label_type"] == "negative")
    from collections import Counter
    ns_cov = Counter(e["namespace"] for e in examples)
    summary = {
        "generated_by": "scripts/tr3_export_training_data.py",
        "num_examples": len(examples), "label_histogram": label_hist,
        "num_positive": n_pos, "num_negative": n_neg,
        "excluded": excluded, "namespaces_covered": dict(ns_cov.most_common()),
        "new_positives": n_pos, "new_negatives": n_neg,
        "helps_tr4": (n_pos >= 2),
        "tr4_note": ("TR3 adds verified positive retrieval-depth-delta labels"
                     if n_pos >= 2 else
                     "TR3 adds mostly negative depth-gap labels; positives too few to "
                     "scale TR4 training alone — combine with SF5 positives and a fresh sweep."),
    }
    json.dump(summary, open(os.path.join(out_dir, "tr3_training_delta_summary.json"), "w"),
              ensure_ascii=False, indent=2)
    md = ["# TR3 training delta summary", "",
          f"- examples exported (additive): **{len(examples)}**",
          f"- positives: {n_pos} | negatives: {n_neg}",
          f"- label histogram: {label_hist}",
          f"- excluded: {excluded}",
          f"- namespaces: {dict(ns_cov.most_common())}", "",
          f"**Helps TR4 scale training:** {summary['helps_tr4']} — {summary['tr4_note']}",
          "", "Does NOT modify TR1/TR2/SF5 datasets."]
    open(os.path.join(out_dir, "tr3_training_delta_summary.md"), "w").write("\n".join(md) + "\n")

    print(f"[tr3-export] {len(examples)} examples (pos {n_pos}/neg {n_neg}), "
          f"labels={label_hist}, excluded={excluded}")


if __name__ == "__main__":
    main()
