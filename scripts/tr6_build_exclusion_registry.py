#!/usr/bin/env python3
"""TR6 Part 2 — build the exclusion registry.

Collects every theorem full_name already used in TR1/TR2/SF4/SF5/TR3/TR4/TR5
train/eval sets + the RC4A known-wins set, so the fresh-frontier builder can guarantee
zero in-sample leakage. Records per-source counts and cross-source overlap.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def _names_from_jsonl(path, key="full_name"):
    fp = _p(path)
    out = []
    if os.path.exists(fp):
        for l in open(fp):
            if l.strip():
                try:
                    r = json.loads(l)
                    if r.get(key):
                        out.append(r[key])
                except json.JSONDecodeError:
                    pass
    return out


def _names_from_json(path, key="full_name"):
    fp = _p(path)
    if not os.path.exists(fp):
        return []
    d = json.load(open(fp))
    rows = d if isinstance(d, list) else (d.get("results") or d.get("records")
                                          or d.get("theorems") or d.get("queue") or [])
    return [r[key] for r in rows if isinstance(r, dict) and r.get(key)]


SOURCES = [
    ("tr1", "jsonl", "project/evolve/experiments/tr1/data/tr1_examples.jsonl"),
    ("tr2", "jsonl", "project/evolve/experiments/tr2/data/tr2_added_examples.jsonl"),
    ("sf4", "jsonl", "project/evolve/experiments/sf4/cases/rc2_failure_pool.jsonl"),
    ("sf5", "json", "project/evolve/experiments/sf5/cases/sf5_missing_bridge_targets.json"),
    ("tr3", "jsonl", "project/evolve/experiments/tr3/cases/tr3_case_pool.jsonl"),
    ("tr4", "jsonl", "project/evolve/experiments/tr4/data/tr4_program_examples.jsonl"),
    ("tr5_pool", "jsonl", "project/evolve/experiments/tr5/cases/tr5_target_pool.jsonl"),
    ("tr5_examples", "jsonl", "project/evolve/experiments/tr5/data/tr5_program_examples.jsonl"),
    ("rc4a_known_wins", "json",
     "project/evolve/experiments/rc4_candidates/def_unfold_simp/theorem_sets/known_wins.json"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    per_source = {}
    name_to_sources = defaultdict(set)
    for tag, kind, path in SOURCES:
        names = _names_from_jsonl(path) if kind == "jsonl" else _names_from_json(path)
        per_source[tag] = {"path": path, "count_rows": len(names),
                           "unique": len(set(names)), "present": os.path.exists(_p(path))}
        for n in set(names):
            name_to_sources[n].add(tag)

    excluded = sorted(name_to_sources.keys())
    overlap = Counter(len(s) for s in name_to_sources.values())  # how many sources each name in
    multi = {n: sorted(s) for n, s in name_to_sources.items() if len(s) > 1}

    out = {
        "generated_by": "scripts/tr6_build_exclusion_registry.py",
        "num_excluded": len(excluded),
        "per_source": per_source,
        "names_in_n_sources_histogram": {str(k): v for k, v in sorted(overlap.items())},
        "num_in_multiple_sources": len(multi),
        "excluded_full_names": excluded,
    }
    os.makedirs(os.path.dirname(_p(args.out_json)), exist_ok=True)
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR6 exclusion registry", "",
          f"- **{len(excluded)} unique theorem full_names excluded** (zero-leakage guard)",
          f"- names appearing in multiple sources: {len(multi)}", "",
          "| source | path present | rows | unique names |", "|---|---|---|---|"]
    for tag, info in per_source.items():
        md.append(f"| {tag} | {info['present']} | {info['count_rows']} | {info['unique']} |")
    md += ["", f"- overlap histogram (names × #sources): {out['names_in_n_sources_histogram']}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    counts = {k: v["unique"] for k, v in per_source.items()}
    print(f"[tr6-exclusion] {len(excluded)} excluded names; per_source_unique={counts}")


if __name__ == "__main__":
    main()
