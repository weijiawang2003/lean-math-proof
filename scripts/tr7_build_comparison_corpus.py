#!/usr/bin/env python3
"""TR7 Part 2 — build the TR6-vs-RC4R comparison corpus.

Merges the TR6 ranker-search records (137 searched, 18 fresh true deltas) with the RC4R
benchmark theorems (271) into one per-theorem corpus row, recomputing the RC4 static
ordered-union gate (rc4d_gate) for every theorem. Each row records TR6 success/attribution/
winning-program + RC4 static coverage (gate fired? which component? RC4 benchmark result?) so
the downstream analyses can locate exactly where dynamic generalization disappears under the
static wrapper.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
import rc4d_gate as G  # noqa: E402

TR6 = "project/evolve/experiments/tr6"
RC4R = "project/evolve/experiments/rc4_release_candidate"
POLICY = "project/evolve/experiments/rc4_candidates/composition_rc4d/rc4d_composition_policy.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def _j(*a):
    return json.load(open(_p(*a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    ap.add_argument("--out-summary-md", required=True)
    args = ap.parse_args()

    policy = G.load_policy(POLICY)

    # ---- TR6 ----
    tr6_attr = _j(TR6, "out/tr6_attribution.json")
    tr6_rec = {r["full_name"]: r for r in tr6_attr["records"]}
    fresh18 = set(tr6_attr["fresh_true_delta_targets"])
    # statements / file_paths / features from eval batch + fresh pool
    meta = {}
    batch = _j(TR6, "cases/tr6_eval_batch.json")
    for r in (batch.get("theorems", batch) if isinstance(batch, dict) else batch):
        meta.setdefault(r["full_name"], r)
    for l in open(_p(TR6, "cases/tr6_fresh_frontier_pool.jsonl")):
        r = json.loads(l)
        meta.setdefault(r["full_name"], r)
    tr6_conf = {r["full_name"]: r for r in _j(TR6, "out/tr6_rc2_confirmation.json").get("results", [])}

    # ---- RC4R ----
    rc2_bench = {r["full_name"]: r for r in _j(RC4R, "out/rc2_benchmark_results.json")["results"]}
    rc4_bench = {r["full_name"]: r for r in _j(RC4R, "out/rc4_benchmark_results.json")["results"]}
    man = _j(RC4R, "theorem_sets/benchmark_manifest.json")
    rc4r_entry, rc4r_sets = {}, {}
    for setname, rel in man["set_files"].items():
        for e in _j(rel):
            rc4r_entry.setdefault(e["full_name"], e)
            rc4r_sets.setdefault(e["full_name"], []).append(setname)

    names = set(tr6_rec) | set(rc4_bench) | set(rc2_bench)

    rows = []
    for fn in sorted(names):
        t6 = tr6_rec.get(fn)
        ent = rc4r_entry.get(fn) or meta.get(fn) or {}
        ns = ent.get("namespace") or fn.split(".")[0]
        goal = ent.get("goal_text") or ent.get("statement_text") or (meta.get(fn) or {}).get("statement_text")
        fp = ent.get("file_path") or (meta.get(fn) or {}).get("file_path")
        in_tr6 = t6 is not None
        in_rc4r = fn in rc4r_sets
        fires, em = G.gate_fires(policy, ns, goal, fn)
        comps = G.components_firing(em)
        wp = (t6 or {}).get("winning_program") or {}
        rc4_res = rc4_bench.get(fn, {}).get("status") if fn in rc4_bench else (
            "not_applicable" if not in_rc4r else "not_run")
        row = {
            "full_name": fn, "file_path": fp, "namespace": G.namespace_of(ns, fn),
            "source": "both" if (in_tr6 and in_rc4r) else ("TR6" if in_tr6 else "RC4R"),
            "in_tr6": in_tr6, "in_rc4r": in_rc4r,
            "tr6_rc2_status": (t6 or {}).get("rc2_status") or (tr6_conf.get(fn) or {}).get("classification"),
            "rc4r_rc2_status": rc2_bench.get(fn, {}).get("status"),
            "tr6_success": bool(t6 and t6.get("classification") == "FRESH_TRUE_DELTA"),
            "tr6_attribution": (t6 or {}).get("classification"),
            "tr6_winning_program": wp.get("tactic"),
            "tr6_winning_family": wp.get("family"),
            "tr6_winning_lemma": (wp.get("used_lemmas") or [None])[0],
            "tr6_winning_lemmas": wp.get("used_lemmas") or [],
            "tr6_first_success_rank": (t6 or {}).get("first_success_rank"),
            "tr6_rc4a_evidence": (t6 or {}).get("rc4a_evidence"),
            "tr6_rc4b_evidence": (t6 or {}).get("rc4b_evidence"),
            "tr6_rc4c_evidence": (t6 or {}).get("rc4c_evidence"),
            "rc4_static_gate_fired": fires,
            "rc4_static_components": comps,
            "rc4_static_component": comps[0] if comps else None,
            "rc4_static_tactics": G.tactics_of(em),
            "rc4_static_result": rc4_res,
            "rc4r_sets": rc4r_sets.get(fn, []),
            "is_tr6_fresh_win": fn in fresh18,
            "features": ent.get("features") or (meta.get(fn) or {}).get("features") or {},
        }
        rows.append(row)

    with open(_p(args.out_jsonl), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # summary
    tr6_fresh = [r for r in rows if r["is_tr6_fresh_win"]]
    tr6_fail = [r for r in rows if r["in_tr6"] and r["tr6_attribution"] == "NO_WIN_UNDER_BUDGET"]
    rc4r_fresh = [r for r in rows if "fresh_out_of_sample_frontier" in r["rc4r_sets"]]
    rc4r_known = [r for r in rows if "rc4_known_wins" in r["rc4r_sets"]]
    summary = {
        "generated_by": "scripts/tr7_build_comparison_corpus.py",
        "num_rows": len(rows),
        "source_split": dict(Counter(r["source"] for r in rows)),
        "tr6_fresh_wins": len(tr6_fresh),
        "tr6_fresh_win_in_rc4r_fresh": sum(1 for r in tr6_fresh if "fresh_out_of_sample_frontier" in r["rc4r_sets"]),
        "tr6_fresh_win_in_rc4r_known": sum(1 for r in tr6_fresh if "rc4_known_wins" in r["rc4r_sets"]),
        "tr6_fresh_win_static_gate_fires": sum(1 for r in tr6_fresh if r["rc4_static_gate_fired"]),
        "tr6_no_win_failures": len(tr6_fail),
        "rc4r_fresh_cases": len(rc4r_fresh),
        "rc4r_fresh_gate_firing": sum(1 for r in rc4r_fresh if r["rc4_static_gate_fired"]),
        "rc4r_known_wins": len(rc4r_known),
        "tr6_fresh_win_namespaces": dict(Counter(r["namespace"] for r in tr6_fresh)),
        "tr6_fresh_win_families": dict(Counter(r["tr6_winning_family"] for r in tr6_fresh)),
    }
    json.dump(summary, open(_p(args.out_summary_json), "w"), ensure_ascii=False, indent=2)
    md = ["# TR7 comparison corpus", "",
          f"- rows: {summary['num_rows']} | source split: {summary['source_split']}",
          f"- TR6 fresh wins: {summary['tr6_fresh_wins']}",
          f"  - in RC4R fresh frontier: **{summary['tr6_fresh_win_in_rc4r_fresh']}** "
          f"(this is the cohort artifact)",
          f"  - in RC4R known wins: {summary['tr6_fresh_win_in_rc4r_known']}",
          f"  - RC4 static gate fires on them: {summary['tr6_fresh_win_static_gate_fires']}",
          f"- TR6 no-win failures: {summary['tr6_no_win_failures']}",
          f"- RC4R fresh cases: {summary['rc4r_fresh_cases']} (gate-firing {summary['rc4r_fresh_gate_firing']})",
          f"- RC4R known wins: {summary['rc4r_known_wins']}",
          f"- TR6 fresh-win namespaces: {summary['tr6_fresh_win_namespaces']}",
          f"- TR6 fresh-win families: {summary['tr6_fresh_win_families']}"]
    open(_p(args.out_summary_md), "w").write("\n".join(md) + "\n")
    print(f"[tr7-corpus] rows={len(rows)} tr6_fresh={len(tr6_fresh)} "
          f"in_rc4r_fresh={summary['tr6_fresh_win_in_rc4r_fresh']} "
          f"in_rc4r_known={summary['tr6_fresh_win_in_rc4r_known']}")
    print(f"[tr7-corpus] rc4r_fresh={len(rc4r_fresh)} (firing {summary['rc4r_fresh_gate_firing']})")


if __name__ == "__main__":
    main()
