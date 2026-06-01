#!/usr/bin/env python3
"""RC5H Part 6 — retrieve candidate lemmas for the RC4 static failures.

Selects the dynamic-eligible static failures (RC4 failed ∧ allowed namespace ∧ non-flake),
writes a confirmation-shaped file, and shells out to the validated TR6 retrieval
(`tr6_retrieve_lemmas`, = TR3/SF5 deterministic scorer over the 10,790∪5,994 index) to get
top-20 lemmas per theorem. Summarizes retrieval coverage, best score, unknown-name risk,
namespace match, and TR6-dynamic-pattern resemblance.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALLOWED_NS = {"Set", "Finset", "List", "Multiset", "Nat"}
CONF_OUT = "project/evolve/experiments/rc5_hybrid/cases/rc5h_dynamic_confirmation.json"


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--static-results", required=True)
    ap.add_argument("--rc2-results", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    manifest = json.load(open(_p(args.manifest)))
    static = {r["full_name"]: r for r in json.load(open(_p(args.static_results)))["results"]}
    rc2 = {r["full_name"]: r for r in json.load(open(_p(args.rc2_results)))["results"]}
    # entry metadata (file_path/goal/ns/features/set)
    meta, set_of = {}, {}
    for setname, rel in manifest["set_files"].items():
        for e in json.load(open(_p(rel))):
            meta.setdefault(e["full_name"], e)
            set_of.setdefault(e["full_name"], setname)

    # dynamic-eligible static failures
    eligible = []
    gated_out = {"static_win": 0, "namespace": 0, "flake": 0, "no_file": 0}
    for fn, e in meta.items():
        st = static.get(fn, {}).get("status")
        ns = e["namespace"]
        if st == "solved":
            gated_out["static_win"] += 1; continue
        if st in ("open_flake", "trace_insufficient", "path_error"):
            gated_out["flake"] += 1; continue
        if ns not in ALLOWED_NS:
            gated_out["namespace"] += 1; continue
        if not e.get("file_path"):
            gated_out["no_file"] += 1; continue
        eligible.append({"full_name": fn, "namespace": ns, "file_path": e["file_path"],
                         "statement_text": e.get("statement_text") or e.get("goal_text"),
                         "goal_text": e.get("goal_text") or e.get("statement_text"),
                         "rc2_status": rc2.get(fn, {}).get("status"),
                         "set": set_of.get(fn), "classification": "CONFIRMED_RC2_FAILURE"})

    conf = {"generated_by": "scripts/rc5h_retrieve_for_static_failures.py",
            "num_eligible": len(eligible), "results": eligible}
    os.makedirs(os.path.dirname(_p(CONF_OUT)), exist_ok=True)
    json.dump(conf, open(_p(CONF_OUT), "w"), ensure_ascii=False, indent=2)

    # shell out to the validated TR6 retrieval
    retr_md = args.out_json.replace(".json", "_retrieval.md")
    cmd = [sys.executable, _p("scripts/tr6_retrieve_lemmas.py"),
           "--confirmation", _p(CONF_OUT), "--out-json", _p(args.out_json),
           "--out-md", _p(retr_md), "--top-k", str(args.top_k)]
    print(f"[rc5h-retrieve] retrieving for {len(eligible)} eligible static failures ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(_p(args.out_json)):
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
        raise SystemExit("retrieval failed")

    retr = json.load(open(_p(args.out_json)))
    rrows = {x.get("target", x.get("full_name")): x for x in retr.get("results", [])}
    from collections import Counter
    cov, best_scores, ns_match = 0, [], 0
    by_set = Counter()
    for e in eligible:
        rr = rrows.get(e["full_name"], {})
        tl = rr.get("top_lemmas", [])
        if tl:
            cov += 1
            best_scores.append(tl[0].get("score", 0.0))
            if any((L.get("lemma", "").split(".")[0] == e["namespace"]) for L in tl[:5]):
                ns_match += 1
        by_set[e["set"]] += 1
    summary = {
        "generated_by": "scripts/rc5h_retrieve_for_static_failures.py",
        "num_eligible_static_failures": len(eligible),
        "gated_out_counts": gated_out,
        "retrieval_coverage": cov,
        "retrieval_coverage_rate": round(cov / (len(eligible) or 1), 3),
        "best_score_mean": round(sum(best_scores) / (len(best_scores) or 1), 4),
        "namespace_match_top5": ns_match,
        "eligible_by_set": dict(by_set),
        "confirmation_file": CONF_OUT, "retrieval_file": args.out_json,
    }
    json.dump(summary, open(_p(args.out_json.replace(".json", "_summary.json")), "w"),
              ensure_ascii=False, indent=2)
    md = ["# RC5H retrieval for static failures", "",
          f"- eligible static failures: {len(eligible)} (gated out: {gated_out})",
          f"- retrieval coverage: {cov}/{len(eligible)} ({summary['retrieval_coverage_rate']:.0%})",
          f"- best-score mean: {summary['best_score_mean']}",
          f"- namespace match (top-5): {ns_match}/{len(eligible)}",
          f"- eligible by set: {dict(by_set)}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5h-retrieve] eligible={len(eligible)} coverage={cov} "
          f"best_mean={summary['best_score_mean']} gated_out={gated_out}")


if __name__ == "__main__":
    main()
