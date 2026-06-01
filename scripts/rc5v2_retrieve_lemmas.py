#!/usr/bin/env python3
"""RC5V2 Part 7 — retrieve top-20 lemmas for the dynamic-eligible cases.

Shells out to the validated TR6 retrieval (TR3∪SF5 index). Summarizes coverage, best score,
namespace match, unknown-name risk.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    args = ap.parse_args()

    cmd = [sys.executable, _p("scripts/tr6_retrieve_lemmas.py"),
           "--confirmation", _p(args.eligible), "--out-json", _p(args.out_json),
           "--out-md", _p(args.out_md.replace(".md", "_raw.md")), "--top-k", str(args.top_k)]
    print(f"[rc5v2-retrieve] retrieving top-{args.top_k} for eligible cases ...", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if not os.path.exists(_p(args.out_json)):
        print(r.stdout[-1500:]); print(r.stderr[-1500:]); raise SystemExit("retrieval failed")

    retr = json.load(open(_p(args.out_json)))
    elig = {e["full_name"]: e for e in json.load(open(_p(args.eligible)))["results"]}
    rows = retr.get("results", [])
    cov = sum(1 for r in rows if r.get("top_lemmas"))
    best = [r.get("best_score", 0.0) for r in rows if r.get("top_lemmas")]
    nsmatch = sum(1 for r in rows if r.get("top_lemmas")
                  and any((L.get("lemma", "").split(".")[0] == elig.get(r["target"], {}).get("namespace"))
                          for L in r["top_lemmas"][:5]))
    summary = {"generated_by": "scripts/rc5v2_retrieve_lemmas.py",
               "num_targets": len(rows), "coverage": cov,
               "coverage_rate": round(cov / (len(rows) or 1), 3),
               "best_score_mean": round(sum(best) / (len(best) or 1), 4),
               "namespace_match_top5": nsmatch}
    json.dump(summary, open(_p(args.out_json.replace(".json", "_summary.json")), "w"),
              ensure_ascii=False, indent=2)
    md = ["# RC5V2 retrieval", "",
          f"- targets: {len(rows)} | coverage: {cov} ({summary['coverage_rate']:.0%})",
          f"- best-score mean: {summary['best_score_mean']} | namespace match (top-5): {nsmatch}"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc5v2-retrieve] targets={len(rows)} coverage={cov} best_mean={summary['best_score_mean']} nsmatch={nsmatch}")


if __name__ == "__main__":
    main()
