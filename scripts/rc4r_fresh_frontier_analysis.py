#!/usr/bin/env python3
"""RC4R Part 10 — fresh out-of-sample frontier analysis.

Restricts the RC2-vs-RC4 comparison to the fresh_out_of_sample_frontier set and reports whether
RC4 improves beyond known-win replay on genuinely fresh theorems. Classifies:
  FRESH_FRONTIER_DELTA_FOUND        >=1 fresh new win, 0 fresh regressions.
  NO_FRESH_DELTA_BUT_SAFE           0 fresh new wins, 0 fresh regressions (RC4 safe on fresh).
  FRESH_FRONTIER_REGRESSION         >=1 fresh regression.
  INCONCLUSIVE_TOO_FEW_FRESH_CASES  fewer than MIN_FRESH analyzable fresh cases.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_FRESH = 25
FRESH_SET = "fresh_out_of_sample_frontier"


def _p(*a):
    return os.path.join(_REPO, *a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rc2", required=True)
    ap.add_argument("--rc4", required=True)
    ap.add_argument("--comparison", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-md", required=True)
    args = ap.parse_args()

    comp = json.load(open(_p(args.comparison)))
    fresh_rows = [r for r in comp["rows"] if FRESH_SET in r["sets"]]
    n = len(fresh_rows)
    analyzable = [r for r in fresh_rows if r["classification"] not in ("FLAKE", "PATH_ERROR")]
    rc2_solved = sum(1 for r in fresh_rows if r["rc2_status"] == "solved")
    rc4_solved = sum(1 for r in fresh_rows if r["rc4_status"] == "solved")
    new_wins = [r for r in fresh_rows if r["classification"] == "RC4_NEW_WIN"]
    regr = [r for r in fresh_rows if r["classification"] == "RC4_REGRESSION"]
    by_comp = Counter(r["component"] for r in new_wins)
    by_ns = Counter(r["namespace"] for r in new_wins)

    if len(analyzable) < MIN_FRESH:
        verdict = "INCONCLUSIVE_TOO_FEW_FRESH_CASES"
    elif regr:
        verdict = "FRESH_FRONTIER_REGRESSION"
    elif new_wins:
        verdict = "FRESH_FRONTIER_DELTA_FOUND"
    else:
        verdict = "NO_FRESH_DELTA_BUT_SAFE"

    out = {"generated_by": "scripts/rc4r_fresh_frontier_analysis.py",
           "fresh_total": n, "analyzable": len(analyzable),
           "rc2_solved": rc2_solved, "rc4_solved": rc4_solved,
           "fresh_new_wins": len(new_wins), "fresh_regressions": len(regr),
           "fresh_net_delta": len(new_wins) - len(regr),
           "new_wins_by_component": dict(by_comp), "new_wins_by_namespace": dict(by_ns),
           "improves_beyond_known_replay": len(new_wins) > 0,
           "new_win_targets": [r["full_name"] for r in new_wins],
           "regression_targets": [r["full_name"] for r in regr],
           "verdict": verdict}
    json.dump(out, open(_p(args.out_json), "w"), ensure_ascii=False, indent=2)
    md = ["# RC4 fresh out-of-sample frontier analysis", "",
          f"- fresh theorems: {n} (analyzable {len(analyzable)})",
          f"- RC2 solved: {rc2_solved} | RC4 solved: {rc4_solved}",
          f"- **fresh new wins: {len(new_wins)}** | fresh regressions: {len(regr)} | "
          f"net: {out['fresh_net_delta']}",
          f"- by component: {dict(by_comp)} | by namespace: {dict(by_ns)}",
          f"- improves beyond known-win replay: {out['improves_beyond_known_replay']}",
          f"- new wins: {out['new_win_targets']}",
          f"- **verdict: {verdict}**"]
    open(_p(args.out_md), "w").write("\n".join(md) + "\n")
    print(f"[rc4r-fresh] fresh={n} analyzable={len(analyzable)} rc2={rc2_solved} rc4={rc4_solved} "
          f"new={len(new_wins)} regr={len(regr)} verdict={verdict}")
    print(f"[rc4r-fresh] new_wins={out['new_win_targets']} by_comp={dict(by_comp)}")


if __name__ == "__main__":
    main()
